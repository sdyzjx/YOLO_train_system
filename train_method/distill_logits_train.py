#!/usr/bin/env python
"""
Logits-only knowledge distillation trainer.

Usage example:
    python distill_logits_train.py \
        student_model=yolov8n-obb.pt \
        teacher_model=runs/obb/teacher/weights/best.pt \
        data=datasets/cfg/ttpla.yaml \
        distill_logits_alpha=1.0 \
        temperature=2.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.utils import LOGGER, RANK


def _resolve_custom_ultralytics():
    """Ensure local ./modules/ultralytics is importable before importing Ultralytics."""
    project_root = Path(__file__).resolve().parents[1]
    candidate = project_root / "modules"
    custom_pkg = candidate / "ultralytics"
    if custom_pkg.exists() and (custom_pkg / "__init__.py").exists():
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_resolve_custom_ultralytics()


class LogitsKDTrainer(OBBTrainer):
    """Trainer that only applies logits KL distillation against a frozen teacher."""

    def __init__(self, overrides=None, _callbacks=None):
        overrides = overrides.copy() if overrides else {}
        self.teacher_model_path = overrides.pop("teacher_model", None)
        self.teacher_imgsz = int(overrides.pop("teacher_imgsz", 1024))
        self.distill_logits_alpha = float(overrides.pop("distill_logits_alpha", 1.0))
        self.temperature = float(overrides.pop("temperature", 2.0))
        self.logits_debug = bool(overrides.pop("logits_debug", False))

        student_model = overrides.pop("student_model", None)
        if student_model and "model" not in overrides:
            overrides["model"] = student_model

        if self.teacher_model_path is None:
            raise ValueError("必须提供 teacher_model 用于蒸馏。")
        self.teacher_model_path = self._resolve_teacher_weights(Path(self.teacher_model_path))

        super().__init__(overrides=overrides, _callbacks=_callbacks)

        if RANK in [-1, 0]:
            print(f"⏳ 正在加载教师模型: {self.teacher_model_path}")

        teacher_yolo = YOLO(str(self.teacher_model_path))
        self.teacher_model = teacher_yolo.model.to(self.device)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self.teacher_head = getattr(self.teacher_model, "head", None) or self.teacher_model.model[-1]
        self.reg_channel_count = self.teacher_head.reg_max * 4

        self.distill_logits_loss = nn.KLDivLoss(reduction="batchmean")
        self._sync_teacher_head()

        if RANK in [-1, 0]:
            print("✅ 教师模型加载完毕，启用 logits 蒸馏")
            print(f"   - 教师输入尺寸: {self.teacher_imgsz}")
            print(f"   - Logits 蒸馏系数: {self.distill_logits_alpha}, 温度: {self.temperature}")

    def _resolve_teacher_weights(self, raw_path: Path) -> Path:
        raw_path = raw_path.expanduser().resolve()
        if raw_path.is_file() and raw_path.suffix == ".pt":
            return raw_path
        weights_dir = raw_path.parent if raw_path.is_file() else raw_path
        for name in ("best.pt", "last.pt"):
            candidate = weights_dir / name
            if candidate.exists():
                return candidate
            candidate = weights_dir / "weights" / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"无法在 {raw_path} 寻找可用教师权重 (.pt)")

    def _sync_teacher_head(self):
        """Align teacher head metadata with current dataset."""
        detect = self.teacher_model.model[-1]
        detect.nc = self.data["nc"]
        detect.names = self.data.get("names", {})
        detect.args = getattr(self.model, "args", None) or self.args
        self.teacher_model.nc = detect.nc
        self.teacher_model.names = detect.names

    def _setup_train(self):
        super()._setup_train()
        if RANK in [-1, 0]:
            print("✅ 学生模型构建完成，准备包裹 loss 以添加 logits 蒸馏")
        self._wrap_model_loss()

    def _wrap_model_loss(self):
        base_loss_fn = self.model.loss
        trainer = self

        def loss_with_distill(model_self, batch, preds=None):
            if preds is None:
                preds = model_self.forward(batch["img"])
            loss_original, loss_items = base_loss_fn(batch, preds)
            return trainer._augment_with_distill(preds, batch, loss_original, loss_items)

        self.model.loss = MethodType(loss_with_distill, self.model)

    def _forward_teacher(self, batch):
        with torch.no_grad():
            teacher_input_img = F.interpolate(
                batch["img"],
                size=(self.teacher_imgsz, self.teacher_imgsz),
                mode="bilinear",
                align_corners=False,
            )
            head_state = self.teacher_head.training
            self.teacher_head.training = True
            try:
                outputs = self.teacher_model(teacher_input_img)
            finally:
                self.teacher_head.training = head_state
            if isinstance(outputs, (list, tuple)):
                return outputs[0]
            return outputs

    def _stack_head_outputs(self, outputs):
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        return torch.cat([m.view(m.shape[0], m.shape[1], -1) for m in outputs], dim=2)

    def _augment_with_distill(self, preds, batch, loss_original, loss_items):
        if not self.distill_logits_alpha:
            return loss_original, loss_items

        teacher_maps = self._forward_teacher(batch)
        student_maps = preds[0] if isinstance(preds, (list, tuple)) else preds

        student_tensor = self._stack_head_outputs(student_maps)
        teacher_tensor = self._stack_head_outputs(teacher_maps)

        student_head = getattr(self.model, "head", None) or self.model.model[-1]
        student_reg_channels = student_head.reg_max * 4
        s_cls_logits = student_tensor[:, student_reg_channels:, :]
        t_cls_logits = teacher_tensor[:, self.reg_channel_count :, :]

        loss_distill_logits = torch.zeros(1, device=self.device)
        if s_cls_logits.shape[1] > 0 and t_cls_logits.shape[1] > 0:
            if s_cls_logits.shape[-1] != t_cls_logits.shape[-1]:
                t_cls_logits = F.interpolate(
                    t_cls_logits,
                    size=s_cls_logits.shape[-1],
                    mode="linear",
                    align_corners=False,
                )
            s_logits = s_cls_logits.permute(0, 2, 1)
            t_logits = t_cls_logits.permute(0, 2, 1)
            loss_distill_logits = (
                self.distill_logits_loss(
                    F.log_softmax(s_logits / self.temperature, dim=-1),
                    F.softmax(t_logits / self.temperature, dim=-1),
                )
                * (self.temperature**2)
            )

        if self.logits_debug and RANK in [-1, 0] and getattr(self, "batch_idx", 0) == 0:
            print(
                f"[logits] student {tuple(s_cls_logits.shape)} teacher {tuple(t_cls_logits.shape)} "
                f"loss={loss_distill_logits.item():.4f}"
            )

        total_loss = loss_original + self.distill_logits_alpha * loss_distill_logits
        return total_loss, loss_items


def parse_key_value_args(args: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in args:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def smart_cast(value: str):
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


SCRIPT_SPECIAL_KEYS = {
    "student_model": {"label": "学生模型 (.pt)", "type": "path", "required": True},
    "teacher_model": {"label": "教师模型 (.pt)", "type": "path", "required": True},
    "teacher_imgsz": {"label": "教师输入尺寸", "type": "int", "default": 1024},
    "distill_logits_alpha": {"label": "Logits 蒸馏系数", "type": "float", "default": 1.0},
    "temperature": {"label": "蒸馏温度", "type": "float", "default": 2.0},
    "logits_debug": {"label": "打印调试信息", "type": "bool", "default": False},
}


def main():
    raw_args = parse_key_value_args(sys.argv[1:])
    if not raw_args:
        raise SystemExit("请使用 key=value 形式传参，例如 data=xx.yaml student_model=xxx.pt teacher_model=yyy.pt")

    for key, meta in SCRIPT_SPECIAL_KEYS.items():
        if key not in raw_args and meta.get("required"):
            raise SystemExit(f"缺少必要参数：{key}")
        if key not in raw_args and "default" in meta:
            raw_args[key] = str(meta["default"])

    if "model" not in raw_args and "student_model" in raw_args:
        raw_args["model"] = raw_args["student_model"]

    required = ["model", "data", "teacher_model"]
    missing = [key for key in required if key not in raw_args]
    if missing:
        raise SystemExit(f"缺少必要参数：{', '.join(missing)}")

    overrides = {key: smart_cast(value) for key, value in raw_args.items()}
    LOGGER.info("Logits 蒸馏训练参数：%s", overrides)

    trainer = LogitsKDTrainer(overrides=overrides)
    trainer.train()
    print("✅ Logits 蒸馏训练完成。")


if __name__ == "__main__":
    main()
