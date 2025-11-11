#!/usr/bin/env python
"""
自定义蒸馏训练脚本：复制参考 KDTrainer 逻辑，实现 P3/P4 特征与 logits 蒸馏。

示例：
    python distill_p3p4_train.py model=yolov8n-obb.pt data=ttpla.yaml teacher_model=runs/obb/train85/weights/best.pt
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable
from types import MethodType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_ULTRALYTICS_DIRS = [
    PROJECT_ROOT / "ultralytics",
    PROJECT_ROOT / "modules" / "ultralytics",
]
for candidate in CANDIDATE_ULTRALYTICS_DIRS:
    pkg_root = candidate / "ultralytics"
    if pkg_root.exists() and (pkg_root / "__init__.py").exists():
        path_str = str(candidate)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        break

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.obb.train import OBBTrainer
from ultralytics.nn.modules import C2f, Conv
from ultralytics.utils import LOGGER, RANK


def get_feature_hook(name, features_dict):
    """通用前向 hook，保存模块输出。"""

    def hook(_module, _inputs, output):
        features_dict[name] = output

    return hook


class KDTrainer(OBBTrainer):
    """完全沿用参考代码思路的 KD Trainer。"""

    def __init__(self, overrides=None, _callbacks=None):
        overrides = overrides.copy() if overrides else {}
        self.teacher_model_path = overrides.pop("teacher_model", None)
        self.teacher_imgsz = int(overrides.pop("teacher_imgsz", 1024))
        self.distill_p3_p2_alpha = float(overrides.pop("distill_p3_p2_alpha", 1.0))
        self.distill_p4_p3_alpha = float(overrides.pop("distill_p4_p3_alpha", 1.0))
        self.distill_logits_alpha = float(overrides.pop("distill_logits_alpha", 1.0))
        self.temperature = float(overrides.pop("temperature", 2.0))
        self.logits_debug = bool(overrides.pop("logits_debug", False))

        student_model = overrides.pop("student_model", None)
        if student_model and "model" not in overrides:
            overrides["model"] = student_model

        if self.teacher_model_path is None:
            raise ValueError("錯誤：知識蒸餾需要提供 'teacher_model' 參數。")

        self.teacher_model_path = self._resolve_teacher_weights(Path(self.teacher_model_path))

        super().__init__(overrides=overrides, _callbacks=_callbacks)

        if RANK in [-1, 0]:
            print(f"⏳ 正在從 '{self.teacher_model_path}' 加載教師模型...")
        teacher_yolo = YOLO(str(self.teacher_model_path))
        self.teacher_model = teacher_yolo.model.to(self.device)
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        self.teacher_head = getattr(self.teacher_model, "head", None)
        if self.teacher_head is None:
            self.teacher_head = self.teacher_model.model[-1]
            setattr(self.teacher_model, "head", self.teacher_head)
        self.reg_channel_count = self.teacher_head.reg_max * 4
        if RANK in [-1, 0]:
            print("✅ 教師模型已加載並凍結。")

        self.distill_feature_loss = nn.MSELoss()
        self.distill_logits_loss = nn.KLDivLoss(reduction="batchmean")
        self.teacher_features: Dict[str, torch.Tensor] = {}
        self.student_features: Dict[str, torch.Tensor] = {}
        self.adaptor_p2: nn.Module | None = None
        self.adaptor_p3: nn.Module | None = None

        self._sync_teacher_head()

        if RANK in [-1, 0]:
            print("✅ 蒸餾已啟用:")
            print(f"   - 教師輸入尺寸: {self.teacher_imgsz}")
            print(f"   - 特徵蒸餾 1: 教師 P3 -> 學生 P2 (MSELoss), alpha = {self.distill_p3_p2_alpha}")
            print(f"   - 特徵蒸餾 2: 教師 P4 -> 學生 P3 (MSELoss), alpha = {self.distill_p4_p3_alpha}")
            print(f"   - Logits 蒸餾 (KLDivLoss), alpha = {self.distill_logits_alpha}, T = {self.temperature}")

    def _resolve_conv_module(self, module: nn.Module) -> nn.Module | None:
        if isinstance(module, (C2f, Conv)):
            return module
        if isinstance(module, nn.Sequential):
            for child in module.children():
                resolved = self._resolve_conv_module(child)
                if resolved is not None:
                    return resolved
        return None

    def _get_out_channels(self, module: nn.Module) -> int:
        resolved = self._resolve_conv_module(module)
        if isinstance(resolved, C2f):
            return resolved.cv2.conv.out_channels
        if isinstance(resolved, Conv):
            return resolved.conv.out_channels
        raise TypeError(f"未支援的模組類型: {type(module)}")

    def _setup_train(self):
        """重寫以在基礎設置後註冊 hook 並添加適配器。"""
        super()._setup_train()

        if RANK in [-1, 0]:
            print("✅ 基礎模型與優化器已構建，現在註冊 hooks 並創建通道適配器...")

        student_p2_module_idx, student_p3_module_idx = 3, 4
        teacher_p3_module_idx, teacher_p4_module_idx = 4, 5

        student_p2_module = self.model.model[student_p2_module_idx]
        student_p3_module = self.model.model[student_p3_module_idx]
        teacher_p3_module = self.teacher_model.model[teacher_p3_module_idx]
        teacher_p4_module = self.teacher_model.model[teacher_p4_module_idx]

        student_p2_module.register_forward_hook(get_feature_hook("p2", self.student_features))
        student_p3_module.register_forward_hook(get_feature_hook("p3", self.student_features))
        teacher_p3_module.register_forward_hook(get_feature_hook("p3", self.teacher_features))
        teacher_p4_module.register_forward_hook(get_feature_hook("p4", self.teacher_features))

        s_p2_channels = self._get_out_channels(student_p2_module)
        s_p3_channels = self._get_out_channels(student_p3_module)
        t_p3_channels = self._get_out_channels(teacher_p3_module)
        t_p4_channels = self._get_out_channels(teacher_p4_module)

        self.adaptor_p2 = nn.Conv2d(s_p2_channels, t_p3_channels, kernel_size=1).to(self.device)
        self.adaptor_p3 = nn.Conv2d(s_p3_channels, t_p4_channels, kernel_size=1).to(self.device)

        self.optimizer.add_param_group(
            {
                "params": list(self.adaptor_p2.parameters()) + list(self.adaptor_p3.parameters()),
                "name": "adaptors",
                "lr": 0.01,
                "initial_lr": 0.01,
            }
        )

        if RANK in [-1, 0]:
            print(f"✅ Hooks 已註冊 (學生): P2 (索引 {student_p2_module_idx}), P3 (索引 {student_p3_module_idx})")
            print(f"✅ Hooks 已註冊 (教師): P3 (索引 {teacher_p3_module_idx}), P4 (索引 {teacher_p4_module_idx})")
            print(f"✅ 適配器已創建 (P3->P2): {s_p2_channels} -> {t_p3_channels}")
            print(f"✅ 適配器已創建 (P4->P3): {s_p3_channels} -> {t_p4_channels}")
            print("✅ 所有通道適配器的參數已成功添加到優化器。")

        self._wrap_model_loss()

    def _resolve_teacher_weights(self, raw_path: Path) -> Path:
        raw_path = raw_path.expanduser().resolve()
        if raw_path.is_file() and raw_path.suffix == ".pt":
            return raw_path
        candidates = []
        base = raw_path
        if raw_path.is_file():
            base = raw_path.parent
        weights_dir = base / "weights"
        for name in ("best.pt", "last.pt"):
            candidate = weights_dir / name
            if candidate.exists():
                return candidate
            candidate = base / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"無法在 {raw_path} 及其 weights 子目錄中找到教師權重 (.pt)。")

    def _sync_teacher_head(self):
        """Ensure teacher head knows dataset metadata so classification channels exist."""
        teacher_detect = self.teacher_model.model[-1]
        teacher_detect.nc = self.data["nc"]
        teacher_detect.names = self.data.get("names", {})
        teacher_detect.args = getattr(self.model, "args", None) or self.args
        self.teacher_model.nc = teacher_detect.nc
        self.teacher_model.names = teacher_detect.names

    def _wrap_model_loss(self):
        base_loss_fn = self.model.loss
        trainer = self

        def loss_with_distill(model_self, batch, preds=None):
            if preds is None:
                preds = model_self.forward(batch["img"])
            loss_original, loss_items = base_loss_fn(batch, preds)
            return trainer._augment_with_distill(preds, batch, loss_original, loss_items)

        self.model.loss = MethodType(loss_with_distill, self.model)

    def _augment_with_distill(self, preds, batch, loss_original, loss_items):
        with torch.no_grad():
            teacher_input_img = F.interpolate(
                batch["img"],
                size=(self.teacher_imgsz, self.teacher_imgsz),
                mode="bilinear",
                align_corners=False,
            )
            head_training_state = self.teacher_head.training
            self.teacher_head.training = True  # force raw logits path without touching BN stats
            try:
                teacher_outputs = self.teacher_model(teacher_input_img)
            finally:
                self.teacher_head.training = head_training_state
            t_maps = teacher_outputs[0] if isinstance(teacher_outputs, (list, tuple)) else teacher_outputs
            if not isinstance(t_maps, (list, tuple)):
                t_maps = [t_maps]

        s_p2_features = self.student_features.get("p2")
        s_p3_features = self.student_features.get("p3")
        t_p3_features = self.teacher_features.get("p3")
        t_p4_features = self.teacher_features.get("p4")

        loss_distill_p3_p2 = torch.zeros(1, device=self.device)
        if self.distill_p3_p2_alpha and s_p2_features is not None and t_p3_features is not None:
            teacher_feat = t_p3_features
            if teacher_feat.shape[2:] != s_p2_features.shape[2:]:
                teacher_feat = F.interpolate(
                    teacher_feat,
                    size=s_p2_features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            s_p2_adapted = self.adaptor_p2(s_p2_features)
            loss_distill_p3_p2 = self.distill_feature_loss(s_p2_adapted, teacher_feat)

        loss_distill_p4_p3 = torch.zeros(1, device=self.device)
        if self.distill_p4_p3_alpha and s_p3_features is not None and t_p4_features is not None:
            teacher_feat = t_p4_features
            if teacher_feat.shape[2:] != s_p3_features.shape[2:]:
                teacher_feat = F.interpolate(
                    teacher_feat,
                    size=s_p3_features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            s_p3_adapted = self.adaptor_p3(s_p3_features)
            loss_distill_p4_p3 = self.distill_feature_loss(s_p3_adapted, teacher_feat)

        loss_distill_logits = torch.zeros(1, device=self.device)
        if self.distill_logits_alpha:
            student_maps = preds[0] if isinstance(preds, (list, tuple)) else preds
            if not isinstance(student_maps, (list, tuple)):
                student_maps = [student_maps]
            student_tensor = torch.cat(
                [m.view(m.shape[0], m.shape[1], -1) for m in student_maps],
                dim=2,
            )
            teacher_tensor = torch.cat(
                [m.view(m.shape[0], m.shape[1], -1) for m in t_maps],
                dim=2,
            )
            student_head = getattr(self.model, "head", None)
            if student_head is None:
                student_head = self.model.model[-1]
            student_reg_channels = student_head.reg_max * 4
            s_cls_logits = student_tensor[:, student_reg_channels:, :]
            t_cls_logits = teacher_tensor[:, self.reg_channel_count :, :]
            if s_cls_logits.shape[1] > 0 and t_cls_logits.shape[1] > 0:
                if s_cls_logits.shape[-1] != t_cls_logits.shape[-1]:
                    t_cls_logits = F.interpolate(
                        t_cls_logits,
                        size=s_cls_logits.shape[-1],
                        mode="linear",
                        align_corners=False,
                    )
                s_cls_logits = s_cls_logits.permute(0, 2, 1)
                t_cls_logits = t_cls_logits.permute(0, 2, 1)
                loss_distill_logits = (
                    self.distill_logits_loss(
                        F.log_softmax(s_cls_logits / self.temperature, dim=-1),
                        F.softmax(t_cls_logits / self.temperature, dim=-1),
                    )
                    * (self.temperature**2)
                )

        if self.logits_debug and RANK in [-1, 0] and getattr(self, "epoch", 0) == 0 and getattr(self, "batch_idx", 0) == 0:
            print(
                f"[logits-debug] student shape: {tuple(s_cls_logits.shape)} teacher shape: {tuple(t_cls_logits.shape)}"
            )
            if s_cls_logits.numel():
                print(
                    f"[logits-debug] student min/max: {s_cls_logits.min().item():.6f} / {s_cls_logits.max().item():.6f}"
                )
            if t_cls_logits.numel():
                print(
                    f"[logits-debug] teacher min/max: {t_cls_logits.min().item():.6f} / {t_cls_logits.max().item():.6f}"
                )
            if not t_cls_logits.numel():
                print("[logits-debug] teacher classification logits empty")

        loss = (
            loss_original
            + self.distill_p3_p2_alpha * loss_distill_p3_p2
            + self.distill_p4_p3_alpha * loss_distill_p4_p3
            + self.distill_logits_alpha * loss_distill_logits
        )

        if RANK in [-1, 0] and getattr(self.args, "verbose", False) and getattr(self, "batch_idx", 0) % 10 == 0:
            log_str = f" (Epoch {getattr(self, 'epoch', 0) + 1}, Batch {getattr(self, 'batch_idx', 0)}) -"
            if self.distill_p3_p2_alpha:
                log_str += f" P3->P2 Loss: {self.distill_p3_p2_alpha * loss_distill_p3_p2.item():.4f}"
            if self.distill_p4_p3_alpha:
                log_str += f" | P4->P3 Loss: {self.distill_p4_p3_alpha * loss_distill_p4_p3.item():.4f}"
            if self.distill_logits_alpha:
                log_str += f" | Logits Loss: {self.distill_logits_alpha * loss_distill_logits.item():.4f}"
            print(log_str)

        return loss, loss_items


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
    "student_model": {"label": "學生模型 (.pt)", "type": "path", "required": True},
    "teacher_model": {"label": "老師模型 (.pt)", "type": "path", "required": True},
    "teacher_imgsz": {"label": "老師輸入尺寸", "type": "int", "default": 1024},
    "distill_p3_p2_alpha": {"label": "P3→P2 蒸餾係數", "type": "float", "default": 1.0},
    "distill_p4_p3_alpha": {"label": "P4→P3 蒸餾係數", "type": "float", "default": 1.0},
    "distill_logits_alpha": {"label": "Logits 蒸餾係數", "type": "float", "default": 1.0},
    "temperature": {"label": "蒸餾溫度", "type": "float", "default": 2.0},
    "logits_debug": {"label": "輸出 Logits 調試", "type": "bool", "default": False},
}


def main():
    raw_args = parse_key_value_args(sys.argv[1:])
    if not raw_args:
        raise SystemExit("請以 key=value 形式傳入參數，例如：model=yolo.pt data=xxx.yaml teacher_model=teacher.pt")

    for key, meta in SCRIPT_SPECIAL_KEYS.items():
        if key not in raw_args and meta.get("required"):
            raise SystemExit(f"缺少必要參數：{key}")
        if key not in raw_args and "default" in meta:
            raw_args[key] = str(meta["default"])

    student_model = raw_args.get("student_model")
    if student_model and "model" not in raw_args:
        raw_args["model"] = student_model

    required = ["model", "data", "teacher_model"]
    missing = [key for key in required if key not in raw_args]
    if missing:
        raise SystemExit(f"缺少必要參數：{', '.join(missing)}")

    overrides = {key: smart_cast(value) for key, value in raw_args.items()}
    LOGGER.info("蒸餾訓練參數：%s", overrides)

    trainer = KDTrainer(overrides=overrides)
    trainer.train()
    print("✅ 訓練完成。")


if __name__ == "__main__":
    main()
