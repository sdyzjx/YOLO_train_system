from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from .visualization import VisualizationPayload, VisualizationProvider, register_provider

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _format_label(folder: Path) -> str:
    try:
        modified = dt.datetime.fromtimestamp(folder.stat().st_mtime).astimezone()
        timestamp = modified.strftime("%Y-%m-%d %H:%M")
    except OSError:
        timestamp = "未知时间"
    return f"{folder.name} · {timestamp}"


def _collect_images(folder: Path) -> List[str]:
    images: List[str] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(str(path.resolve()))
    return images


def list_feature_map_records(run_path: Path) -> List[Dict[str, object]]:
    base = run_path / "feature_maps"
    if not base.exists():
        return []
    records: List[Dict[str, object]] = []
    for folder in sorted(base.iterdir(), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if not folder.is_dir():
            continue
        images = _collect_images(folder)
        if not images:
            continue
        records.append(
            {
                "id": folder.name,
                "label": _format_label(folder),
                "path": str(folder.resolve()),
                "images": images,
            }
        )
    return records


class FeatureMapProvider(VisualizationProvider):
    name = "feature_maps"

    def render(self, payload: VisualizationPayload) -> Dict[str, object]:
        return {"records": list_feature_map_records(payload.run_path)}


register_provider(FeatureMapProvider())


def generate_feature_map_visualization(
    run_path: Path,
    weight_path: Path,
    image_path: Path,
    imgsz: int = 512,
) -> Tuple[str, List[Dict[str, object]]]:
    if not weight_path.exists():
        raise FileNotFoundError(f"权重文件不存在：{weight_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在：{image_path}")

    output_root = run_path / "feature_maps"
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().astimezone().strftime("%y%m%d_%H%M%S")
    output_name = f"{weight_path.stem}_{image_path.stem}_{timestamp}"

    def _ensure_vendored_ultralytics() -> bool:
        project_root = Path(__file__).resolve().parents[2]
        vendored_root = project_root / "ultralytics"
        package_dir = vendored_root / "ultralytics"
        if not package_dir.exists():
            return False
        path_str = str(vendored_root)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
        # Remove any namespace-package stubs that may already be cached to force a reload.
        for name in list(sys.modules):
            if name == "ultralytics" or name.startswith("ultralytics."):
                sys.modules.pop(name, None)
        return True

    try:
        from ultralytics import YOLO
    except Exception as exc:  # noqa: BLE001
        if not _ensure_vendored_ultralytics():
            raise RuntimeError("无法导入 Ultralytics YOLO，请确认环境依赖。") from exc
        try:
            from ultralytics import YOLO
        except Exception as retry_exc:  # noqa: BLE001
            raise RuntimeError("无法导入 Ultralytics YOLO，请确认环境依赖。") from retry_exc

    model = YOLO(str(weight_path))
    results = model.predict(
        source=str(image_path),
        visualize=True,
        imgsz=int(imgsz),
        project=str(output_root),
        name=output_name,
        save=True,
    )
    if not results:
        raise RuntimeError("未能生成特征图结果。")
    save_dir = Path(results[0].save_dir)
    if not save_dir.exists():
        raise RuntimeError("特征图输出目录生成失败。")
    records = list_feature_map_records(run_path)
    return output_name, records
