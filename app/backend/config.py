from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


def _detect_base_dir() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AppConfig:
    """统一管理项目的关键配置路径与默认参数。"""

    base_dir: Path = field(default_factory=_detect_base_dir)
    models_cfg_dir: Path = field(default_factory=lambda: _detect_base_dir() / "models_cfg")
    train_script_dir: Path = field(default_factory=lambda: _detect_base_dir() / "train_method")
    dataset_cfg_dir: Path = field(default_factory=lambda: _detect_base_dir() / "datasets" / "cfg")
    runs_dir: Path = field(default_factory=lambda: _detect_base_dir() / "runs")
    yolo_cli: str = field(default_factory=lambda: os.getenv("YOLO_CLI", "yolo"))
    python_executable: str = field(default_factory=lambda: os.getenv("PYTHON_EXECUTABLE", os.sys.executable))

    default_train_params: Dict[str, str] = field(
        default_factory=lambda: {
            "epochs": "100",
            "batch": "16",
            "lr0": "0.01",
            "imgsz": "640",
        }
    )

    realtime_refresh_sec: float = 2.0
    log_history_limit: int = 5000

    @classmethod
    def from_env(cls) -> "AppConfig":
        """允许通过环境变量覆盖路径。"""
        base_dir = Path(os.getenv("APP_BASE_DIR", _detect_base_dir()))
        return cls(
            base_dir=base_dir,
            models_cfg_dir=Path(os.getenv("MODELS_CFG_DIR", base_dir / "models_cfg")),
            train_script_dir=Path(os.getenv("TRAIN_SCRIPT_DIR", base_dir / "train_method")),
            dataset_cfg_dir=Path(os.getenv("DATASET_CFG_DIR", base_dir / "datasets" / "cfg")),
            runs_dir=Path(os.getenv("RUNS_DIR", base_dir / "runs")),
            yolo_cli=os.getenv("YOLO_CLI", "yolo"),
            python_executable=os.getenv("PYTHON_EXECUTABLE", os.sys.executable),
        )

    def ensure_directories(self) -> None:
        """确保关键目录存在。"""
        for path in (self.models_cfg_dir, self.train_script_dir, self.dataset_cfg_dir, self.runs_dir):
            path.mkdir(parents=True, exist_ok=True)

    def models_cfg_files(self) -> List[Path]:
        return sorted(p for p in self.models_cfg_dir.glob("*.yaml") if p.is_file())

    def available_train_scripts(self) -> List[Path]:
        return sorted(p for p in self.train_script_dir.glob("*.py") if p.is_file())

    def dataset_cfg_files(self) -> List[Path]:
        return sorted(p for p in self.dataset_cfg_dir.glob("*.yaml") if p.is_file())
