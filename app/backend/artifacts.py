from __future__ import annotations

import csv
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .config import AppConfig
from .models import TrainingTask


_SAFE_CHAR_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")


def _slugify(value: str) -> str:
    value = value.strip()
    value = value.replace(" ", "_")
    value = _SAFE_CHAR_PATTERN.sub("_", value)
    return value.strip("_") or "run"


@dataclass
class RunDescriptor:
    name: str
    path: Path


class RunArtifactManager:
    """负责 run 目录命名、定位与结果读取。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self.config.ensure_directories()
        for sub in ("train", "val"):
            (self.config.runs_dir / sub).mkdir(parents=True, exist_ok=True)

    def assign_run(self, task: TrainingTask, category: str = "train") -> RunDescriptor:
        with self._lock:
            timestamp = datetime.now(timezone.utc).astimezone().strftime("%y.%m.%d.%H%M%S")
            model_name = task.parameters.get("model", "model")
            model_name = Path(model_name).stem
            epoch = task.parameters.get("epochs", "")
            batch = task.parameters.get("batch", "")
            lr0 = task.parameters.get("lr0", task.parameters.get("lr", ""))
            script_label = task.script_path.stem if task.script_path else ""
            note = _slugify(task.user_note) if task.user_note else ""

            parts: List[str] = [
                _slugify(model_name),
                timestamp,
                f"epoch{epoch}" if epoch else "",
                f"batch{batch}" if batch else "",
                f"lr0{lr0}" if lr0 else "",
                script_label,
                note,
            ]
            run_name = "-".join([p for p in parts if p])
            base_dir = self.config.runs_dir / category
            base_dir.mkdir(parents=True, exist_ok=True)
            run_dir = base_dir / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            task.run_name = run_name
            task.run_dir = run_dir
            return RunDescriptor(name=run_name, path=run_dir)

    @staticmethod
    def results_file(run_dir: Path) -> Path:
        return run_dir / "results.csv"

    def read_results(self, run_dir: Path) -> List[Dict[str, float]]:
        file_path = self.results_file(run_dir)
        if not file_path.exists():
            return []
        rows: List[Dict[str, float]] = []
        with file_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                numeric_row: Dict[str, float] = {}
                for key, value in row.items():
                    if value is None or value == "":
                        continue
                    try:
                        numeric_row[key] = float(value)
                    except ValueError:
                        continue
                if numeric_row:
                    rows.append(numeric_row)
        return rows

    def list_all_runs(self) -> Iterable[RunDescriptor]:
        train_dir = self.config.runs_dir / "train"
        if not train_dir.exists():
            return []
        return [
            RunDescriptor(name=path.name, path=path)
            for path in sorted(train_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if path.is_dir()
        ]

    @staticmethod
    def load_args(run_dir: Path) -> Dict[str, Any]:
        for filename in ("args.yaml", "args.json"):
            file_path = run_dir / filename
            if not file_path.exists():
                continue
            try:
                if file_path.suffix == ".yaml":
                    import yaml

                    with file_path.open("r", encoding="utf-8") as fh:
                        return yaml.safe_load(fh) or {}
                if file_path.suffix == ".json":
                    import json

                    with file_path.open("r", encoding="utf-8") as fh:
                        return json.load(fh) or {}
            except Exception:
                continue
        return {}
