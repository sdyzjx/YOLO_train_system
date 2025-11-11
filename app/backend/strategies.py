from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Dict, List, Tuple

from .artifacts import RunDescriptor
from .config import AppConfig
from .models import TaskMode, TrainingTask


class TrainingStrategy(ABC):
    """策略基类，根据任务模式构建命令。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @abstractmethod
    def build_command(self, task: TrainingTask, run: RunDescriptor) -> Tuple[List[str], Dict[str, str]]:
        """返回命令行及环境变量。"""

    def _compose_cli_args(self, params: Dict[str, str]) -> List[str]:
        args: List[str] = []
        for key, value in params.items():
            if value in (None, "", False):
                continue
            args.append(f"{key}={value}")
        return args


class QuickTrainStrategy(TrainingStrategy):
    """快速模式：直接调用 yolo train。"""

    def build_command(self, task: TrainingTask, run: RunDescriptor) -> Tuple[List[str], Dict[str, str]]:
        params = dict(task.parameters)
        params.setdefault("name", run.name)
        params.setdefault("project", str(run.path.parent))

        model_path = params.get("model")
        if model_path:
            model_path = Path(model_path)
            if not model_path.is_absolute():
                model_path = self.config.models_cfg_dir / model_path
            params["model"] = str(model_path.resolve())

        data_path = params.get("data")
        if data_path:
            data_path = Path(data_path)
            if not data_path.is_absolute():
                candidate = self.config.dataset_cfg_dir / data_path
                if candidate.exists():
                    data_path = candidate
            params["data"] = str(data_path.resolve())

        params.update(task.extra_args)

        cli_args = self._compose_cli_args(params)
        command = [self.config.yolo_cli, "train", *cli_args]
        return command, {}


class ScriptTrainStrategy(TrainingStrategy):
    """脚本模式：调用用户自定义训练脚本。"""

    def build_command(self, task: TrainingTask, run: RunDescriptor) -> Tuple[List[str], Dict[str, str]]:
        if not task.script_path:
            raise ValueError("脚本模式需要提供脚本路径。")

        params = dict(task.parameters)
        params.setdefault("name", run.name)
        params.setdefault("project", str(run.path.parent))
        params.update(task.extra_args)
        params.update(task.script_args)

        data_path = params.get("data")
        if data_path:
            data_path = Path(data_path)
            if not data_path.is_absolute():
                candidate = self.config.dataset_cfg_dir / data_path
                if candidate.exists():
                    data_path = candidate
            params["data"] = str(data_path.resolve())

        cli_args = self._compose_cli_args(params)
        command = [self.config.python_executable, str(task.script_path), *cli_args]
        pythonpath_entries = [str(self.config.base_dir)]
        existing = os.environ.get("PYTHONPATH")
        if existing:
            pythonpath_entries.append(existing)
        env = {"PYTHONPATH": os.pathsep.join(pythonpath_entries)}
        return command, env


def build_strategy(task: TrainingTask, config: AppConfig) -> TrainingStrategy:
    if task.mode is TaskMode.QUICK:
        return QuickTrainStrategy(config)
    if task.mode is TaskMode.SCRIPT:
        return ScriptTrainStrategy(config)
    raise ValueError(f"不支持的训练模式: {task.mode}")
