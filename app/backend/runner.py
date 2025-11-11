from __future__ import annotations

import os
import threading
from pathlib import Path
from subprocess import Popen, PIPE
from typing import Callable, Dict, Optional, Tuple

from .artifacts import RunArtifactManager
from .config import AppConfig
from .models import TaskStatus, TrainingTask, ValidationSummary
from .strategies import build_strategy


class YOLOProcessRunner:
    """负责子进程调度与日志流转。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run_process(
        self,
        task: TrainingTask,
        command: Tuple[str, ...],
        env: Optional[Dict[str, str]] = None,
        on_start: Optional[Callable[[Popen], None]] = None,
        on_exit: Optional[Callable[[], None]] = None,
    ) -> int:
        task.append_log("INFO", f"启动命令: {' '.join(command)}")
        process_env = None
        if env:
            process_env = {**os.environ, **env}
        process = Popen(
            command,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=process_env,
        )

        if on_start:
            on_start(process)

        assert process.stdout is not None
        assert process.stderr is not None

        def pump(stream, level: str) -> None:
            with stream:
                for line in stream:
                    task.append_log(level, line.rstrip())

        threads = [
            threading.Thread(target=pump, args=(process.stdout, "OUT"), daemon=True),
            threading.Thread(target=pump, args=(process.stderr, "ERR"), daemon=True),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        return_code = process.wait()
        task.append_log("INFO", f"进程退出，代码 {return_code}")
        if on_exit:
            on_exit()
        return return_code


class ValidationExecutor:
    """负责自动执行 Val 流程。"""

    def __init__(self, config: AppConfig, runner: YOLOProcessRunner) -> None:
        self.config = config
        self.runner = runner

    def run_validation(
        self,
        task: TrainingTask,
        *,
        weight_path: Optional[Path] = None,
        name_suffix: str = "val",
        on_start: Optional[Callable[[Popen], None]] = None,
        on_exit: Optional[Callable[[], None]] = None,
    ) -> ValidationSummary:
        if not task.run_dir:
            raise ValueError("任务尚未绑定 run 目录，无法验证。")

        weights_dir = task.run_dir / "weights"
        model_path = Path(weight_path) if weight_path else self._pick_weights(weights_dir)
        if not model_path:
            raise FileNotFoundError("未找到权重文件，跳过验证。")

        params = {
            "model": str(model_path),
            "data": task.parameters.get("data", ""),
            "imgsz": task.parameters.get("imgsz", ""),
            "batch": task.parameters.get("batch", ""),
            "project": str(task.run_dir.parent),
            "name": f"{task.run_name}-{name_suffix}" if task.run_name else f"validation-{name_suffix}",
        }

        if task.extra_args:
            params.update(task.extra_args)

        cli_args = [f"{k}={v}" for k, v in params.items() if v]
        command = (self.config.yolo_cli, "val", *cli_args)
        task.append_log("INFO", f"开始验证: {' '.join(command)}")
        return_code = self.runner.run_process(
            task,
            command,
            on_start=on_start,
            on_exit=on_exit,
        )

        val_run_dir = Path(params["project"]) / params["name"]
        metrics = self._collect_metrics(val_run_dir)
        success = return_code == 0
        summary = ValidationSummary(
            metrics=metrics,
            raw_output="\n".join(log.message for log in task.logs[-50:]),
            success=success,
        )
        if not success:
            task.append_log("ERROR", "验证阶段失败。")
        return summary

    @staticmethod
    def _pick_weights(weights_dir: Path) -> Optional[Path]:
        if not weights_dir.exists():
            return None
        for filename in ("best.pt", "last.pt"):
            candidate = weights_dir / filename
            if candidate.exists():
                return candidate
        candidates = list(weights_dir.glob("*.pt"))
        return candidates[0] if candidates else None

    @staticmethod
    def _collect_metrics(results_dir: Path) -> Dict[str, float]:
        results_file = results_dir / "results.csv"
        if not results_file.exists():
            return {}
        try:
            import csv
        except ImportError:  # pragma: no cover
            return {}

        metrics: Dict[str, float] = {}
        with results_file.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
            if not rows:
                return {}
            last = rows[-1]
            for key, value in last.items():
                if not value:
                    continue
                try:
                    metrics[key] = float(value)
                except ValueError:
                    continue
        return metrics


class TrainingOrchestrator:
    """封装训练、验证的完整流程供队列调用。"""

    def __init__(self, config: AppConfig, artifact_manager: RunArtifactManager) -> None:
        self.config = config
        self.artifact_manager = artifact_manager
        self.runner = YOLOProcessRunner(config)
        self.validator = ValidationExecutor(config, self.runner)
        self._lock = threading.Lock()
        self._active_processes: Dict[str, Popen] = {}
        self._active_lock = threading.Lock()

    def execute(self, task: TrainingTask) -> TrainingTask:
        with self._lock:
            run_descriptor = self.artifact_manager.assign_run(task, category="train")
            strategy = build_strategy(task, self.config)
            command, env = strategy.build_command(task, run_descriptor)

        task.status = TaskStatus.RUNNING
        return_code = self.runner.run_process(
            task,
            tuple(command),
            env,
            on_start=lambda proc: self._register_process(task.id, proc),
            on_exit=lambda: self._clear_process(task.id),
        )
        task.exit_code = return_code

        if task.cancel_requested:
            task.status = TaskStatus.CANCELED
            task.append_log("INFO", "任务已取消。")
            return task

        if return_code != 0:
            task.status = TaskStatus.FAILED
            return task

        task.status = TaskStatus.COMPLETED
        return task

    def _register_process(self, task_id: str, process: Popen) -> None:
        with self._active_lock:
            self._active_processes[task_id] = process

    def _clear_process(self, task_id: str) -> None:
        with self._active_lock:
            self._active_processes.pop(task_id, None)

    def request_cancel(self, task: TrainingTask) -> bool:
        task.cancel_requested = True
        with self._active_lock:
            process = self._active_processes.get(task.id)
        if not process:
            return False
        task.append_log("WARN", "正在尝试终止任务子进程...")
        process.terminate()
        try:
            process.wait(timeout=10)
        except Exception:  # noqa: BLE001
            task.append_log("WARN", "子进程未能及时退出，强制结束。")
            process.kill()
        return True
