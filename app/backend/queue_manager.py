from __future__ import annotations

import threading
from collections import deque
from queue import Empty, Queue
from pathlib import Path
from typing import Deque, Dict, List, Optional

from .artifacts import RunArtifactManager
from .config import AppConfig
from .models import TaskMode, TaskStatus, TrainingTask
from .runner import TrainingOrchestrator


class TrainingQueueManager:
    """训练队列调度核心。"""

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig.from_env()
        self.config.ensure_directories()
        self.artifact_manager = RunArtifactManager(self.config)
        self.orchestrator = TrainingOrchestrator(self.config, self.artifact_manager)

        self._task_queue: "Queue[TrainingTask]" = Queue()
        self._tasks: Dict[str, TrainingTask] = {}
        self._history: Deque[str] = deque(maxlen=200)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def submit(self, task: TrainingTask) -> TrainingTask:
        with self._lock:
            self._tasks[task.id] = task
        task.append_log("INFO", "任务已加入队列。")
        self._task_queue.put(task)
        return task

    def cancel_task(self, task_id: str) -> str:
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return "未找到指定任务。"
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELED):
            return f"任务当前状态为 {task.status.label}，无需取消。"

        task.cancel_requested = True
        task.append_log("WARN", "收到用户取消请求。")

        if task.status is TaskStatus.PENDING:
            task.status = TaskStatus.CANCELED
            return "已取消排队中的任务。"

        if task.status in (TaskStatus.RUNNING, TaskStatus.VALIDATING):
            stopped = self.orchestrator.request_cancel(task)
            if stopped:
                return "已发送停止信号，任务即将终止。"
            return "正在等待任务进入可取消状态。"

        return "任务取消请求已记录。"

    def list_tasks(self) -> List[TrainingTask]:
        with self._lock:
            return sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)

    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                task = self._task_queue.get(timeout=0.5)
            except Empty:
                continue

            if task.status is TaskStatus.CANCELED:
                task.append_log("WARN", "任务已被取消，跳过执行。")
                self._task_queue.task_done()
                continue

            try:
                self.orchestrator.execute(task)
            except Exception as exc:  # noqa: BLE001
                task.status = TaskStatus.FAILED
                task.append_log("ERROR", f"执行过程中出现异常: {exc}")
            finally:
                self._history.appendleft(task.id)
                self._trim_logs(task)
                self._task_queue.task_done()

    def _trim_logs(self, task: TrainingTask) -> None:
        limit = self.config.log_history_limit
        if len(task.logs) > limit:
            task.logs[:] = task.logs[-limit:]

    def stop(self) -> None:
        self._stop_event.set()
        self._worker.join(timeout=2)

    def recent_history(self) -> List[TrainingTask]:
        with self._lock:
            return [self._tasks[task_id] for task_id in self._history if task_id in self._tasks]

    def manual_validate(
        self,
        *,
        weight_path: Path,
        dataset_path: Path,
        imgsz: Optional[int] = None,
        batch: Optional[int] = None,
        note: str = "",
        extra_params: Optional[Dict[str, str]] = None,
    ):
        if not weight_path.exists():
            raise FileNotFoundError(f"未找到权重文件: {weight_path}")
        if not dataset_path.exists():
            raise FileNotFoundError(f"未找到数据集配置文件: {dataset_path}")

        task = TrainingTask(
            mode=TaskMode.QUICK,
            parameters={
                "model": weight_path.stem,
                "data": str(dataset_path),
            },
            user_note=note or "manual-val",
        )
        if imgsz:
            task.parameters["imgsz"] = str(int(imgsz))
        if batch:
            task.parameters["batch"] = str(int(batch))
        task.extra_args = extra_params or {}

        run_descriptor = self.artifact_manager.assign_run(task, category="val")
        task.append_log("INFO", f"使用数据集 {dataset_path}")
        if task.extra_args:
            task.append_log("INFO", f"附加参数: {task.extra_args}")
        task.append_log("INFO", f"手动验证 {weight_path.name}")

        summary = self.orchestrator.validator.run_validation(
            task,
            weight_path=weight_path,
            name_suffix="manual",
        )
        return summary, task.logs, run_descriptor
