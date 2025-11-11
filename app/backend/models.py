from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class TaskMode(str, Enum):
    QUICK = "quick"
    SCRIPT = "script"

    @property
    def label(self) -> str:
        return "快速添加" if self is TaskMode.QUICK else "脚本添加"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def label(self) -> str:
        mapping = {
            TaskStatus.PENDING: "排队中",
            TaskStatus.RUNNING: "训练中",
            TaskStatus.VALIDATING: "验证中",
            TaskStatus.COMPLETED: "已完成",
            TaskStatus.FAILED: "失败",
            TaskStatus.CANCELED: "已取消",
        }
        return mapping[self]


@dataclass
class LogEntry:
    timestamp: dt.datetime
    level: str
    message: str


@dataclass
class ValidationSummary:
    metrics: Dict[str, float] = field(default_factory=dict)
    raw_output: str = ""
    success: bool = True


@dataclass
class TrainingTask:
    """训练任务的统一描述。"""

    mode: TaskMode
    parameters: Dict[str, str]
    extra_args: Dict[str, str] = field(default_factory=dict)
    script_path: Optional[Path] = None
    script_args: Dict[str, str] = field(default_factory=dict)
    user_note: str = ""
    created_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    logs: List[LogEntry] = field(default_factory=list)
    run_name: Optional[str] = None
    run_dir: Optional[Path] = None
    validation: Optional[ValidationSummary] = None
    exit_code: Optional[int] = None
    cancel_requested: bool = False

    def append_log(self, level: str, message: str) -> None:
        self.logs.append(LogEntry(timestamp=dt.datetime.now(dt.timezone.utc), level=level, message=message))

    def short_description(self) -> str:
        base = self.parameters.get("model", "未指定模型")
        return f"{self.mode.label}-{base}-{self.id[:6]}"
