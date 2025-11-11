from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass
class VisualizationPayload:
    """传递给可视化实现的数据载体。"""

    run_name: str
    run_path: Path
    metrics: Sequence[Dict[str, float]]


class VisualizationProvider(ABC):
    """可视化插件接口，后续可按需扩展。"""

    name: str

    @abstractmethod
    def render(self, payload: VisualizationPayload) -> Dict[str, object]:
        """返回绘图所需的数据/figure，交由前端自行选择控件展示。"""


_registry: List[VisualizationProvider] = []


def register_provider(provider: VisualizationProvider) -> None:
    _registry.append(provider)


def iter_providers() -> Iterable[VisualizationProvider]:
    return tuple(_registry)
