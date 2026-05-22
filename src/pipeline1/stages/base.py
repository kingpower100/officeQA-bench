from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class StageInput:
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageOutput:
    stage_name: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStage(ABC):
    stage_name: str

    @abstractmethod
    def run(self, stage_input: StageInput) -> StageOutput:
        raise NotImplementedError
