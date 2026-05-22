from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any


class EventType(StrEnum):
    PIPELINE_START = "pipeline_start"
    PIPELINE_END = "pipeline_end"
    PIPELINE_ERROR = "pipeline_error"
    DOCUMENT_LOAD_START = "document_load_start"
    DOCUMENT_LOAD_END = "document_load_end"
    CHUNKING_START = "chunking_start"
    CHUNKING_END = "chunking_end"
    EMBEDDING_START = "embedding_start"
    EMBEDDING_END = "embedding_end"
    INDEX_BUILD_START = "index_build_start"
    INDEX_BUILD_END = "index_build_end"
    RETRIEVAL_START = "retrieval_start"
    RETRIEVAL_END = "retrieval_end"
    RERANK_START = "rerank_start"
    RERANK_END = "rerank_end"
    GENERATION_START = "generation_start"
    GENERATION_END = "generation_end"
    GENERATION_ERROR = "generation_error"
    OUTPUT_WRITE_START = "output_write_start"
    OUTPUT_WRITE_END = "output_write_end"


@dataclass(frozen=True)
class PipelineEvent:
    timestamp: str
    experiment_id: str
    run_id: str | None
    question_id: str | None
    stage: str
    event_type: str
    message: str
    duration_ms: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        experiment_id: str,
        stage: str,
        event_type: EventType | str,
        message: str,
        run_id: str | None = None,
        question_id: str | None = None,
        duration_ms: float | None = None,
        metrics: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> "PipelineEvent":
        return cls(
            timestamp=datetime.now(timezone.utc).isoformat(),
            experiment_id=experiment_id,
            run_id=run_id,
            question_id=question_id,
            stage=stage,
            event_type=str(event_type.value if isinstance(event_type, EventType) else event_type),
            message=message,
            duration_ms=duration_ms,
            metrics=dict(metrics or {}),
            diagnostics=dict(diagnostics or {}),
        )

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventWriter:
    def __init__(self, path: Path, experiment_id: str, run_id: str | None = None) -> None:
        self.path = path
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def write(
        self,
        stage: str,
        event_type: EventType | str,
        message: str,
        question_id: str | None = None,
        duration_ms: float | None = None,
        metrics: dict[str, Any] | None = None,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        event = PipelineEvent.create(
            experiment_id=self.experiment_id,
            run_id=self.run_id,
            question_id=question_id,
            stage=stage,
            event_type=event_type,
            message=message,
            duration_ms=duration_ms,
            metrics=metrics,
            diagnostics=diagnostics,
        )
        self.write_event(event)

    def write_event(self, event: PipelineEvent) -> None:
        self._file.write(json.dumps(event.to_json_dict(), ensure_ascii=False, default=str) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
