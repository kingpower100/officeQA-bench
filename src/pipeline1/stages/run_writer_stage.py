from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.pipeline1.io.result_writer import ResultWriter
from src.pipeline1.stages.base import BaseStage, StageInput, StageOutput


@dataclass(frozen=True)
class RunWriterStageOutput(StageOutput):
    writer: ResultWriter | None = None
    existing_question_ids: set[str] | None = None


class RunWriterStage(BaseStage):
    stage_name = "run_writer"

    def __init__(self, run_dir: Path, save_csv: bool = True, logger=None, resume: bool = False) -> None:
        self.run_dir = run_dir
        self.save_csv = save_csv
        self.logger = logger
        self.resume = resume
        self.writer: ResultWriter | None = None

    def run(self, stage_input: StageInput | None = None) -> RunWriterStageOutput:
        self.writer = ResultWriter(self.run_dir, save_csv=self.save_csv, logger=self.logger)
        existing_ids = self.writer.load_existing_question_ids() if self.resume else set()
        return RunWriterStageOutput(
            stage_name=self.stage_name,
            artifacts={"writer": self.writer},
            diagnostics={"existing_question_ids": len(existing_ids)},
            metadata={"run_dir": str(self.run_dir), "save_csv": self.save_csv, "resume": self.resume},
            writer=self.writer,
            existing_question_ids=existing_ids,
        )

    def write(self, record) -> None:
        if self.writer is None:
            raise RuntimeError("RunWriterStage.run() must be called before write().")
        self.writer.write(record)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

    @staticmethod
    def output_row_counts(run_dir: Path) -> dict[str, int | None]:
        counts: dict[str, int | None] = {}
        for name in ("results.jsonl", "results.csv"):
            path = run_dir / name
            if not path.exists():
                counts[name] = None
                continue
            with path.open("r", encoding="utf-8") as f:
                row_count = sum(1 for line in f if line.strip())
            counts[name] = max(0, row_count - 1) if name.endswith(".csv") and row_count else row_count
        return counts
