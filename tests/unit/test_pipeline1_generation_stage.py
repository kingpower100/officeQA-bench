import json

from src.pipeline1.generation.base import GenerationResult
from src.pipeline1.observability.events import EventWriter
from src.pipeline1.schemas.config_schema import PipelineConfig
from src.pipeline1.schemas.query import QueryRecord
from src.pipeline1.schemas.retrieval import RetrievalItem
from src.pipeline1.stages.base import StageInput
from src.pipeline1.stages.generation_stage import GenerationStage
from src.pipeline1.stages.retrieval_stage import RetrievalRow


def test_generation_stage_produces_output_record_compatible_row():
    cfg = _cfg()
    output = GenerationStage(cfg, _Retriever(), generator_factory=lambda config: _Generator("42")).run(
        StageInput({"retrieval_rows": [_retrieval_row()], "final_top_k": 1})
    )

    record = output.generation_rows[0].output_record
    assert record.question_id == "q1"
    assert record.generated_answer == "42"
    assert record.retrieved_chunk_ids == ["c1"]
    assert record.retrieved_original_context_ids == ["ctx1"]
    assert record.raw_retrieved_context_ids == ["c1"]
    assert record.input_tokens == 5
    assert record.output_tokens == 1
    assert record.total_tokens == 6
    assert record.error is None


def test_generation_stage_error_path_is_preserved(tmp_path):
    cfg = _cfg()
    events = EventWriter(tmp_path / "events.jsonl", experiment_id="exp")
    output = GenerationStage(cfg, _Retriever(), event_writer=events, generator_factory=lambda config: _FailingGenerator()).run(
        StageInput({"retrieval_rows": [_retrieval_row()], "final_top_k": 1})
    )
    events.close()

    record = output.generation_rows[0].output_record
    event_rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert record.generated_answer == ""
    assert record.input_tokens == 0
    assert record.output_tokens == 0
    assert record.total_tokens == 0
    assert record.error == "boom"
    assert any(row["event_type"] == "generation_error" for row in event_rows)
    assert any(row["event_type"] == "generation_end" and row["metrics"]["generation_failed"] is True for row in event_rows)


def test_generation_stage_preserves_prompt_context_diagnostics():
    cfg = _cfg(max_context_chars=5)
    output = GenerationStage(cfg, _Retriever(), generator_factory=lambda config: _Generator("ok")).run(
        StageInput({"retrieval_rows": [_retrieval_row(text="alpha beta gamma")], "final_top_k": 1})
    )

    record = output.generation_rows[0].output_record
    assert record.prompt_stats["chunks_before"] == 1
    assert record.prompt_stats["chunks_after"] in {0, 1}
    assert record.context_chars_before > record.context_chars_after
    assert record.prompt_tokens is not None


class _Generator:
    def __init__(self, answer):
        self.answer = answer

    def generate(self, prompt):
        return GenerationResult(answer=self.answer, input_tokens=5, output_tokens=1)


class _FailingGenerator:
    def generate(self, prompt):
        raise RuntimeError("boom")


class _Retriever:
    def extract_query_metadata(self, question):
        return None


def _retrieval_row(text="alpha"):
    item = RetrievalItem(
        chunk_id="c1",
        original_context_id="ctx1",
        text=text,
        score=1.0,
        dense_score=1.0,
        metadata={"document_id": "doc1", "file_name": "source.txt"},
    )
    return RetrievalRow(
        query=QueryRecord(question_id="q1", question="What is the answer?"),
        raw_retrieved=[item],
        raw_dense_retrieved=[item],
        raw_bm25_retrieved=[],
        fused_retrieved=[],
        retrieved=[item],
        retrieval_time_ms=2.0,
        reranker_used=False,
        retrieval_warnings=[],
        retrieval_diagnostics={"diagnostic": "ok"},
    )


def _cfg(max_context_chars=24000):
    return PipelineConfig.model_validate(
        {
            "experiment": {"experiment_id": "exp", "output_dir": "runs"},
            "data": {"documents_path": "documents.jsonl", "questions_path": "questions.jsonl"},
            "chunking": {"strategy": "fixed_word", "chunk_size": 10, "chunk_overlap": 0},
            "embedding": {"provider": "sentence_transformers", "model_name": "fake"},
            "index": {"type": "faiss", "metric": "cosine"},
            "retrieval": {"retriever_type": "dense", "top_k": 1, "fetch_k": 1},
            "reranker": {"enabled": False},
            "generation": {
                "provider": "ollama",
                "model_name": "fake",
                "system_prompt": "Use context.",
                "max_context_chars": max_context_chars,
                "max_chunk_chars": 8000,
            },
            "telemetry": {},
            "runtime": {},
        }
    )
