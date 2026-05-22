import json

from src.pipeline1.observability.events import EventType, EventWriter, PipelineEvent


def test_event_writer_writes_valid_jsonl(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, experiment_id="exp", run_id="run-1")

    writer.write(
        stage="retrieval",
        event_type=EventType.RETRIEVAL_END,
        message="Retrieval completed.",
        question_id="q1",
        duration_ms=12.5,
        metrics={"top_k": 5},
        diagnostics={"backend": "faiss"},
    )
    writer.close()

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "exp"
    assert rows[0]["run_id"] == "run-1"
    assert rows[0]["question_id"] == "q1"
    assert rows[0]["stage"] == "retrieval"
    assert rows[0]["event_type"] == "retrieval_end"
    assert rows[0]["message"] == "Retrieval completed."
    assert rows[0]["duration_ms"] == 12.5
    assert rows[0]["metrics"] == {"top_k": 5}
    assert rows[0]["diagnostics"] == {"backend": "faiss"}
    assert rows[0]["timestamp"]


def test_event_writer_optional_fields_do_not_break_writing(tmp_path):
    path = tmp_path / "events.jsonl"
    writer = EventWriter(path, experiment_id="exp")

    writer.write(stage="pipeline", event_type=EventType.PIPELINE_START, message="Started.")
    writer.close()

    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["run_id"] is None
    assert row["question_id"] is None
    assert row["duration_ms"] is None
    assert row["metrics"] == {}
    assert row["diagnostics"] == {}


def test_pipeline_event_contains_required_fields():
    event = PipelineEvent.create(
        experiment_id="exp",
        run_id="run-1",
        question_id=None,
        stage="chunking",
        event_type=EventType.CHUNKING_END,
        message="Chunking completed.",
    )

    payload = event.to_json_dict()
    for field in (
        "timestamp",
        "experiment_id",
        "run_id",
        "question_id",
        "stage",
        "event_type",
        "message",
        "duration_ms",
        "metrics",
        "diagnostics",
    ):
        assert field in payload
