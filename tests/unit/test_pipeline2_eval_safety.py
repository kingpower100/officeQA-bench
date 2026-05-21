import pytest

from src.pipeline2.aggregation.summarizer import summarize_by_experiment
from src.pipeline2.orchestrator import (
    EvaluationOrchestrator,
    _validate_eval_diagnostics,
    build_eval_diagnostics,
)
from src.pipeline2.schemas.eval_config_schema import EvalConfig


def _cfg(retrieval_only: bool = False) -> EvalConfig:
    return EvalConfig.model_validate(
        {
            "evaluation": {
                "eval_run_id": "eval",
                "retrieval_only": retrieval_only,
                "retrieval_eval_field": "retrieved_file_names",
            },
            "inputs": {"rag_outputs": []},
            "retrieval": {"k": 1, "ks": [1]},
        }
    )


def _rag_row(qid: str = "q1", answer: str = "100", files=None) -> dict:
    return {
        "question_id": qid,
        "experiment_id": "exp",
        "question": "Q?",
        "generated_answer": answer,
        "retrieved_original_context_ids": ["chunk1"],
        "retrieved_file_names": ["doc.txt"] if files is None else files,
    }


def test_empty_pipeline1_results_fail():
    cfg = _cfg()
    diagnostics = build_eval_diagnostics([], [{"uid": "q1", "answer": "100"}], [{"id": "q1", "context_id": ["doc.txt"]}], {"q1": {"uid": "q1", "answer": "100"}}, {"q1": ["doc.txt"]}, cfg)

    with pytest.raises(ValueError, match="zero Pipeline 1 result rows"):
        _validate_eval_diagnostics(diagnostics, cfg)


def test_zero_qa_intersection_fails():
    cfg = _cfg()
    diagnostics = build_eval_diagnostics([_rag_row("q_missing")], [{"uid": "q1", "answer": "100"}], [{"id": "q_missing", "context_id": ["doc.txt"]}], {"q1": {"uid": "q1", "answer": "100"}}, {"q_missing": ["doc.txt"]}, cfg)

    with pytest.raises(ValueError, match="zero matching question IDs.*QA"):
        _validate_eval_diagnostics(diagnostics, cfg)


def test_zero_gold_intersection_fails():
    cfg = _cfg()
    diagnostics = build_eval_diagnostics([_rag_row("q1")], [{"uid": "q1", "answer": "100"}], [], {"q1": {"uid": "q1", "answer": "100"}}, {}, cfg)

    with pytest.raises(ValueError, match="zero matching question IDs.*gold"):
        _validate_eval_diagnostics(diagnostics, cfg)


def test_missing_generated_answer_for_all_rows_fails():
    cfg = _cfg()
    diagnostics = build_eval_diagnostics([_rag_row("q1", answer="")], [{"uid": "q1", "answer": "100"}], [{"id": "q1", "context_id": ["doc.txt"]}], {"q1": {"uid": "q1", "answer": "100"}}, {"q1": ["doc.txt"]}, cfg)

    with pytest.raises(ValueError, match="no generated_answer"):
        _validate_eval_diagnostics(diagnostics, cfg)


def test_missing_retrieved_field_for_all_rows_fails():
    cfg = _cfg()
    diagnostics = build_eval_diagnostics([_rag_row("q1", files=[])], [{"uid": "q1", "answer": "100"}], [{"id": "q1", "context_id": ["doc.txt"]}], {"q1": {"uid": "q1", "answer": "100"}}, {"q1": ["doc.txt"]}, cfg)

    with pytest.raises(ValueError, match="no non-empty values"):
        _validate_eval_diagnostics(diagnostics, cfg)


def test_missing_retrieval_eval_field_raises_during_evaluation():
    cfg = _cfg()
    row = _rag_row("q1")
    del row["retrieved_file_names"]

    with pytest.raises(ValueError, match="retrieval_eval_field='retrieved_file_names' is missing"):
        EvaluationOrchestrator()._evaluate_rows([row], {"q1": {"uid": "q1", "answer": "100"}}, {"q1": ["doc.txt"]}, cfg)


def test_normal_small_fixture_evaluates_correct_row_count_and_summary():
    cfg = _cfg()
    rows = [_rag_row("q1", "100"), _rag_row("q2", "200", ["wrong.txt"])]
    qa_by_id = {"q1": {"uid": "q1", "answer": "100"}, "q2": {"uid": "q2", "answer": "200"}}
    gold_by_id = {"q1": ["doc.txt"], "q2": ["doc.txt"]}

    diagnostics = build_eval_diagnostics(rows, list(qa_by_id.values()), [{"id": "q1"}, {"id": "q2"}], qa_by_id, gold_by_id, cfg)
    _validate_eval_diagnostics(diagnostics, cfg)
    evaluated = EvaluationOrchestrator()._evaluate_rows(rows, qa_by_id, gold_by_id, cfg)
    summary = summarize_by_experiment(evaluated)

    assert len(evaluated) == 2
    assert summary[0]["n_questions"] == len(evaluated)
    assert summary[0]["mean_hit_at_1"] == 0.5
