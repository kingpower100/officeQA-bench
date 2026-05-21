from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline2.io.jsonl import read_jsonl
from src.pipeline2.metrics.answer_metrics import compute_answer_metrics, resolve_ground_truth_answer
from src.pipeline2.orchestrator import (
    EvaluationOrchestrator,
    _gold_by_question,
    _index_by_id,
    _merge_gold_with_qa_fallback,
    _resolve,
    _validate_eval_diagnostics,
    build_eval_diagnostics,
)
from src.pipeline2.schemas.eval_config_schema import EvalConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether Pipeline 2 evaluates real Pipeline 1 outputs.")
    parser.add_argument("--config", required=True, help="Path to Pipeline 2 YAML config.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = EvalConfig.from_yaml(str(config_path))
    project_root = Path(__file__).resolve().parents[1]
    eval_dir = project_root / cfg.evaluation.output_dir / cfg.evaluation.eval_run_id

    print(f"resolved_config_path: {config_path}")
    print(f"eval_id: {cfg.evaluation.eval_run_id}")
    print(f"eval_output_dir: {eval_dir}")
    print(f"retrieval_eval_field: {cfg.evaluation.retrieval_eval_field}")

    rag_rows = []
    print("pipeline1_results:")
    for raw_path in cfg.inputs.rag_outputs:
        path = _resolve(project_root, raw_path)
        try:
            rows = read_jsonl(path)
        except Exception as ex:
            print(f"  - path: {path}")
            print(f"    diagnostic_status: FAIL: {ex}")
            raise SystemExit(1) from None
        rag_rows.extend(rows)
        experiments = sorted({str(row.get("experiment_id", "")) for row in rows})
        print(f"  - path: {path}")
        print(f"    rows: {len(rows)}")
        print(f"    experiment_ids: {experiments}")

    qa_path = _resolve(project_root, cfg.inputs.qa_path)
    gold_path = _resolve(project_root, cfg.inputs.gold_contexts_path)
    qa_rows = read_jsonl(qa_path)
    gold_rows = read_jsonl(gold_path) if gold_path.exists() else []
    qa_by_id = _index_by_id(qa_rows, require_answer=not cfg.evaluation.retrieval_only)
    gold_by_id = _merge_gold_with_qa_fallback(_gold_by_question(gold_rows), qa_by_id)
    diagnostics = build_eval_diagnostics(rag_rows, qa_rows, gold_rows, qa_by_id, gold_by_id, cfg)

    print(f"qa_path: {qa_path}")
    print(f"qa_rows_loaded: {len(qa_rows)}")
    print(f"gold_contexts_path: {gold_path}")
    print(f"gold_rows_loaded: {len(gold_rows)}")
    print(f"matched_question_ids_qa: {diagnostics['qa_intersection_size']}")
    print(f"matched_question_ids_gold: {diagnostics['gold_intersection_size']}")
    print(f"evaluated_rows_expected: {diagnostics['evaluated_rows_expected']}")
    print(f"skipped_rows: {diagnostics['skipped_rows']}")
    print(f"generated_answer_coverage: {diagnostics['generated_answer_coverage']:.3f}")
    print(f"retrieved_field_coverage: {diagnostics['retrieved_field_coverage']:.3f}")
    print(f"first_5_pipeline1_question_ids: {diagnostics['first_5_pipeline1_question_ids']}")
    print(f"first_5_qa_question_ids: {diagnostics['first_5_qa_question_ids']}")
    print(f"first_5_gold_question_ids: {diagnostics['first_5_gold_question_ids']}")
    print(f"missing_in_qa_examples: {diagnostics['missing_in_qa_examples']}")
    print(f"missing_in_gold_examples: {diagnostics['missing_in_gold_examples']}")

    try:
        _validate_eval_diagnostics(diagnostics, cfg)
        status = "PASS"
    except Exception as ex:
        status = f"FAIL: {ex}"
    print(f"diagnostic_status: {status}")

    if status == "PASS":
        evaluated = EvaluationOrchestrator()._evaluate_rows(rag_rows, qa_by_id, gold_by_id, cfg)
        print("first_5_joined_examples:")
        for row, evaluated_row in zip(rag_rows[:5], evaluated[:5]):
            qid = str(row.get("question_id", ""))
            gold_answer = resolve_ground_truth_answer(row, qa_by_id)
            answer_metrics = compute_answer_metrics(str(row.get("generated_answer", "")), gold_answer)
            print(f"  - question_id: {qid}")
            print(f"    generated_answer: {row.get('generated_answer', '')}")
            print(f"    gold_answer: {gold_answer}")
            print(f"    exact_match: {answer_metrics['exact_match']}")
            print(f"    numeric_accuracy: {answer_metrics['numeric_accuracy']}")
            print(f"    retrieved_values: {evaluated_row.get('retrieval_eval_ids')}")
            print(f"    gold_values: {evaluated_row.get('gold_context_ids')}")


if __name__ == "__main__":
    main()
