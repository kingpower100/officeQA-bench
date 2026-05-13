from __future__ import annotations

from statistics import mean
from typing import Any


def summarize_by_experiment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_experiment: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_experiment.setdefault(str(row.get("experiment_id", "")), []).append(row)

    summaries = []
    for experiment_id, group in sorted(by_experiment.items()):
        metric_cols = _dynamic_metric_columns(group)
        successful_rows = [row for row in group if not row.get("pipeline1_error")]
        summary = {
            "experiment_id": experiment_id,
            "n_questions": len(group),
            "pipeline_success_rate": _mean([1.0 if not row.get("pipeline1_error") else 0.0 for row in group]),
            "eval_success_rate": _mean([1.0 if not row.get("evaluation_errors") else 0.0 for row in group]),
        }
        for col in metric_cols:
            summary[f"mean_{col}"] = _mean([row.get(col) for row in successful_rows if row.get(col) is not None])
        for col in ("numeric_accuracy",):
            summary[f"mean_{col}"] = _mean([row.get(col) for row in successful_rows if row.get(col) is not None])
        for col in ("total_latency_ms", "total_tokens", "estimated_cost"):
            summary[f"mean_{col}"] = _mean([row.get(col) for row in successful_rows if row.get(col) is not None])
        summaries.append(summary)
    return summaries


def _dynamic_metric_columns(rows: list[dict[str, Any]]) -> list[str]:
    prefixes = ("hit_at_", "recall_at_", "precision_at_", "mrr_at_")
    cols = []
    for prefix in prefixes:
        names = sorted({key for row in rows for key in row if key.startswith(prefix)})
        if names:
            cols.append(names[0])
    return cols


def _mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return mean(numeric)
