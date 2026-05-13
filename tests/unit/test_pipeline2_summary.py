from src.pipeline2.aggregation.summarizer import summarize_by_experiment


def test_summary_aggregates_final_metrics_and_success_rates():
    rows = [
        {
            "experiment_id": "exp",
            "hit_at_3": 1.0,
            "recall_at_3": 0.5,
            "precision_at_3": 0.5,
            "mrr_at_3": 1.0,
            "numeric_accuracy": 1.0,
            "total_latency_ms": 100.0,
            "total_tokens": 10,
            "estimated_cost": 0.01,
            "pipeline1_error": None,
            "evaluation_errors": [],
        },
        {
            "experiment_id": "exp",
            "hit_at_3": 0.0,
            "recall_at_3": 0.0,
            "precision_at_3": 0.0,
            "mrr_at_3": 0.0,
            "numeric_accuracy": 0.0,
            "total_latency_ms": 0.0,
            "total_tokens": 0,
            "estimated_cost": 0.0,
            "pipeline1_error": "generation failed",
            "evaluation_errors": ["bad retrieved ids"],
        },
    ]

    summary = summarize_by_experiment(rows)[0]

    assert summary["n_questions"] == 2
    assert summary["pipeline_success_rate"] == 0.5
    assert summary["eval_success_rate"] == 0.5
    assert summary["mean_hit_at_3"] == 1.0
    assert summary["mean_recall_at_3"] == 0.5
    assert summary["mean_precision_at_3"] == 0.5
    assert summary["mean_mrr_at_3"] == 1.0
    assert summary["mean_numeric_accuracy"] == 1.0
    assert summary["mean_total_latency_ms"] == 100.0
    assert summary["mean_total_tokens"] == 10.0
    assert summary["mean_estimated_cost"] == 0.01
