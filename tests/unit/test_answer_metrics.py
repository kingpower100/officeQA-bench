from src.pipeline2.metrics.answer_metrics import compute_answer_metrics


def test_answer_metrics_command_alias_smoke():
    metrics = compute_answer_metrics("100", "100")

    assert metrics["numeric_accuracy"] == 1.0
