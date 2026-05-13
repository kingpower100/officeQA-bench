from src.pipeline2.metrics.answer_metrics import compute_answer_metrics


def test_numeric_accuracy_matches_numbers_with_commas_and_currency():
    metrics = compute_answer_metrics("Revenue was $1,250.00.", "1250")

    assert metrics["numeric_accuracy"] == 1.0


def test_numeric_accuracy_is_none_without_numeric_ground_truth():
    metrics = compute_answer_metrics("Paris", "Paris")

    assert metrics["numeric_accuracy"] is None


def test_numeric_accuracy_detects_mismatch():
    metrics = compute_answer_metrics("42", "41")

    assert metrics["numeric_accuracy"] == 0.0


def test_yes_no_normalization_matches_one_zero():
    assert compute_answer_metrics("yes", "1")["numeric_accuracy"] == 1.0
    assert compute_answer_metrics("no", "0")["numeric_accuracy"] == 1.0


def test_percentage_decimal_normalization_matches_ratio():
    metrics = compute_answer_metrics("65.7%", "0.657")

    assert metrics["numeric_accuracy"] == 1.0
    assert metrics["generated_number"] == 0.657
    assert metrics["gold_number"] == 0.657


def test_numeric_debug_fields_explain_mismatch():
    metrics = compute_answer_metrics("2 million", "1000000")

    assert metrics["numeric_accuracy"] == 0.0
    assert metrics["generated_number"] == 2000000.0
    assert metrics["gold_number"] == 1000000.0
    assert metrics["absolute_error"] == 1000000.0
    assert metrics["answer_match_status"] == "mismatch"
