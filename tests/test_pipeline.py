from churn_intelligence.data import generate_customer_data, validate_customer_data
from churn_intelligence.model import train_and_evaluate


def test_synthetic_data_is_valid_and_reproducible():
    first = generate_customer_data(500, seed=7)
    second = generate_customer_data(500, seed=7)
    validate_customer_data(first)
    assert first.equals(second)
    assert first["customer_id"].is_unique
    assert set(first["churn"].unique()).issubset({0, 1})


def test_training_produces_useful_outputs():
    result = train_and_evaluate(generate_customer_data(1_500, seed=11), seed=11)
    assert 0.65 <= result.metrics["roc_auc"] <= 1.0
    assert 0.0 <= result.metrics["precision"] <= 1.0
    assert 0.0 <= result.metrics["recall"] <= 1.0
    assert result.retention_queue["churn_risk"].is_monotonic_decreasing
    assert not result.retention_queue.empty

