"""Tests for the naive baselines and the improvement over them."""

from __future__ import annotations

import pytest

from ml.evaluation.metrics import get_metric
from ml.features.types import TaskType
from ml.models.baselines import (
    build_baseline_estimator,
    compare_to_baseline,
    evaluate_baseline,
)
from ml.pipelines.result import PreparedDataset

F1 = get_metric("f1", TaskType.CLASSIFICATION)
RMSE = get_metric("rmse", TaskType.REGRESSION)


def test_majority_baseline_predicts_the_commonest_training_class(
    classification_prepared: PreparedDataset,
) -> None:
    """The classification baseline always answers with the majority label."""
    estimator = build_baseline_estimator(TaskType.CLASSIFICATION)
    estimator.fit(classification_prepared.X_train_raw, classification_prepared.y_train)
    predictions = estimator.predict(classification_prepared.X_test_raw)

    majority = classification_prepared.y_train.value_counts().idxmax()
    assert set(predictions) == {majority}


def test_majority_baseline_accuracy_is_the_majority_share(
    classification_prepared: PreparedDataset,
) -> None:
    """Its accuracy is exactly the share of test rows holding that label."""
    baseline = evaluate_baseline(classification_prepared)

    majority = classification_prepared.y_train.value_counts().idxmax()
    expected = float((classification_prepared.y_test == majority).mean())
    assert baseline.metrics.get("accuracy") == pytest.approx(expected)


def test_classification_baseline_is_described(
    classification_prepared: PreparedDataset,
) -> None:
    """The reference identifies itself and its strategy."""
    baseline = evaluate_baseline(classification_prepared)

    assert baseline.identifier == "majority_class_baseline"
    assert baseline.strategy == "most_frequent"
    assert baseline.task_type is TaskType.CLASSIFICATION
    assert baseline.metrics.get("f1") is not None


def test_mean_baseline_predicts_the_training_mean(
    regression_prepared: PreparedDataset,
) -> None:
    """The regression baseline always answers with the training mean."""
    estimator = build_baseline_estimator(TaskType.REGRESSION)
    estimator.fit(regression_prepared.X_train_raw, regression_prepared.y_train)
    predictions = estimator.predict(regression_prepared.X_test_raw)

    assert predictions == pytest.approx(regression_prepared.y_train.mean())


def test_mean_baseline_error_matches_the_arithmetic(
    regression_prepared: PreparedDataset,
) -> None:
    """Its MAE is the average distance from the training mean."""
    baseline = evaluate_baseline(regression_prepared)

    train_mean = regression_prepared.y_train.mean()
    expected = float((regression_prepared.y_test - train_mean).abs().mean())
    assert baseline.metrics.get("mae") == pytest.approx(expected)
    assert baseline.identifier == "mean_baseline"
    assert baseline.strategy == "mean"


def test_baseline_is_serialisable(classification_prepared: PreparedDataset) -> None:
    """The baseline renders as plain values."""
    payload = evaluate_baseline(classification_prepared).as_dict()

    assert payload["identifier"] == "majority_class_baseline"
    assert payload["metrics"]["task_type"] == "classification"


def test_improvement_is_positive_when_a_score_metric_rises() -> None:
    """For a higher-is-better metric, beating the baseline is positive."""
    comparison = compare_to_baseline(F1, model_value=0.9, baseline_value=0.6)

    assert comparison.absolute_improvement == pytest.approx(0.3)
    assert comparison.relative_improvement == pytest.approx(0.5)
    assert comparison.beats_baseline is True


def test_improvement_is_negative_when_a_score_metric_falls() -> None:
    """A model worse than the baseline reports a negative improvement."""
    comparison = compare_to_baseline(F1, model_value=0.4, baseline_value=0.6)

    assert comparison.absolute_improvement == pytest.approx(-0.2)
    assert comparison.beats_baseline is False


def test_improvement_is_positive_when_an_error_metric_falls() -> None:
    """For a lower-is-better metric, reducing the error is still positive."""
    comparison = compare_to_baseline(RMSE, model_value=4.0, baseline_value=10.0)

    assert comparison.absolute_improvement == pytest.approx(6.0)
    assert comparison.relative_improvement == pytest.approx(0.6)
    assert comparison.beats_baseline is True


def test_improvement_is_negative_when_an_error_metric_rises() -> None:
    """A model with more error than the baseline is worse, not better."""
    comparison = compare_to_baseline(RMSE, model_value=12.0, baseline_value=10.0)

    assert comparison.absolute_improvement == pytest.approx(-2.0)
    assert comparison.beats_baseline is False


def test_improvement_is_unknown_when_a_value_is_missing() -> None:
    """An unavailable metric produces no improvement claim."""
    comparison = compare_to_baseline(F1, model_value=None, baseline_value=0.6)

    assert comparison.absolute_improvement is None
    assert comparison.beats_baseline is None


def test_relative_improvement_is_skipped_when_the_baseline_is_zero() -> None:
    """Dividing by a zero baseline is avoided rather than producing infinity."""
    comparison = compare_to_baseline(F1, model_value=0.4, baseline_value=0.0)

    assert comparison.absolute_improvement == pytest.approx(0.4)
    assert comparison.relative_improvement is None


def test_comparison_is_serialisable() -> None:
    """The comparison renders as plain values, direction included."""
    payload = compare_to_baseline(RMSE, model_value=4.0, baseline_value=10.0).as_dict()

    assert payload["metric"] == "rmse"
    assert payload["direction"] == "lower_is_better"
    assert payload["beats_baseline"] is True
