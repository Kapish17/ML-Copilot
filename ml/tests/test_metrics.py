"""Tests for the evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ml.errors import InvalidMetricError
from ml.evaluation.metrics import (
    DEFAULT_PRIMARY_METRIC,
    MetricDirection,
    evaluate_classification,
    evaluate_predictions,
    evaluate_regression,
    get_metric,
    is_better,
    metrics_for_task,
    resolve_primary_metric,
)
from ml.features.types import TaskType

BINARY_TRUTH = [0, 0, 1, 1, 1, 1]
BINARY_PREDICTIONS = [0, 1, 1, 1, 0, 1]


def test_binary_classification_metrics() -> None:
    """Accuracy, precision, recall and F1 match the confusion matrix."""
    metrics = evaluate_classification(BINARY_TRUTH, BINARY_PREDICTIONS)

    assert metrics.get("accuracy") == pytest.approx(4 / 6)
    assert metrics.get("precision") == pytest.approx(0.75)
    assert metrics.get("recall") == pytest.approx(0.75)
    assert metrics.get("f1") == pytest.approx(0.75)


def test_binary_confusion_matrix_and_context() -> None:
    """A score is reported with the context needed to read it."""
    details = evaluate_classification(BINARY_TRUTH, BINARY_PREDICTIONS).classification

    assert details is not None
    assert details.confusion_matrix == ((1, 1), (1, 3))
    assert details.averaging == "binary"
    assert details.positive_label == "1"
    assert details.class_count == 2
    assert details.class_distribution == {"1": 4, "0": 2}


def test_accuracy_is_not_the_only_metric() -> None:
    """Every classification metric is attempted, not just accuracy."""
    metrics = evaluate_classification(BINARY_TRUTH, BINARY_PREDICTIONS)
    reported = set(metrics.values) | set(metrics.unavailable)

    assert reported == {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_roc_auc_uses_the_positive_class_column() -> None:
    """ROC-AUC is computed from the probability of the positive class."""
    scores = np.array([[0.9, 0.1], [0.6, 0.4], [0.65, 0.35], [0.2, 0.8]])
    metrics = evaluate_classification(
        [0, 0, 1, 1], [0, 0, 1, 1], y_score=scores, score_labels=[0, 1]
    )
    assert metrics.get("roc_auc") == pytest.approx(0.75)


def test_roc_auc_is_unavailable_without_probabilities() -> None:
    """A model that cannot produce scores reports why, instead of crashing."""
    metrics = evaluate_classification(BINARY_TRUTH, BINARY_PREDICTIONS)

    assert metrics.get("roc_auc") is None
    assert "probabilities" in (metrics.reason_unavailable("roc_auc") or "")


def test_roc_auc_is_unavailable_for_a_single_class() -> None:
    """ROC-AUC is undefined when the test set holds one class."""
    scores = np.array([[0.2, 0.8], [0.3, 0.7]])
    metrics = evaluate_classification(
        [1, 1], [1, 1], y_score=scores, score_labels=[0, 1]
    )

    assert metrics.get("roc_auc") is None
    assert "one class" in (metrics.reason_unavailable("roc_auc") or "")
    assert metrics.get("accuracy") == pytest.approx(1.0), "other metrics still computed"


def test_multiclass_uses_macro_averaging() -> None:
    """More than two classes are averaged so no class dominates."""
    metrics = evaluate_classification(
        ["a", "b", "c", "a", "b", "c"], ["a", "b", "c", "a", "c", "c"]
    )
    details = metrics.classification

    assert details is not None
    assert details.averaging == "macro"
    assert details.class_count == 3
    assert details.positive_label is None
    assert metrics.get("f1") is not None


def test_multiclass_roc_auc_is_computed_one_versus_rest() -> None:
    """Multiclass ROC-AUC uses one-vs-rest with macro averaging."""
    scores = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.7, 0.2, 0.1]]
    )
    metrics = evaluate_classification(
        ["a", "b", "c", "a"],
        ["a", "b", "c", "a"],
        y_score=scores,
        score_labels=["a", "b", "c"],
    )
    assert metrics.get("roc_auc") == pytest.approx(1.0)


def test_regression_metrics() -> None:
    """MAE, MSE, RMSE and R² match the arithmetic."""
    metrics = evaluate_regression([1.0, 2.0, 3.0, 4.0], [1.0, 3.0, 2.0, 4.0])

    assert metrics.get("mae") == pytest.approx(0.5)
    assert metrics.get("mse") == pytest.approx(0.5)
    assert metrics.get("rmse") == pytest.approx(math.sqrt(0.5))
    assert metrics.get("r2") == pytest.approx(0.6)


def test_perfect_regression_predictions() -> None:
    """A perfect fit has zero error and an R² of one."""
    metrics = evaluate_regression([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])

    assert metrics.get("rmse") == pytest.approx(0.0)
    assert metrics.get("r2") == pytest.approx(1.0)


def test_r2_is_unavailable_for_a_constant_target() -> None:
    """R² divides by the target's variance, so a flat target has none."""
    metrics = evaluate_regression([5.0, 5.0, 5.0], [4.0, 5.0, 6.0])

    assert metrics.get("r2") is None
    assert "does not vary" in (metrics.reason_unavailable("r2") or "")
    assert metrics.get("mae") == pytest.approx(2 / 3), "error metrics still computed"


def test_evaluate_predictions_dispatches_on_task() -> None:
    """The right metric set is chosen from the task type."""
    classification = evaluate_predictions(TaskType.CLASSIFICATION, ["a", "b"], ["a", "b"])
    regression = evaluate_predictions(TaskType.REGRESSION, [1.0, 2.0], [1.0, 2.0])

    assert classification.classification is not None
    assert regression.classification is None
    assert "rmse" in regression.values


def test_evaluate_predictions_rejects_an_unscorable_task() -> None:
    """An unresolved task type has no metrics and says so."""
    with pytest.raises(InvalidMetricError):
        evaluate_predictions(TaskType.AUTO, [1], [1])


def test_metric_directions_are_declared() -> None:
    """Error metrics are lower-is-better; score metrics are higher-is-better."""
    assert get_metric("rmse", TaskType.REGRESSION).direction is (
        MetricDirection.LOWER_IS_BETTER
    )
    assert get_metric("mae", TaskType.REGRESSION).higher_is_better is False
    assert get_metric("r2", TaskType.REGRESSION).higher_is_better is True
    assert get_metric("f1", TaskType.CLASSIFICATION).higher_is_better is True


def test_unknown_metric_is_rejected() -> None:
    """A metric that does not exist for the task fails clearly."""
    with pytest.raises(InvalidMetricError) as exc_info:
        get_metric("rmse", TaskType.CLASSIFICATION)
    assert "f1" in exc_info.value.details["available"]


def test_default_primary_metrics() -> None:
    """The defaults are F1 for classification and RMSE for regression."""
    assert DEFAULT_PRIMARY_METRIC[TaskType.CLASSIFICATION] == "f1"
    assert DEFAULT_PRIMARY_METRIC[TaskType.REGRESSION] == "rmse"
    assert resolve_primary_metric(TaskType.CLASSIFICATION, None).key == "f1"
    assert resolve_primary_metric(TaskType.REGRESSION, None).key == "rmse"


def test_primary_metric_is_configurable() -> None:
    """A caller can rank by something other than the default."""
    assert resolve_primary_metric(TaskType.REGRESSION, "mae").key == "mae"
    assert resolve_primary_metric(TaskType.CLASSIFICATION, "roc_auc").key == "roc_auc"


def test_primary_metric_override_is_validated() -> None:
    """A nonsense primary metric is refused rather than ignored."""
    with pytest.raises(InvalidMetricError):
        resolve_primary_metric(TaskType.CLASSIFICATION, "rmse")


@pytest.mark.parametrize(
    ("candidate", "incumbent", "direction", "expected"),
    [
        (0.9, 0.8, MetricDirection.HIGHER_IS_BETTER, True),
        (0.7, 0.8, MetricDirection.HIGHER_IS_BETTER, False),
        (1.0, 2.0, MetricDirection.LOWER_IS_BETTER, True),
        (3.0, 2.0, MetricDirection.LOWER_IS_BETTER, False),
        (None, 2.0, MetricDirection.LOWER_IS_BETTER, False),
        (2.0, None, MetricDirection.LOWER_IS_BETTER, True),
    ],
)
def test_is_better_respects_direction(
    candidate: float | None,
    incumbent: float | None,
    direction: MetricDirection,
    expected: bool,
) -> None:
    """Comparison never assumes larger is better."""
    assert is_better(candidate, incumbent, direction) is expected


def test_metrics_are_serialisable() -> None:
    """Metrics render as plain values for reporting."""
    payload = evaluate_classification(BINARY_TRUTH, BINARY_PREDICTIONS).as_dict()

    assert payload["task_type"] == "classification"
    assert payload["sample_count"] == 6
    assert payload["classification"]["confusion_matrix"] == [[1, 1], [1, 3]]
    assert "roc_auc" in payload["unavailable"]


def test_every_task_has_metrics_defined() -> None:
    """Both supported tasks expose a metric set."""
    assert len(metrics_for_task(TaskType.CLASSIFICATION)) == 5
    assert len(metrics_for_task(TaskType.REGRESSION)) == 4
    assert metrics_for_task(TaskType.AUTO) == ()
