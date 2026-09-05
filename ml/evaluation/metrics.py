"""Task-appropriate evaluation metrics.

Two ideas run through this module.

**Direction matters.** Every metric declares whether higher or lower is better.
Ranking code reads that declaration instead of assuming, which is what stops a
comparison from proudly selecting the worst RMSE.

**A missing metric is not a crash.** ROC-AUC is undefined for a test set with a
single class, and R² is undefined when the target never varies. Rather than
raising or silently returning a meaningless number, those metrics are reported
as unavailable with the reason, and everything else is still computed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from ml.errors import InvalidMetricError
from ml.features.types import TaskType

#: Averaging used when a classification problem has exactly two classes.
BINARY_AVERAGING = "binary"
#: Averaging used for more than two classes; treats every class equally.
MULTICLASS_AVERAGING = "macro"


class MetricDirection(str, Enum):
    """Whether a larger or a smaller value means a better model."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


@dataclass(frozen=True)
class MetricDefinition:
    """One metric: its identity, the task it belongs to, and its direction."""

    key: str
    display_name: str
    task_type: TaskType
    direction: MetricDirection
    description: str

    @property
    def higher_is_better(self) -> bool:
        """True when a larger value is a better result."""
        return self.direction is MetricDirection.HIGHER_IS_BETTER


CLASSIFICATION_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        "accuracy",
        "Accuracy",
        TaskType.CLASSIFICATION,
        MetricDirection.HIGHER_IS_BETTER,
        "Share of correct predictions. Misleading on imbalanced data.",
    ),
    MetricDefinition(
        "precision",
        "Precision",
        TaskType.CLASSIFICATION,
        MetricDirection.HIGHER_IS_BETTER,
        "Of the rows predicted positive, how many were positive.",
    ),
    MetricDefinition(
        "recall",
        "Recall",
        TaskType.CLASSIFICATION,
        MetricDirection.HIGHER_IS_BETTER,
        "Of the truly positive rows, how many were found.",
    ),
    MetricDefinition(
        "f1",
        "F1",
        TaskType.CLASSIFICATION,
        MetricDirection.HIGHER_IS_BETTER,
        "Harmonic mean of precision and recall; the default primary metric.",
    ),
    MetricDefinition(
        "roc_auc",
        "ROC-AUC",
        TaskType.CLASSIFICATION,
        MetricDirection.HIGHER_IS_BETTER,
        "Ranking quality across thresholds. Needs predicted probabilities.",
    ),
)

REGRESSION_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        "mae",
        "Mean absolute error",
        TaskType.REGRESSION,
        MetricDirection.LOWER_IS_BETTER,
        "Average absolute error, in the target's own units. Lower is better.",
    ),
    MetricDefinition(
        "mse",
        "Mean squared error",
        TaskType.REGRESSION,
        MetricDirection.LOWER_IS_BETTER,
        "Average squared error; punishes large mistakes. Lower is better.",
    ),
    MetricDefinition(
        "rmse",
        "Root mean squared error",
        TaskType.REGRESSION,
        MetricDirection.LOWER_IS_BETTER,
        "Square root of the MSE, back in the target's units. Lower is better.",
    ),
    MetricDefinition(
        "r2",
        "R²",
        TaskType.REGRESSION,
        MetricDirection.HIGHER_IS_BETTER,
        "Share of variance explained; 0 matches a mean prediction. Higher is better.",
    ),
)

_METRICS_BY_TASK = {
    TaskType.CLASSIFICATION: CLASSIFICATION_METRICS,
    TaskType.REGRESSION: REGRESSION_METRICS,
}

#: The metric used to rank models when the caller does not choose one.
#:
#: F1 rather than accuracy for classification, because accuracy flatters a
#: model that only ever predicts the majority class. RMSE rather than MAE for
#: regression, because it is in the target's units and penalises the large
#: errors that usually matter most; ``mae`` is one override away.
DEFAULT_PRIMARY_METRIC = {
    TaskType.CLASSIFICATION: "f1",
    TaskType.REGRESSION: "rmse",
}


#: Every defined metric, indexed by key. Keys are unique across tasks, which
#: lets a label be found without knowing which task produced the number.
_METRICS_BY_KEY = {
    definition.key: definition
    for definitions in _METRICS_BY_TASK.values()
    for definition in definitions
}


def metrics_for_task(task_type: TaskType) -> tuple[MetricDefinition, ...]:
    """Return every metric defined for a task type."""
    return _METRICS_BY_TASK.get(task_type, ())


def metric_label(key: str) -> str:
    """Return a metric's display name for a heading or a sentence.

    Anything reading a stored record — a comparison table, an experiment
    summary — has the metric key but not always the task that produced it, and
    ``CV F1`` reads better than ``CV f1``.

    Args:
        key: A metric identifier such as ``"f1"`` or ``"rmse"``.

    Returns:
        str: The display name, or the key unchanged when no metric of that
        name is defined. Unlike :func:`get_metric` this never raises: a label
        is presentation, and a record naming a metric this code no longer
        defines should still be readable.
    """
    definition = _METRICS_BY_KEY.get(key)
    return definition.display_name if definition is not None else key


def format_metric_value(
    value: Any, *, digits: int = 4, missing: str = "-"
) -> str:
    """Render one metric value for a table or a sentence.

    Args:
        value: The number, or ``None`` when the metric was not available.
        digits: Decimal places for values below a thousand. Larger values —
            an RMSE in dollars, say — are shown with thousands separators and
            two decimals instead, where four would be noise.
        missing: What to print when there is no number.

    Returns:
        str: The formatted value, or ``missing``. A non-finite value is
        treated as missing rather than printed as ``nan``.
    """
    number = _finite(value)
    if number is None:
        return missing
    if abs(number) >= 1000:
        return f"{number:,.2f}"
    return f"{number:.{digits}f}"


def format_metric_spread(
    value: Any, spread: Any, *, digits: int = 4, missing: str = "-"
) -> str:
    """Render a score with its variability, as ``0.8421 ± 0.0310``.

    The two numbers belong together: a mean across folds without the spread
    between them invites more confidence than the run earned. The ``±`` is a
    standard deviation across folds — a measure of how much the folds
    disagreed, not a confidence interval.
    """
    number = _finite(value)
    centre = format_metric_value(value, digits=digits, missing=missing)
    width = _finite(spread)
    if number is None or width is None:
        return centre
    # A spread is read against its centre, so both are rendered at the same
    # precision: "5,118.33 ± 221.43", never "5,118.33 ± 221.4277".
    if abs(number) >= 1000:
        return f"{centre} ± {width:,.2f}"
    return f"{centre} ± {format_metric_value(width, digits=digits)}"


def get_metric(key: str, task_type: TaskType) -> MetricDefinition:
    """Look up a metric definition for a task.

    Args:
        key: Metric identifier, such as ``"f1"`` or ``"rmse"``.
        task_type: The task the metric must belong to.

    Returns:
        MetricDefinition: The matching definition.

    Raises:
        InvalidMetricError: If the task has no such metric.
    """
    for definition in metrics_for_task(task_type):
        if definition.key == key:
            return definition
    available = [item.key for item in metrics_for_task(task_type)]
    raise InvalidMetricError(
        f"Unknown {task_type.value} metric '{key}'. Available: "
        + ", ".join(available)
        + ".",
        details={"metric": key, "task_type": task_type.value, "available": available},
    )


def resolve_primary_metric(task_type: TaskType, requested: str | None) -> MetricDefinition:
    """Return the metric used to rank models, honouring an override.

    Args:
        task_type: The task being solved.
        requested: An explicit metric key, or ``None`` for the default.

    Returns:
        MetricDefinition: The primary metric.

    Raises:
        InvalidMetricError: If the requested metric does not exist.
    """
    key = requested or DEFAULT_PRIMARY_METRIC.get(task_type)
    if key is None:
        raise InvalidMetricError(
            f"No primary metric is defined for task type '{task_type.value}'.",
            details={"task_type": task_type.value},
        )
    return get_metric(key, task_type)


def is_better(
    candidate: float | None, incumbent: float | None, direction: MetricDirection
) -> bool:
    """Return True when ``candidate`` beats ``incumbent`` for this direction.

    A missing candidate never wins; a missing incumbent always loses.
    """
    if candidate is None:
        return False
    if incumbent is None:
        return True
    if direction is MetricDirection.HIGHER_IS_BETTER:
        return candidate > incumbent
    return candidate < incumbent


def _finite(value: Any) -> float | None:
    """Return a finite float, or ``None`` when the value cannot be reported."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class ClassificationDetails:
    """Context a classification score cannot be read without."""

    class_count: int
    class_labels: tuple[str, ...]
    class_distribution: dict[str, int]
    confusion_matrix: tuple[tuple[int, ...], ...]
    averaging: str
    positive_label: str | None

    def as_dict(self) -> dict[str, Any]:
        """Render the details as plain, JSON-friendly values."""
        return {
            "class_count": self.class_count,
            "class_labels": list(self.class_labels),
            "class_distribution": dict(self.class_distribution),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "averaging": self.averaging,
            "positive_label": self.positive_label,
        }


@dataclass(frozen=True)
class EvaluationMetrics:
    """Every metric that could be computed, plus why the rest could not."""

    task_type: TaskType
    sample_count: int
    values: dict[str, float] = field(default_factory=dict)
    unavailable: dict[str, str] = field(default_factory=dict)
    classification: ClassificationDetails | None = None

    def get(self, key: str) -> float | None:
        """Return a metric value, or ``None`` when it was not available."""
        return self.values.get(key)

    def reason_unavailable(self, key: str) -> str | None:
        """Return why a metric is missing, if it is."""
        return self.unavailable.get(key)

    def as_dict(self) -> dict[str, Any]:
        """Render the metrics as plain, JSON-friendly values."""
        return {
            "task_type": self.task_type.value,
            "sample_count": self.sample_count,
            "values": dict(self.values),
            "unavailable": dict(self.unavailable),
            "classification": self.classification.as_dict()
            if self.classification
            else None,
        }


def _label_strings(labels: Sequence[Any]) -> tuple[str, ...]:
    """Render class labels as strings for a stable, serialisable payload."""
    return tuple(str(label) for label in labels)


def _sorted_labels(*arrays: Sequence[Any]) -> list[Any]:
    """Return the union of the labels appearing in the given arrays, sorted."""
    seen: set[Any] = set()
    for array in arrays:
        seen.update(pd.Series(list(array)).dropna().tolist())
    try:
        return sorted(seen)
    except TypeError:  # pragma: no cover - mixed label types are unusual
        return sorted(seen, key=str)


def _roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray | None,
    score_labels: Sequence[Any] | None,
    positive_label: Any | None,
    observed_labels: list[Any],
) -> tuple[float | None, str | None]:
    """Compute ROC-AUC when it is mathematically meaningful.

    Returns the value, or ``None`` together with the reason it was skipped.
    ROC-AUC needs predicted scores, at least two classes actually present in
    the test set, and — for multiclass — a score column for every class.
    """
    if y_score is None or score_labels is None:
        return None, "the model does not expose predicted probabilities"
    if len(observed_labels) < 2:
        return None, "only one class is present in the test set"

    score_labels = list(score_labels)
    try:
        if len(score_labels) == 2:
            positive = positive_label if positive_label is not None else score_labels[-1]
            column = score_labels.index(positive)
            return _finite(roc_auc_score(y_true, y_score[:, column])), None
        if not set(observed_labels).issubset(score_labels):
            return None, "the test set contains classes the model never saw"
        value = roc_auc_score(
            y_true,
            y_score,
            multi_class="ovr",
            average=MULTICLASS_AVERAGING,
            labels=score_labels,
        )
        return _finite(value), None
    except (ValueError, IndexError) as exc:
        return None, f"scikit-learn could not compute it: {exc}"


def evaluate_classification(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    y_score: np.ndarray | None = None,
    score_labels: Sequence[Any] | None = None,
) -> EvaluationMetrics:
    """Score classification predictions.

    Binary problems use ``average="binary"`` against a positive class; anything
    with more classes uses macro averaging, which weights every class equally
    rather than letting the largest class dominate. The averaging that was used
    and the positive class are both reported, so a number is never ambiguous.

    Args:
        y_true: Observed labels.
        y_pred: Predicted labels.
        y_score: Predicted probabilities, one column per class, if available.
        score_labels: The classes matching the columns of ``y_score``, in order.

    Returns:
        EvaluationMetrics: Scores, class context and any skipped metrics.
    """
    truth = np.asarray(list(y_true))
    predictions = np.asarray(list(y_pred))
    observed = _sorted_labels(truth)
    # The label set includes the classes the model knows about, not only those
    # that happen to appear in the test set, so the confusion matrix keeps its
    # full shape when a rare class is absent from the sample.
    labels = _sorted_labels(truth, predictions, score_labels or ())
    is_binary = len(labels) <= 2

    averaging = BINARY_AVERAGING if is_binary else MULTICLASS_AVERAGING
    positive_label = labels[-1] if is_binary and labels else None

    shared: dict[str, Any] = {"zero_division": 0}
    if is_binary:
        shared |= {"average": BINARY_AVERAGING, "pos_label": positive_label}
    else:
        shared |= {"average": MULTICLASS_AVERAGING, "labels": labels}

    values: dict[str, float] = {}
    unavailable: dict[str, str] = {}

    for key, value in (
        ("accuracy", accuracy_score(truth, predictions)),
        ("precision", precision_score(truth, predictions, **shared)),
        ("recall", recall_score(truth, predictions, **shared)),
        ("f1", f1_score(truth, predictions, **shared)),
    ):
        number = _finite(value)
        if number is None:  # pragma: no cover - these are always finite
            unavailable[key] = "the metric could not be computed for this data"
        else:
            values[key] = number

    auc, reason = _roc_auc(truth, y_score, score_labels, positive_label, observed)
    if auc is None:
        unavailable["roc_auc"] = reason or "unavailable"
    else:
        values["roc_auc"] = auc

    counts = pd.Series(truth).value_counts()
    details = ClassificationDetails(
        class_count=len(observed),
        class_labels=_label_strings(labels),
        class_distribution={str(label): int(count) for label, count in counts.items()},
        confusion_matrix=tuple(
            tuple(int(cell) for cell in row)
            for row in confusion_matrix(truth, predictions, labels=labels)
        ),
        averaging=averaging,
        positive_label=str(positive_label) if positive_label is not None else None,
    )

    return EvaluationMetrics(
        task_type=TaskType.CLASSIFICATION,
        sample_count=int(truth.shape[0]),
        values=values,
        unavailable=unavailable,
        classification=details,
    )


def evaluate_regression(
    y_true: Sequence[Any], y_pred: Sequence[Any]
) -> EvaluationMetrics:
    """Score regression predictions.

    MAE, MSE and RMSE are error measures, so **lower is better**; R² is a share
    of explained variance, so **higher is better**, with 0 meaning "no better
    than always predicting the mean". R² is skipped when the target does not
    vary, because the quantity it divides by is then zero.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        EvaluationMetrics: Scores and any skipped metrics.
    """
    truth = np.asarray(list(y_true), dtype="float64")
    predictions = np.asarray(list(y_pred), dtype="float64")

    values: dict[str, float] = {}
    unavailable: dict[str, str] = {}

    mse = _finite(mean_squared_error(truth, predictions))
    for key, value in (
        ("mae", _finite(mean_absolute_error(truth, predictions))),
        ("mse", mse),
        ("rmse", math.sqrt(mse) if mse is not None and mse >= 0 else None),
    ):
        if value is None:  # pragma: no cover - defensive
            unavailable[key] = "the metric could not be computed for this data"
        else:
            values[key] = value

    if truth.size < 2 or float(np.var(truth)) == 0.0:
        unavailable["r2"] = "the target does not vary in the test set"
    else:
        r2 = _finite(r2_score(truth, predictions))
        if r2 is None:  # pragma: no cover - defensive
            unavailable["r2"] = "the metric could not be computed for this data"
        else:
            values["r2"] = r2

    return EvaluationMetrics(
        task_type=TaskType.REGRESSION,
        sample_count=int(truth.shape[0]),
        values=values,
        unavailable=unavailable,
    )


def evaluate_predictions(
    task_type: TaskType,
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    y_score: np.ndarray | None = None,
    score_labels: Sequence[Any] | None = None,
) -> EvaluationMetrics:
    """Score predictions with the metrics appropriate to the task.

    Raises:
        InvalidMetricError: If the task type has no metric set.
    """
    if task_type is TaskType.CLASSIFICATION:
        return evaluate_classification(
            y_true, y_pred, y_score=y_score, score_labels=score_labels
        )
    if task_type is TaskType.REGRESSION:
        return evaluate_regression(y_true, y_pred)
    raise InvalidMetricError(
        f"No metrics are defined for task type '{task_type.value}'.",
        details={"task_type": task_type.value},
    )
