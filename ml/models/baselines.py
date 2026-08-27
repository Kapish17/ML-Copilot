"""Trivial baselines, and the improvement a real model makes over them.

"F1 = 0.82" means nothing on its own. On a dataset where 82% of rows share one
label, always predicting that label scores about the same, and the model has
learned nothing. Every trained model is therefore scored against a deliberately
naive reference: the majority class for classification, the training mean for
regression.

The baseline runs through the same ``Pipeline(preprocessing, estimator)`` as a
real model. It ignores its inputs, so the preprocessing is redundant work, but
keeping one code path means the baseline is scored on exactly the same rows,
with exactly the same metric code, as everything it is compared against.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.pipeline import Pipeline

from ml.errors import InvalidMetricError
from ml.evaluation.metrics import (
    EvaluationMetrics,
    MetricDefinition,
    MetricDirection,
    evaluate_predictions,
)
from ml.features.types import TaskType
from ml.pipelines.preprocessing import clone_preprocessor
from ml.pipelines.result import PreparedDataset

#: Identifier and strategy of the reference model for each task.
BASELINE_STRATEGIES = {
    TaskType.CLASSIFICATION: "most_frequent",
    TaskType.REGRESSION: "mean",
}

BASELINE_IDENTIFIERS = {
    TaskType.CLASSIFICATION: "majority_class_baseline",
    TaskType.REGRESSION: "mean_baseline",
}

BASELINE_DISPLAY_NAMES = {
    TaskType.CLASSIFICATION: "Majority class baseline",
    TaskType.REGRESSION: "Mean prediction baseline",
}


@dataclass(frozen=True)
class BaselineResult:
    """How a deliberately naive model scores on the same test set."""

    identifier: str
    display_name: str
    strategy: str
    task_type: TaskType
    metrics: EvaluationMetrics

    def as_dict(self) -> dict[str, Any]:
        """Render the baseline as plain, JSON-friendly values."""
        return {
            "identifier": self.identifier,
            "display_name": self.display_name,
            "strategy": self.strategy,
            "task_type": self.task_type.value,
            "metrics": self.metrics.as_dict(),
        }


@dataclass(frozen=True)
class BaselineComparison:
    """A model's primary metric set beside the baseline's.

    ``absolute_improvement`` and ``relative_improvement`` are always signed so
    that **positive means the model is better than the baseline**, whichever
    direction the metric runs in. For an error metric such as RMSE, an
    improvement of 12.0 means the model's error is 12.0 lower.
    """

    metric: str
    direction: MetricDirection
    model_value: float | None
    baseline_value: float | None
    absolute_improvement: float | None
    relative_improvement: float | None
    beats_baseline: bool | None

    def as_dict(self) -> dict[str, Any]:
        """Render the comparison as plain, JSON-friendly values."""
        return {
            "metric": self.metric,
            "direction": self.direction.value,
            "model_value": self.model_value,
            "baseline_value": self.baseline_value,
            "absolute_improvement": self.absolute_improvement,
            "relative_improvement": self.relative_improvement,
            "beats_baseline": self.beats_baseline,
        }


def build_baseline_estimator(
    task_type: TaskType, *, random_state: int | None = None
) -> BaseEstimator:
    """Return the naive estimator used as the reference for a task.

    Args:
        task_type: The task being solved.
        random_state: Seed; the chosen strategies are deterministic, but it is
            passed through so the behaviour holds if a strategy changes.

    Returns:
        BaseEstimator: An unfitted dummy estimator.

    Raises:
        InvalidMetricError: If the task type has no defined baseline.
    """
    strategy = BASELINE_STRATEGIES.get(task_type)
    if strategy is None:
        raise InvalidMetricError(
            f"No baseline is defined for task type '{task_type.value}'.",
            details={"task_type": task_type.value},
        )
    if task_type is TaskType.CLASSIFICATION:
        return DummyClassifier(strategy=strategy, random_state=random_state)
    return DummyRegressor(strategy=strategy)


def evaluate_baseline(prepared: PreparedDataset) -> BaselineResult:
    """Fit the naive reference on the training rows and score it on the test set.

    Args:
        prepared: The dataset produced by the preprocessing layer.

    Returns:
        BaselineResult: The reference scores every model is measured against.
    """
    task_type = prepared.task_type
    estimator = build_baseline_estimator(
        task_type, random_state=prepared.config.random_state
    )
    pipeline = Pipeline(
        [
            ("preprocessing", clone_preprocessor(prepared.preprocessor)),
            ("model", estimator),
        ]
    )
    pipeline.fit(prepared.X_train_raw, prepared.y_train)
    predictions = pipeline.predict(prepared.X_test_raw)

    scores = None
    labels = None
    if task_type is TaskType.CLASSIFICATION and hasattr(pipeline, "predict_proba"):
        scores = pipeline.predict_proba(prepared.X_test_raw)
        labels = list(pipeline.classes_)

    metrics = evaluate_predictions(
        task_type,
        prepared.y_test,
        predictions,
        y_score=scores,
        score_labels=labels,
    )
    return BaselineResult(
        identifier=BASELINE_IDENTIFIERS[task_type],
        display_name=BASELINE_DISPLAY_NAMES[task_type],
        strategy=BASELINE_STRATEGIES[task_type],
        task_type=task_type,
        metrics=metrics,
    )


def compare_to_baseline(
    metric: MetricDefinition,
    model_value: float | None,
    baseline_value: float | None,
) -> BaselineComparison:
    """Express a model's score as an improvement over the baseline's.

    The sign convention is fixed by the metric's direction, so a caller never
    has to remember whether larger is better for the metric in hand.

    Args:
        metric: The metric being compared, carrying its direction.
        model_value: The model's score, or ``None`` if unavailable.
        baseline_value: The baseline's score, or ``None`` if unavailable.

    Returns:
        BaselineComparison: Both values and the signed improvement between them.
    """
    if model_value is None or baseline_value is None:
        return BaselineComparison(
            metric=metric.key,
            direction=metric.direction,
            model_value=model_value,
            baseline_value=baseline_value,
            absolute_improvement=None,
            relative_improvement=None,
            beats_baseline=None,
        )

    if metric.higher_is_better:
        absolute = model_value - baseline_value
    else:
        absolute = baseline_value - model_value

    relative = absolute / abs(baseline_value) if baseline_value else None

    return BaselineComparison(
        metric=metric.key,
        direction=metric.direction,
        model_value=model_value,
        baseline_value=baseline_value,
        absolute_improvement=absolute,
        relative_improvement=relative,
        beats_baseline=absolute > 0,
    )
