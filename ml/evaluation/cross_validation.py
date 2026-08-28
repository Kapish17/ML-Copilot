"""Cross-validation over the training portion of a prepared dataset.

Why this exists: a single train/test split gives one number per model, and
that number carries the luck of one particular division of the rows. Choosing
a winner by it means choosing partly by luck, and — worse — a test set used to
*pick* a model is no longer an honest estimate of how that model will do on
data it has never seen. Cross-validation moves selection onto the training
data, where it belongs, and leaves the test set untouched for a single final
measurement.

The rule this module enforces:

    cross-validation selects the model; the held-out test set is reserved
    for the final evaluation.

Nothing here reads ``X_test_raw`` or ``y_test``. Every fold builds its own
pipeline, fits the preprocessing on that fold's training rows only, and scores
the rows it never saw — so a validation fold cannot influence the imputation
values, scaler statistics or encoder categories used against it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from ml.errors import InvalidFoldCountError
from ml.evaluation.metrics import (
    EvaluationMetrics,
    MetricDefinition,
    evaluate_predictions,
    metrics_for_task,
    resolve_primary_metric,
)
from ml.features.types import TaskType
from ml.models.registry import ModelRegistry, default_registry
from ml.models.spec import ModelSpec, build_estimator, get_model_spec, validate_spec
from ml.models.training import build_pipeline, class_scores, clone_pipeline
from ml.pipelines.result import PreparedDataset

#: Folds used when the caller does not choose. Five is the usual compromise
#: between a stable estimate and the cost of fitting the model that many times.
DEFAULT_FOLDS = 5
#: Fewer than two folds is not cross-validation.
MIN_FOLDS = 2


class FoldStatus(str, Enum):
    """Whether one fold completed."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


def validate_fold_count(
    target: pd.Series, *, task_type: TaskType, folds: int
) -> None:
    """Check that a fold count can actually be honoured by this data.

    Three ways it cannot: fewer than two folds, more folds than rows, or — for
    classification — more folds than members of the smallest class, since
    stratification then cannot place that class in every validation fold.
    Failing here is better than producing folds that quietly misrepresent a
    rare class.

    Args:
        target: The training target column.
        task_type: The task being solved.
        folds: Requested number of folds.

    Raises:
        InvalidFoldCountError: If the fold count cannot be used.
    """
    if folds < MIN_FOLDS:
        raise InvalidFoldCountError(
            f"Cross-validation needs at least {MIN_FOLDS} folds, got {folds}.",
            details={"reason": "too_few_folds", "folds": folds, "minimum": MIN_FOLDS},
        )

    row_count = int(target.shape[0])
    if folds > row_count:
        raise InvalidFoldCountError(
            f"Cannot split {row_count} training row(s) into {folds} folds.",
            details={
                "reason": "more_folds_than_rows",
                "folds": folds,
                "row_count": row_count,
            },
        )

    if task_type is TaskType.CLASSIFICATION:
        counts = target.value_counts(dropna=True)
        if counts.empty:
            raise InvalidFoldCountError(
                "The training target has no values to stratify.",
                details={"reason": "empty_target", "folds": folds},
            )
        smallest = int(counts.min())
        if smallest < folds:
            raise InvalidFoldCountError(
                f"Class '{counts.idxmin()}' has only {smallest} training row(s), "
                f"which is fewer than the {folds} folds requested. Reduce the "
                "number of folds or collect more examples of that class.",
                details={
                    "reason": "class_smaller_than_folds",
                    "folds": folds,
                    "smallest_class": str(counts.idxmin()),
                    "smallest_class_count": smallest,
                    "class_counts": {
                        str(label): int(count) for label, count in counts.items()
                    },
                },
            )


def build_splitter(
    task_type: TaskType, *, folds: int, random_state: int
) -> StratifiedKFold | KFold:
    """Return the cross-validation splitter appropriate to the task.

    Classification uses ``StratifiedKFold`` so every fold keeps the dataset's
    class proportions — without it, an imbalanced dataset can produce a fold
    with almost none of the minority class and a meaningless score. Regression
    uses plain ``KFold``, because a continuous target has no classes to
    balance.

    Both shuffle before splitting, using the dataset's seed, so the folds are
    random with respect to row order but identical on every re-run.

    Args:
        task_type: A resolved task type, never ``AUTO``.
        folds: Number of folds.
        random_state: Seed, so the same inputs give the same folds.

    Returns:
        StratifiedKFold | KFold: The splitter to iterate.
    """
    if task_type is TaskType.CLASSIFICATION:
        return StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=random_state
        )
    return KFold(n_splits=folds, shuffle=True, random_state=random_state)


@dataclass(frozen=True)
class FoldResult:
    """The outcome of one fold: what it scored, or why it failed."""

    fold: int
    status: FoldStatus
    train_size: int
    validation_size: int
    metrics: EvaluationMetrics | None = None
    training_seconds: float | None = None
    error: str | None = None
    error_type: str | None = None

    @property
    def succeeded(self) -> bool:
        """True when the fold trained and was scored."""
        return self.status is FoldStatus.SUCCEEDED

    def as_dict(self) -> dict[str, Any]:
        """Render the fold as plain, JSON-friendly values."""
        return {
            "fold": self.fold,
            "status": self.status.value,
            "train_size": self.train_size,
            "validation_size": self.validation_size,
            "metrics": self.metrics.as_dict() if self.metrics else None,
            "training_seconds": self.training_seconds,
            "error": self.error,
            "error_type": self.error_type,
        }


@dataclass(frozen=True)
class MetricSummary:
    """One metric across the folds: its spread as well as its average.

    The standard deviation is the population spread over the folds that
    actually ran. It is the honest half of a cross-validation score: a mean of
    0.84 with a spread of 0.01 is a different claim from the same mean with a
    spread of 0.09.
    """

    metric: str
    values: tuple[float, ...]
    mean: float | None
    std: float | None
    minimum: float | None
    maximum: float | None

    @property
    def fold_count(self) -> int:
        """How many folds produced this metric."""
        return len(self.values)

    def as_dict(self) -> dict[str, Any]:
        """Render the summary as plain, JSON-friendly values."""
        return {
            "metric": self.metric,
            "fold_count": self.fold_count,
            "values": list(self.values),
            "mean": self.mean,
            "std": self.std,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


def summarise_metric(metric: str, values: Sequence[float]) -> MetricSummary:
    """Aggregate one metric's fold values into mean, spread and range."""
    usable = tuple(
        float(value) for value in values if value is not None and math.isfinite(value)
    )
    if not usable:
        return MetricSummary(
            metric=metric, values=(), mean=None, std=None, minimum=None, maximum=None
        )
    array = np.asarray(usable, dtype="float64")
    return MetricSummary(
        metric=metric,
        values=usable,
        mean=float(array.mean()),
        std=float(array.std()),
        minimum=float(array.min()),
        maximum=float(array.max()),
    )


def aggregate_folds(
    fold_results: Sequence[FoldResult], task_type: TaskType
) -> dict[str, MetricSummary]:
    """Aggregate every metric of the task across the successful folds."""
    summaries: dict[str, MetricSummary] = {}
    for definition in metrics_for_task(task_type):
        values = [
            fold.metrics.get(definition.key)
            for fold in fold_results
            if fold.succeeded and fold.metrics is not None
        ]
        summaries[definition.key] = summarise_metric(
            definition.key, [value for value in values if value is not None]
        )
    return summaries


def pooled_confusion_matrix(
    fold_results: Sequence[FoldResult],
) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]] | None:
    """Sum the fold confusion matrices, when they share a class layout.

    Every row of the training data appears in exactly one validation fold, so
    the summed matrix covers the whole training set once. It is returned only
    when all folds agree on the label order, since otherwise the cells would
    not line up.
    """
    matrices = []
    labels: tuple[str, ...] | None = None
    for fold in fold_results:
        details = fold.metrics.classification if fold.metrics else None
        if details is None:
            continue
        if labels is None:
            labels = details.class_labels
        elif labels != details.class_labels:
            return None
        matrices.append(np.asarray(details.confusion_matrix, dtype="int64"))

    if labels is None or not matrices:
        return None
    total = np.sum(matrices, axis=0)
    return labels, tuple(tuple(int(cell) for cell in row) for row in total)


@dataclass(frozen=True)
class CrossValidationResult:
    """How a model scored across the folds of the training data.

    Every number here comes from the training portion. No field is derived
    from the held-out test set.
    """

    spec: ModelSpec
    display_name: str
    task_type: TaskType
    folds: int
    primary_metric: MetricDefinition
    fold_results: tuple[FoldResult, ...]
    aggregates: dict[str, MetricSummary] = field(default_factory=dict)
    total_seconds: float = 0.0
    class_labels: tuple[str, ...] | None = None
    confusion_matrix: tuple[tuple[int, ...], ...] | None = None

    @property
    def model_name(self) -> str:
        """The registry identifier of the model that was validated."""
        return self.spec.model_name

    @property
    def successful_folds(self) -> tuple[FoldResult, ...]:
        """The folds that trained and were scored."""
        return tuple(fold for fold in self.fold_results if fold.succeeded)

    @property
    def failed_folds(self) -> tuple[FoldResult, ...]:
        """The folds that raised, with their errors."""
        return tuple(fold for fold in self.fold_results if not fold.succeeded)

    @property
    def succeeded(self) -> bool:
        """True when at least one fold produced the primary metric."""
        return self.mean_primary_metric is not None

    @property
    def mean_primary_metric(self) -> float | None:
        """The score models are ranked by: the mean over the folds."""
        summary = self.aggregates.get(self.primary_metric.key)
        return summary.mean if summary else None

    @property
    def std_primary_metric(self) -> float | None:
        """The spread of the ranking score over the folds."""
        summary = self.aggregates.get(self.primary_metric.key)
        return summary.std if summary else None

    @property
    def errors(self) -> dict[int, str]:
        """Fold number to error message, for the folds that failed."""
        return {
            fold.fold: fold.error
            for fold in self.failed_folds
            if fold.error is not None
        }

    def summary(self) -> dict[str, Any]:
        """Return a serialisable description of the cross-validation run.

        No estimator and no fitted pipeline appears here — the folds' models
        are discarded once they have been scored, because cross-validation
        exists to compare models, not to produce one.
        """
        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "task_type": self.task_type.value,
            "spec": self.spec.as_dict(),
            "folds": self.folds,
            "successful_fold_count": len(self.successful_folds),
            "failed_fold_count": len(self.failed_folds),
            "primary_metric": {
                "key": self.primary_metric.key,
                "display_name": self.primary_metric.display_name,
                "direction": self.primary_metric.direction.value,
                "mean": self.mean_primary_metric,
                "std": self.std_primary_metric,
            },
            "aggregates": {
                key: summary.as_dict() for key, summary in self.aggregates.items()
            },
            "fold_results": [fold.as_dict() for fold in self.fold_results],
            "class_labels": list(self.class_labels) if self.class_labels else None,
            "confusion_matrix": [list(row) for row in self.confusion_matrix]
            if self.confusion_matrix
            else None,
            "total_seconds": self.total_seconds,
            "errors": {str(fold): message for fold, message in self.errors.items()},
            "evaluated_on": "training_folds",
        }


def _run_fold(
    *,
    fold_number: int,
    pipeline_factory: Callable[[], Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    task_type: TaskType,
) -> FoldResult:
    """Fit, predict and score one fold, capturing any failure.

    The pipeline is built fresh here, so the preprocessing statistics are
    learned from this fold's training rows and from nothing else.
    """
    sizes = {
        "train_size": int(X_train.shape[0]),
        "validation_size": int(X_validation.shape[0]),
    }
    pipeline = pipeline_factory()
    started = perf_counter()
    try:
        pipeline.fit(X_train, y_train)
        elapsed = perf_counter() - started
        predictions = pipeline.predict(X_validation)
        scores, score_labels = class_scores(pipeline, X_validation, task_type)
        metrics = evaluate_predictions(
            task_type,
            y_validation,
            predictions,
            y_score=scores,
            score_labels=score_labels,
        )
    except Exception as exc:  # noqa: BLE001 - a fold failure must not stop the run
        return FoldResult(
            fold=fold_number,
            status=FoldStatus.FAILED,
            error=str(exc),
            error_type=type(exc).__name__,
            **sizes,
        )

    return FoldResult(
        fold=fold_number,
        status=FoldStatus.SUCCEEDED,
        metrics=metrics,
        training_seconds=elapsed,
        **sizes,
    )


def cross_validate_pipeline(
    pipeline_factory: Callable[[], Pipeline],
    features: pd.DataFrame,
    target: pd.Series,
    *,
    task_type: TaskType,
    folds: int,
    random_state: int,
) -> tuple[tuple[FoldResult, ...], float]:
    """Run the fold loop over already-selected training data.

    Args:
        pipeline_factory: Builds a fresh, unfitted pipeline for each fold.
        features: Raw training features. Never the test set.
        target: Training target, aligned with ``features``.
        task_type: A resolved task type, never ``AUTO``.
        folds: Number of folds; already validated.
        random_state: Seed for the shuffle.

    Returns:
        tuple: The fold results and the total wall-clock seconds.
    """
    splitter = build_splitter(task_type, folds=folds, random_state=random_state)
    results: list[FoldResult] = []
    started = perf_counter()

    for number, (train_index, validation_index) in enumerate(
        splitter.split(features, target), start=1
    ):
        results.append(
            _run_fold(
                fold_number=number,
                pipeline_factory=pipeline_factory,
                X_train=features.iloc[train_index],
                y_train=target.iloc[train_index],
                X_validation=features.iloc[validation_index],
                y_validation=target.iloc[validation_index],
                task_type=task_type,
            )
        )

    return tuple(results), perf_counter() - started


def cross_validate_model(
    prepared: PreparedDataset,
    spec: ModelSpec | str,
    *,
    folds: int = DEFAULT_FOLDS,
    registry: ModelRegistry | None = None,
    primary_metric: str | None = None,
) -> CrossValidationResult:
    """Cross-validate one model on the training portion of a dataset.

    Only ``X_train_raw`` and ``y_train`` are read. The test set is not touched,
    so the result can be used to choose a model without spending the one
    honest estimate of its performance.

    Each fold gets its own clone of the ``Pipeline(preprocessing, estimator)``
    and fits it on that fold's training rows alone, which is what keeps the
    validation rows out of every imputation value, scaler statistic and
    encoder category applied to them.

    Args:
        prepared: The dataset produced by the preprocessing layer.
        spec: A model specification, or a registry identifier for the defaults.
        folds: Number of folds.
        registry: Registry to resolve the model in; the default when omitted.
        primary_metric: Metric used to rank; the task default when omitted.

    Returns:
        CrossValidationResult: Fold scores, aggregates and any fold failures.
        A run where every fold failed is returned with the errors recorded
        rather than raised, so a comparison can report it alongside the models
        that worked; check ``result.succeeded``.

    Raises:
        UnknownModelError: The model is not in the registry.
        IncompatibleTaskError: The model does not solve the dataset's task.
        InvalidHyperparameterError: A hyperparameter is not accepted.
        InvalidMetricError: The requested primary metric does not exist.
        InvalidFoldCountError: The fold count does not suit this data.
    """
    active = registry or default_registry()
    resolved_spec = (
        get_model_spec(spec, registry=active) if isinstance(spec, str) else spec
    )
    definition = validate_spec(resolved_spec, prepared.task_type, registry=active)
    metric = resolve_primary_metric(
        prepared.task_type, primary_metric or resolved_spec.primary_metric
    )

    features = prepared.X_train_raw
    target = prepared.y_train
    validate_fold_count(target, task_type=prepared.task_type, folds=folds)

    def make_pipeline() -> Pipeline:
        """Build an unfitted pipeline for one fold."""
        estimator = build_estimator(
            definition,
            resolved_spec,
            fallback_random_state=prepared.config.random_state,
        )
        return clone_pipeline(build_pipeline(prepared, estimator))

    fold_results, total_seconds = cross_validate_pipeline(
        make_pipeline,
        features,
        target,
        task_type=prepared.task_type,
        folds=folds,
        random_state=prepared.config.random_state,
    )

    pooled = pooled_confusion_matrix(fold_results)
    return CrossValidationResult(
        spec=resolved_spec,
        display_name=definition.display_name,
        task_type=prepared.task_type,
        folds=folds,
        primary_metric=metric,
        fold_results=fold_results,
        aggregates=aggregate_folds(fold_results, prepared.task_type),
        total_seconds=total_seconds,
        class_labels=pooled[0] if pooled else None,
        confusion_matrix=pooled[1] if pooled else None,
    )


__all__ = [
    "DEFAULT_FOLDS",
    "MIN_FOLDS",
    "CrossValidationResult",
    "FoldResult",
    "FoldStatus",
    "MetricSummary",
    "aggregate_folds",
    "build_splitter",
    "cross_validate_model",
    "cross_validate_pipeline",
    "pooled_confusion_matrix",
    "summarise_metric",
    "validate_fold_count",
]
