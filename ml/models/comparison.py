"""Training several models and ranking them.

Two rules shape this module.

**One failure must not sink the run.** Each model is trained inside its own
error boundary. A model that fails is recorded with its error and the others
carry on, so a comparison always returns whatever could be learned.

**Ranking reads the metric's direction.** Entries are ordered by the primary
metric, ascending for error metrics and descending for score metrics, which is
what stops "best" from meaning "worst RMSE".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ml.errors import MLError, NoSuccessfulModelError
from ml.evaluation.metrics import (
    EvaluationMetrics,
    MetricDefinition,
    MetricDirection,
    resolve_primary_metric,
)
from ml.features.types import TaskType
from ml.models.baselines import BaselineComparison, BaselineResult, evaluate_baseline
from ml.models.registry import ModelRegistry, default_registry
from ml.models.result import TrainedModel
from ml.models.spec import ModelSpec
from ml.models.training import train_model
from ml.pipelines.result import PreparedDataset


class ModelStatus(str, Enum):
    """Whether a model in a comparison run finished."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


#: Ranking groups: scored models first, then unscored, then failures.
_RANK_SCORED = 0
_RANK_UNSCORED = 1
_RANK_FAILED = 2


@dataclass(frozen=True)
class ComparisonEntry:
    """One row of a comparison: a model, how it scored, or why it failed."""

    model_name: str
    display_name: str
    task_type: TaskType
    status: ModelStatus
    primary_metric: str
    primary_metric_value: float | None = None
    metrics: EvaluationMetrics | None = None
    baseline_comparison: BaselineComparison | None = None
    training_seconds: float | None = None
    trained_model: TrainedModel | None = None
    error: str | None = None
    error_type: str | None = None

    @property
    def succeeded(self) -> bool:
        """True when the model trained and was scored."""
        return self.status is ModelStatus.SUCCEEDED

    def as_dict(self) -> dict[str, Any]:
        """Render the row as plain, JSON-friendly values.

        The fitted model is deliberately left out; ``trained_model`` stays
        available on the object for callers that need the estimator itself.
        """
        return {
            "model_name": self.model_name,
            "display_name": self.display_name,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "primary_metric": self.primary_metric,
            "primary_metric_value": self.primary_metric_value,
            "metrics": self.metrics.as_dict() if self.metrics else None,
            "baseline_comparison": self.baseline_comparison.as_dict()
            if self.baseline_comparison
            else None,
            "training_seconds": self.training_seconds,
            "error": self.error,
            "error_type": self.error_type,
        }


def _sort_key(entry: ComparisonEntry, direction: MetricDirection) -> tuple[int, float]:
    """Order entries best-first, honouring the metric's direction."""
    if not entry.succeeded:
        return (_RANK_FAILED, 0.0)
    value = entry.primary_metric_value
    if value is None:
        return (_RANK_UNSCORED, 0.0)
    ordered = -value if direction is MetricDirection.HIGHER_IS_BETTER else value
    return (_RANK_SCORED, ordered)


@dataclass(frozen=True)
class ModelComparison:
    """Every model that was tried, ranked, with the shared baseline."""

    task_type: TaskType
    primary_metric: MetricDefinition
    baseline: BaselineResult
    entries: tuple[ComparisonEntry, ...]

    def successful(self) -> tuple[ComparisonEntry, ...]:
        """The entries that trained and were scored, best first."""
        return tuple(entry for entry in self.entries if entry.succeeded)

    def failed(self) -> tuple[ComparisonEntry, ...]:
        """The entries that raised, with their errors."""
        return tuple(entry for entry in self.entries if not entry.succeeded)

    def best(self) -> ComparisonEntry | None:
        """The best scored entry, or ``None`` when nothing could be ranked."""
        ranked = [
            entry for entry in self.successful() if entry.primary_metric_value is not None
        ]
        return ranked[0] if ranked else None

    def as_table(self) -> list[dict[str, Any]]:
        """Render the comparison as a list of serialisable rows."""
        return [entry.as_dict() for entry in self.entries]

    def summary(self) -> dict[str, Any]:
        """Return a serialisable description of the whole comparison."""
        best = self.best()
        return {
            "task_type": self.task_type.value,
            "primary_metric": {
                "key": self.primary_metric.key,
                "display_name": self.primary_metric.display_name,
                "direction": self.primary_metric.direction.value,
            },
            "baseline": self.baseline.as_dict(),
            "model_count": len(self.entries),
            "succeeded_count": len(self.successful()),
            "failed_count": len(self.failed()),
            "best_model": best.model_name if best else None,
            "models": self.as_table(),
        }


def compare_models(
    prepared: PreparedDataset,
    *,
    models: Sequence[str | ModelSpec] | None = None,
    registry: ModelRegistry | None = None,
    primary_metric: str | None = None,
) -> ModelComparison:
    """Train several models on one prepared dataset and rank them.

    The baseline is evaluated once and shared, so every model is measured
    against the same reference and the naive fit is not repeated.

    Args:
        prepared: The dataset produced by the preprocessing layer.
        models: Registry identifiers or specifications to try. Defaults to
            every registered model for the dataset's task.
        registry: Registry to resolve models in; the default when omitted.
        primary_metric: Metric used to rank; the task default when omitted.

    Returns:
        ModelComparison: One entry per model, best first, failures last.

    Raises:
        InvalidMetricError: If the requested primary metric does not exist.
    """
    active = registry or default_registry()
    metric = resolve_primary_metric(prepared.task_type, primary_metric)
    requested: Sequence[str | ModelSpec] = (
        models
        if models is not None
        else active.identifiers(prepared.task_type)
    )

    baseline = evaluate_baseline(prepared)
    entries: list[ComparisonEntry] = []

    for item in requested:
        model_name = item if isinstance(item, str) else item.model_name
        spec = (
            item
            if isinstance(item, ModelSpec)
            else ModelSpec(model_name=model_name, primary_metric=metric.key)
        )
        display_name = (
            active.get(model_name).display_name
            if active.contains(model_name)
            else model_name
        )
        try:
            trained = train_model(prepared, spec, registry=active, baseline=baseline)
        except MLError as exc:
            entries.append(
                ComparisonEntry(
                    model_name=model_name,
                    display_name=display_name,
                    task_type=prepared.task_type,
                    status=ModelStatus.FAILED,
                    primary_metric=metric.key,
                    error=exc.message,
                    error_type=type(exc).__name__,
                )
            )
            continue

        entries.append(
            ComparisonEntry(
                model_name=model_name,
                display_name=trained.display_name,
                task_type=trained.task_type,
                status=ModelStatus.SUCCEEDED,
                primary_metric=metric.key,
                primary_metric_value=trained.metrics.get(metric.key),
                metrics=trained.metrics,
                baseline_comparison=trained.baseline_comparison,
                training_seconds=trained.training_seconds,
                trained_model=trained,
            )
        )

    entries.sort(key=lambda entry: _sort_key(entry, metric.direction))
    return ModelComparison(
        task_type=prepared.task_type,
        primary_metric=metric,
        baseline=baseline,
        entries=tuple(entries),
    )


def select_best_model(comparison: ModelComparison) -> ComparisonEntry:
    """Return the best entry of a comparison.

    "Best" means the highest value for a score metric and the lowest for an
    error metric, decided from the primary metric's declared direction. Models
    that failed, or that could not produce the primary metric, are not
    candidates.

    Args:
        comparison: The result of :func:`compare_models`.

    Returns:
        ComparisonEntry: The winning entry.

    Raises:
        NoSuccessfulModelError: If no model produced a rankable score.
    """
    best = comparison.best()
    if best is None:
        raise NoSuccessfulModelError(
            "No model produced a usable "
            f"{comparison.primary_metric.key} score, so none can be selected.",
            details={
                "primary_metric": comparison.primary_metric.key,
                "attempted_models": [entry.model_name for entry in comparison.entries],
                "errors": {
                    entry.model_name: entry.error
                    for entry in comparison.failed()
                    if entry.error
                },
            },
        )
    return best
