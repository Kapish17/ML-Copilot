"""Training several models and ranking them.

Two strategies decide *what* a model is ranked by:

``holdout``
    Each model is trained once on the training rows and scored on the held-out
    test set. Quick, and the original behaviour, but it spends the test set on
    the choice of model — after which that same test score is no longer an
    honest estimate of the winner's performance.

``cross_validation``
    Each model is cross-validated over the training rows only, and ranked by
    the mean of the folds. The test set is not read at all, so it stays
    available for one final, unbiased measurement of whichever model wins.

Three rules shape the module.

**One failure must not sink the run.** Each model is handled inside its own
error boundary; a model that fails is recorded with its error and the others
carry on.

**Ranking reads the metric's direction.** Entries are ordered by the primary
metric, ascending for error metrics and descending for score metrics, which is
what stops "best" from meaning "worst RMSE".

**The strategy decides which numbers exist.** Under cross-validation the
comparison carries no test-set metrics and no test-set baseline at all — not
merely unused, but absent — so selection cannot accidentally read them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ml.evaluation.cross_validation import CrossValidationResult


class ModelStatus(str, Enum):
    """Whether a model in a comparison run finished."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SelectionStrategy(str, Enum):
    """How models are scored for the purpose of ranking them."""

    HOLDOUT = "holdout"
    CROSS_VALIDATION = "cross_validation"


#: Ranking groups: scored models first, then unscored, then failures.
_RANK_SCORED = 0
_RANK_UNSCORED = 1
_RANK_FAILED = 2


@dataclass(frozen=True)
class ComparisonEntry:
    """One row of a comparison: a model, how it scored, or why it failed.

    ``primary_metric_value`` is always the number the ranking used — the test
    score under the holdout strategy, the cross-validated mean under
    cross-validation. ``primary_metric_std`` is the spread over the folds, and
    is set only for cross-validation.
    """

    model_name: str
    display_name: str
    task_type: TaskType
    status: ModelStatus
    primary_metric: str
    primary_metric_value: float | None = None
    primary_metric_std: float | None = None
    metrics: EvaluationMetrics | None = None
    baseline_comparison: BaselineComparison | None = None
    cross_validation: CrossValidationResult | None = None
    training_seconds: float | None = None
    spec: ModelSpec | None = None
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
            "primary_metric_std": self.primary_metric_std,
            "metrics": self.metrics.as_dict() if self.metrics else None,
            "baseline_comparison": self.baseline_comparison.as_dict()
            if self.baseline_comparison
            else None,
            "cross_validation": self.cross_validation.summary()
            if self.cross_validation
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
    """Every model that was tried, ranked, and how they were scored."""

    task_type: TaskType
    primary_metric: MetricDefinition
    entries: tuple[ComparisonEntry, ...]
    strategy: SelectionStrategy = SelectionStrategy.HOLDOUT
    baseline: BaselineResult | None = None
    folds: int | None = None

    @property
    def uses_test_data(self) -> bool:
        """True when the ranking scores came from the held-out test set."""
        return self.strategy is SelectionStrategy.HOLDOUT

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
            "strategy": self.strategy.value,
            "folds": self.folds,
            "uses_test_data": self.uses_test_data,
            "task_type": self.task_type.value,
            "primary_metric": {
                "key": self.primary_metric.key,
                "display_name": self.primary_metric.display_name,
                "direction": self.primary_metric.direction.value,
            },
            "baseline": self.baseline.as_dict() if self.baseline else None,
            "model_count": len(self.entries),
            "succeeded_count": len(self.successful()),
            "failed_count": len(self.failed()),
            "best_model": best.model_name if best else None,
            "models": self.as_table(),
        }


def _resolve_specs(
    models: Sequence[str | ModelSpec] | None,
    *,
    registry: ModelRegistry,
    task_type: TaskType,
    metric_key: str,
) -> list[tuple[str, ModelSpec]]:
    """Normalise the requested models into (identifier, specification) pairs."""
    requested: Sequence[str | ModelSpec] = (
        models if models is not None else registry.identifiers(task_type)
    )
    pairs: list[tuple[str, ModelSpec]] = []
    for item in requested:
        if isinstance(item, ModelSpec):
            pairs.append((item.model_name, item))
        else:
            pairs.append((item, ModelSpec(model_name=item, primary_metric=metric_key)))
    return pairs


def _display_name(registry: ModelRegistry, model_name: str) -> str:
    """Return a model's human-readable name, falling back to its identifier."""
    return registry.get(model_name).display_name if registry.contains(model_name) else model_name


def _failure_entry(
    *,
    model_name: str,
    display_name: str,
    task_type: TaskType,
    metric_key: str,
    spec: ModelSpec,
    error: str,
    error_type: str,
) -> ComparisonEntry:
    """Build the row recorded for a model that could not be evaluated."""
    return ComparisonEntry(
        model_name=model_name,
        display_name=display_name,
        task_type=task_type,
        status=ModelStatus.FAILED,
        primary_metric=metric_key,
        spec=spec,
        error=error,
        error_type=error_type,
    )


def _holdout_entries(
    prepared: PreparedDataset,
    pairs: Sequence[tuple[str, ModelSpec]],
    *,
    registry: ModelRegistry,
    metric: MetricDefinition,
    baseline: BaselineResult,
) -> list[ComparisonEntry]:
    """Train each model once and score it on the held-out test set."""
    entries: list[ComparisonEntry] = []
    for model_name, spec in pairs:
        display_name = _display_name(registry, model_name)
        try:
            trained = train_model(prepared, spec, registry=registry, baseline=baseline)
        except MLError as exc:
            entries.append(
                _failure_entry(
                    model_name=model_name,
                    display_name=display_name,
                    task_type=prepared.task_type,
                    metric_key=metric.key,
                    spec=spec,
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
                spec=spec,
                trained_model=trained,
            )
        )
    return entries


def _cross_validation_entries(
    prepared: PreparedDataset,
    pairs: Sequence[tuple[str, ModelSpec]],
    *,
    registry: ModelRegistry,
    metric: MetricDefinition,
    folds: int,
) -> list[ComparisonEntry]:
    """Cross-validate each model on the training rows only.

    The resulting rows carry no test-set metrics, no test-set baseline and no
    fitted model — cross-validation exists to choose between models, and the
    winner is trained again afterwards on the full training data.
    """
    # Imported here rather than at module level: the cross-validation module
    # builds training pipelines from ``ml.models``, so a module-level import
    # in either direction would make the two packages import-order dependent.
    from ml.evaluation.cross_validation import cross_validate_model

    entries: list[ComparisonEntry] = []
    for model_name, spec in pairs:
        display_name = _display_name(registry, model_name)
        try:
            result = cross_validate_model(
                prepared,
                spec,
                folds=folds,
                registry=registry,
                primary_metric=metric.key,
            )
        except MLError as exc:
            entries.append(
                _failure_entry(
                    model_name=model_name,
                    display_name=display_name,
                    task_type=prepared.task_type,
                    metric_key=metric.key,
                    spec=spec,
                    error=exc.message,
                    error_type=type(exc).__name__,
                )
            )
            continue

        if not result.succeeded:
            first_failure = result.failed_folds[0] if result.failed_folds else None
            entries.append(
                ComparisonEntry(
                    model_name=model_name,
                    display_name=result.display_name,
                    task_type=result.task_type,
                    status=ModelStatus.FAILED,
                    primary_metric=metric.key,
                    cross_validation=result,
                    training_seconds=result.total_seconds,
                    spec=spec,
                    error=(
                        f"No fold produced a {metric.key} score. "
                        + (first_failure.error or "" if first_failure else "")
                    ).strip(),
                    error_type=(
                        first_failure.error_type if first_failure else "NoSuccessfulFolds"
                    ),
                )
            )
            continue

        entries.append(
            ComparisonEntry(
                model_name=model_name,
                display_name=result.display_name,
                task_type=result.task_type,
                status=ModelStatus.SUCCEEDED,
                primary_metric=metric.key,
                primary_metric_value=result.mean_primary_metric,
                primary_metric_std=result.std_primary_metric,
                cross_validation=result,
                training_seconds=result.total_seconds,
                spec=spec,
            )
        )
    return entries


def compare_models(
    prepared: PreparedDataset,
    *,
    models: Sequence[str | ModelSpec] | None = None,
    registry: ModelRegistry | None = None,
    primary_metric: str | None = None,
    strategy: SelectionStrategy | str = SelectionStrategy.HOLDOUT,
    folds: int | None = None,
) -> ModelComparison:
    """Compare several models on one prepared dataset.

    Args:
        prepared: The dataset produced by the preprocessing layer.
        models: Registry identifiers or specifications to try. Defaults to
            every registered model for the dataset's task.
        registry: Registry to resolve models in; the default when omitted.
        primary_metric: Metric used to rank; the task default when omitted.
        strategy: ``"holdout"`` (default, unchanged behaviour) scores each
            model on the held-out test set. ``"cross_validation"`` scores each
            model over folds of the training data and never reads the test set.
        folds: Number of cross-validation folds. Ignored by the holdout
            strategy; defaults to five for cross-validation.

    Returns:
        ModelComparison: One entry per model, best first, failures last.

    Raises:
        InvalidMetricError: If the requested primary metric does not exist.
        InvalidFoldCountError: If the fold count does not suit this dataset.
    """
    active = registry or default_registry()
    metric = resolve_primary_metric(prepared.task_type, primary_metric)
    resolved_strategy = SelectionStrategy(strategy)
    pairs = _resolve_specs(
        models, registry=active, task_type=prepared.task_type, metric_key=metric.key
    )

    if resolved_strategy is SelectionStrategy.CROSS_VALIDATION:
        from ml.evaluation.cross_validation import DEFAULT_FOLDS, validate_fold_count

        fold_count = folds if folds is not None else DEFAULT_FOLDS
        # Checked once, before any model runs: an unusable fold count is a
        # property of the dataset, not of one model, so it should be reported
        # immediately rather than as an identical failure on every row.
        validate_fold_count(
            prepared.y_train, task_type=prepared.task_type, folds=fold_count
        )
        entries = _cross_validation_entries(
            prepared, pairs, registry=active, metric=metric, folds=fold_count
        )
        baseline = None
    else:
        fold_count = None
        baseline = evaluate_baseline(prepared)
        entries = _holdout_entries(
            prepared, pairs, registry=active, metric=metric, baseline=baseline
        )

    entries.sort(key=lambda entry: _sort_key(entry, metric.direction))
    return ModelComparison(
        task_type=prepared.task_type,
        primary_metric=metric,
        entries=tuple(entries),
        strategy=resolved_strategy,
        baseline=baseline,
        folds=fold_count,
    )


def select_best_model(comparison: ModelComparison) -> ComparisonEntry:
    """Return the best entry of a comparison.

    "Best" means the highest value for a score metric and the lowest for an
    error metric, decided from the primary metric's declared direction. Models
    that failed, or that could not produce the primary metric, are not
    candidates.

    Under the cross-validation strategy the value being compared is the mean
    over the training folds, so the winner is chosen without the test set
    having been read.

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
                "strategy": comparison.strategy.value,
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


def format_comparison_table(
    comparison: ModelComparison, *, final_model: TrainedModel | None = None
) -> str:
    """Render a comparison as a readable text table.

    The score column is labelled with where the number came from, so a reader
    can never mistake a cross-validated mean for a test score.

    Args:
        comparison: The comparison to render.
        final_model: The winner retrained and evaluated on the held-out test
            set, when that step has happened.

    Returns:
        str: A plain-text table.
    """
    metric = comparison.primary_metric
    is_cv = comparison.strategy is SelectionStrategy.CROSS_VALIDATION
    label = metric.key.upper()
    score_header = f"CV Mean {label}" if is_cv else f"Test {label}"
    def render_score(value: float | None, fallback: str) -> str:
        """Format one cell, falling back when the number is missing."""
        return f"{value:.4f}" if value is not None else fallback

    scores = [
        render_score(entry.primary_metric_value, entry.status.value)
        for entry in comparison.entries
    ]
    spreads = [
        render_score(entry.primary_metric_std, "-") for entry in comparison.entries
    ]

    name_width = max(
        [len("Model"), *(len(entry.display_name) for entry in comparison.entries)]
    )
    score_width = max([len(score_header), *(len(item) for item in scores)])
    std_width = max([len("CV Std"), *(len(item) for item in spreads)])

    header = f"{'Model':<{name_width}}  {score_header:>{score_width}}"
    if is_cv:
        header += f"  {'CV Std':>{std_width}}"
    lines = [header, "-" * len(header)]

    for entry, score, spread in zip(comparison.entries, scores, spreads, strict=True):
        row = f"{entry.display_name:<{name_width}}  {score:>{score_width}}"
        if is_cv:
            row += f"  {spread:>{std_width}}"
        lines.append(row)

    best = comparison.best()
    lines.append("")
    lines.append(f"Winner: {best.display_name if best else 'none'}")
    if is_cv:
        lines.append(
            f"Selected on {comparison.folds}-fold cross-validation of the "
            "training data; the test set was not used."
        )

    if final_model is not None:
        value = final_model.primary_metric_value
        baseline_value = final_model.baseline.metrics.get(final_model.primary_metric.key)
        improvement = final_model.baseline_comparison.absolute_improvement
        lines.append("")
        lines.append(
            f"Final held-out test {final_model.primary_metric.key.upper()}: "
            + (f"{value:.4f}" if value is not None else "unavailable")
        )
        if baseline_value is not None and improvement is not None:
            lines.append(
                f"Baseline ({final_model.baseline.display_name}): "
                f"{baseline_value:.4f}  |  improvement: {improvement:+.4f}"
            )
    return "\n".join(lines)
