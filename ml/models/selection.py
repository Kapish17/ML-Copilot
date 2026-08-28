"""Choosing a model, then measuring it once.

This module ties the two halves of an honest evaluation together:

1. **Selection** — models are cross-validated on the training rows and ranked
   by the mean of their folds. The test set is not read.
2. **Final evaluation** — the winner is retrained on the *complete* training
   portion and evaluated exactly once on the untouched test set, alongside the
   naive baseline.

Keeping those steps apart is the whole point. A test set used to pick the
winner has already been spent: the score it then reports is the best of
several draws, not an estimate of future performance. Because selection here
never reads it, the final number is the first and only time the model meets
that data.

The holdout strategy is still available and unchanged, but it cannot offer the
same guarantee — under holdout, selection and final evaluation are the same
measurement, and the result says so.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ml.models.comparison import (
    ComparisonEntry,
    ModelComparison,
    SelectionStrategy,
    compare_models,
    format_comparison_table,
    select_best_model,
)
from ml.models.registry import ModelRegistry, default_registry
from ml.models.result import TrainedModel
from ml.models.spec import ModelSpec
from ml.models.training import train_model
from ml.pipelines.result import PreparedDataset

#: Cross-validation is the default because it is the strategy that keeps the
#: final test score meaningful.
DEFAULT_SELECTION_STRATEGY = SelectionStrategy.CROSS_VALIDATION


@dataclass(frozen=True)
class ModelSelectionResult:
    """A chosen model, how it was chosen, and its one test measurement."""

    strategy: SelectionStrategy
    comparison: ModelComparison
    selected_entry: ComparisonEntry
    final_model: TrainedModel

    @property
    def selected_model_name(self) -> str:
        """The registry identifier of the winning model."""
        return self.selected_entry.model_name

    @property
    def folds(self) -> int | None:
        """Number of cross-validation folds used, when applicable."""
        return self.comparison.folds

    @property
    def final_evaluation_is_unbiased(self) -> bool:
        """True when the test set played no part in choosing this model.

        False under the holdout strategy, where the reported test score is the
        best of several and is therefore optimistic.
        """
        return self.strategy is SelectionStrategy.CROSS_VALIDATION

    @property
    def selection_score(self) -> float | None:
        """The score the winner was chosen by."""
        return self.selected_entry.primary_metric_value

    @property
    def final_test_score(self) -> float | None:
        """The winner's primary metric on the held-out test set."""
        return self.final_model.primary_metric_value

    def as_text(self) -> str:
        """Render the comparison and the final measurement as a text table."""
        return format_comparison_table(self.comparison, final_model=self.final_model)

    def summary(self) -> dict[str, Any]:
        """Return a serialisable description, selection separated from testing.

        The two sections are deliberately distinct: ``selection`` holds the
        numbers that chose the model, ``final_evaluation`` holds the single
        untouched-test measurement. Neither contains a fitted estimator.
        """
        metric = self.comparison.primary_metric
        return {
            "strategy": self.strategy.value,
            "folds": self.folds,
            "selection": {
                "primary_metric": {
                    "key": metric.key,
                    "display_name": metric.display_name,
                    "direction": metric.direction.value,
                },
                "scored_on": "held_out_test_set"
                if self.comparison.uses_test_data
                else "training_folds",
                "uses_test_data": self.comparison.uses_test_data,
                "winner": self.selected_model_name,
                "winner_score": self.selection_score,
                "winner_score_std": self.selected_entry.primary_metric_std,
                "candidates": [
                    {
                        "model_name": entry.model_name,
                        "display_name": entry.display_name,
                        "status": entry.status.value,
                        "score": entry.primary_metric_value,
                        "score_std": entry.primary_metric_std,
                        "error": entry.error,
                    }
                    for entry in self.comparison.entries
                ],
            },
            "final_evaluation": {
                "model_name": self.final_model.model_name,
                "display_name": self.final_model.display_name,
                "trained_on": "full_training_data",
                "evaluated_on": "held_out_test_set",
                "is_unbiased": self.final_evaluation_is_unbiased,
                "test_row_count": self.final_model.dataset.test_row_count,
                "primary_metric": {
                    "key": self.final_model.primary_metric.key,
                    "value": self.final_test_score,
                },
                "metrics": self.final_model.metrics.as_dict(),
                "baseline": self.final_model.baseline.as_dict(),
                "baseline_comparison": self.final_model.baseline_comparison.as_dict(),
            },
            "comparison": self.comparison.summary(),
        }


def select_and_evaluate_best_model(
    prepared: PreparedDataset,
    *,
    models: Sequence[str | ModelSpec] | None = None,
    registry: ModelRegistry | None = None,
    primary_metric: str | None = None,
    strategy: SelectionStrategy | str = DEFAULT_SELECTION_STRATEGY,
    folds: int | None = None,
) -> ModelSelectionResult:
    """Choose the best model, then measure it once on the untouched test set.

    Under the default cross-validation strategy the sequence is:

    1. cross-validate every candidate on the training rows,
    2. rank them by the mean of their folds and pick the winner,
    3. retrain that winner on the **complete** training portion,
    4. evaluate it **once** on the held-out test set, against the baseline.

    Steps 1 and 2 never read the test set, so step 4 is an unbiased estimate.

    Under the holdout strategy the winner was already trained on the full
    training portion and scored on the test set, so that model is reused rather
    than refitted — and ``final_evaluation_is_unbiased`` reports ``False``,
    because the same numbers chose it.

    Args:
        prepared: The dataset produced by the preprocessing layer.
        models: Registry identifiers or specifications to consider. Defaults to
            every registered model for the dataset's task.
        registry: Registry to resolve models in; the default when omitted.
        primary_metric: Metric used to rank; the task default when omitted.
        strategy: ``"cross_validation"`` (default) or ``"holdout"``.
        folds: Number of cross-validation folds; five when omitted.

    Returns:
        ModelSelectionResult: The comparison, the winner, and its final
        held-out evaluation.

    Raises:
        NoSuccessfulModelError: If no candidate produced a rankable score.
        InvalidMetricError: If the requested primary metric does not exist.
        InvalidFoldCountError: If the fold count does not suit this dataset.
        ModelTrainingError: If the winner fails while being retrained.
    """
    active = registry or default_registry()
    resolved_strategy = SelectionStrategy(strategy)

    comparison = compare_models(
        prepared,
        models=models,
        registry=active,
        primary_metric=primary_metric,
        strategy=resolved_strategy,
        folds=folds,
    )
    selected = select_best_model(comparison)

    if selected.trained_model is not None:
        # Holdout: the winner is already fitted on the full training portion
        # and scored on the test set. Refitting would change nothing.
        final_model = selected.trained_model
    else:
        spec = selected.spec or ModelSpec(
            model_name=selected.model_name,
            primary_metric=comparison.primary_metric.key,
        )
        final_model = train_model(prepared, spec, registry=active)

    return ModelSelectionResult(
        strategy=resolved_strategy,
        comparison=comparison,
        selected_entry=selected,
        final_model=final_model,
    )
