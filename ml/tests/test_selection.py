"""Tests for cross-validated model selection and the final test evaluation."""

from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from ml.errors import InvalidFoldCountError, NoSuccessfulModelError
from ml.features.types import TaskType
from ml.models.comparison import (
    ModelStatus,
    SelectionStrategy,
    compare_models,
    format_comparison_table,
    select_best_model,
)
from ml.models.registry import ModelDefinition, ModelRegistry, default_registry
from ml.models.selection import (
    DEFAULT_SELECTION_STRATEGY,
    select_and_evaluate_best_model,
)
from ml.pipelines.result import PreparedDataset

CV = SelectionStrategy.CROSS_VALIDATION
FAILING_MODEL = "always_fails"
#: A non-default fold count, so the tests prove the setting is honoured.
FOLDS = 3


class _FailingClassifier(BaseEstimator, ClassifierMixin):
    """An estimator that always raises, used to exercise the error boundary."""

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn signature
        """Fail loudly, as a broken estimator would."""
        raise RuntimeError("this estimator always fails")

    def predict(self, X):  # noqa: ANN001, ANN201 - sklearn signature
        """Never reached."""
        raise RuntimeError("this estimator always fails")


def _registry_with_failure() -> ModelRegistry:
    """The default registry plus one model that cannot train."""
    return default_registry().extend(
        ModelDefinition(
            identifier=FAILING_MODEL,
            display_name="Always Fails",
            task_type=TaskType.CLASSIFICATION,
            factory=_FailingClassifier,
        )
    )


def _poison_test_set(prepared: PreparedDataset) -> PreparedDataset:
    """Return the same dataset with its test labels deliberately scrambled.

    The label values are reversed while the feature rows stay put, so every
    test row is now paired with someone else's answer. The training half is
    untouched, so anything that depends only on training data must be
    completely unaffected.
    """
    scrambled_target = pd.Series(
        prepared.y_test.to_numpy()[::-1],
        index=prepared.y_test.index,
        name=prepared.y_test.name,
    )
    return replace(prepared, y_test=scrambled_target)


# --------------------------------------------------------------------------
# The cross-validation strategy
# --------------------------------------------------------------------------


def test_cross_validation_strategy_ranks_by_fold_means(
    classification_prepared: PreparedDataset,
) -> None:
    """Every model is scored by the mean of its folds, with the spread beside it."""
    comparison = compare_models(classification_prepared, strategy=CV, folds=FOLDS)

    assert comparison.strategy is CV
    assert comparison.folds == FOLDS
    assert len(comparison.entries) == 3
    for entry in comparison.successful():
        assert entry.cross_validation is not None
        assert entry.primary_metric_value == entry.cross_validation.mean_primary_metric
        assert entry.primary_metric_std == entry.cross_validation.std_primary_metric
        assert len(entry.cross_validation.successful_folds) == FOLDS


def test_cross_validation_comparison_holds_no_test_numbers(
    classification_prepared: PreparedDataset,
) -> None:
    """The comparison cannot leak the test set because it never contains it.

    Under this strategy the test metrics, the test baseline and the fitted
    model are absent from the result — not merely unused.
    """
    comparison = compare_models(classification_prepared, strategy=CV, folds=FOLDS)

    assert comparison.baseline is None
    assert comparison.uses_test_data is False
    for entry in comparison.entries:
        assert entry.metrics is None
        assert entry.baseline_comparison is None
        assert entry.trained_model is None


def test_strategy_accepts_a_plain_string(
    classification_prepared: PreparedDataset,
) -> None:
    """A stored configuration can name the strategy as text."""
    comparison = compare_models(
        classification_prepared, strategy="cross_validation", folds=3
    )
    assert comparison.strategy is CV


def test_default_strategy_is_still_holdout(
    classification_prepared: PreparedDataset,
) -> None:
    """Existing callers keep the behaviour they had."""
    comparison = compare_models(classification_prepared)

    assert comparison.strategy is SelectionStrategy.HOLDOUT
    assert comparison.uses_test_data is True
    assert comparison.baseline is not None
    assert comparison.folds is None
    assert all(entry.cross_validation is None for entry in comparison.entries)
    assert all(entry.metrics is not None for entry in comparison.successful())


def test_invalid_fold_count_is_reported_once(
    classification_prepared: PreparedDataset,
) -> None:
    """A fold count the data cannot support fails before any model runs."""
    with pytest.raises(InvalidFoldCountError):
        compare_models(classification_prepared, strategy=CV, folds=1)


# --------------------------------------------------------------------------
# Direction-aware selection
# --------------------------------------------------------------------------


def test_f1_selection_takes_the_highest_mean(
    classification_prepared: PreparedDataset,
) -> None:
    """F1 is a score, so the winner has the largest cross-validated mean."""
    comparison = compare_models(classification_prepared, strategy=CV, folds=FOLDS)
    means = [entry.primary_metric_value for entry in comparison.successful()]
    best = select_best_model(comparison)

    assert comparison.primary_metric.key == "f1"
    assert best.primary_metric_value == max(means)
    assert comparison.entries[0] is best


def test_rmse_selection_takes_the_lowest_mean(
    regression_prepared: PreparedDataset,
) -> None:
    """RMSE is an error, so the winner has the smallest cross-validated mean."""
    comparison = compare_models(regression_prepared, strategy=CV, folds=FOLDS)
    means = [entry.primary_metric_value for entry in comparison.successful()]
    best = select_best_model(comparison)

    assert comparison.primary_metric.key == "rmse"
    assert comparison.primary_metric.higher_is_better is False
    assert best.primary_metric_value == min(means)
    assert best.primary_metric_value < max(means), "the ranking is not accidental"


def test_r2_selection_takes_the_highest_mean(
    regression_prepared: PreparedDataset,
) -> None:
    """R² is a score even though the task default is an error metric."""
    comparison = compare_models(
        regression_prepared, strategy=CV, folds=FOLDS, primary_metric="r2"
    )
    means = [entry.primary_metric_value for entry in comparison.successful()]

    assert comparison.primary_metric.key == "r2"
    assert select_best_model(comparison).primary_metric_value == max(means)


# --------------------------------------------------------------------------
# Failures
# --------------------------------------------------------------------------


def test_a_failing_model_does_not_stop_the_others(
    classification_prepared: PreparedDataset,
) -> None:
    """A model whose every fold fails is recorded; the rest still run."""
    comparison = compare_models(
        classification_prepared,
        models=[FAILING_MODEL, "logistic_regression", "random_forest_classifier"],
        registry=_registry_with_failure(),
        strategy=CV,
        folds=3,
    )

    assert len(comparison.successful()) == 2
    failure = comparison.failed()[0]
    assert failure.model_name == FAILING_MODEL
    assert failure.status is ModelStatus.FAILED
    assert failure.primary_metric_value is None
    assert failure.cross_validation is not None
    assert len(failure.cross_validation.failed_folds) == 3
    assert failure.cross_validation.errors
    assert "always fails" in (failure.error or "")
    assert comparison.entries[-1] is failure, "failures rank last"


def test_an_unknown_model_is_recorded_as_a_failure(
    classification_prepared: PreparedDataset,
) -> None:
    """A bad name fails that row alone."""
    comparison = compare_models(
        classification_prepared,
        models=["not_a_model", "logistic_regression"],
        strategy=CV,
        folds=3,
    )

    assert comparison.failed()[0].error_type == "UnknownModelError"
    assert len(comparison.successful()) == 1


def test_selection_fails_loudly_when_nothing_worked(
    classification_prepared: PreparedDataset,
) -> None:
    """No silent ``None``: an unrankable comparison raises with the reasons."""
    comparison = compare_models(
        classification_prepared,
        models=[FAILING_MODEL],
        registry=_registry_with_failure(),
        strategy=CV,
        folds=3,
    )

    with pytest.raises(NoSuccessfulModelError) as exc_info:
        select_best_model(comparison)
    assert exc_info.value.details["strategy"] == "cross_validation"
    assert FAILING_MODEL in exc_info.value.details["errors"]


# --------------------------------------------------------------------------
# Selection followed by one final measurement
# --------------------------------------------------------------------------


def test_winner_is_chosen_by_cross_validation_and_measured_on_the_test_set(
    classification_prepared: PreparedDataset,
) -> None:
    """The two steps are distinct and both are reported."""
    outcome = select_and_evaluate_best_model(
        classification_prepared, strategy=CV, folds=FOLDS
    )

    assert DEFAULT_SELECTION_STRATEGY is CV
    assert outcome.strategy is CV
    assert outcome.selected_entry.metrics is None, "selection saw no test metrics"
    assert outcome.selection_score == outcome.selected_entry.cross_validation.mean_primary_metric
    assert outcome.final_model.metrics.sample_count == (
        classification_prepared.test_row_count
    )
    assert outcome.final_test_score is not None
    assert outcome.final_evaluation_is_unbiased is True


def test_the_winner_is_retrained_on_the_complete_training_data(
    classification_prepared: PreparedDataset,
) -> None:
    """The final model uses every training row, not a fold's subset."""
    outcome = select_and_evaluate_best_model(
        classification_prepared, strategy=CV, folds=FOLDS
    )
    branch = outcome.final_model.preprocessor.named_transformers_["numeric"]
    imputer = dict(branch.transformer_list)["values"].named_steps["impute"]

    assert outcome.final_model.dataset.train_row_count == (
        classification_prepared.train_row_count
    )
    assert imputer.statistics_[0] == pytest.approx(
        classification_prepared.X_train_raw["income"].median()
    )


def test_the_final_model_is_the_selected_model(
    classification_prepared: PreparedDataset,
) -> None:
    """The model measured is the one cross-validation chose."""
    outcome = select_and_evaluate_best_model(
        classification_prepared, strategy=CV, folds=FOLDS
    )

    assert outcome.final_model.model_name == outcome.selected_model_name


def test_the_final_model_is_compared_with_the_baseline(
    classification_prepared: PreparedDataset,
) -> None:
    """The naive reference still frames the final number."""
    outcome = select_and_evaluate_best_model(
        classification_prepared, strategy=CV, folds=FOLDS
    )
    comparison = outcome.final_model.baseline_comparison

    assert outcome.final_model.baseline.identifier == "majority_class_baseline"
    assert comparison.baseline_value is not None
    assert comparison.beats_baseline is True
    assert comparison.absolute_improvement > 0


def test_regression_selection_end_to_end(
    regression_prepared: PreparedDataset,
) -> None:
    """The same flow works for a continuous target."""
    outcome = select_and_evaluate_best_model(
        regression_prepared, strategy=CV, folds=FOLDS
    )

    assert outcome.comparison.primary_metric.key == "rmse"
    assert outcome.final_model.task_type is TaskType.REGRESSION
    assert outcome.final_test_score > 0
    assert outcome.final_model.baseline.identifier == "mean_baseline"


def test_holdout_selection_reuses_the_already_trained_model(
    classification_prepared: PreparedDataset,
) -> None:
    """Under holdout there is no separate final step, and the result says so."""
    outcome = select_and_evaluate_best_model(
        classification_prepared, strategy="holdout"
    )

    assert outcome.strategy is SelectionStrategy.HOLDOUT
    assert outcome.final_evaluation_is_unbiased is False
    assert outcome.final_model is outcome.selected_entry.trained_model
    assert outcome.selection_score == outcome.final_test_score


# --------------------------------------------------------------------------
# Leakage: the test set cannot reach the choice of model
# --------------------------------------------------------------------------


def test_ruining_the_test_set_leaves_the_cv_ranking_unchanged(
    classification_prepared: PreparedDataset,
) -> None:
    """The decisive check: selection is blind to the test set.

    The test half is reversed against its own index, which scrambles the
    pairing of rows and labels. If any part of cross-validated selection read
    it, the ranking or the fold means would move. Neither does.
    """
    baseline = compare_models(classification_prepared, strategy=CV, folds=FOLDS)
    after = compare_models(_poison_test_set(classification_prepared), strategy=CV, folds=FOLDS)

    assert [entry.model_name for entry in after.entries] == [
        entry.model_name for entry in baseline.entries
    ]
    assert [entry.primary_metric_value for entry in after.entries] == [
        entry.primary_metric_value for entry in baseline.entries
    ]
    assert [entry.primary_metric_std for entry in after.entries] == [
        entry.primary_metric_std for entry in baseline.entries
    ]
    assert select_best_model(after).model_name == select_best_model(baseline).model_name


def test_ruining_the_test_set_does_change_the_final_score(
    classification_prepared: PreparedDataset,
) -> None:
    """The complement: the test set is what the final measurement reads.

    Together with the test above, this shows the split of responsibilities is
    real — the test data moves the final number and nothing else.
    """
    original = select_and_evaluate_best_model(
        classification_prepared, strategy=CV, folds=FOLDS
    )
    poisoned = select_and_evaluate_best_model(
        _poison_test_set(classification_prepared), strategy=CV, folds=FOLDS
    )

    assert poisoned.selected_model_name == original.selected_model_name
    assert poisoned.selection_score == original.selection_score
    assert poisoned.final_test_score != pytest.approx(original.final_test_score)


def test_cv_comparison_is_reproducible(
    classification_prepared: PreparedDataset,
) -> None:
    """The same dataset, models, folds and seed rank the same way twice."""
    first = compare_models(classification_prepared, strategy=CV, folds=FOLDS)
    second = compare_models(classification_prepared, strategy=CV, folds=FOLDS)

    assert [entry.model_name for entry in first.entries] == [
        entry.model_name for entry in second.entries
    ]
    assert [entry.primary_metric_value for entry in first.entries] == [
        entry.primary_metric_value for entry in second.entries
    ]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_selection_summary_separates_the_two_measurements(
    classification_prepared: PreparedDataset,
) -> None:
    """Selection numbers and the final test number never share a section."""
    outcome = select_and_evaluate_best_model(
        classification_prepared, strategy=CV, folds=FOLDS
    )
    summary = outcome.summary()

    json.dumps(summary)
    selection = summary["selection"]
    final = summary["final_evaluation"]

    assert selection["scored_on"] == "training_folds"
    assert selection["uses_test_data"] is False
    assert selection["winner"] == outcome.selected_model_name
    assert selection["winner_score_std"] is not None
    assert len(selection["candidates"]) == 3

    assert final["trained_on"] == "full_training_data"
    assert final["evaluated_on"] == "held_out_test_set"
    assert final["is_unbiased"] is True
    assert final["test_row_count"] == classification_prepared.test_row_count
    assert final["baseline_comparison"]["beats_baseline"] is True
    assert "pipeline" not in json.dumps(summary)


def test_holdout_summary_admits_it_is_biased(
    classification_prepared: PreparedDataset,
) -> None:
    """The holdout path reports honestly that selection used the test set."""
    summary = select_and_evaluate_best_model(
        classification_prepared, strategy="holdout"
    ).summary()

    assert summary["selection"]["scored_on"] == "held_out_test_set"
    assert summary["selection"]["uses_test_data"] is True
    assert summary["final_evaluation"]["is_unbiased"] is False


def test_text_table_labels_where_the_score_came_from(
    classification_prepared: PreparedDataset,
) -> None:
    """A reader cannot mistake a cross-validated mean for a test score."""
    outcome = select_and_evaluate_best_model(
        classification_prepared, strategy=CV, folds=FOLDS
    )
    text = outcome.as_text()

    assert "CV Mean F1" in text
    assert "CV Std" in text
    assert f"Winner: {outcome.final_model.display_name}" in text
    assert "the test set was not used" in text
    assert "Final held-out test F1" in text
    assert "Baseline" in text


def test_text_table_for_the_holdout_strategy(
    classification_prepared: PreparedDataset,
) -> None:
    """The holdout table is labelled as a test score, with no spread column."""
    comparison = compare_models(classification_prepared)
    text = format_comparison_table(comparison)

    assert "Test F1" in text
    assert "CV Std" not in text
