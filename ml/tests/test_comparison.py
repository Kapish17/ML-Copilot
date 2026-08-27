"""Tests for training several models and picking the best one."""

from __future__ import annotations

import json

import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from ml.errors import NoSuccessfulModelError
from ml.features.types import TaskType
from ml.models.comparison import ModelStatus, compare_models, select_best_model
from ml.models.registry import ModelDefinition, ModelRegistry, default_registry
from ml.models.spec import ModelSpec
from ml.pipelines.result import PreparedDataset

FAILING_MODEL = "always_fails"


class _FailingClassifier(BaseEstimator, ClassifierMixin):
    """An estimator that always raises, used to exercise the error boundary."""

    def fit(self, X, y=None):  # noqa: ANN001, ANN201 - sklearn signature
        """Fail loudly, as a broken estimator would."""
        raise RuntimeError("this estimator always fails")

    def predict(self, X):  # noqa: ANN001, ANN201 - sklearn signature
        """Never reached; present so the class looks like an estimator."""
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


def test_comparison_trains_every_registered_model(
    classification_prepared: PreparedDataset,
) -> None:
    """By default every model for the dataset's task is tried."""
    comparison = compare_models(classification_prepared)

    assert len(comparison.entries) == 3
    assert {entry.model_name for entry in comparison.entries} == {
        "logistic_regression",
        "random_forest_classifier",
        "hist_gradient_boosting_classifier",
    }
    assert all(entry.status is ModelStatus.SUCCEEDED for entry in comparison.entries)


def test_comparison_rows_carry_the_full_picture(
    classification_prepared: PreparedDataset,
) -> None:
    """Each row has the metrics, the baseline and the timing behind it."""
    entry = compare_models(classification_prepared).entries[0]

    assert entry.primary_metric == "f1"
    assert entry.primary_metric_value is not None
    assert entry.metrics is not None
    assert entry.metrics.get("accuracy") is not None
    assert entry.baseline_comparison is not None
    assert entry.baseline_comparison.baseline_value is not None
    assert entry.training_seconds is not None and entry.training_seconds >= 0
    assert entry.trained_model is not None
    assert entry.error is None


def test_one_baseline_is_shared_by_every_model(
    classification_prepared: PreparedDataset,
) -> None:
    """Models are measured against the same reference, not one each."""
    comparison = compare_models(classification_prepared)
    baseline_value = comparison.baseline.metrics.get("f1")

    for entry in comparison.successful():
        assert entry.baseline_comparison is not None
        assert entry.baseline_comparison.baseline_value == baseline_value


def test_higher_is_better_metric_selects_the_maximum(
    classification_prepared: PreparedDataset,
) -> None:
    """F1 is a score, so the best model is the one with the largest value."""
    comparison = compare_models(classification_prepared)
    values = [
        entry.primary_metric_value
        for entry in comparison.successful()
        if entry.primary_metric_value is not None
    ]
    best = select_best_model(comparison)

    assert comparison.primary_metric.key == "f1"
    assert best.primary_metric_value == max(values)
    assert comparison.entries[0] is best, "entries are ordered best first"


def test_lower_is_better_metric_selects_the_minimum(
    regression_prepared: PreparedDataset,
) -> None:
    """RMSE is an error, so the best model is the one with the smallest value."""
    comparison = compare_models(regression_prepared)
    values = [
        entry.primary_metric_value
        for entry in comparison.successful()
        if entry.primary_metric_value is not None
    ]
    best = select_best_model(comparison)

    assert comparison.primary_metric.key == "rmse"
    assert comparison.primary_metric.higher_is_better is False
    assert best.primary_metric_value == min(values)
    assert best.primary_metric_value < max(values), "the ranking is not accidental"


def test_selection_follows_the_configured_metric(
    regression_prepared: PreparedDataset,
) -> None:
    """Ranking by R² maximises it, even though the default minimises RMSE."""
    comparison = compare_models(regression_prepared, primary_metric="r2")
    values = [
        entry.primary_metric_value
        for entry in comparison.successful()
        if entry.primary_metric_value is not None
    ]

    assert comparison.primary_metric.key == "r2"
    assert select_best_model(comparison).primary_metric_value == max(values)


def test_explicit_model_list_is_respected(
    classification_prepared: PreparedDataset,
) -> None:
    """A caller can compare a subset."""
    comparison = compare_models(
        classification_prepared, models=["logistic_regression", "random_forest_classifier"]
    )
    assert {entry.model_name for entry in comparison.entries} == {
        "logistic_regression",
        "random_forest_classifier",
    }


def test_specifications_can_be_compared(
    classification_prepared: PreparedDataset,
) -> None:
    """Models may be given as full specifications, not just names."""
    comparison = compare_models(
        classification_prepared,
        models=[
            ModelSpec(
                model_name="random_forest_classifier",
                hyperparameters={"n_estimators": 15},
            ),
            "logistic_regression",
        ],
    )
    forest = next(
        entry for entry in comparison.entries if entry.model_name == "random_forest_classifier"
    )

    assert forest.status is ModelStatus.SUCCEEDED
    assert forest.trained_model is not None
    assert forest.trained_model.estimator.get_params()["n_estimators"] == 15


def test_one_failure_does_not_stop_the_others(
    classification_prepared: PreparedDataset,
) -> None:
    """A broken model is recorded; every compatible model still runs."""
    comparison = compare_models(
        classification_prepared,
        models=[FAILING_MODEL, "logistic_regression", "random_forest_classifier"],
        registry=_registry_with_failure(),
    )

    assert len(comparison.successful()) == 2
    failure = comparison.failed()[0]
    assert failure.model_name == FAILING_MODEL
    assert failure.status is ModelStatus.FAILED
    assert "always fails" in (failure.error or "")
    assert failure.error_type == "ModelTrainingError"
    assert failure.primary_metric_value is None


def test_failures_are_ranked_last(classification_prepared: PreparedDataset) -> None:
    """A model that failed can never appear above one that worked."""
    comparison = compare_models(
        classification_prepared,
        models=[FAILING_MODEL, "logistic_regression"],
        registry=_registry_with_failure(),
    )

    assert comparison.entries[-1].model_name == FAILING_MODEL
    assert select_best_model(comparison).model_name == "logistic_regression"


def test_an_unknown_model_is_recorded_as_a_failure(
    classification_prepared: PreparedDataset,
) -> None:
    """A bad name in the list fails that row alone."""
    comparison = compare_models(
        classification_prepared, models=["not_a_model", "logistic_regression"]
    )
    failure = comparison.failed()[0]

    assert failure.model_name == "not_a_model"
    assert failure.error_type == "UnknownModelError"
    assert len(comparison.successful()) == 1


def test_an_incompatible_model_is_recorded_as_a_failure(
    classification_prepared: PreparedDataset,
) -> None:
    """Asking for a regressor on a classification dataset fails that row."""
    comparison = compare_models(
        classification_prepared, models=["linear_regression", "logistic_regression"]
    )
    failure = comparison.failed()[0]

    assert failure.model_name == "linear_regression"
    assert failure.error_type == "IncompatibleTaskError"
    assert len(comparison.successful()) == 1


def test_selection_fails_loudly_when_nothing_worked(
    classification_prepared: PreparedDataset,
) -> None:
    """No silent ``None``: an empty comparison raises with the reasons."""
    comparison = compare_models(
        classification_prepared,
        models=[FAILING_MODEL],
        registry=_registry_with_failure(),
    )

    assert comparison.best() is None
    with pytest.raises(NoSuccessfulModelError) as exc_info:
        select_best_model(comparison)
    assert FAILING_MODEL in exc_info.value.details["errors"]


def test_comparison_table_is_serialisable(
    classification_prepared: PreparedDataset,
) -> None:
    """The table is plain data; the fitted models stay on the objects."""
    comparison = compare_models(
        classification_prepared,
        models=[FAILING_MODEL, "logistic_regression"],
        registry=_registry_with_failure(),
    )
    table = comparison.as_table()

    json.dumps(table)
    assert all("trained_model" not in row for row in table)
    assert {row["status"] for row in table} == {"succeeded", "failed"}


def test_comparison_summary_is_serialisable(
    classification_prepared: PreparedDataset,
) -> None:
    """The whole comparison renders as one JSON-friendly document."""
    comparison = compare_models(classification_prepared)
    summary = comparison.summary()

    json.dumps(summary)
    assert summary["task_type"] == "classification"
    assert summary["primary_metric"]["direction"] == "higher_is_better"
    assert summary["succeeded_count"] == 3
    assert summary["failed_count"] == 0
    assert summary["best_model"] == select_best_model(comparison).model_name
    assert summary["baseline"]["identifier"] == "majority_class_baseline"


def test_comparison_is_reproducible(classification_prepared: PreparedDataset) -> None:
    """The same dataset ranks the same way twice."""
    first = compare_models(classification_prepared)
    second = compare_models(classification_prepared)

    assert [entry.model_name for entry in first.entries] == [
        entry.model_name for entry in second.entries
    ]
    assert [entry.primary_metric_value for entry in first.entries] == [
        entry.primary_metric_value for entry in second.entries
    ]
