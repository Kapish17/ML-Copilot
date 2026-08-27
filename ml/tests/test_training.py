"""Tests for training a single model end to end."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from ml.errors import (
    IncompatibleTaskError,
    InsufficientDataError,
    InvalidHyperparameterError,
    ModelTrainingError,
    UnknownModelError,
)
from ml.features.types import TaskType
from ml.models.registry import default_registry
from ml.models.result import MODEL_STEP, PREPROCESSING_STEP
from ml.models.spec import ModelSpec
from ml.models.training import train_model
from ml.pipelines.result import PreparedDataset
from ml.tests.conftest import classification_config
from ml.tests.factories import learnable_classification_frame

CLASSIFIERS = (
    "logistic_regression",
    "random_forest_classifier",
    "hist_gradient_boosting_classifier",
)
REGRESSORS = (
    "linear_regression",
    "random_forest_regressor",
    "hist_gradient_boosting_regressor",
)
RAW_FEATURE_COLUMNS = ["income", "tenure_months", "segment"]


@pytest.mark.parametrize("model_name", CLASSIFIERS)
def test_every_classifier_trains_and_scores(
    classification_prepared: PreparedDataset, model_name: str
) -> None:
    """Each registered classifier produces a full metric set."""
    trained = train_model(classification_prepared, model_name)

    assert trained.task_type is TaskType.CLASSIFICATION
    assert trained.model_name == model_name
    assert trained.metrics.get("accuracy") is not None
    assert trained.metrics.get("f1") is not None
    assert trained.metrics.get("roc_auc") is not None, "all three expose probabilities"
    assert trained.metrics.classification is not None
    assert trained.metrics.classification.class_count == 2
    assert trained.training_seconds >= 0


@pytest.mark.parametrize("model_name", REGRESSORS)
def test_every_regressor_trains_and_scores(
    regression_prepared: PreparedDataset, model_name: str
) -> None:
    """Each registered regressor produces a full metric set."""
    trained = train_model(regression_prepared, model_name)

    assert trained.task_type is TaskType.REGRESSION
    assert set(trained.metrics.values) == {"mae", "mse", "rmse", "r2"}
    assert trained.metrics.get("rmse") > 0
    assert trained.metrics.classification is None


def test_classification_beats_the_majority_baseline(
    classification_prepared: PreparedDataset,
) -> None:
    """On a learnable problem a real model should clear the naive reference."""
    trained = train_model(classification_prepared, "logistic_regression")

    assert trained.baseline_comparison.beats_baseline is True
    assert trained.baseline_comparison.absolute_improvement > 0
    assert trained.primary_metric.key == "f1"


def test_regression_beats_the_mean_baseline(
    regression_prepared: PreparedDataset,
) -> None:
    """A fitted regressor should have less error than predicting the mean."""
    trained = train_model(regression_prepared, "linear_regression")

    assert trained.primary_metric.key == "rmse"
    assert trained.primary_metric_value < trained.baseline.metrics.get("rmse")
    assert trained.baseline_comparison.beats_baseline is True


def test_multiclass_training_uses_macro_averaging(
    multiclass_prepared: PreparedDataset,
) -> None:
    """A three-class problem is handled explicitly, not as a binary one."""
    trained = train_model(multiclass_prepared, "random_forest_classifier")
    details = trained.metrics.classification

    assert details is not None
    assert details.class_count == 3
    assert details.averaging == "macro"
    assert trained.metrics.get("roc_auc") is not None


def test_trained_pipeline_contains_preprocessing_and_estimator(
    classification_prepared: PreparedDataset,
) -> None:
    """The artefact is a pipeline, not a bare estimator."""
    trained = train_model(classification_prepared, "logistic_regression")

    assert isinstance(trained.pipeline, Pipeline)
    assert list(trained.pipeline.named_steps) == [PREPROCESSING_STEP, MODEL_STEP]
    assert trained.preprocessor is trained.pipeline.named_steps[PREPROCESSING_STEP]
    assert trained.estimator is trained.pipeline.named_steps[MODEL_STEP]


def test_pipeline_reproduces_the_prepared_transformation(
    classification_prepared: PreparedDataset,
) -> None:
    """The preprocessing inside the model matches the preparation step exactly."""
    trained = train_model(classification_prepared, "logistic_regression")
    transformed = trained.preprocessor.transform(classification_prepared.X_test_raw)

    assert list(transformed.columns) == list(classification_prepared.feature_names)
    assert np.allclose(
        transformed.to_numpy(), classification_prepared.X_test.to_numpy()
    )


def test_trained_model_accepts_raw_feature_rows(
    classification_prepared: PreparedDataset,
) -> None:
    """A caller passes original columns, not a pre-transformed matrix."""
    trained = train_model(classification_prepared, "logistic_regression")
    raw = learnable_classification_frame().loc[:4, RAW_FEATURE_COLUMNS]

    predictions = trained.predict(raw)

    assert len(predictions) == 5
    assert set(predictions) <= {"yes", "no"}


def test_raw_rows_with_gaps_and_unseen_categories_still_predict(
    classification_prepared: PreparedDataset,
) -> None:
    """Inference-time surprises are handled by the preserved preprocessing."""
    trained = train_model(classification_prepared, "random_forest_classifier")
    awkward = pd.DataFrame(
        {
            "income": [55_000.0, np.nan],
            "tenure_months": [np.nan, 12.0],
            "segment": ["retail", "a_segment_never_seen"],
        }
    )

    predictions = trained.predict(awkward)

    assert len(predictions) == 2


def test_probabilities_are_available_for_classifiers(
    classification_prepared: PreparedDataset,
) -> None:
    """Probability output survives the pipeline wrapper."""
    trained = train_model(classification_prepared, "logistic_regression")
    raw = learnable_classification_frame().loc[:2, RAW_FEATURE_COLUMNS]

    probabilities = trained.predict_proba(raw)

    assert probabilities.shape == (3, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_the_target_never_reaches_the_model_pipeline(
    classification_prepared: PreparedDataset,
) -> None:
    """The fitted pipeline was never shown the column it predicts."""
    trained = train_model(classification_prepared, "logistic_regression")
    seen = list(trained.preprocessor.feature_names_in_)

    assert "renewed" not in seen
    assert seen == list(classification_prepared.config.feature_columns)


def test_training_ignores_the_test_rows() -> None:
    """Corrupting the test features leaves the fitted model untouched.

    The split depends only on the target and the seed, so the same rows stay on
    each side. If any statistic or split of the estimator were learned from the
    test set, the training predictions would move. They do not.
    """
    from ml.pipelines.preparation import prepare_dataset

    frame = learnable_classification_frame()
    baseline_prepared = prepare_dataset(frame, classification_config())
    baseline_model = train_model(baseline_prepared, "random_forest_classifier")

    corrupted = frame.copy()
    corrupted.loc[baseline_prepared.X_test_raw.index, "income"] = 10_000_000.0
    corrupted.loc[baseline_prepared.X_test_raw.index, "segment"] = "corrupted"
    corrupted_prepared = prepare_dataset(corrupted, classification_config())
    corrupted_model = train_model(corrupted_prepared, "random_forest_classifier")

    assert list(corrupted_prepared.X_test_raw.index) == list(
        baseline_prepared.X_test_raw.index
    )
    assert np.array_equal(
        baseline_model.predict(baseline_prepared.X_train_raw),
        corrupted_model.predict(baseline_prepared.X_train_raw),
    )
    assert np.allclose(
        baseline_model.preprocessor.transform(
            baseline_prepared.X_train_raw
        ).to_numpy(),
        corrupted_model.preprocessor.transform(
            baseline_prepared.X_train_raw
        ).to_numpy(),
    )


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest_classifier"])
def test_training_is_reproducible(
    classification_prepared: PreparedDataset, model_name: str
) -> None:
    """The same data and specification give the same model twice."""
    first = train_model(classification_prepared, model_name)
    second = train_model(classification_prepared, model_name)

    assert np.array_equal(
        first.predict(classification_prepared.X_test_raw),
        second.predict(classification_prepared.X_test_raw),
    )
    assert first.metrics.values == second.metrics.values


def test_a_different_seed_can_change_a_stochastic_model(
    classification_prepared: PreparedDataset,
) -> None:
    """The seed is genuinely wired into estimators that use one."""
    trained = train_model(
        classification_prepared,
        ModelSpec(model_name="random_forest_classifier", random_state=7),
    )
    assert trained.estimator.get_params()["random_state"] == 7


def test_dataset_seed_is_used_when_the_spec_has_none(
    classification_prepared: PreparedDataset,
) -> None:
    """One seed on the dataset makes the whole run reproducible."""
    trained = train_model(classification_prepared, "random_forest_classifier")
    assert trained.estimator.get_params()["random_state"] == (
        classification_prepared.config.random_state
    )


def test_primary_metric_can_be_overridden(
    regression_prepared: PreparedDataset,
) -> None:
    """Ranking does not have to use the task default."""
    trained = train_model(
        regression_prepared,
        ModelSpec(model_name="linear_regression", primary_metric="mae"),
    )

    assert trained.primary_metric.key == "mae"
    assert trained.primary_metric_value == trained.metrics.get("mae")


def test_hyperparameters_reach_the_estimator(
    classification_prepared: PreparedDataset,
) -> None:
    """A specification's hyperparameters are actually applied."""
    trained = train_model(
        classification_prepared,
        ModelSpec(
            model_name="random_forest_classifier",
            hyperparameters={"n_estimators": 12, "max_depth": 3},
        ),
    )
    params = trained.estimator.get_params()

    assert params["n_estimators"] == 12
    assert params["max_depth"] == 3


def test_dataset_information_is_recorded(
    classification_prepared: PreparedDataset,
) -> None:
    """The result says what it was trained on."""
    trained = train_model(classification_prepared, "logistic_regression")
    dataset = trained.dataset

    assert dataset.target_column == "renewed"
    assert dataset.train_row_count == classification_prepared.train_row_count
    assert dataset.test_row_count == classification_prepared.test_row_count
    assert dataset.transformed_feature_count == len(classification_prepared.feature_names)
    assert dataset.stratified is True


def test_feature_names_are_carried_through(
    classification_prepared: PreparedDataset,
) -> None:
    """Encoded feature names survive into the trained model."""
    trained = train_model(classification_prepared, "logistic_regression")

    assert trained.feature_names == classification_prepared.feature_names
    assert "segment_retail" in trained.feature_names


def test_summary_is_serialisable_and_omits_the_pipeline(
    classification_prepared: PreparedDataset,
) -> None:
    """The summary is the boundary; sklearn objects stay behind it."""
    trained = train_model(classification_prepared, "logistic_regression")
    summary = trained.summary()

    json.dumps(summary)
    assert "pipeline" not in summary
    assert "estimator" not in summary
    assert summary["model_name"] == "logistic_regression"
    assert summary["primary_metric"]["key"] == "f1"
    assert summary["primary_metric"]["direction"] == "higher_is_better"
    assert summary["baseline"]["identifier"] == "majority_class_baseline"
    assert summary["baseline_comparison"]["beats_baseline"] is True
    assert summary["dataset"]["target_column"] == "renewed"


def test_unknown_model_is_rejected(classification_prepared: PreparedDataset) -> None:
    """A model outside the registry cannot be trained."""
    with pytest.raises(UnknownModelError):
        train_model(classification_prepared, "xgboost")


def test_incompatible_model_is_rejected(
    classification_prepared: PreparedDataset,
) -> None:
    """A regressor cannot be trained on a classification dataset."""
    with pytest.raises(IncompatibleTaskError):
        train_model(classification_prepared, "linear_regression")


def test_invalid_hyperparameter_is_rejected(
    classification_prepared: PreparedDataset,
) -> None:
    """A misspelled hyperparameter fails before any fitting happens."""
    with pytest.raises(InvalidHyperparameterError):
        train_model(
            classification_prepared,
            ModelSpec(
                model_name="logistic_regression", hyperparameters={"max_iterations": 10}
            ),
        )


def test_estimator_failure_becomes_a_training_error(
    classification_prepared: PreparedDataset,
) -> None:
    """A failure inside scikit-learn is reported as a typed ML error."""
    registry = default_registry()
    spec = ModelSpec(
        model_name="logistic_regression",
        hyperparameters={"C": -1.0},
    )
    with pytest.raises(ModelTrainingError) as exc_info:
        train_model(classification_prepared, spec, registry=registry)

    assert exc_info.value.details["model_name"] == "logistic_regression"
    assert exc_info.value.details["error_type"]


def test_empty_split_is_rejected(classification_prepared: PreparedDataset) -> None:
    """Training refuses a dataset with nothing to learn from or test on."""
    from dataclasses import replace

    empty = replace(
        classification_prepared,
        X_train_raw=classification_prepared.X_train_raw.iloc[:0],
        train_row_count=0,
    )
    with pytest.raises(InsufficientDataError):
        train_model(empty, "logistic_regression")
