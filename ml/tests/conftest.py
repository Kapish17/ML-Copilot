"""Shared fixtures for the ML test suite.

The prepared datasets are built once per session: preparation is deterministic
and the results are read-only, so rebuilding them for every test would only
cost time.
"""

from __future__ import annotations

import pytest

from ml.features.config import PreprocessingConfig
from ml.models.result import TrainedModel
from ml.models.training import train_model
from ml.pipelines.preparation import prepare_dataset
from ml.pipelines.result import PreparedDataset
from ml.tests.factories import (
    imbalanced_classification_frame,
    learnable_classification_frame,
    multiclass_frame,
    regression_frame,
)

CLASSIFICATION_FEATURES = ("income", "tenure_months")
CLASSIFICATION_CATEGORICALS = ("segment",)
REGRESSION_FEATURES = ("size_sqm", "rooms")
REGRESSION_CATEGORICALS = ("district",)


def classification_config(**overrides: object) -> PreprocessingConfig:
    """Configuration for the learnable binary dataset."""
    config = PreprocessingConfig(
        target_column="renewed",
        numeric_columns=CLASSIFICATION_FEATURES,
        categorical_columns=CLASSIFICATION_CATEGORICALS,
        task_type="classification",
    )
    return config.with_overrides(**overrides) if overrides else config


def regression_config(**overrides: object) -> PreprocessingConfig:
    """Configuration for the housing dataset."""
    config = PreprocessingConfig(
        target_column="price",
        numeric_columns=REGRESSION_FEATURES,
        categorical_columns=REGRESSION_CATEGORICALS,
        task_type="regression",
    )
    return config.with_overrides(**overrides) if overrides else config


@pytest.fixture(scope="session")
def classification_prepared() -> PreparedDataset:
    """A prepared binary-classification dataset with a learnable signal."""
    return prepare_dataset(learnable_classification_frame(), classification_config())


@pytest.fixture(scope="session")
def regression_prepared() -> PreparedDataset:
    """A prepared regression dataset with a genuine linear relationship."""
    return prepare_dataset(regression_frame(rows=200), regression_config())


@pytest.fixture(scope="session")
def imbalanced_prepared() -> PreparedDataset:
    """A prepared dataset with a small but workable minority class."""
    config = PreprocessingConfig(
        target_column="outcome",
        numeric_columns=("measure", "noise"),
        task_type="classification",
    )
    return prepare_dataset(imbalanced_classification_frame(), config)


@pytest.fixture(scope="session")
def multiclass_prepared() -> PreparedDataset:
    """A prepared three-class dataset."""
    config = PreprocessingConfig(
        target_column="grade",
        numeric_columns=("first_measure", "second_measure"),
        task_type="classification",
    )
    return prepare_dataset(multiclass_frame(), config)


# ---------------------------------------------------------------------------
# Trained models
#
# Training is deterministic and the results are read-only, so each model is
# fitted once for the whole session. The explainability tests depend on that:
# several of them assert a model is unchanged after being explained, which is
# only meaningful if they share one instance.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def logistic_model(classification_prepared: PreparedDataset) -> TrainedModel:
    """A fitted linear classifier."""
    return train_model(classification_prepared, "logistic_regression")


@pytest.fixture(scope="session")
def forest_model(classification_prepared: PreparedDataset) -> TrainedModel:
    """A fitted tree-ensemble classifier."""
    return train_model(classification_prepared, "random_forest_classifier")


@pytest.fixture(scope="session")
def boosting_model(classification_prepared: PreparedDataset) -> TrainedModel:
    """A fitted boosted-tree classifier, which emits a single SHAP output."""
    return train_model(classification_prepared, "hist_gradient_boosting_classifier")


@pytest.fixture(scope="session")
def linear_regression_model(regression_prepared: PreparedDataset) -> TrainedModel:
    """A fitted linear regressor."""
    return train_model(regression_prepared, "linear_regression")


@pytest.fixture(scope="session")
def forest_regression_model(regression_prepared: PreparedDataset) -> TrainedModel:
    """A fitted tree-ensemble regressor."""
    return train_model(regression_prepared, "random_forest_regressor")


@pytest.fixture(scope="session")
def multiclass_model(multiclass_prepared: PreparedDataset) -> TrainedModel:
    """A fitted classifier on a three-class problem."""
    return train_model(multiclass_prepared, "random_forest_classifier")
