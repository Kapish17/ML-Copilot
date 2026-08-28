"""Shared fixtures for the ML test suite.

The prepared datasets are built once per session: preparation is deterministic
and the results are read-only, so rebuilding them for every test would only
cost time.
"""

from __future__ import annotations

import pytest

from ml.features.config import PreprocessingConfig
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
