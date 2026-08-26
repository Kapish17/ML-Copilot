"""End-to-end tests for dataset preparation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml.errors import EmptyFeatureSetError, InsufficientDataError, MLError, MissingTargetError
from ml.features.config import PreprocessingConfig
from ml.features.types import ColumnRole, TaskType
from ml.pipelines.preparation import prepare_dataset, separate_target
from ml.tests.factories import classification_frame, regression_frame


def _classification_config(**overrides: object) -> PreprocessingConfig:
    """The configuration used for the churn dataset in these tests."""
    config = PreprocessingConfig(
        target_column="churn",
        numeric_columns=("age", "monthly_charges"),
        categorical_columns=("contract", "payment_method"),
        boolean_columns=("is_active",),
        datetime_columns=("signup_date",),
        identifier_columns=("customer_id",),
        excluded_columns=("notes", "plan"),
        task_type=TaskType.CLASSIFICATION,
    )
    return config.with_overrides(**overrides) if overrides else config


def _regression_config() -> PreprocessingConfig:
    """The configuration used for the housing dataset in these tests."""
    return PreprocessingConfig(
        target_column="price",
        numeric_columns=("size_sqm", "rooms"),
        categorical_columns=("district",),
        task_type=TaskType.REGRESSION,
    )


def test_classification_dataset_is_prepared() -> None:
    """A mixed-type dataset produces complete, numeric train and test frames."""
    prepared = prepare_dataset(classification_frame(), _classification_config())

    assert prepared.train_row_count == 96
    assert prepared.test_row_count == 24
    assert prepared.X_train.shape[1] == prepared.X_test.shape[1] == prepared.feature_count
    assert prepared.X_train.isna().to_numpy().sum() == 0
    assert prepared.X_test.isna().to_numpy().sum() == 0
    assert set(prepared.X_train.dtypes.astype(str)) == {"float64"}


def test_regression_dataset_is_prepared() -> None:
    """A continuous target is prepared without stratification."""
    prepared = prepare_dataset(regression_frame(), _regression_config())

    assert prepared.task_type is TaskType.REGRESSION
    assert prepared.stratified is False
    assert prepared.target_train.numeric_summary is not None
    assert prepared.target_train.class_counts is None


def test_classification_split_is_stratified() -> None:
    """Class proportions survive the split for a classification task."""
    prepared = prepare_dataset(classification_frame(), _classification_config())

    assert prepared.stratified is True
    train_share = prepared.target_train.class_percentages
    test_share = prepared.target_test.class_percentages
    assert train_share is not None and test_share is not None
    assert train_share["yes"] == pytest.approx(test_share["yes"], abs=2.0)


def test_target_is_separated_from_the_features() -> None:
    """X never contains the target, and y keeps the original values."""
    frame = classification_frame()
    prepared = prepare_dataset(frame, _classification_config())

    assert "churn" not in prepared.X_train.columns
    assert "churn" not in prepared.X_test.columns
    assert not any(name.startswith("churn") for name in prepared.feature_names)
    assert prepared.y_train.tolist() == frame.loc[prepared.y_train.index, "churn"].tolist()


def test_target_never_reaches_a_transformer() -> None:
    """The fitted preprocessor was never shown the target column."""
    prepared = prepare_dataset(classification_frame(), _classification_config())
    assert "churn" not in list(prepared.preprocessor.feature_names_in_)


def test_separate_target_selects_only_feature_columns() -> None:
    """Feature selection is positive, not "everything except the target"."""
    frame = classification_frame()
    features, target = separate_target(frame, _classification_config())

    assert list(features.columns) == list(_classification_config().feature_columns)
    assert target.name == "churn"


def test_excluded_and_identifier_columns_are_absent() -> None:
    """Nothing excluded reaches the feature matrix, under any name."""
    prepared = prepare_dataset(classification_frame(), _classification_config())

    for column in ("customer_id", "notes", "plan"):
        assert column not in prepared.X_train.columns
        assert not any(name.startswith(column) for name in prepared.feature_names)


def test_excluded_columns_are_reported() -> None:
    """The result says what was left out, so the choice stays visible."""
    prepared = prepare_dataset(classification_frame(), _classification_config())

    assert prepared.identifier_columns == ("customer_id",)
    assert prepared.excluded_columns == ("notes", "plan")


def test_every_column_has_a_decision_even_without_a_profile() -> None:
    """A hand-written configuration still explains each column."""
    frame = classification_frame()
    prepared = prepare_dataset(frame, _classification_config())
    decisions = {item.column: item for item in prepared.column_decisions}

    assert set(decisions) == set(frame.columns)
    assert decisions["churn"].role is ColumnRole.TARGET
    assert decisions["customer_id"].role is ColumnRole.IDENTIFIER
    assert decisions["notes"].role is ColumnRole.EXCLUDED
    assert decisions["age"].role is ColumnRole.FEATURE
    assert all(item.reason for item in prepared.column_decisions)


def test_feature_names_are_readable() -> None:
    """Encoded features keep the name of the column they came from."""
    prepared = prepare_dataset(classification_frame(), _classification_config())

    assert "monthly_charges" in prepared.feature_names
    assert "contract_Month-to-month" in prepared.feature_names
    assert "signup_date_day_of_week" in prepared.feature_names
    assert "missingindicator_age" in prepared.feature_names
    assert list(prepared.X_train.columns) == list(prepared.feature_names)


def test_rows_without_a_target_are_dropped_and_counted() -> None:
    """Unlabelled rows cannot be used, and their removal is reported."""
    frame = classification_frame()
    frame.loc[frame.index[:5], "churn"] = np.nan
    prepared = prepare_dataset(frame, _classification_config())

    assert prepared.rows_dropped_missing_target == 5
    assert prepared.train_row_count + prepared.test_row_count == len(frame) - 5


def test_class_distribution_is_reported_but_not_changed() -> None:
    """No resampling happens: the class balance is measured, not corrected."""
    frame = classification_frame()
    prepared = prepare_dataset(frame, _classification_config())

    overall = prepared.target_overall
    assert overall.class_counts == frame["churn"].value_counts().to_dict()
    assert overall.imbalance_ratio is not None and overall.imbalance_ratio > 1


def test_preparation_is_reproducible() -> None:
    """The same frame, configuration and seed give the same result twice."""
    first = prepare_dataset(classification_frame(), _classification_config())
    second = prepare_dataset(classification_frame(), _classification_config())

    assert list(first.X_train.index) == list(second.X_train.index)
    assert np.allclose(first.X_train.to_numpy(), second.X_train.to_numpy())


def test_a_different_seed_changes_the_split() -> None:
    """The seed is honoured end to end."""
    first = prepare_dataset(classification_frame(), _classification_config())
    second = prepare_dataset(
        classification_frame(), _classification_config(random_state=99)
    )
    assert list(first.X_train.index) != list(second.X_train.index)


def test_test_size_is_configurable() -> None:
    """The split fraction comes from the configuration."""
    prepared = prepare_dataset(
        classification_frame(), _classification_config(test_size=0.5)
    )
    assert prepared.train_row_count == prepared.test_row_count == 60


def test_missing_target_column_is_rejected() -> None:
    """A target that is not in the dataset fails before any work is done."""
    with pytest.raises(MissingTargetError):
        prepare_dataset(
            classification_frame(), _classification_config(target_column="revenue")
        )


def test_empty_feature_configuration_is_rejected() -> None:
    """A configuration selecting nothing cannot be prepared."""
    config = PreprocessingConfig(target_column="churn")
    with pytest.raises(EmptyFeatureSetError):
        prepare_dataset(classification_frame(), config)


def test_too_few_labelled_rows_is_rejected() -> None:
    """A dataset that cannot be split is refused with a clear error."""
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "y": ["x", "y", "x"]})
    config = PreprocessingConfig(target_column="y", numeric_columns=("a",))

    with pytest.raises(InsufficientDataError) as exc_info:
        prepare_dataset(frame, config)
    assert exc_info.value.details["labelled_row_count"] == 3


def test_non_dataframe_input_is_rejected() -> None:
    """The ML layer accepts a standardised DataFrame and nothing else."""
    with pytest.raises(MLError) as exc_info:
        prepare_dataset({"a": [1, 2, 3]}, _classification_config())
    assert exc_info.value.details["received_type"] == "dict"


def test_summary_is_serialisable_and_hides_internals() -> None:
    """The summary is the boundary between ML objects and API responses."""
    prepared = prepare_dataset(classification_frame(), _classification_config())
    summary = prepared.summary()

    json.dumps(summary)
    assert "preprocessor" not in summary
    assert "X_train" not in summary
    assert summary["target_column"] == "churn"
    assert summary["task_type"] == "classification"
    assert summary["feature_count"] == len(prepared.feature_names)
    assert summary["split"]["stratified"] is True
    assert summary["column_decisions"][0]["reason"]
