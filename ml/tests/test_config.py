"""Tests for the preprocessing configuration and its validation."""

from __future__ import annotations

import pytest

from ml.errors import (
    ConfigurationError,
    DuplicateColumnAssignmentError,
    EmptyFeatureSetError,
    MissingTargetError,
    TargetLeakageError,
    UnknownColumnError,
)
from ml.features.config import PreprocessingConfig, validate_config
from ml.features.types import (
    CategoricalImputation,
    FeatureType,
    NumericImputation,
    ScalingStrategy,
    TaskType,
)


def test_feature_columns_are_ordered_by_group() -> None:
    """Feature order is stable and grouped, which keeps output names stable."""
    config = PreprocessingConfig(
        target_column="y",
        numeric_columns=["a", "b"],
        categorical_columns=["c"],
        boolean_columns=["d"],
        datetime_columns=["e"],
    )
    assert config.feature_columns == ("a", "b", "c", "d", "e")


def test_sequences_are_normalised_to_tuples() -> None:
    """Lists are accepted but stored immutably."""
    config = PreprocessingConfig(target_column="y", numeric_columns=["a"])
    assert isinstance(config.numeric_columns, tuple)


def test_feature_type_lookup() -> None:
    """Each column reports the pipeline branch that will handle it."""
    config = PreprocessingConfig(
        target_column="y", numeric_columns=["a"], categorical_columns=["c"]
    )
    assert config.feature_type_of("a") is FeatureType.NUMERIC
    assert config.feature_type_of("c") is FeatureType.CATEGORICAL
    assert config.feature_type_of("missing") is None


def test_strategies_accept_plain_strings() -> None:
    """Strings are coerced, so a stored configuration round-trips cleanly."""
    config = PreprocessingConfig(
        target_column="y",
        numeric_columns=["a"],
        scaling_strategy="minmax",
        numeric_imputation="mean",
        categorical_imputation="constant",
        task_type="regression",
    )
    assert config.scaling_strategy is ScalingStrategy.MINMAX
    assert config.numeric_imputation is NumericImputation.MEAN
    assert config.categorical_imputation is CategoricalImputation.CONSTANT
    assert config.task_type is TaskType.REGRESSION


def test_invalid_strategy_is_rejected() -> None:
    """An unusable strategy fails immediately, with the allowed values listed."""
    with pytest.raises(ConfigurationError) as exc_info:
        PreprocessingConfig(target_column="y", scaling_strategy="quantum")
    assert "scaling_strategy" in str(exc_info.value)


def test_with_overrides_replaces_inferred_values() -> None:
    """Explicit configuration wins over whatever was inferred."""
    config = PreprocessingConfig(target_column="y", numeric_columns=["a"])
    updated = config.with_overrides(scaling_strategy="none", test_size=0.3)

    assert updated.scaling_strategy is ScalingStrategy.NONE
    assert updated.test_size == 0.3
    assert config.scaling_strategy is ScalingStrategy.STANDARD, "original is unchanged"


def test_with_overrides_rejects_unknown_fields() -> None:
    """A typo in an override is an error rather than a silent no-op."""
    config = PreprocessingConfig(target_column="y", numeric_columns=["a"])
    with pytest.raises(ConfigurationError) as exc_info:
        config.with_overrides(scaling_stratergy="none")
    assert "scaling_stratergy" in str(exc_info.value)


def test_validate_requires_a_target() -> None:
    """A blank target is refused before anything else is checked."""
    with pytest.raises(MissingTargetError):
        validate_config(PreprocessingConfig(target_column="  "), ["a"])


def test_validate_requires_the_target_to_exist() -> None:
    """A target that is not a column of the dataset is refused."""
    config = PreprocessingConfig(target_column="y", numeric_columns=["a"])
    with pytest.raises(MissingTargetError) as exc_info:
        validate_config(config, ["a", "b"])
    assert exc_info.value.details["available_columns"] == ["a", "b"]


def test_validate_rejects_target_used_as_a_feature() -> None:
    """The target must never be handed to a feature transformer."""
    config = PreprocessingConfig(target_column="y", numeric_columns=["a", "y"])
    with pytest.raises(TargetLeakageError):
        validate_config(config, ["a", "y"])


def test_validate_rejects_a_column_in_two_groups() -> None:
    """A column cannot be both numeric and categorical."""
    config = PreprocessingConfig(
        target_column="y", numeric_columns=["a"], categorical_columns=["a"]
    )
    with pytest.raises(DuplicateColumnAssignmentError) as exc_info:
        validate_config(config, ["a", "y"])
    assert exc_info.value.details["columns"] == ["a"]


def test_validate_rejects_unknown_columns() -> None:
    """Configuring a column the dataset does not have is an error."""
    config = PreprocessingConfig(target_column="y", numeric_columns=["ghost"])
    with pytest.raises(UnknownColumnError) as exc_info:
        validate_config(config, ["a", "y"])
    assert exc_info.value.details["columns"] == ["ghost"]


def test_validate_rejects_an_empty_feature_set() -> None:
    """A configuration that selects nothing cannot be preprocessed."""
    with pytest.raises(EmptyFeatureSetError):
        validate_config(PreprocessingConfig(target_column="y"), ["a", "y"])


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.2, 1.5])
def test_validate_rejects_out_of_range_test_size(test_size: float) -> None:
    """A test fraction must leave rows on both sides of the split."""
    config = PreprocessingConfig(
        target_column="y", numeric_columns=["a"], test_size=test_size
    )
    with pytest.raises(ConfigurationError):
        validate_config(config, ["a", "y"])


def test_valid_configuration_passes() -> None:
    """A well-formed configuration validates without raising."""
    config = PreprocessingConfig(
        target_column="y", numeric_columns=["a"], excluded_columns=["b"]
    )
    validate_config(config, ["a", "b", "y"])
