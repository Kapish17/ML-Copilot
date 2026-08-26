"""Tests for deriving a configuration from a dataset profile."""

from __future__ import annotations

from enum import Enum

import pytest

from ml.errors import MissingTargetError
from ml.features.inference import infer_configuration
from ml.features.types import ColumnRole, ExclusionReason, FeatureType, TaskType
from ml.tests.factories import (
    FakeProfile,
    FakeProfiledColumn,
    FakeQualityIssue,
    FakeQualityReport,
    FakeTargetProfile,
    churn_profile,
)


def _decision(inferred, column: str):
    """Return the decision recorded for one column."""
    return next(item for item in inferred.decisions if item.column == column)


def test_feature_groups_follow_the_profiled_types() -> None:
    """Each profiled type is routed to the matching pipeline branch."""
    config = infer_configuration(churn_profile(), target_column="churn").config

    assert config.numeric_columns == ("age", "monthly_charges")
    assert config.categorical_columns == ("contract",)
    assert config.boolean_columns == ("is_active",)
    assert config.datetime_columns == ("signup_date",)


def test_target_is_never_a_feature() -> None:
    """The declared target is recorded as the target and nothing else."""
    inferred = infer_configuration(churn_profile(), target_column="churn")

    assert "churn" not in inferred.config.feature_columns
    assert _decision(inferred, "churn").role is ColumnRole.TARGET


def test_possible_id_column_is_excluded_with_a_reason() -> None:
    """A column profiling flagged as a possible id is kept out of the features."""
    inferred = infer_configuration(churn_profile(), target_column="churn")
    decision = _decision(inferred, "customer_id")

    assert decision.role is ColumnRole.IDENTIFIER
    assert decision.reason_code is ExclusionReason.PROFILE_POSSIBLE_ID
    assert "customer_id" not in inferred.config.feature_columns
    assert inferred.config.identifier_columns == ("customer_id",)
    assert "identifier" in decision.reason.lower()


def test_suspicious_column_is_excluded_with_a_reason() -> None:
    """A column flagged as potentially suspicious is not used by default."""
    profile = churn_profile()
    profile.quality.issues.append(
        FakeQualityIssue("potentially_suspicious_column", ["monthly_charges"])
    )
    inferred = infer_configuration(profile, target_column="churn")

    assert _decision(inferred, "monthly_charges").reason_code is (
        ExclusionReason.PROFILE_SUSPICIOUS
    )
    assert "monthly_charges" not in inferred.config.feature_columns


def test_text_constant_and_empty_columns_are_excluded() -> None:
    """Columns that cannot carry signal are excluded, each with its reason."""
    inferred = infer_configuration(churn_profile(), target_column="churn")

    assert _decision(inferred, "notes").reason_code is ExclusionReason.FREE_TEXT
    assert _decision(inferred, "plan").reason_code is ExclusionReason.CONSTANT_COLUMN
    assert _decision(inferred, "blank").reason_code is ExclusionReason.NO_VALUES


def test_high_cardinality_categorical_is_excluded() -> None:
    """A categorical column with too many values is not one-hot encoded."""
    inferred = infer_configuration(churn_profile(), target_column="churn")
    decision = _decision(inferred, "region_code")

    assert decision.reason_code is ExclusionReason.HIGH_CARDINALITY
    assert "region_code" not in inferred.config.categorical_columns


def test_cardinality_limit_is_configurable() -> None:
    """Raising the limit lets a wide categorical column through."""
    inferred = infer_configuration(
        churn_profile(), target_column="churn", max_categorical_cardinality=1000
    )
    assert "region_code" in inferred.config.categorical_columns


def test_caller_exclusions_win_over_inference() -> None:
    """An explicitly excluded column is dropped even though it profiles well."""
    inferred = infer_configuration(
        churn_profile(), target_column="churn", excluded_columns=["age"]
    )
    decision = _decision(inferred, "age")

    assert decision.reason_code is ExclusionReason.EXCLUDED_BY_CALLER
    assert "age" not in inferred.config.feature_columns


def test_caller_identifiers_are_recorded_as_identifiers() -> None:
    """A column the caller knows is an identifier is treated as one."""
    inferred = infer_configuration(
        churn_profile(), target_column="churn", identifier_columns=["monthly_charges"]
    )
    decision = _decision(inferred, "monthly_charges")

    assert decision.role is ColumnRole.IDENTIFIER
    assert decision.reason_code is ExclusionReason.IDENTIFIER_BY_CALLER


def test_flagged_column_can_be_reinstated_by_override() -> None:
    """Nothing is deleted: an excluded column can be put back explicitly."""
    inferred = infer_configuration(churn_profile(), target_column="churn")
    config = inferred.config.with_overrides(
        numeric_columns=(*inferred.config.numeric_columns, "customer_id"),
        identifier_columns=(),
    )
    assert "customer_id" in config.feature_columns


def test_task_type_comes_from_the_profile() -> None:
    """The suggested task is reused rather than recomputed."""
    config = infer_configuration(churn_profile(), target_column="churn").config
    assert config.task_type is TaskType.CLASSIFICATION


def test_task_type_falls_back_to_auto() -> None:
    """Without a usable suggestion the task is left for the split to decide."""
    profile = churn_profile()
    profile.target = FakeTargetProfile("churn", "undetermined")
    assert (
        infer_configuration(profile, target_column="churn").config.task_type
        is TaskType.AUTO
    )


def test_inferred_type_may_be_an_enum() -> None:
    """The adapter reads either a plain string or an enum's value."""

    class SemanticType(str, Enum):
        INTEGER = "integer"
        CATEGORICAL = "categorical"

    profile = FakeProfile(
        columns=[
            FakeProfiledColumn("a", SemanticType.INTEGER),
            FakeProfiledColumn("y", SemanticType.CATEGORICAL, unique_count=2),
        ],
        quality=FakeQualityReport(),
    )
    config = infer_configuration(profile, target_column="y").config

    assert config.numeric_columns == ("a",)


def test_unknown_target_is_rejected() -> None:
    """A target missing from the profile is refused up front."""
    with pytest.raises(MissingTargetError) as exc_info:
        infer_configuration(churn_profile(), target_column="revenue")
    assert "churn" in exc_info.value.details["available_columns"]


def test_every_column_gets_exactly_one_decision() -> None:
    """No column is dropped without a recorded explanation."""
    profile = churn_profile()
    inferred = infer_configuration(profile, target_column="churn")

    assert len(inferred.decisions) == len(profile.columns)
    assert {item.column for item in inferred.decisions} == {
        column.name for column in profile.columns
    }
    assert all(item.reason for item in inferred.decisions)


def test_decisions_can_be_filtered_by_role() -> None:
    """Callers can list exactly what became a feature."""
    inferred = infer_configuration(churn_profile(), target_column="churn")
    features = inferred.decisions_for(ColumnRole.FEATURE)

    assert {item.column for item in features} == set(inferred.config.feature_columns)
    assert all(item.feature_type in set(FeatureType) for item in features)
