"""Deriving a preprocessing configuration from a dataset profile.

This module is the single point where the ML layer consumes profiling output.
It depends on the *shape* of a profile rather than on the class that produces
one: the protocols below describe the handful of attributes that are read, so
the profiling implementation stays free to live in another package without the
ML layer importing it. Profiling logic is never repeated here — the inferred
types and quality findings are taken as given and only mapped onto feature
groups.

Inference produces a starting point. Anything a caller states explicitly wins,
either through the arguments here or through
:meth:`~ml.features.config.PreprocessingConfig.with_overrides`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ml.errors import MissingTargetError
from ml.features.config import (
    DEFAULT_MAX_CATEGORICAL_CARDINALITY,
    PreprocessingConfig,
)
from ml.features.decisions import ColumnDecision
from ml.features.types import (
    ColumnRole,
    ExclusionReason,
    FeatureType,
    TaskType,
)

#: Profile column types that map directly onto a pipeline branch.
_TYPE_TO_FEATURE_GROUP = {
    "integer": FeatureType.NUMERIC,
    "float": FeatureType.NUMERIC,
    "boolean": FeatureType.BOOLEAN,
    "datetime": FeatureType.DATETIME,
    "categorical": FeatureType.CATEGORICAL,
}

#: Profile column types that are never used as features, with the reason.
_TYPE_TO_EXCLUSION = {
    "text": ExclusionReason.FREE_TEXT,
    "empty": ExclusionReason.NO_VALUES,
    "unknown": ExclusionReason.UNSUPPORTED_TYPE,
}

#: Quality findings that keep a column out of the feature set by default.
_QUALITY_EXCLUSIONS = {
    "possible_id_column": ExclusionReason.PROFILE_POSSIBLE_ID,
    "potentially_suspicious_column": ExclusionReason.PROFILE_SUSPICIOUS,
}

_EXCLUSION_MESSAGES = {
    ExclusionReason.DECLARED_TARGET: "Declared as the target column.",
    ExclusionReason.EXCLUDED_BY_CALLER: "Excluded explicitly by the caller.",
    ExclusionReason.IDENTIFIER_BY_CALLER: "Marked as an identifier by the caller.",
    ExclusionReason.PROFILE_POSSIBLE_ID: (
        "Profiling flagged this as a possible identifier; identifiers do not "
        "generalise, so it is excluded unless you include it explicitly."
    ),
    ExclusionReason.PROFILE_SUSPICIOUS: (
        "Profiling flagged this column as potentially suspicious, so it is "
        "excluded unless you include it explicitly."
    ),
    ExclusionReason.CONSTANT_COLUMN: "Holds a single value, so it carries no signal.",
    ExclusionReason.NO_VALUES: "Has no values at all.",
    ExclusionReason.FREE_TEXT: (
        "Looks like free text. Text features are not supported yet; a text "
        "encoder arrives in a later commit."
    ),
    ExclusionReason.HIGH_CARDINALITY: (
        "Has more distinct values than the configured cardinality limit, so "
        "one-hot encoding it would create an unmanageable number of features."
    ),
    ExclusionReason.UNSUPPORTED_TYPE: "Type is not supported by the pipeline.",
}


@runtime_checkable
class ProfiledColumnLike(Protocol):
    """The column attributes the ML layer reads from a dataset profile."""

    name: str
    inferred_type: Any
    is_constant: bool
    unique_count: int
    non_null_count: int


@runtime_checkable
class QualityIssueLike(Protocol):
    """The quality-finding attributes the ML layer reads."""

    code: str
    columns: Sequence[str]


class QualityReportLike(Protocol):
    """A collection of quality findings."""

    issues: Sequence[QualityIssueLike]


class DatasetProfileLike(Protocol):
    """The subset of a dataset profile the ML layer depends on."""

    columns: Sequence[ProfiledColumnLike]
    quality: QualityReportLike


@dataclass(frozen=True)
class InferredConfiguration:
    """A configuration together with the reasoning that produced it."""

    config: PreprocessingConfig
    decisions: tuple[ColumnDecision, ...]

    def decisions_for(self, role: ColumnRole) -> tuple[ColumnDecision, ...]:
        """Return every decision with the given role."""
        return tuple(item for item in self.decisions if item.role is role)


def _semantic_type(column: ProfiledColumnLike) -> str:
    """Read a column's inferred type as a plain string.

    Accepts either a string or an enum whose ``value`` is the string, which
    keeps the ML layer independent of the profiling package's enum class.
    """
    return str(getattr(column.inferred_type, "value", column.inferred_type))


def _columns_flagged_by_quality(profile: DatasetProfileLike) -> dict[str, ExclusionReason]:
    """Map column names to the first quality finding that excludes them."""
    flagged: dict[str, ExclusionReason] = {}
    for issue in getattr(profile.quality, "issues", ()):
        reason = _QUALITY_EXCLUSIONS.get(str(getattr(issue, "code", "")))
        if reason is None:
            continue
        for column in getattr(issue, "columns", ()):
            flagged.setdefault(str(column), reason)
    return flagged


def _excluded_decision(column: str, reason: ExclusionReason) -> ColumnDecision:
    """Build an exclusion decision with its standard explanation."""
    role = (
        ColumnRole.IDENTIFIER
        if reason
        in (ExclusionReason.IDENTIFIER_BY_CALLER, ExclusionReason.PROFILE_POSSIBLE_ID)
        else ColumnRole.EXCLUDED
    )
    return ColumnDecision(
        column=column,
        role=role,
        reason=_EXCLUSION_MESSAGES[reason],
        reason_code=reason,
    )


def _decide_column(
    column: ProfiledColumnLike,
    *,
    target_column: str,
    excluded: set[str],
    identifiers: set[str],
    quality_flags: dict[str, ExclusionReason],
    max_categorical_cardinality: int,
) -> ColumnDecision:
    """Decide the role of a single profiled column.

    Caller intent is honoured first, then profiling findings, then the column's
    own statistics, then its inferred type.
    """
    name = column.name

    if name == target_column:
        return ColumnDecision(
            column=name,
            role=ColumnRole.TARGET,
            reason=_EXCLUSION_MESSAGES[ExclusionReason.DECLARED_TARGET],
            reason_code=ExclusionReason.DECLARED_TARGET,
        )
    if name in excluded:
        return _excluded_decision(name, ExclusionReason.EXCLUDED_BY_CALLER)
    if name in identifiers:
        return _excluded_decision(name, ExclusionReason.IDENTIFIER_BY_CALLER)
    if name in quality_flags:
        return _excluded_decision(name, quality_flags[name])

    if column.non_null_count == 0:
        return _excluded_decision(name, ExclusionReason.NO_VALUES)
    if column.is_constant:
        return _excluded_decision(name, ExclusionReason.CONSTANT_COLUMN)

    semantic = _semantic_type(column)
    if semantic in _TYPE_TO_EXCLUSION:
        return _excluded_decision(name, _TYPE_TO_EXCLUSION[semantic])

    feature_type = _TYPE_TO_FEATURE_GROUP.get(semantic)
    if feature_type is None:
        return _excluded_decision(name, ExclusionReason.UNSUPPORTED_TYPE)

    if (
        feature_type is FeatureType.CATEGORICAL
        and column.unique_count > max_categorical_cardinality
    ):
        return _excluded_decision(name, ExclusionReason.HIGH_CARDINALITY)

    return ColumnDecision(
        column=name,
        role=ColumnRole.FEATURE,
        reason=f"Profiled as {semantic}; handled by the {feature_type.value} branch.",
        feature_type=feature_type,
    )


def _task_type_from_profile(profile: DatasetProfileLike, target_column: str) -> TaskType:
    """Read the suggested task for the target from the profile, if present.

    Profiling already decides whether a target looks like a classification or a
    regression problem; that answer is reused rather than recomputed. When the
    profile carries no usable suggestion the task stays ``AUTO`` and the split
    decides on its own.
    """
    target = getattr(profile, "target", None)
    if target is None or str(getattr(target, "name", "")) != target_column:
        return TaskType.AUTO
    suggestion = str(
        getattr(
            getattr(target, "task_suggestion", ""),
            "value",
            getattr(target, "task_suggestion", ""),
        )
    )
    if suggestion in (TaskType.CLASSIFICATION.value, TaskType.REGRESSION.value):
        return TaskType(suggestion)
    return TaskType.AUTO


def infer_configuration(
    profile: DatasetProfileLike,
    *,
    target_column: str,
    excluded_columns: Sequence[str] = (),
    identifier_columns: Sequence[str] = (),
    max_categorical_cardinality: int = DEFAULT_MAX_CATEGORICAL_CARDINALITY,
) -> InferredConfiguration:
    """Derive a preprocessing configuration from a dataset profile.

    Args:
        profile: A dataset profile, as produced by the profiling layer.
        target_column: Name of the column to predict.
        excluded_columns: Columns the caller wants kept out of the features.
        identifier_columns: Columns the caller knows to be identifiers.
        max_categorical_cardinality: Above this many distinct values, a
            categorical column is excluded instead of one-hot encoded.

    Returns:
        InferredConfiguration: The configuration plus one decision per column.

    Raises:
        MissingTargetError: If the profile has no such column.
    """
    names = [column.name for column in profile.columns]
    if target_column not in names:
        raise MissingTargetError(
            f"Target column '{target_column}' is not in the dataset profile.",
            details={"target_column": target_column, "available_columns": names},
        )

    quality_flags = _columns_flagged_by_quality(profile)
    excluded = set(excluded_columns)
    identifiers = set(identifier_columns)

    decisions = tuple(
        _decide_column(
            column,
            target_column=target_column,
            excluded=excluded,
            identifiers=identifiers,
            quality_flags=quality_flags,
            max_categorical_cardinality=max_categorical_cardinality,
        )
        for column in profile.columns
    )

    grouped: dict[FeatureType, list[str]] = {
        feature_type: [] for feature_type in FeatureType
    }
    for decision in decisions:
        if decision.is_feature and decision.feature_type is not None:
            grouped[decision.feature_type].append(decision.column)

    config = PreprocessingConfig(
        target_column=target_column,
        numeric_columns=tuple(grouped[FeatureType.NUMERIC]),
        categorical_columns=tuple(grouped[FeatureType.CATEGORICAL]),
        boolean_columns=tuple(grouped[FeatureType.BOOLEAN]),
        datetime_columns=tuple(grouped[FeatureType.DATETIME]),
        identifier_columns=tuple(
            decision.column
            for decision in decisions
            if decision.role is ColumnRole.IDENTIFIER
        ),
        excluded_columns=tuple(
            decision.column
            for decision in decisions
            if decision.role is ColumnRole.EXCLUDED
        ),
        task_type=_task_type_from_profile(profile, target_column),
        max_categorical_cardinality=max_categorical_cardinality,
    )
    return InferredConfiguration(config=config, decisions=decisions)
