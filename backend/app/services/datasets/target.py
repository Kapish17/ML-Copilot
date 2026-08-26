"""Analysis of an optional target column.

No model is trained here. The task suggestion is a documented heuristic based
on the column's inferred type and how many distinct values it holds, and the
reason behind it is always returned alongside the suggestion.
"""

from __future__ import annotations

import pandas as pd

from app.core.config import Settings
from app.core.errors import TargetColumnNotFoundError
from app.schemas.dataset import (
    ClassBalance,
    ColumnProfile,
    InferredType,
    TargetProfile,
    TaskSuggestion,
)
from app.services.datasets.conversions import percentage, stringify
from app.services.datasets.profiler import numeric_stats, value_distribution


def resolve_target_column(frame: pd.DataFrame, target_column: str) -> str:
    """Validate that the requested target exists in the dataset.

    Args:
        frame: The parsed dataset.
        target_column: Column name requested by the caller.

    Returns:
        str: The validated column name.

    Raises:
        TargetColumnNotFoundError: If the dataset has no such column.
    """
    names = [str(name) for name in frame.columns]
    if target_column not in names:
        raise TargetColumnNotFoundError(
            f"Target column '{target_column}' is not in the dataset.",
            details={"target_column": target_column, "available_columns": names},
        )
    return target_column


def suggest_task(
    profile: ColumnProfile, settings: Settings
) -> tuple[TaskSuggestion, str]:
    """Suggest a modelling task for the target column.

    Args:
        profile: The target column's profile.
        settings: Active application settings.

    Returns:
        tuple[TaskSuggestion, str]: The suggestion and the reason for it.
    """
    limit = settings.max_classification_classes

    if profile.non_null_count == 0:
        return TaskSuggestion.UNDETERMINED, "The target column has no values."
    if profile.is_constant:
        return (
            TaskSuggestion.UNDETERMINED,
            "The target column holds a single value, so there is nothing to predict.",
        )

    if profile.inferred_type is InferredType.BOOLEAN:
        return TaskSuggestion.CLASSIFICATION, "The target is boolean."
    if profile.inferred_type is InferredType.CATEGORICAL:
        return (
            TaskSuggestion.CLASSIFICATION,
            f"The target is categorical with {profile.unique_count} distinct values.",
        )
    if profile.inferred_type is InferredType.TEXT:
        if profile.unique_count > limit:
            return (
                TaskSuggestion.UNDETERMINED,
                f"The target is free text with {profile.unique_count} distinct values, "
                f"more than the {limit} treated as classes.",
            )
        return (
            TaskSuggestion.CLASSIFICATION,
            f"The target is text with {profile.unique_count} distinct values.",
        )
    if profile.inferred_type is InferredType.INTEGER:
        if profile.unique_count <= limit:
            return (
                TaskSuggestion.CLASSIFICATION,
                f"The target is integer with only {profile.unique_count} distinct "
                f"values, at or below the {limit} treated as classes.",
            )
        return (
            TaskSuggestion.REGRESSION,
            f"The target is integer with {profile.unique_count} distinct values, "
            f"more than the {limit} treated as classes.",
        )
    if profile.inferred_type is InferredType.FLOAT:
        return TaskSuggestion.REGRESSION, "The target is continuous and numeric."
    if profile.inferred_type is InferredType.DATETIME:
        return (
            TaskSuggestion.UNDETERMINED,
            "The target looks like a date, which is not a supported task type.",
        )
    return TaskSuggestion.UNDETERMINED, "The target column type could not be determined."


def class_balance(series: pd.Series, settings: Settings) -> ClassBalance | None:
    """Summarise how evenly the target's classes are represented.

    Args:
        series: The target column; missing values are ignored.
        settings: Active application settings.

    Returns:
        ClassBalance | None: Majority and minority shares, or ``None`` when the
        column has no values.
    """
    counts = series.value_counts(dropna=True)
    if counts.empty:
        return None

    total = int(counts.sum())
    majority_share = percentage(int(counts.iloc[0]), total)
    minority_share = percentage(int(counts.iloc[-1]), total)

    return ClassBalance(
        class_count=int(counts.size),
        majority_class=stringify(counts.index[0]),
        majority_percentage=majority_share,
        minority_class=stringify(counts.index[-1]),
        minority_percentage=minority_share,
        is_imbalanced=majority_share > settings.imbalance_ratio * 100,
    )


def analyse_target(
    frame: pd.DataFrame,
    columns: list[ColumnProfile],
    target_column: str,
    settings: Settings,
) -> TargetProfile:
    """Build the target section of a dataset profile.

    Args:
        frame: The parsed dataset.
        columns: Column profiles produced by the profiler.
        target_column: Column name requested by the caller.
        settings: Active application settings.

    Returns:
        TargetProfile: Type, missingness, suggested task and distribution.

    Raises:
        TargetColumnNotFoundError: If the dataset has no such column.
    """
    name = resolve_target_column(frame, target_column)
    profile = next(column for column in columns if column.name == name)
    series = frame[name]

    suggestion, reason = suggest_task(profile, settings)

    target = TargetProfile(
        name=name,
        dtype=profile.dtype,
        inferred_type=profile.inferred_type,
        missing_count=profile.missing_count,
        missing_percentage=profile.missing_percentage,
        task_suggestion=suggestion,
        task_reason=reason,
    )

    if suggestion is TaskSuggestion.CLASSIFICATION:
        target.distribution = value_distribution(
            series, settings.max_classification_classes
        )
        target.class_balance = class_balance(series, settings)
    elif suggestion is TaskSuggestion.REGRESSION:
        target.numeric_stats = profile.numeric_stats or numeric_stats(series)

    return target
