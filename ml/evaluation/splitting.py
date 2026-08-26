"""Train/test splitting and target-distribution reporting.

Splitting lives next to evaluation because it defines the protocol a model is
judged under; cross-validation will join it here in a later commit. The
functions are deliberately small and pure so they can be reused by a future
cross-validation strategy without change.

Class imbalance is measured and reported, never corrected: no resampling of any
kind happens here, so the class distribution a caller sees is the one the data
actually has.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pandas.api import types as pdt
from sklearn.model_selection import train_test_split

from ml.features.types import TaskType

#: A class needs at least this many rows to appear on both sides of a split.
MIN_CLASS_MEMBERS_FOR_STRATIFY = 2

PERCENTAGE_PRECISION = 4


def _percentage(part: float, whole: float) -> float:
    """Return ``part`` as a percentage of ``whole``, rounded and zero-safe."""
    if not whole:
        return 0.0
    return round((part / whole) * 100, PERCENTAGE_PRECISION)


def _finite(value: Any) -> float | None:
    """Return a finite float, or ``None`` for NaN, infinity and non-numbers."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def looks_discrete(target: pd.Series, max_classes: int) -> bool:
    """Return True when a target has few enough distinct values to be classes.

    This is a splitting decision, not a re-run of profiling: it answers only
    "can this column be stratified?", and it is used solely when the caller
    left the task type unset.

    Args:
        target: The target column.
        max_classes: Most distinct values still considered class labels.

    Returns:
        bool: True when the target should be treated as discrete.
    """
    if pdt.is_bool_dtype(target):
        return True
    if not pdt.is_numeric_dtype(target):
        return True
    if pdt.is_integer_dtype(target):
        return int(target.nunique(dropna=True)) <= max_classes
    return False


def resolve_task_type(
    target: pd.Series, declared: TaskType, max_classes: int
) -> TaskType:
    """Settle on a concrete task type for a target column.

    An explicitly declared task always wins; profiling normally supplies it.
    ``AUTO`` falls back to :func:`looks_discrete`.

    Args:
        target: The target column.
        declared: The task type from the configuration.
        max_classes: Most distinct values still considered class labels.

    Returns:
        TaskType: Either ``CLASSIFICATION`` or ``REGRESSION``.
    """
    if declared is not TaskType.AUTO:
        return declared
    return (
        TaskType.CLASSIFICATION
        if looks_discrete(target, max_classes)
        else TaskType.REGRESSION
    )


@dataclass(frozen=True)
class TargetDistribution:
    """How the target is distributed over a set of rows.

    Provided so a later commit can decide whether to weight classes or
    resample. This commit only measures.
    """

    task_type: TaskType
    row_count: int
    class_counts: dict[str, int] | None = None
    class_percentages: dict[str, float] | None = None
    majority_class: str | None = None
    majority_percentage: float | None = None
    minority_class: str | None = None
    minority_percentage: float | None = None
    imbalance_ratio: float | None = None
    numeric_summary: dict[str, float | None] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Render the distribution as plain, JSON-friendly values."""
        return {
            "task_type": self.task_type.value,
            "row_count": self.row_count,
            "class_counts": self.class_counts,
            "class_percentages": self.class_percentages,
            "majority_class": self.majority_class,
            "majority_percentage": self.majority_percentage,
            "minority_class": self.minority_class,
            "minority_percentage": self.minority_percentage,
            "imbalance_ratio": self.imbalance_ratio,
            "numeric_summary": self.numeric_summary,
        }


def describe_target(target: pd.Series, task_type: TaskType) -> TargetDistribution:
    """Summarise a target column for the given task type.

    Args:
        target: The target column; missing values are ignored.
        task_type: A resolved task type, never ``AUTO``.

    Returns:
        TargetDistribution: Class counts for classification, descriptive
        statistics for regression.
    """
    row_count = int(target.shape[0])

    if task_type is TaskType.REGRESSION:
        values = pd.to_numeric(target, errors="coerce").dropna()
        return TargetDistribution(
            task_type=task_type,
            row_count=row_count,
            numeric_summary={
                "mean": _finite(values.mean()) if not values.empty else None,
                "median": _finite(values.median()) if not values.empty else None,
                "std": _finite(values.std()) if not values.empty else None,
                "minimum": _finite(values.min()) if not values.empty else None,
                "maximum": _finite(values.max()) if not values.empty else None,
            },
        )

    counts = target.value_counts(dropna=True)
    if counts.empty:
        return TargetDistribution(task_type=task_type, row_count=row_count)

    total = int(counts.sum())
    class_counts = {str(label): int(count) for label, count in counts.items()}
    majority_count = int(counts.iloc[0])
    minority_count = int(counts.iloc[-1])

    return TargetDistribution(
        task_type=task_type,
        row_count=row_count,
        class_counts=class_counts,
        class_percentages={
            label: _percentage(count, total) for label, count in class_counts.items()
        },
        majority_class=str(counts.index[0]),
        majority_percentage=_percentage(majority_count, total),
        minority_class=str(counts.index[-1]),
        minority_percentage=_percentage(minority_count, total),
        imbalance_ratio=round(majority_count / minority_count, PERCENTAGE_PRECISION)
        if minority_count
        else None,
    )


@dataclass(frozen=True)
class DatasetSplit:
    """A train/test division of features and target, with how it was made."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    stratified: bool
    stratification_note: str | None = None


def resolve_stratification(
    target: pd.Series,
    *,
    task_type: TaskType,
    test_size: float,
) -> tuple[bool, str | None]:
    """Decide whether a split can be stratified, and explain the answer.

    Stratification keeps class proportions equal in both halves, which matters
    for imbalanced classification. It is skipped — with a reason — for
    regression targets, for single-class targets, and whenever a class is too
    small to appear on both sides of the split.

    Args:
        target: The resolved target column.
        task_type: A resolved task type, never ``AUTO``.
        test_size: Fraction of rows held out for testing.

    Returns:
        tuple[bool, str | None]: Whether to stratify, and why not when False.
    """
    if task_type is TaskType.REGRESSION:
        return False, "Regression target: class stratification does not apply."

    counts = target.value_counts(dropna=True)
    if counts.size < 2:
        return False, "The target has fewer than two classes, so there is nothing to stratify."

    smallest = int(counts.min())
    if smallest < MIN_CLASS_MEMBERS_FOR_STRATIFY:
        return False, (
            f"Class '{counts.idxmin()}' has only {smallest} row(s), too few to "
            "appear in both the training and the test set."
        )

    row_count = int(target.shape[0])
    test_rows = math.ceil(test_size * row_count)
    train_rows = row_count - test_rows
    if test_rows < counts.size or train_rows < counts.size:
        return False, (
            f"A {test_size:.0%} test split of {row_count} rows cannot hold all "
            f"{counts.size} classes on both sides."
        )

    return True, None


def split_dataset(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    test_size: float,
    random_state: int,
    task_type: TaskType,
) -> DatasetSplit:
    """Divide features and target into a training and a test set.

    The split happens before any transformer is fitted, which is what makes
    leakage prevention structural rather than a matter of discipline.

    Args:
        features: Feature columns only; the target must not be present.
        target: The target column, aligned with ``features``.
        test_size: Fraction of rows held out for testing.
        random_state: Seed, so the same inputs always give the same split.
        task_type: A resolved task type, never ``AUTO``.

    Returns:
        DatasetSplit: The four parts plus whether stratification was applied.
    """
    stratify, note = resolve_stratification(
        target, task_type=task_type, test_size=test_size
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=target if stratify else None,
    )
    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        stratified=stratify,
        stratification_note=note,
    )
