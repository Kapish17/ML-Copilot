"""Tests for optional target-column analysis."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import TargetColumnNotFoundError
from app.schemas.dataset import InferredType, TargetProfile, TaskSuggestion
from app.services.datasets.loader import load_csv
from app.services.datasets.profiler import profile_columns
from app.services.datasets.target import analyse_target
from tests.factories import build_csv


def _target(content: bytes, settings: Settings, name: str) -> TargetProfile:
    """Analyse a named target column of CSV bytes."""
    frame = load_csv(content, settings)
    columns = profile_columns(frame, settings)
    return analyse_target(frame, columns, name, settings)


def test_missing_target_column_is_rejected(settings: Settings) -> None:
    """An unknown target name is a typed error listing the real columns."""
    content = build_csv(["a", "b"], [[1, 2]])
    with pytest.raises(TargetColumnNotFoundError) as exc_info:
        _target(content, settings, "nope")

    assert exc_info.value.status_code == 422
    assert exc_info.value.details["available_columns"] == ["a", "b"]


def test_categorical_target_suggests_classification(settings: Settings) -> None:
    """A categorical target yields a classification suggestion and a reason."""
    content = build_csv(
        ["feature", "churn"], [[1, "yes"], [2, "no"], [3, "yes"], [4, "no"]]
    )
    target = _target(content, settings, "churn")

    assert target.task_suggestion is TaskSuggestion.CLASSIFICATION
    assert target.inferred_type is InferredType.CATEGORICAL
    assert target.task_reason
    assert target.numeric_stats is None


def test_classification_target_reports_distribution(settings: Settings) -> None:
    """Every class and its share is reported for a classification target."""
    rows = [[index, "a" if index < 3 else "b"] for index in range(4)]
    target = _target(build_csv(["x", "y"], rows), settings, "y")

    assert target.distribution is not None
    assert {entry.value for entry in target.distribution} == {"a", "b"}
    assert sum(entry.count for entry in target.distribution) == 4


def test_class_balance_flags_imbalance(settings: Settings) -> None:
    """A dominant majority class is reported as imbalanced."""
    rows = [[index, "a" if index < 9 else "b"] for index in range(10)]
    balance = _target(build_csv(["x", "y"], rows), settings, "y").class_balance

    assert balance is not None
    assert balance.class_count == 2
    assert balance.majority_class == "a"
    assert balance.majority_percentage == pytest.approx(90.0)
    assert balance.minority_class == "b"
    assert balance.is_imbalanced is True


def test_balanced_target_is_not_flagged(settings: Settings) -> None:
    """An even split is not reported as imbalanced."""
    rows = [[index, "a" if index % 2 else "b"] for index in range(10)]
    balance = _target(build_csv(["x", "y"], rows), settings, "y").class_balance

    assert balance is not None
    assert balance.is_imbalanced is False


def test_continuous_target_suggests_regression(settings: Settings) -> None:
    """A float target yields a regression suggestion with summary statistics."""
    content = build_csv(["x", "price"], [[1, 10.5], [2, 20.25], [3, 30.75]])
    target = _target(content, settings, "price")

    assert target.task_suggestion is TaskSuggestion.REGRESSION
    assert target.distribution is None
    assert target.numeric_stats is not None
    assert target.numeric_stats.minimum == pytest.approx(10.5)
    assert target.numeric_stats.maximum == pytest.approx(30.75)


def test_low_cardinality_integer_target_is_classification(settings: Settings) -> None:
    """Integers with few distinct values are read as class labels."""
    rows = [[index, index % 3] for index in range(12)]
    target = _target(build_csv(["x", "y"], rows), settings, "y")

    assert target.task_suggestion is TaskSuggestion.CLASSIFICATION


def test_high_cardinality_integer_target_is_regression(settings: Settings) -> None:
    """Integers with many distinct values are read as a continuous target."""
    rows = [[index, index * 7] for index in range(40)]
    target = _target(build_csv(["x", "y"], rows), settings, "y")

    assert target.task_suggestion is TaskSuggestion.REGRESSION
    assert target.numeric_stats is not None


def test_constant_target_is_undetermined(settings: Settings) -> None:
    """A target with a single value gives no usable task."""
    content = build_csv(["x", "y"], [[1, "a"], [2, "a"], [3, "a"]])
    target = _target(content, settings, "y")

    assert target.task_suggestion is TaskSuggestion.UNDETERMINED
    assert "single value" in target.task_reason


def test_empty_target_is_undetermined(settings: Settings) -> None:
    """A target with no values at all gives no usable task."""
    content = build_csv(["x", "y"], [[1, None], [2, None]])
    target = _target(content, settings, "y")

    assert target.task_suggestion is TaskSuggestion.UNDETERMINED
    assert target.missing_percentage == pytest.approx(100.0)


def test_target_missing_values_are_reported(settings: Settings) -> None:
    """Missing labels are counted so they can be dealt with before training."""
    content = build_csv(["x", "y"], [[1, "a"], [2, None], [3, "b"], [4, "a"]])
    target = _target(content, settings, "y")

    assert target.missing_count == 1
    assert target.missing_percentage == pytest.approx(25.0)


def test_free_text_target_is_undetermined() -> None:
    """A target with more distinct values than allowed classes is undetermined."""
    settings = Settings(max_classification_classes=3)
    rows = [[index, f"note-{index}"] for index in range(60)]
    target = _target(build_csv(["x", "y"], rows), settings, "y")

    assert target.task_suggestion is TaskSuggestion.UNDETERMINED
    assert "distinct values" in target.task_reason
