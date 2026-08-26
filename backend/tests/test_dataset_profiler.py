"""Tests for dataset- and column-level profiling."""

from __future__ import annotations

import pandas as pd
import pytest

from app.core.config import Settings
from app.schemas.dataset import ColumnProfile, InferredType
from app.services.datasets.profiler import (
    profile_columns,
    summarise_dataset,
)
from tests.factories import build_csv, high_cardinality_csv
from app.services.datasets.loader import load_csv


def _profiles(content: bytes, settings: Settings) -> dict[str, ColumnProfile]:
    """Profile CSV bytes and index the column profiles by name."""
    frame = load_csv(content, settings)
    return {column.name: column for column in profile_columns(frame, settings)}


def test_numeric_statistics(settings: Settings) -> None:
    """Numeric columns get mean, median, spread, extremes and quartiles."""
    content = build_csv(["value"], [[1], [2], [3], [4], [100]])
    profile = _profiles(content, settings)["value"]

    assert profile.inferred_type is InferredType.INTEGER
    assert profile.categorical_stats is None
    stats = profile.numeric_stats
    assert stats is not None
    assert stats.mean == pytest.approx(22.0)
    assert stats.median == pytest.approx(3.0)
    assert stats.minimum == pytest.approx(1.0)
    assert stats.maximum == pytest.approx(100.0)
    assert stats.q1 == pytest.approx(2.0)
    assert stats.q3 == pytest.approx(4.0)
    assert stats.std is not None and stats.std > 0


def test_numeric_statistics_count_zero_and_negative(settings: Settings) -> None:
    """Zero and negative counts are reported for numeric columns."""
    content = build_csv(["value"], [[-2], [0], [0], [5]])
    stats = _profiles(content, settings)["value"].numeric_stats
    assert stats is not None
    assert stats.zero_count == 2
    assert stats.negative_count == 1


def test_single_row_numeric_std_is_null(settings: Settings) -> None:
    """A standard deviation that is undefined becomes ``None``, never NaN."""
    stats = _profiles(build_csv(["value"], [[7]]), settings)["value"].numeric_stats
    assert stats is not None
    assert stats.std is None
    assert stats.mean == pytest.approx(7.0)


def test_categorical_profiling(settings: Settings) -> None:
    """Categorical columns get value frequencies and no numeric statistics."""
    content = build_csv(
        ["colour"], [["red"], ["red"], ["blue"], ["green"], ["red"], ["blue"]]
    )
    profile = _profiles(content, settings)["colour"]

    assert profile.inferred_type is InferredType.CATEGORICAL
    assert profile.numeric_stats is None
    stats = profile.categorical_stats
    assert stats is not None
    assert stats.top_values[0].value == "red"
    assert stats.top_values[0].count == 3
    assert stats.top_values[0].percentage == pytest.approx(50.0)
    assert stats.truncated is False


def test_top_values_are_truncated_by_configuration() -> None:
    """Only the configured number of top values is returned."""
    settings = Settings(profile_top_values=2)
    content = build_csv(["colour"], [["a"], ["b"], ["c"], ["d"]])
    stats = _profiles(content, settings)["colour"].categorical_stats
    assert stats is not None
    assert len(stats.top_values) == 2
    assert stats.truncated is True


def test_missing_values_are_counted(settings: Settings) -> None:
    """Missing cells are counted per column and as a percentage."""
    content = build_csv(
        ["value", "label"], [[1, "a"], [None, "b"], [None, "c"], [4, "d"]]
    )
    profile = _profiles(content, settings)["value"]

    assert profile.missing_count == 2
    assert profile.missing_percentage == pytest.approx(50.0)
    assert profile.non_null_count == 2


def test_fully_empty_column_is_typed_empty(settings: Settings) -> None:
    """A column with no values at all is typed ``empty`` and marked constant."""
    content = build_csv(["a", "blank"], [[1, None], [2, None]])
    profile = _profiles(content, settings)["blank"]

    assert profile.inferred_type is InferredType.EMPTY
    assert profile.missing_percentage == pytest.approx(100.0)
    assert profile.is_constant is True


def test_constant_column_is_flagged(settings: Settings) -> None:
    """A single-valued column is reported as constant."""
    content = build_csv(["fixed"], [["x"], ["x"], ["x"]])
    assert _profiles(content, settings)["fixed"].is_constant is True


def test_boolean_column_gets_frequencies(settings: Settings) -> None:
    """Boolean columns are summarised as frequencies, not as numbers."""
    content = build_csv(["flag"], [[True], [False], [True]])
    profile = _profiles(content, settings)["flag"]

    assert profile.inferred_type is InferredType.BOOLEAN
    assert profile.numeric_stats is None
    assert profile.categorical_stats is not None


def test_date_like_column_is_detected(settings: Settings) -> None:
    """Text that parses as dates is typed as datetime with a range."""
    content = build_csv(
        ["day"], [["2023-01-05"], ["2023-02-11"], ["2023-03-02"], ["2023-04-19"]]
    )
    profile = _profiles(content, settings)["day"]

    assert profile.inferred_type is InferredType.DATETIME
    assert profile.datetime_stats is not None
    assert profile.datetime_stats.minimum is not None
    assert profile.datetime_stats.minimum.startswith("2023-01-05")
    assert profile.datetime_stats.maximum is not None
    assert profile.datetime_stats.maximum.startswith("2023-04-19")


def test_numeric_codes_are_not_treated_as_dates(settings: Settings) -> None:
    """Numeric-looking text is never reinterpreted as a timestamp."""
    content = build_csv(["code"], [[20230101], [20230102], [20230103]])
    assert _profiles(content, settings)["code"].inferred_type is InferredType.INTEGER


def test_high_cardinality_column_is_typed_text(settings: Settings) -> None:
    """A column that is distinct in every row is treated as free text."""
    profile = _profiles(high_cardinality_csv(), settings)["code"]
    assert profile.inferred_type is InferredType.TEXT
    assert profile.unique_percentage == pytest.approx(100.0)


def test_dataset_summary(settings: Settings) -> None:
    """The dataset summary reports shape, duplicates, missingness and memory."""
    content = build_csv(
        ["a", "b"], [[1, "x"], [2, None], [1, "x"], [3, "y"]]
    )
    frame = load_csv(content, settings)
    summary = summarise_dataset(frame, profile_columns(frame, settings))

    assert summary.row_count == 4
    assert summary.column_count == 2
    assert summary.duplicate_row_count == 1
    assert summary.duplicate_row_percentage == pytest.approx(25.0)
    assert summary.missing_cell_count == 1
    assert summary.missing_cell_percentage == pytest.approx(12.5)
    assert summary.memory_usage_bytes is not None
    assert summary.memory_usage_bytes > 0
    assert sum(summary.column_type_counts.values()) == 2


def test_profile_columns_preserves_order(settings: Settings) -> None:
    """Column profiles are returned in the order the columns appear."""
    frame = pd.DataFrame({"z": [1], "a": [2], "m": [3]})
    assert [column.name for column in profile_columns(frame, settings)] == ["z", "a", "m"]
