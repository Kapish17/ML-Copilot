"""Dataset and column profiling.

The profiler turns a parsed DataFrame into plain Pydantic models. Statistics
are chosen per column so that numeric summaries are never computed for
categorical data and frequency tables are never computed for continuous data.
"""

from __future__ import annotations

import warnings

import pandas as pd
from pandas.api import types as pdt

from app.core.config import Settings
from app.schemas.dataset import (
    CategoricalStats,
    ColumnProfile,
    DatasetSummary,
    DatetimeStats,
    InferredType,
    NumericStats,
    ValueCount,
)
from app.services.datasets.conversions import percentage, safe_float, stringify

DATETIME_SAMPLE_SIZE = 200
DATETIME_MATCH_RATIO = 0.95
QUARTILE_LOW = 0.25
QUARTILE_HIGH = 0.75


def _is_numeric_like(value: object) -> bool:
    """Return True when a value is a plain number written as text."""
    try:
        float(str(value))
    except (TypeError, ValueError):
        return False
    return True


def is_text_dtype(series: pd.Series) -> bool:
    """Return True for text-holding columns.

    pandas 2 parses text into ``object`` columns while pandas 3 uses a
    dedicated ``str`` dtype, so both are accepted.
    """
    return pdt.is_object_dtype(series) or pdt.is_string_dtype(series)


def as_datetime(series: pd.Series) -> pd.Series | None:
    """Return ``series`` parsed as datetimes, or ``None`` if it is not dates.

    CSV parsing never infers dates, so text columns are sampled and only
    treated as datetimes when nearly all sampled values parse. Purely numeric
    text (identifiers, codes) is excluded so that numbers are not mistaken for
    timestamps.

    Args:
        series: The column to inspect.

    Returns:
        pandas.Series | None: A datetime series, or ``None`` when the column
        does not look like dates.
    """
    if pdt.is_datetime64_any_dtype(series):
        return series
    if not is_text_dtype(series):
        return None

    non_null = series.dropna()
    if non_null.empty:
        return None

    sample = non_null.head(DATETIME_SAMPLE_SIZE)
    if all(_is_numeric_like(value) for value in sample):
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed_sample = pd.to_datetime(sample, errors="coerce", format="mixed")
            if parsed_sample.notna().mean() < DATETIME_MATCH_RATIO:
                return None
            return pd.to_datetime(series, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return None


def infer_type(
    series: pd.Series,
    *,
    unique_count: int,
    row_count: int,
    settings: Settings,
    datetime_series: pd.Series | None,
) -> InferredType:
    """Infer the semantic type of a column.

    Args:
        series: The raw column.
        unique_count: Distinct non-missing values in the column.
        row_count: Total rows in the dataset.
        settings: Active application settings.
        datetime_series: Result of :func:`as_datetime` for this column.

    Returns:
        InferredType: The semantic type used to pick statistics downstream.
    """
    if series.isna().all():
        return InferredType.EMPTY
    if pdt.is_bool_dtype(series):
        return InferredType.BOOLEAN
    if pdt.is_integer_dtype(series):
        return InferredType.INTEGER
    if pdt.is_float_dtype(series):
        return InferredType.FLOAT
    if datetime_series is not None:
        return InferredType.DATETIME
    if is_text_dtype(series):
        # A column is only treated as free text when it has both many distinct
        # values and a high share of distinct values. Requiring both keeps
        # small datasets, where every ratio is high, from being called text.
        ratio = unique_count / row_count if row_count else 0.0
        many_distinct = unique_count > settings.max_categorical_distinct
        mostly_distinct = ratio > settings.categorical_max_unique_ratio
        if many_distinct and mostly_distinct:
            return InferredType.TEXT
        return InferredType.CATEGORICAL
    return InferredType.UNKNOWN


def numeric_stats(series: pd.Series) -> NumericStats:
    """Compute descriptive statistics for a numeric column.

    Args:
        series: A numeric column; missing values are ignored.

    Returns:
        NumericStats: Statistics with non-finite results normalised to ``None``.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return NumericStats()

    return NumericStats(
        mean=safe_float(values.mean()),
        median=safe_float(values.median()),
        std=safe_float(values.std()),
        minimum=safe_float(values.min()),
        maximum=safe_float(values.max()),
        q1=safe_float(values.quantile(QUARTILE_LOW)),
        q3=safe_float(values.quantile(QUARTILE_HIGH)),
        zero_count=int((values == 0).sum()),
        negative_count=int((values < 0).sum()),
    )


def datetime_stats(series: pd.Series) -> DatetimeStats:
    """Compute the observed range of a datetime column."""
    values = series.dropna()
    if values.empty:
        return DatetimeStats()
    return DatetimeStats(
        minimum=stringify(values.min()),
        maximum=stringify(values.max()),
    )


def value_distribution(series: pd.Series, limit: int) -> list[ValueCount]:
    """Return the most frequent values of a column.

    Args:
        series: The column to count; missing values are ignored.
        limit: Maximum number of entries to return.

    Returns:
        list[ValueCount]: Values ordered by descending frequency.
    """
    counts = series.value_counts(dropna=True)
    total = int(counts.sum())
    return [
        ValueCount(
            value=stringify(value),
            count=int(count),
            percentage=percentage(int(count), total),
        )
        for value, count in counts.head(limit).items()
    ]


def categorical_stats(
    series: pd.Series, *, unique_count: int, settings: Settings
) -> CategoricalStats:
    """Compute frequency statistics for a categorical, boolean or text column."""
    top_values = value_distribution(series, settings.profile_top_values)
    return CategoricalStats(
        top_values=top_values,
        truncated=unique_count > len(top_values),
    )


def profile_column(series: pd.Series, *, row_count: int, settings: Settings) -> ColumnProfile:
    """Build the profile of a single column.

    Args:
        series: The column to profile.
        row_count: Total rows in the dataset, used for percentages.
        settings: Active application settings.

    Returns:
        ColumnProfile: Counts, inferred type and type-appropriate statistics.
    """
    non_null_count = int(series.count())
    missing_count = row_count - non_null_count
    unique_count = int(series.nunique(dropna=True))

    datetime_series = as_datetime(series)
    inferred = infer_type(
        series,
        unique_count=unique_count,
        row_count=row_count,
        settings=settings,
        datetime_series=datetime_series,
    )

    profile = ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        inferred_type=inferred,
        non_null_count=non_null_count,
        missing_count=missing_count,
        missing_percentage=percentage(missing_count, row_count),
        unique_count=unique_count,
        unique_percentage=percentage(unique_count, row_count),
        is_constant=unique_count <= 1,
    )

    if inferred in (InferredType.INTEGER, InferredType.FLOAT):
        profile.numeric_stats = numeric_stats(series)
    elif inferred is InferredType.DATETIME and datetime_series is not None:
        profile.datetime_stats = datetime_stats(datetime_series)
    elif inferred in (
        InferredType.BOOLEAN,
        InferredType.CATEGORICAL,
        InferredType.TEXT,
    ):
        profile.categorical_stats = categorical_stats(
            series, unique_count=unique_count, settings=settings
        )

    return profile


def profile_columns(frame: pd.DataFrame, settings: Settings) -> list[ColumnProfile]:
    """Profile every column of a dataset."""
    row_count = int(frame.shape[0])
    return [
        profile_column(frame[column], row_count=row_count, settings=settings)
        for column in frame.columns
    ]


def _memory_usage(frame: pd.DataFrame) -> int | None:
    """Return the deep memory footprint of a frame, or ``None`` if unavailable."""
    try:
        return int(frame.memory_usage(deep=True).sum())
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _duplicate_row_count(frame: pd.DataFrame) -> int:
    """Return the number of duplicated rows, or 0 if they cannot be compared."""
    try:
        return int(frame.duplicated().sum())
    except TypeError:  # pragma: no cover - unhashable cells cannot occur from CSV
        return 0


def summarise_dataset(frame: pd.DataFrame, columns: list[ColumnProfile]) -> DatasetSummary:
    """Build the dataset-level section of the profile.

    Args:
        frame: The parsed dataset.
        columns: Already computed column profiles, reused for the type counts.

    Returns:
        DatasetSummary: Shape, memory, duplicate and missingness totals.
    """
    row_count = int(frame.shape[0])
    column_count = int(frame.shape[1])
    total_cells = row_count * column_count
    duplicate_rows = _duplicate_row_count(frame)
    missing_cells = int(frame.isna().sum().sum())

    type_counts: dict[str, int] = {}
    for column in columns:
        key = column.inferred_type.value
        type_counts[key] = type_counts.get(key, 0) + 1

    return DatasetSummary(
        row_count=row_count,
        column_count=column_count,
        memory_usage_bytes=_memory_usage(frame),
        duplicate_row_count=duplicate_rows,
        duplicate_row_percentage=percentage(duplicate_rows, row_count),
        missing_cell_count=missing_cells,
        missing_cell_percentage=percentage(missing_cells, total_cells),
        column_type_counts=type_counts,
    )
