"""Schemas describing the dataset profiling API.

These models are the public contract of the profiling endpoint. The service
layer builds them directly, so pandas objects never leak out of the backend.
Non-finite floats (``NaN``/``inf``) are normalised to ``null`` before they
reach these models, keeping every response valid JSON.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class InferredType(str, Enum):
    """Semantic column type inferred from the parsed data."""

    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    TEXT = "text"
    EMPTY = "empty"
    UNKNOWN = "unknown"


class TaskSuggestion(str, Enum):
    """Suggested modelling task for a target column."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UNDETERMINED = "undetermined"


class IssueSeverity(str, Enum):
    """How strongly a data-quality finding should be acted upon."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ValueCount(BaseModel):
    """One observed value and how often it occurs.

    ``value`` is always a string representation so the payload stays uniform
    regardless of the column's underlying dtype.
    """

    value: str = Field(..., description="String representation of the value.")
    count: int = Field(..., description="Number of rows holding this value.")
    percentage: float = Field(..., description="Share of non-missing rows, 0-100.")


class NumericStats(BaseModel):
    """Descriptive statistics for a numeric column."""

    mean: float | None = None
    median: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    q1: float | None = Field(None, description="25th percentile.")
    q3: float | None = Field(None, description="75th percentile.")
    zero_count: int = Field(0, description="Number of exact zero values.")
    negative_count: int = Field(0, description="Number of values below zero.")


class DatetimeStats(BaseModel):
    """Range statistics for a column recognised as datetime-like."""

    minimum: str | None = Field(None, description="Earliest value, ISO 8601.")
    maximum: str | None = Field(None, description="Latest value, ISO 8601.")


class CategoricalStats(BaseModel):
    """Frequency statistics for a categorical or text column."""

    top_values: list[ValueCount] = Field(
        default_factory=list,
        description="Most frequent values, longest first, truncated by config.",
    )
    truncated: bool = Field(
        False,
        description="True when the column has more distinct values than were returned.",
    )


class ColumnProfile(BaseModel):
    """Profile of a single dataset column."""

    name: str
    dtype: str = Field(..., description="Underlying pandas dtype, e.g. 'int64'.")
    inferred_type: InferredType
    non_null_count: int
    missing_count: int
    missing_percentage: float = Field(..., description="Share of all rows, 0-100.")
    unique_count: int = Field(..., description="Distinct non-missing values.")
    unique_percentage: float = Field(..., description="Share of all rows, 0-100.")
    is_constant: bool = Field(
        ..., description="True when the column holds at most one distinct value."
    )
    numeric_stats: NumericStats | None = None
    datetime_stats: DatetimeStats | None = None
    categorical_stats: CategoricalStats | None = None


class DatasetSummary(BaseModel):
    """Dataset-level profile information."""

    row_count: int
    column_count: int
    memory_usage_bytes: int | None = Field(
        None, description="Deep memory footprint of the parsed frame, when available."
    )
    duplicate_row_count: int
    duplicate_row_percentage: float
    missing_cell_count: int
    missing_cell_percentage: float
    column_type_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Number of columns per inferred type.",
    )


class QualityIssue(BaseModel):
    """A single heuristic data-quality finding.

    Findings are deliberately worded as observations, not verdicts: codes such
    as ``possible_id_column`` signal that the detection is a heuristic.
    """

    code: str = Field(..., description="Stable issue code, e.g. 'constant_column'.")
    severity: IssueSeverity
    message: str = Field(..., description="Plain-language description of the finding.")
    columns: list[str] = Field(
        default_factory=list, description="Columns the finding applies to."
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Numbers behind the finding, so the heuristic stays explainable.",
    )


class QualityReport(BaseModel):
    """Collected data-quality findings for a dataset."""

    issue_count: int
    issues: list[QualityIssue] = Field(default_factory=list)


class ClassBalance(BaseModel):
    """Class balance summary for a classification-like target."""

    class_count: int
    majority_class: str | None = None
    majority_percentage: float | None = None
    minority_class: str | None = None
    minority_percentage: float | None = None
    is_imbalanced: bool = Field(
        False,
        description="True when the majority class exceeds the configured share.",
    )


class TargetProfile(BaseModel):
    """Analysis of the optional target column.

    No model is trained in this commit; ``task_suggestion`` is a heuristic
    reading of the column's type and cardinality.
    """

    name: str
    dtype: str
    inferred_type: InferredType
    missing_count: int
    missing_percentage: float
    task_suggestion: TaskSuggestion
    task_reason: str = Field(..., description="Why this task type was suggested.")
    distribution: list[ValueCount] | None = Field(
        None, description="Value distribution for classification-like targets."
    )
    class_balance: ClassBalance | None = None
    numeric_stats: NumericStats | None = Field(
        None, description="Summary statistics for regression-like targets."
    )


class DatasetProfileResponse(BaseModel):
    """Full response of the dataset profiling endpoint."""

    filename: str
    source_format: str | None = Field(
        default=None,
        description=(
            "The format the upload was read as: 'csv', 'xlsx' or 'json'. "
            "Reported as context only — profiling is identical for all "
            "three, because every format becomes the same standardised "
            "table before anything is measured."
        ),
        examples=["csv"],
    )
    generated_at: datetime
    dataset: DatasetSummary
    columns: list[ColumnProfile]
    quality: QualityReport
    target: TargetProfile | None = None
