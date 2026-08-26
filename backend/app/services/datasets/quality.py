"""Heuristic data-quality detection.

Every finding is an observation backed by numbers, not a verdict. Codes that
rest on a guess are named accordingly (``possible_id_column``,
``potentially_suspicious_column``) and each issue carries the counts that
triggered it in ``details``, so a reader can always see why it fired.
"""

from __future__ import annotations

import pandas as pd
from pandas.api import types as pdt

from app.core.config import Settings
from app.schemas.dataset import (
    ColumnProfile,
    InferredType,
    IssueSeverity,
    QualityIssue,
    QualityReport,
)
from app.services.datasets.conversions import percentage

#: Column names that, on their own, hint at a row identifier.
ID_NAME_EXACT = frozenset({"id", "index", "key", "uuid", "guid", "row_id", "rowid"})
#: Suffixes that hint at a row identifier, e.g. ``customer_id``.
ID_NAME_SUFFIXES = ("_id", "_key", "_uuid", "_guid")
#: Column names that often hold an outcome and may leak into a model.
OUTCOME_NAME_HINTS = frozenset(
    {"target", "label", "labels", "outcome", "result", "class", "y"}
)
#: Sample size used when checking an object column for mixed value types.
MIXED_TYPE_SAMPLE_SIZE = 500

_SEVERITY_ORDER = {
    IssueSeverity.CRITICAL: 0,
    IssueSeverity.WARNING: 1,
    IssueSeverity.INFO: 2,
}

_TEXT_LIKE_TYPES = (InferredType.CATEGORICAL, InferredType.TEXT)


def _looks_like_id_name(name: str) -> bool:
    """Return True when a column name itself suggests an identifier."""
    lowered = name.strip().lower()
    return lowered in ID_NAME_EXACT or lowered.endswith(ID_NAME_SUFFIXES)


def _is_consecutive_integers(series: pd.Series) -> bool:
    """Return True when a column is an unbroken, increasing integer run.

    This is what a row number looks like: sorted, gap-free and distinct. A
    numeric column that merely happens to be sorted will have gaps and is not
    matched.
    """
    if not pdt.is_integer_dtype(series):
        return False
    values = series.dropna()
    if values.empty or not values.is_monotonic_increasing:
        return False
    span = int(values.max()) - int(values.min()) + 1
    return span == len(values)


def _is_numeric_text(value: object) -> bool:
    """Return True when a value is a number or parses as one."""
    try:
        float(str(value))
    except (TypeError, ValueError):
        return False
    return True


def detect_missing_values(
    columns: list[ColumnProfile], settings: Settings
) -> list[QualityIssue]:
    """Report columns that are entirely or heavily missing.

    A column with no values at all is reported as ``empty_column``; a column
    above the configured missingness ratio as ``high_missing_values``.
    """
    issues: list[QualityIssue] = []
    threshold_percentage = settings.high_missing_ratio * 100

    for column in columns:
        if column.non_null_count == 0:
            issues.append(
                QualityIssue(
                    code="empty_column",
                    severity=IssueSeverity.CRITICAL,
                    message=f"Column '{column.name}' has no values at all.",
                    columns=[column.name],
                    details={"missing_percentage": column.missing_percentage},
                )
            )
        elif column.missing_percentage >= threshold_percentage:
            issues.append(
                QualityIssue(
                    code="high_missing_values",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Column '{column.name}' is {column.missing_percentage:.1f}% "
                        f"missing, at or above the {threshold_percentage:.0f}% threshold."
                    ),
                    columns=[column.name],
                    details={
                        "missing_count": column.missing_count,
                        "missing_percentage": column.missing_percentage,
                        "threshold_percentage": threshold_percentage,
                    },
                )
            )
        elif column.missing_count > 0:
            issues.append(
                QualityIssue(
                    code="missing_values",
                    severity=IssueSeverity.INFO,
                    message=(
                        f"Column '{column.name}' has {column.missing_count} missing "
                        f"value(s) ({column.missing_percentage:.1f}%)."
                    ),
                    columns=[column.name],
                    details={
                        "missing_count": column.missing_count,
                        "missing_percentage": column.missing_percentage,
                    },
                )
            )
    return issues


def detect_duplicate_rows(frame: pd.DataFrame) -> list[QualityIssue]:
    """Report fully duplicated rows."""
    try:
        duplicate_count = int(frame.duplicated().sum())
    except TypeError:  # pragma: no cover - unhashable cells cannot occur from CSV
        return []
    if duplicate_count == 0:
        return []

    row_count = int(frame.shape[0])
    return [
        QualityIssue(
            code="duplicate_rows",
            severity=IssueSeverity.WARNING,
            message=(
                f"{duplicate_count} of {row_count} rows are exact duplicates of an "
                "earlier row."
            ),
            details={
                "duplicate_row_count": duplicate_count,
                "duplicate_row_percentage": percentage(duplicate_count, row_count),
            },
        )
    ]


def detect_constant_columns(columns: list[ColumnProfile]) -> list[QualityIssue]:
    """Report columns that hold a single value and carry no information.

    Fully empty columns are skipped here; they are reported as
    ``empty_column`` by :func:`detect_missing_values`.
    """
    return [
        QualityIssue(
            code="constant_column",
            severity=IssueSeverity.WARNING,
            message=(
                f"Column '{column.name}' holds the same value in every row, so it "
                "carries no information."
            ),
            columns=[column.name],
            details={"unique_count": column.unique_count},
        )
        for column in columns
        if column.is_constant and column.non_null_count > 0
    ]


def detect_high_cardinality(
    columns: list[ColumnProfile], settings: Settings
) -> list[QualityIssue]:
    """Report categorical or text columns with an unusually high distinct count.

    Two signals are required so small datasets, where every column looks
    distinct, are not flagged: more distinct values than
    ``max_categorical_distinct``, and a distinct share above
    ``high_cardinality_ratio``.
    """
    threshold_percentage = settings.high_cardinality_ratio * 100
    return [
        QualityIssue(
            code="high_cardinality_column",
            severity=IssueSeverity.INFO,
            message=(
                f"Column '{column.name}' has {column.unique_count} distinct values "
                f"({column.unique_percentage:.1f}% of rows). Encoding it directly "
                "would create a very large number of features."
            ),
            columns=[column.name],
            details={
                "unique_count": column.unique_count,
                "unique_percentage": column.unique_percentage,
                "distinct_threshold": settings.max_categorical_distinct,
                "threshold_percentage": threshold_percentage,
            },
        )
        for column in columns
        if column.inferred_type in _TEXT_LIKE_TYPES
        and column.unique_count > settings.max_categorical_distinct
        and column.unique_percentage > threshold_percentage
    ]


def detect_possible_id_columns(
    frame: pd.DataFrame, columns: list[ColumnProfile], settings: Settings
) -> list[QualityIssue]:
    """Flag columns that plausibly hold a row identifier.

    Two signals are required: near-unique values, plus either a name that reads
    like an identifier or an unbroken run of consecutive integers, which is how
    a row number looks. Requiring the sequence to be consecutive keeps any
    sorted numeric column from being mistaken for an identifier. Both reasons
    are reported so the heuristic can be judged by the reader.
    """
    issues: list[QualityIssue] = []
    threshold_percentage = settings.id_uniqueness_ratio * 100

    for column in columns:
        if column.unique_percentage < threshold_percentage or column.non_null_count == 0:
            continue

        reasons: list[str] = []
        if _looks_like_id_name(column.name):
            reasons.append("name_suggests_identifier")
        if _is_consecutive_integers(frame[column.name]):
            reasons.append("consecutive_integer_sequence")

        if not reasons:
            continue

        issues.append(
            QualityIssue(
                code="possible_id_column",
                severity=IssueSeverity.INFO,
                message=(
                    f"Column '{column.name}' looks like it may be a row identifier "
                    f"({column.unique_percentage:.1f}% unique). Identifiers are "
                    "usually excluded from training."
                ),
                columns=[column.name],
                details={
                    "unique_percentage": column.unique_percentage,
                    "reasons": reasons,
                },
            )
        )
    return issues


def detect_suspicious_columns(
    columns: list[ColumnProfile], target_column: str | None
) -> list[QualityIssue]:
    """Flag columns whose name suggests they hold an outcome.

    A second outcome-looking column beside the chosen target is worth a human
    check, since it may duplicate the label and leak into a model.
    """
    return [
        QualityIssue(
            code="potentially_suspicious_column",
            severity=IssueSeverity.INFO,
            message=(
                f"Column '{column.name}' has an outcome-like name. If it is derived "
                "from the target it could leak information into a model."
            ),
            columns=[column.name],
            details={"matched_hint": column.name.strip().lower()},
        )
        for column in columns
        if column.name.strip().lower() in OUTCOME_NAME_HINTS
        and column.name != target_column
    ]


def detect_mixed_types(
    frame: pd.DataFrame, columns: list[ColumnProfile]
) -> list[QualityIssue]:
    """Flag text columns that mix numeric and non-numeric values.

    This usually means a numeric column was polluted by placeholder text such
    as ``"N/A"`` or ``"unknown"``, which stops it from parsing as a number.
    """
    issues: list[QualityIssue] = []
    for column in columns:
        if column.inferred_type not in _TEXT_LIKE_TYPES:
            continue

        sample = frame[column.name].dropna().head(MIXED_TYPE_SAMPLE_SIZE)
        if sample.empty:
            continue

        numeric_count = int(sum(_is_numeric_text(value) for value in sample))
        if 0 < numeric_count < len(sample):
            issues.append(
                QualityIssue(
                    code="mixed_type_column",
                    severity=IssueSeverity.WARNING,
                    message=(
                        f"Column '{column.name}' mixes numeric and non-numeric "
                        "values, so it was not parsed as a number."
                    ),
                    columns=[column.name],
                    details={
                        "numeric_value_count": numeric_count,
                        "sampled_value_count": int(len(sample)),
                    },
                )
            )
    return issues


def analyse_quality(
    frame: pd.DataFrame,
    columns: list[ColumnProfile],
    settings: Settings,
    target_column: str | None = None,
) -> QualityReport:
    """Run every detector and collect the findings.

    Args:
        frame: The parsed dataset.
        columns: Column profiles produced by the profiler.
        settings: Active application settings.
        target_column: The caller's chosen target, excluded from the
            outcome-name check.

    Returns:
        QualityReport: Findings ordered by severity, most serious first.
    """
    issues: list[QualityIssue] = [
        *detect_missing_values(columns, settings),
        *detect_duplicate_rows(frame),
        *detect_constant_columns(columns),
        *detect_high_cardinality(columns, settings),
        *detect_possible_id_columns(frame, columns, settings),
        *detect_suspicious_columns(columns, target_column),
        *detect_mixed_types(frame, columns),
    ]
    issues.sort(key=lambda issue: (_SEVERITY_ORDER[issue.severity], issue.code))
    return QualityReport(issue_count=len(issues), issues=issues)
