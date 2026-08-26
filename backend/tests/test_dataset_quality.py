"""Tests for the heuristic data-quality detectors."""

from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.schemas.dataset import IssueSeverity, QualityReport
from app.services.datasets.loader import load_csv
from app.services.datasets.profiler import profile_columns
from app.services.datasets.quality import analyse_quality
from tests.factories import build_csv, high_cardinality_csv


def _report(
    content: bytes, settings: Settings, target_column: str | None = None
) -> QualityReport:
    """Run the full quality analysis over CSV bytes."""
    frame = load_csv(content, settings)
    columns = profile_columns(frame, settings)
    return analyse_quality(frame, columns, settings, target_column=target_column)


def _codes(report: QualityReport) -> list[str]:
    """Return the issue codes present in a report."""
    return [issue.code for issue in report.issues]


def _issue(report: QualityReport, code: str) -> Any:
    """Return the first issue with the given code."""
    return next(issue for issue in report.issues if issue.code == code)


def test_missing_values_are_reported(settings: Settings) -> None:
    """A column with a few missing values produces an informational finding."""
    content = build_csv(
        ["a", "b"], [[1, "x"], [None, "y"], [3, "x"], [4, "y"], [5, "x"]]
    )
    report = _report(content, settings)

    assert "missing_values" in _codes(report)
    assert _issue(report, "missing_values").details["missing_count"] == 1


def test_high_missingness_is_escalated(settings: Settings) -> None:
    """Missingness at or above the threshold is a warning, not an info note."""
    content = build_csv(["a", "b"], [[1, "x"], [None, "y"], [None, "x"], [None, "y"]])
    issue = _issue(_report(content, settings), "high_missing_values")

    assert issue.severity is IssueSeverity.WARNING
    assert issue.details["missing_percentage"] == 75.0


def test_empty_column_is_critical(settings: Settings) -> None:
    """A column with no values at all is the most serious finding."""
    content = build_csv(["a", "blank"], [[1, None], [2, None]])
    report = _report(content, settings)
    issue = _issue(report, "empty_column")

    assert issue.severity is IssueSeverity.CRITICAL
    assert issue.columns == ["blank"]
    assert report.issues[0].code == "empty_column", "critical issues sort first"


def test_duplicate_rows_are_reported(settings: Settings) -> None:
    """Exact duplicate rows are counted and reported once for the dataset."""
    content = build_csv(["a", "b"], [[1, "x"], [1, "x"], [2, "y"]])
    issue = _issue(_report(content, settings), "duplicate_rows")

    assert issue.details["duplicate_row_count"] == 1
    assert issue.columns == []


def test_constant_column_is_reported(settings: Settings) -> None:
    """A single-valued column is flagged as carrying no information."""
    content = build_csv(["a", "fixed"], [[1, "x"], [2, "x"], [3, "x"]])
    assert _issue(_report(content, settings), "constant_column").columns == ["fixed"]


def test_empty_column_is_not_also_reported_as_constant(settings: Settings) -> None:
    """An all-missing column produces one finding, not two."""
    content = build_csv(["a", "blank"], [[1, None], [2, None]])
    assert "constant_column" not in _codes(_report(content, settings))


def test_high_cardinality_is_reported(settings: Settings) -> None:
    """A column distinct in nearly every row is flagged for encoding cost."""
    issue = _issue(_report(high_cardinality_csv(), settings), "high_cardinality_column")

    assert issue.columns == ["code"]
    assert issue.details["unique_count"] == 60


def test_low_cardinality_column_is_not_flagged(settings: Settings) -> None:
    """A small set of repeated categories is normal and stays unflagged."""
    content = build_csv(["colour"], [["red"], ["blue"], ["red"], ["blue"]])
    assert "high_cardinality_column" not in _codes(_report(content, settings))


def test_possible_id_column_is_detected(settings: Settings) -> None:
    """A unique, increasing, id-named integer column is flagged as a possible id."""
    rows = [[index, index % 2] for index in range(1, 11)]
    issue = _issue(_report(build_csv(["user_id", "y"], rows), settings), "possible_id_column")

    assert issue.severity is IssueSeverity.INFO
    assert "name_suggests_identifier" in issue.details["reasons"]
    assert "consecutive_integer_sequence" in issue.details["reasons"]
    assert "may be" in issue.message, "wording must stay hedged, not a verdict"


def test_sorted_numeric_column_is_not_called_an_id(settings: Settings) -> None:
    """A sorted unique numeric column with gaps is not an identifier."""
    rows = [[value, index] for index, value in enumerate([3, 17, 42, 108, 999])]
    report = _report(build_csv(["measurement", "n"], rows), settings)
    id_issues = [
        issue for issue in report.issues if issue.code == "possible_id_column"
    ]

    assert [issue.columns for issue in id_issues] == [["n"]]


def test_unique_text_without_id_signal_is_not_called_an_id(settings: Settings) -> None:
    """Uniqueness alone is not enough; the heuristic needs a second signal."""
    content = build_csv(["comment"], [["alpha"], ["beta"], ["gamma"], ["delta"]])
    assert "possible_id_column" not in _codes(_report(content, settings))


def test_outcome_named_column_is_potentially_suspicious(settings: Settings) -> None:
    """A second outcome-looking column beside the target is worth a look."""
    content = build_csv(["feature", "label", "target"], [[1, "a", "a"], [2, "b", "b"]])
    issue = _issue(
        _report(content, settings, target_column="target"),
        "potentially_suspicious_column",
    )

    assert issue.columns == ["label"]


def test_chosen_target_is_not_flagged_as_suspicious(settings: Settings) -> None:
    """The declared target is expected to look like an outcome."""
    content = build_csv(["feature", "target"], [[1, "a"], [2, "b"]])
    report = _report(content, settings, target_column="target")

    assert "potentially_suspicious_column" not in _codes(report)


def test_mixed_type_column_is_reported(settings: Settings) -> None:
    """Numbers mixed with words explain why a column did not parse as numeric."""
    content = build_csv(["amount"], [[1], [2], ["unknown"], [4]])
    issue = _issue(_report(content, settings), "mixed_type_column")

    assert issue.columns == ["amount"]
    assert issue.details["numeric_value_count"] == 3


def test_clean_dataset_has_no_issues(settings: Settings) -> None:
    """A tidy dataset produces an empty report."""
    content = build_csv(["a", "b"], [[3, "x"], [1, "y"], [2, "z"]])
    report = _report(content, settings)

    assert report.issue_count == 0
    assert report.issues == []


def test_issues_are_sorted_by_severity(settings: Settings) -> None:
    """Critical findings come first, informational findings last."""
    content = build_csv(
        ["blank", "fixed", "a"], [[None, "x", 1], [None, "x", 1], [None, "x", 2]]
    )
    severities = [issue.severity for issue in _report(content, settings).issues]

    assert severities == sorted(
        severities,
        key=lambda severity: [
            IssueSeverity.CRITICAL,
            IssueSeverity.WARNING,
            IssueSeverity.INFO,
        ].index(severity),
    )
