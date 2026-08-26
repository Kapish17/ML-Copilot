"""Tests for the orchestrating dataset profiling service."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import (
    FileTooLargeError,
    TargetColumnNotFoundError,
    UnsupportedFileTypeError,
)
from app.services.datasets import DatasetProfilingService
from tests.factories import build_csv, sample_csv


def test_profile_content_builds_a_complete_profile(
    service: DatasetProfilingService,
) -> None:
    """The service returns dataset, column and quality sections together."""
    profile = service.profile_content("dataset.csv", sample_csv())

    assert profile.filename == "dataset.csv"
    assert profile.dataset.row_count == 6
    assert len(profile.columns) == profile.dataset.column_count == 6
    assert profile.target is None
    assert profile.quality.issue_count == len(profile.quality.issues)


def test_profile_content_sanitises_the_filename(
    service: DatasetProfilingService,
) -> None:
    """A path-like filename never reaches the response."""
    profile = service.profile_content("../../etc/dataset.csv", sample_csv())
    assert profile.filename == "dataset.csv"


def test_profile_content_rejects_unsupported_type(
    service: DatasetProfilingService,
) -> None:
    """The extension is checked before the content is parsed."""
    with pytest.raises(UnsupportedFileTypeError):
        service.profile_content("dataset.txt", sample_csv())


def test_profile_content_with_target(service: DatasetProfilingService) -> None:
    """A named target column is analysed alongside the profile."""
    profile = service.profile_content("dataset.csv", sample_csv(), target_column="city")

    assert profile.target is not None
    assert profile.target.name == "city"


@pytest.mark.parametrize("target", ["", "   ", None])
def test_blank_target_is_treated_as_absent(
    service: DatasetProfilingService, target: str | None
) -> None:
    """An empty form field must not be mistaken for a column name."""
    profile = service.profile_content("dataset.csv", sample_csv(), target_column=target)
    assert profile.target is None


def test_unknown_target_raises(service: DatasetProfilingService) -> None:
    """An unknown target column is rejected."""
    with pytest.raises(TargetColumnNotFoundError):
        service.profile_content("dataset.csv", sample_csv(), target_column="missing")


def test_service_honours_custom_limits() -> None:
    """Limits come from the settings the service was constructed with."""
    service = DatasetProfilingService(Settings(max_upload_bytes=4))
    content = build_csv(["a", "b"], [[1, 2], [3, 4]])

    with pytest.raises(FileTooLargeError) as exc_info:
        service.profile_content("dataset.csv", content)
    assert exc_info.value.details["max_upload_bytes"] == 4
