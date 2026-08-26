"""Tests for decoding, header validation and CSV parsing."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.core.errors import (
    DatasetTooLargeError,
    DuplicateColumnsError,
    EmptyDatasetError,
    MalformedCSVError,
    MissingHeaderError,
)
from app.services.datasets.loader import decode_content, load_csv
from tests.factories import build_csv


def test_load_csv_parses_valid_content(settings: Settings) -> None:
    """A well-formed CSV becomes a DataFrame with the expected shape."""
    frame = load_csv(build_csv(["a", "b"], [[1, "x"], [2, "y"]]), settings)
    assert list(frame.columns) == ["a", "b"]
    assert frame.shape == (2, 2)


def test_load_csv_handles_utf8_bom(settings: Settings) -> None:
    """A UTF-8 byte order mark does not end up in the first column name."""
    frame = load_csv("﻿a,b\n1,2\n".encode("utf-8"), settings)
    assert list(frame.columns) == ["a", "b"]


def test_decode_content_falls_back_to_latin1() -> None:
    """Bytes that are not valid UTF-8 still decode via the fallback encoding."""
    assert "caf" in decode_content(b"name\ncaf\xe9\n")


def test_load_csv_rejects_malformed_rows(settings: Settings) -> None:
    """A row with more fields than the header is a parse error."""
    with pytest.raises(MalformedCSVError) as exc_info:
        load_csv(b"a,b\n1,2\n3,4,5\n", settings)
    assert exc_info.value.status_code == 422


def test_rows_with_fewer_fields_are_padded(settings: Settings) -> None:
    """Short rows are padded with missing values, following CSV convention.

    Only rows with *more* fields than the header are a parse error. This test
    pins the asymmetry so it stays a documented decision rather than a
    surprise: the padded cells then show up as missing values in the profile.
    """
    frame = load_csv(b"a,b\n1\n", settings)

    assert frame.shape == (1, 2)
    assert bool(frame["b"].isna().all())


def test_load_csv_rejects_duplicate_columns(settings: Settings) -> None:
    """Repeated header names are reported instead of silently renamed."""
    with pytest.raises(DuplicateColumnsError) as exc_info:
        load_csv(b"a,a,b\n1,2,3\n", settings)
    assert exc_info.value.details["duplicate_columns"] == ["a"]


def test_load_csv_rejects_blank_header(settings: Settings) -> None:
    """A file whose first row is blank has no usable column names."""
    with pytest.raises(MissingHeaderError):
        load_csv(b"\n1,2\n", settings)


def test_load_csv_rejects_header_only_file(settings: Settings) -> None:
    """A header with no data rows cannot be profiled."""
    with pytest.raises(EmptyDatasetError) as exc_info:
        load_csv(b"a,b\n", settings)
    assert exc_info.value.code == "empty_dataset"


def test_load_csv_rejects_content_without_data(settings: Settings) -> None:
    """Whitespace-only content is treated as an empty dataset."""
    with pytest.raises((EmptyDatasetError, MissingHeaderError)):
        load_csv(b"\n\n", settings)


def test_load_csv_enforces_row_limit() -> None:
    """A dataset with more rows than allowed is rejected."""
    settings = Settings(max_dataset_rows=2)
    content = build_csv(["a"], [[1], [2], [3]])
    with pytest.raises(DatasetTooLargeError) as exc_info:
        load_csv(content, settings)
    assert exc_info.value.details["max_dataset_rows"] == 2


def test_load_csv_enforces_column_limit() -> None:
    """A dataset with more columns than allowed is rejected before parsing."""
    settings = Settings(max_dataset_columns=2)
    content = build_csv(["a", "b", "c"], [[1, 2, 3]])
    with pytest.raises(DatasetTooLargeError) as exc_info:
        load_csv(content, settings)
    assert exc_info.value.details["column_count"] == 3
