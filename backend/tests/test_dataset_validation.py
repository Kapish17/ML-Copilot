"""Tests for upload-level validation: filenames, extensions and size limits."""

from __future__ import annotations

import asyncio

import pytest

from app.core.config import Settings
from app.core.errors import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.services.datasets.validation import (
    read_upload,
    safe_filename,
    validate_extension,
    validate_size,
)
from tests.factories import FakeUpload


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("dataset.csv", "dataset.csv"),
        ("../../etc/passwd.csv", "passwd.csv"),
        ("/var/tmp/data.csv", "data.csv"),
        ("C:\\Users\\me\\data.csv", "data.csv"),
        ("", "upload"),
        (None, "upload"),
    ],
)
def test_safe_filename_strips_directories(raw: str | None, expected: str) -> None:
    """Client-supplied paths are reduced to a bare filename."""
    assert safe_filename(raw) == expected


@pytest.mark.parametrize(
    ("filename", "extension"),
    [("dataset.xlsx", ".xlsx"), ("dataset.JSON", ".json")],
)
def test_validate_extension_accepts_the_other_supported_formats(
    filename: str, extension: str, settings: Settings
) -> None:
    """Excel and JSON joined the allowlist in Commit 15."""
    assert validate_extension(filename, settings) == extension


def test_validate_extension_accepts_csv(settings: Settings) -> None:
    """A .csv file passes validation regardless of letter case."""
    assert validate_extension("dataset.csv", settings) == ".csv"
    assert validate_extension("DATASET.CSV", settings) == ".csv"


@pytest.mark.parametrize(
    "filename", ["dataset.txt", "dataset.parquet", "dataset", "data.csv.exe"]
)
def test_validate_extension_rejects_other_types(filename: str, settings: Settings) -> None:
    """Anything outside the configured allowlist is rejected with a typed error."""
    with pytest.raises(UnsupportedFileTypeError) as exc_info:
        validate_extension(filename, settings)
    assert exc_info.value.code == "unsupported_file_type"
    assert exc_info.value.status_code == 415


def test_validate_size_rejects_empty(settings: Settings) -> None:
    """A zero-byte upload is rejected before any parsing."""
    with pytest.raises(EmptyFileError):
        validate_size(0, settings)


def test_validate_size_rejects_oversized() -> None:
    """An upload beyond the configured limit is rejected."""
    settings = Settings(max_upload_bytes=10)
    with pytest.raises(FileTooLargeError) as exc_info:
        validate_size(11, settings)
    assert exc_info.value.details["max_upload_bytes"] == 10


def test_validate_size_accepts_limit_exactly() -> None:
    """The limit itself is allowed; only larger uploads fail."""
    validate_size(10, Settings(max_upload_bytes=10))


def test_read_upload_returns_full_content(settings: Settings) -> None:
    """Chunked reading reassembles the original bytes."""
    content = b"a,b\n1,2\n"
    result = asyncio.run(read_upload(FakeUpload(content), settings))
    assert result == content


def test_read_upload_rejects_oversized_stream() -> None:
    """Reading stops as soon as the size limit is passed."""
    settings = Settings(max_upload_bytes=4)
    with pytest.raises(FileTooLargeError):
        asyncio.run(read_upload(FakeUpload(b"far too many bytes"), settings))


def test_read_upload_rejects_empty_stream(settings: Settings) -> None:
    """An upload that yields no bytes is reported as empty."""
    with pytest.raises(EmptyFileError):
        asyncio.run(read_upload(FakeUpload(b""), settings))
