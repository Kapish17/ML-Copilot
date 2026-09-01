"""Safe loading of uploaded CSV content into a pandas DataFrame.

Uploaded files are untrusted input. Nothing here executes file content, no
path from the request reaches the filesystem, and the parse happens entirely
in memory. Structural problems are converted into typed domain errors so the
API can report them precisely.

This module is the CSV *parse*. The checks it shares with the other formats —
column count, duplicate names, emptiness, row limits — live in
:mod:`app.services.datasets.ingestion.normalisation` and are applied by every
adapter, so all three formats agree on what an acceptable dataset is.
:class:`~app.services.datasets.ingestion.csv_adapter.CSVAdapter` is a thin
wrapper around :func:`load_csv`; there is exactly one CSV implementation and
this is it.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable

import pandas as pd

from app.core.config import Settings
from app.core.errors import (
    EmptyDatasetError,
    MalformedCSVError,
    MissingHeaderError,
)
from app.services.datasets.ingestion.normalisation import (
    UNNAMED_COLUMN_PATTERN,
    looks_binary,
    validate_columns,
    validate_frame,
)

CSV_DELIMITER = ","
SUPPORTED_ENCODINGS = ("utf-8-sig", "latin-1")

__all__ = [
    "CSV_DELIMITER",
    "SUPPORTED_ENCODINGS",
    "UNNAMED_COLUMN_PATTERN",
    "decode_content",
    "decode_text",
    "load_csv",
    "parse_csv",
    "read_header",
    "validate_frame",
    "validate_header",
]


def decode_text(content: bytes, on_failure: Callable[[str], Exception]) -> str:
    """Decode raw upload bytes into text, in the encodings this project reads.

    UTF-8 (with optional BOM) is tried first, then Latin-1 as a permissive
    fallback so that a legacy encoding does not fail the whole request. Shared
    by the text-based adapters so that a CSV and a JSON file are decoded the
    same way.

    Args:
        content: Raw bytes of the uploaded file.
        on_failure: Builds the error to raise when nothing decodes, so each
            caller reports the failure in its own format's terms.

    Returns:
        str: The decoded text.

    Raises:
        Exception: Whatever ``on_failure`` returns.
    """
    for encoding in SUPPORTED_ENCODINGS:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise on_failure(  # pragma: no cover - latin-1 decodes any byte string
        "The file could not be decoded as text."
    )


def decode_content(content: bytes) -> str:
    """Decode raw upload bytes as CSV text.

    Args:
        content: Raw bytes of the uploaded file.

    Returns:
        str: The decoded text.

    Raises:
        MalformedCSVError: If no supported encoding can decode the bytes.
    """
    return decode_text(
        content,
        lambda message: MalformedCSVError(
            f"{message} Please upload a UTF-8 encoded CSV."
        ),
    )


def read_header(text: str) -> list[str]:
    """Return the raw header row exactly as written in the file.

    pandas silently renames repeated columns (``a``, ``a.1``), so the header is
    inspected before parsing to detect duplicates faithfully.

    Args:
        text: Decoded file content.

    Returns:
        list[str]: The header cells.

    Raises:
        MissingHeaderError: If the file has no first row, or its first row is
            entirely blank.
        MalformedCSVError: If the first row cannot be tokenised as CSV.
    """
    reader = csv.reader(io.StringIO(text), delimiter=CSV_DELIMITER)
    try:
        header = next(reader)
    except StopIteration:
        raise MissingHeaderError("The file does not contain a header row.") from None
    except csv.Error as exc:
        raise MalformedCSVError(f"The header row could not be parsed: {exc}") from exc

    if not header or all(not cell.strip() for cell in header):
        raise MissingHeaderError(
            "The first row is blank, so the dataset has no usable column names."
        )
    return header


def validate_header(header: list[str], settings: Settings) -> None:
    """Check the header for duplicates and an unreasonable column count.

    Args:
        header: Header cells as written in the file.
        settings: Active application settings.

    Raises:
        DuplicateColumnsError: If any column name appears more than once.
        DatasetTooLargeError: If the header exceeds ``max_dataset_columns``.
    """
    validate_columns(header, settings)


def parse_csv(text: str) -> pd.DataFrame:
    """Parse decoded CSV text into a DataFrame.

    Args:
        text: Decoded file content.

    Returns:
        pandas.DataFrame: The parsed dataset.

    Raises:
        EmptyDatasetError: If the text holds no parsable columns.
        MalformedCSVError: If rows are inconsistent or otherwise unparsable.
    """
    try:
        return pd.read_csv(
            io.StringIO(text),
            sep=CSV_DELIMITER,
            low_memory=False,
            skip_blank_lines=True,
        )
    except pd.errors.EmptyDataError as exc:
        raise EmptyDatasetError("The file contains no data to profile.") from exc
    except pd.errors.ParserError as exc:
        raise MalformedCSVError(
            "The file is not valid CSV. Check for rows with a different number "
            f"of fields than the header: {exc}"
        ) from exc
    except ValueError as exc:
        raise MalformedCSVError(f"The file could not be parsed as CSV: {exc}") from exc


def load_csv(content: bytes, settings: Settings) -> pd.DataFrame:
    """Decode, validate and parse uploaded CSV bytes.

    Args:
        content: Raw bytes of the uploaded file.
        settings: Active application settings.

    Returns:
        pandas.DataFrame: A validated, non-empty dataset.

    Raises:
        DatasetError: Any of the typed loading failures described by the
            functions this orchestrates. Binary content sent under a ``.csv``
            name is refused here rather than decoded into unusable text: the
            extension chose this parser, but only the parser decides whether
            the bytes really are CSV.
    """
    if looks_binary(content):
        raise MalformedCSVError(
            "The file is not text, so it cannot be read as CSV. Upload the "
            "file in a supported format, or export the spreadsheet to CSV."
        )
    text = decode_content(content)
    header = read_header(text)
    validate_header(header, settings)
    frame = parse_csv(text)
    validate_frame(frame, settings)
    return frame
