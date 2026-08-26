"""Safe loading of uploaded CSV content into a pandas DataFrame.

Uploaded files are untrusted input. Nothing here executes file content, no
path from the request reaches the filesystem, and the parse happens entirely
in memory. Structural problems are converted into typed domain errors so the
API can report them precisely.
"""

from __future__ import annotations

import csv
import io
import re

import pandas as pd

from app.core.config import Settings
from app.core.errors import (
    DatasetTooLargeError,
    DuplicateColumnsError,
    EmptyDatasetError,
    MalformedCSVError,
    MissingHeaderError,
)

CSV_DELIMITER = ","
SUPPORTED_ENCODINGS = ("utf-8-sig", "latin-1")
UNNAMED_COLUMN_PATTERN = re.compile(r"^Unnamed: \d+(\.\d+)?$")


def decode_content(content: bytes) -> str:
    """Decode raw upload bytes into text.

    UTF-8 (with optional BOM) is tried first, then Latin-1 as a permissive
    fallback so that a legacy encoding does not fail the whole request.

    Args:
        content: Raw bytes of the uploaded file.

    Returns:
        str: The decoded text.

    Raises:
        MalformedCSVError: If no supported encoding can decode the bytes.
    """
    for encoding in SUPPORTED_ENCODINGS:
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise MalformedCSVError(  # pragma: no cover - latin-1 decodes any byte string
        "The file could not be decoded as text. Please upload a UTF-8 encoded CSV."
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
    if len(header) > settings.max_dataset_columns:
        raise DatasetTooLargeError(
            f"The dataset has {len(header)} columns, more than the "
            f"{settings.max_dataset_columns} column limit.",
            details={
                "column_count": len(header),
                "max_dataset_columns": settings.max_dataset_columns,
            },
        )

    seen: set[str] = set()
    duplicates: list[str] = []
    for cell in header:
        name = cell.strip()
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)

    if duplicates:
        raise DuplicateColumnsError(
            "The header repeats column names: " + ", ".join(duplicates) + ".",
            details={"duplicate_columns": duplicates},
        )


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


def validate_frame(frame: pd.DataFrame, settings: Settings) -> None:
    """Check a parsed frame for emptiness, unusable headers and size limits.

    Args:
        frame: The parsed dataset.
        settings: Active application settings.

    Raises:
        EmptyDatasetError: If the frame has no columns or no rows.
        MissingHeaderError: If every column name is a pandas placeholder,
            which means the file had no real header row.
        DatasetTooLargeError: If the frame exceeds the configured row limit.
    """
    if frame.shape[1] == 0:
        raise EmptyDatasetError("The dataset contains no columns.")

    names = [str(name) for name in frame.columns]
    if all(UNNAMED_COLUMN_PATTERN.match(name) for name in names):
        raise MissingHeaderError(
            "No column names were found. The first row must contain the header."
        )

    if frame.shape[0] == 0:
        raise EmptyDatasetError(
            "The dataset has a header but no data rows.",
            details={"column_count": frame.shape[1]},
        )

    if frame.shape[0] > settings.max_dataset_rows:
        raise DatasetTooLargeError(
            f"The dataset has {frame.shape[0]} rows, more than the "
            f"{settings.max_dataset_rows} row limit.",
            details={
                "row_count": frame.shape[0],
                "max_dataset_rows": settings.max_dataset_rows,
            },
        )


def load_csv(content: bytes, settings: Settings) -> pd.DataFrame:
    """Decode, validate and parse uploaded CSV bytes.

    Args:
        content: Raw bytes of the uploaded file.
        settings: Active application settings.

    Returns:
        pandas.DataFrame: A validated, non-empty dataset.

    Raises:
        DatasetError: Any of the typed loading failures described by the
            functions this orchestrates.
    """
    text = decode_content(content)
    header = read_header(text)
    validate_header(header, settings)
    frame = parse_csv(text)
    validate_frame(frame, settings)
    return frame
