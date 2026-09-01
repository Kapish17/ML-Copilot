"""The checks every parsed dataset passes, whatever format it came from.

Three adapters could easily have become three different definitions of "an
acceptable dataset" — one rejecting an empty file, another returning a frame
with no rows; one capping columns, another not. These functions are the single
definition, applied by all of them.

**Normalisation here is structural, not editorial.** Column names, values and
missing values are left exactly as the file had them. Nothing is renamed,
coerced, trimmed, filled or dropped. What these functions do is *refuse* a
frame that downstream code could not work with, and say why in the caller's
terms.

The unavoidable exceptions, which every reader shares and which are documented
in the READMEs:

* **dtypes are inferred**, so a column of ``1, 2, 3`` becomes ``int64``
  whether it was written in CSV text, an Excel cell or a JSON number;
* **blanks become missing values** — an empty CSV field, an empty Excel cell
  and a JSON ``null`` all arrive as ``NaN``;
* **nested JSON objects are flattened** into dotted column names, which is the
  only way a nested structure can become a table at all.
"""

from __future__ import annotations

import re

import pandas as pd

from app.core.config import Settings
from app.core.errors import (
    DatasetTooLargeError,
    DuplicateColumnsError,
    EmptyDatasetError,
    MissingHeaderError,
)

#: pandas names a column ``Unnamed: 3`` when the header cell was blank. A frame
#: whose every name looks like that had no header row at all.
UNNAMED_COLUMN_PATTERN = re.compile(r"^Unnamed: \d+(\.\d+)?$")

#: The first bytes of a ZIP container, which is what an ``.xlsx`` workbook is.
ZIP_MAGIC = b"PK\x03\x04"
#: The first bytes of the legacy OLE2 container used by ``.xls`` and other
#: Office binaries. Recognised only so it can be refused with a clear message.
OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"


def looks_binary(content: bytes) -> bool:
    """Whether these bytes are a container format rather than text.

    Used by the text-based adapters so that a spreadsheet renamed to
    ``.csv``, or a workbook posted as ``.json``, is refused with a sentence a
    person can act on instead of producing one unreadable column of mojibake.

    Args:
        content: The uploaded bytes.

    Returns:
        bool: True if the content begins with a known binary signature, or
        holds a NUL byte in its first kilobyte — text files do not.
    """
    if content.startswith(ZIP_MAGIC) or content.startswith(OLE2_MAGIC):
        return True
    return b"\x00" in content[:1024]


def validate_columns(header: list[str], settings: Settings) -> None:
    """Check a header for an unreasonable column count and for duplicates.

    Args:
        header: Column names exactly as the file wrote them, before any
            reader had a chance to rename a repeat.
        settings: Active application settings.

    Raises:
        DatasetTooLargeError: If the header exceeds ``max_dataset_columns``.
        DuplicateColumnsError: If any name appears more than once. Readers
            silently rename a repeat (``a``, ``a.1``), which would quietly
            change what the caller asked to model, so it is refused instead.
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
        name = str(cell).strip()
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)

    if duplicates:
        raise DuplicateColumnsError(
            "The header repeats column names: " + ", ".join(duplicates) + ".",
            details={"duplicate_columns": duplicates},
        )


def validate_frame(frame: pd.DataFrame, settings: Settings) -> None:
    """Check a parsed frame for emptiness, unusable headers and size limits.

    Args:
        frame: The parsed dataset.
        settings: Active application settings.

    Raises:
        EmptyDatasetError: If the frame has no columns or no rows.
        MissingHeaderError: If every column name is a reader placeholder,
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


__all__ = [
    "OLE2_MAGIC",
    "UNNAMED_COLUMN_PATTERN",
    "ZIP_MAGIC",
    "looks_binary",
    "validate_columns",
    "validate_frame",
]
