"""The Excel adapter.

Reads an ``.xlsx`` workbook's **first worksheet** into a DataFrame. That
default is documented in every README rather than guessed at: a workbook is
not a table, so *something* has to choose, and choosing the first sheet is the
convention every spreadsheet export follows. A sheet selector is a future
option, not a silent behaviour.

**A workbook is treated as data and only as data.** ``openpyxl`` is opened in
the mode pandas uses for reading values, so a cell holding ``=WEBSERVICE(...)``
or ``=cmd|'/c calc'!A0`` arrives as the string it is or as the value last
cached by the writing application — it is never evaluated, never resolved, and
never reaches a shell, an HTTP client or the filesystem. Macros in a workbook
are not read at all; ``.xlsm`` is not an accepted extension.

Errors are translated at the boundary. Nothing the reader raises — no
traceback, no internal object, no temporary path — reaches a client; each
becomes one of the project's typed dataset errors with a message written for
somebody looking at their own spreadsheet.
"""

from __future__ import annotations

import io

import pandas as pd

from app.core.config import Settings
from app.core.errors import EmptyDatasetError, InvalidExcelError
from app.services.datasets.ingestion.base import BaseDatasetAdapter
from app.services.datasets.ingestion.formats import DatasetFormat
from app.services.datasets.ingestion.normalisation import (
    ZIP_MAGIC,
    validate_columns,
    validate_frame,
)

#: The reader used for the modern Office Open XML format. The only Excel
#: dependency this project has; no second spreadsheet library is involved.
EXCEL_ENGINE = "openpyxl"

#: Which sheet is read when the workbook holds several.
DEFAULT_SHEET_INDEX = 0


def _open_workbook(content: bytes) -> pd.ExcelFile:
    """Open uploaded bytes as a workbook, or say clearly that they are not one.

    An ``.xlsx`` file is a ZIP container, so bytes that do not begin with the
    ZIP signature are refused before the reader is asked to make sense of
    them. This is the check that makes "a file named ``data.xlsx`` holding
    something else must fail" true regardless of what the reader would have
    done with it.

    Args:
        content: The uploaded bytes.

    Returns:
        pandas.ExcelFile: The opened workbook.

    Raises:
        InvalidExcelError: If the bytes are not a readable ``.xlsx`` workbook.
    """
    if not content.startswith(ZIP_MAGIC):
        raise InvalidExcelError(
            "The file is not a valid .xlsx workbook. Save it as Excel "
            "Workbook (.xlsx), or upload it as CSV or JSON."
        )
    try:
        return pd.ExcelFile(io.BytesIO(content), engine=EXCEL_ENGINE)
    except Exception as exc:  # noqa: BLE001 - the reader's exceptions are open-ended
        raise InvalidExcelError(
            "The workbook could not be opened. It may be corrupted, "
            "password-protected, or saved in an older Excel format."
        ) from exc


def _first_sheet(workbook: pd.ExcelFile) -> str:
    """Return the name of the sheet that will be read.

    Args:
        workbook: The opened workbook.

    Returns:
        str: The first worksheet's name.

    Raises:
        InvalidExcelError: If the workbook contains no worksheet at all.
    """
    names = list(workbook.sheet_names)
    if not names:
        raise InvalidExcelError("The workbook contains no worksheets.")
    return str(names[DEFAULT_SHEET_INDEX])


def _read_header(workbook: pd.ExcelFile, sheet: str) -> list[str]:
    """Read the first row as written, before the reader can rename a repeat.

    pandas mangles duplicate column names into ``a``, ``a.1``. Reading the raw
    first row means Excel is held to the same duplicate rule as CSV instead of
    silently modelling a renamed column.

    Args:
        workbook: The opened workbook.
        sheet: The worksheet to read.

    Returns:
        list[str]: The header cells as text, with trailing blanks dropped.

    Raises:
        InvalidExcelError: If the sheet cannot be read.
    """
    try:
        first = workbook.parse(sheet_name=sheet, header=None, nrows=1)
    except Exception as exc:  # noqa: BLE001 - the reader's exceptions are open-ended
        raise InvalidExcelError(
            "The first worksheet could not be read."
        ) from exc

    if first.empty:
        return []

    cells = ["" if pd.isna(value) else str(value) for value in first.iloc[0].tolist()]
    while cells and not cells[-1].strip():
        cells.pop()
    return cells


class ExcelAdapter(BaseDatasetAdapter):
    """Reads the first worksheet of an ``.xlsx`` workbook."""

    format = DatasetFormat.XLSX

    def load(self, content: bytes, settings: Settings) -> pd.DataFrame:
        """Parse uploaded workbook bytes into a standardised DataFrame.

        Args:
            content: The uploaded bytes. Untrusted; treated as data only and
                never evaluated.
            settings: Active application settings, supplying the same row and
                column limits every other format is held to.

        Returns:
            pandas.DataFrame: A validated, non-empty dataset taken from the
            workbook's first worksheet.

        Raises:
            InvalidExcelError: If the bytes are not a readable workbook, or
                the first worksheet cannot be parsed.
            EmptyDatasetError: If the first worksheet holds no usable data.
            DuplicateColumnsError: If its header repeats a column name.
            MissingHeaderError: If its first row holds no column names.
            DatasetTooLargeError: If the sheet exceeds a configured limit.
        """
        workbook = _open_workbook(content)
        try:
            sheet = _first_sheet(workbook)
            header = _read_header(workbook, sheet)
            if not header or all(not cell.strip() for cell in header):
                raise EmptyDatasetError(
                    "The first worksheet is empty, so there is no data to read."
                )
            validate_columns(header, settings)

            try:
                frame = workbook.parse(sheet_name=sheet, header=0)
            except Exception as exc:  # noqa: BLE001 - open-ended reader errors
                raise InvalidExcelError(
                    "The first worksheet could not be read as a table."
                ) from exc
        finally:
            # The workbook holds an open handle on the in-memory buffer; the
            # frame is already independent of it by this point.
            workbook.close()

        validate_frame(frame, settings)
        return frame


__all__ = ["DEFAULT_SHEET_INDEX", "EXCEL_ENGINE", "ExcelAdapter"]
