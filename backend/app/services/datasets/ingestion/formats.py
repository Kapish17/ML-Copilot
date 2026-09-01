"""The formats a dataset may arrive in, and how a file is recognised as one.

This module is the only place in the application that knows a file extension
exists. Everything downstream of ingestion — profiling, the experiment runner,
the ML layer, the agent — works on a standardised DataFrame and has no opinion
about where it came from.

Three formats are implemented: CSV, Excel (``.xlsx``) and JSON. Parquet, SQL,
databases, cloud storage and URL ingestion are **not implemented**; they would
be further adapters behind the same registry, and nothing outside this package
would change to accept them.
"""

from __future__ import annotations

from enum import Enum


class DatasetFormat(str, Enum):
    """A physical format a dataset can be uploaded in.

    The value is the short name reported to clients as ``source_format`` and
    recorded on an experiment, so it is deliberately a plain lowercase word
    rather than an extension or a media type.
    """

    CSV = "csv"
    XLSX = "xlsx"
    JSON = "json"

    @property
    def extension(self) -> str:
        """The canonical file extension for this format, with its dot."""
        return f".{self.value}"


#: Extension to format. The canonical detection route, because it is the one
#: piece of a filename that is a statement about content rather than about
#: where a file lived.
EXTENSIONS: dict[str, DatasetFormat] = {
    ".csv": DatasetFormat.CSV,
    ".xlsx": DatasetFormat.XLSX,
    ".json": DatasetFormat.JSON,
}

#: Media type to format, used only when the filename carries no usable
#: extension. Browsers and HTTP clients disagree about spreadsheet types, so
#: several spellings map to the same format. A media type is a *hint*: it is
#: supplied by the same untrusted client as the filename, so it never
#: overrides an extension and never substitutes for parsing the bytes.
MEDIA_TYPES: dict[str, DatasetFormat] = {
    "text/csv": DatasetFormat.CSV,
    "application/csv": DatasetFormat.CSV,
    "text/comma-separated-values": DatasetFormat.CSV,
    "application/vnd.ms-excel": DatasetFormat.XLSX,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": (
        DatasetFormat.XLSX
    ),
    "application/json": DatasetFormat.JSON,
    "text/json": DatasetFormat.JSON,
}


__all__ = ["DatasetFormat", "EXTENSIONS", "MEDIA_TYPES"]
