"""Builders for the small in-memory datasets used across the test suite.

Everything the tests need is generated here, so no test depends on an external
file, a fixture dataset or network access.
"""

from __future__ import annotations

import io
from typing import Any

CSV_CONTENT_TYPE = "text/csv"


def build_csv(header: list[str], rows: list[list[Any]]) -> bytes:
    """Render a header and rows as UTF-8 CSV bytes.

    ``None`` becomes an empty field, which pandas reads as a missing value.
    """
    lines = [",".join(header)]
    lines.extend(
        ",".join("" if value is None else str(value) for value in row) for row in rows
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def sample_csv() -> bytes:
    """A small dataset touching every profiling path.

    It contains an identifier, numeric columns, a categorical column, a date
    column, a constant column, missing values and one duplicated row.
    """
    header = ["user_id", "age", "score", "city", "signup_date", "plan"]
    rows: list[list[Any]] = [
        [1, 34, 88.5, "Paris", "2023-01-05", "basic"],
        [2, None, 91.0, "Lyon", "2023-02-11", "basic"],
        [3, 29, 79.25, "Paris", "2023-03-02", "basic"],
        [4, 41, 64.0, "Nice", "2023-04-19", "basic"],
        [5, 37, 72.5, "Lyon", "2023-05-23", "basic"],
        [5, 37, 72.5, "Lyon", "2023-05-23", "basic"],
    ]
    return build_csv(header, rows)


def high_cardinality_csv(row_count: int = 60) -> bytes:
    """A dataset whose ``code`` column holds a distinct value in every row."""
    rows: list[list[Any]] = [[f"code-{index}", index % 3] for index in range(row_count)]
    return build_csv(["code", "bucket"], rows)


def upload_payload(content: bytes, filename: str = "dataset.csv") -> dict[str, Any]:
    """Build the ``files=`` argument for a multipart upload."""
    return {"file": (filename, io.BytesIO(content), CSV_CONTENT_TYPE)}


class FakeUpload:
    """Minimal stand-in for ``UploadFile`` with only an async ``read``."""

    def __init__(self, content: bytes) -> None:
        """Store the content the fake upload will stream."""
        self._buffer = io.BytesIO(content)

    async def read(self, size: int = -1) -> bytes:
        """Return up to ``size`` bytes, or everything when ``size`` is -1."""
        return self._buffer.read(size)
