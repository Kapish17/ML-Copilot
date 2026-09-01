"""Builders for the small in-memory datasets used across the test suite.

Everything the tests need is generated here, so no test depends on an external
file, a fixture dataset or network access.
"""

from __future__ import annotations

import io
import json
from typing import Any

import pandas as pd

CSV_CONTENT_TYPE = "text/csv"
XLSX_CONTENT_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
JSON_CONTENT_TYPE = "application/json"

#: Extension to the media type a browser would send for it, so an upload in a
#: test carries the same headers a real client's would.
CONTENT_TYPES: dict[str, str] = {
    ".csv": CSV_CONTENT_TYPE,
    ".xlsx": XLSX_CONTENT_TYPE,
    ".json": JSON_CONTENT_TYPE,
}


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


def learnable_classification_csv(rows: int = 240) -> bytes:
    """A binary target a model can genuinely learn, built deterministically.

    ``income`` and ``tenure_months`` carry the signal and ``segment`` is a
    categorical feature with a mild association, so cross-validation has
    something real to separate. Roughly one label in eleven is flipped, which
    keeps the problem learnable without being perfectly separable — a model
    that scores 1.00 on synthetic data tells you nothing about whether the
    pipeline works.

    No randomness and no external file: the same bytes come out on every
    machine, which is what makes the dataset's content fingerprint stable.
    """
    header = ["income", "tenure_months", "segment", "renewed"]
    body: list[list[Any]] = []
    for index in range(rows):
        high = index % 2 == 0
        income = 30_000 + (index % 40) * 400 + (12_000 if high else 0)
        tenure = 4 + (index % 24) + (18 if high else 0)
        segment = "business" if index % 3 == 0 else "retail"
        label = high if index % 11 else not high
        body.append([income, tenure, segment, "yes" if label else "no"])
    return build_csv(header, body)


def regression_csv(rows: int = 200) -> bytes:
    """A continuous target with a real linear relationship to its features.

    The price is a float with a distinct value in every row, so the target is
    unambiguously continuous rather than a discrete code that happens to be
    stored as a number.
    """
    header = ["size_sqm", "rooms", "district", "price"]
    body: list[list[Any]] = []
    for index in range(rows):
        size = 45.0 + (index % 60) * 2.5 + index * 0.01
        room_count = 1 + (index % 4)
        district = ["north", "south", "centre"][index % 3]
        price = round(1_500 * size + 9_000 * room_count + 37.5 * index, 2)
        body.append([size, room_count, district, price])
    return build_csv(header, body)


def experiment_form(**fields: Any) -> dict[str, Any]:
    """Render experiment options as multipart form fields.

    Sequences stay lists, which the HTTP client sends as a repeated key, and
    booleans become ``"true"``/``"false"`` — exactly how a browser or ``curl``
    would submit them.
    """
    encoded: dict[str, Any] = {}
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, bool):
            encoded[key] = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            encoded[key] = [str(item) for item in value]
        else:
            encoded[key] = str(value)
    return encoded


def upload_payload(
    content: bytes,
    filename: str = "dataset.csv",
    content_type: str | None = None,
) -> dict[str, Any]:
    """Build the ``files=`` argument for a multipart upload.

    The media type is derived from the filename unless one is given, so a
    test that uploads a workbook sends the header a browser would and a test
    about mislabelling can state the mismatch explicitly.
    """
    if content_type is None:
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        content_type = CONTENT_TYPES.get(suffix, CSV_CONTENT_TYPE)
    return {"file": (filename, io.BytesIO(content), content_type)}


def frame_from_csv(content: bytes) -> pd.DataFrame:
    """Parse CSV bytes the way the application would, for re-encoding."""
    return pd.read_csv(io.BytesIO(content))


def build_xlsx(header: list[str], rows: list[list[Any]]) -> bytes:
    """Render a header and rows as a single-worksheet ``.xlsx`` workbook."""
    return frame_to_xlsx(pd.DataFrame(rows, columns=header))


def frame_to_xlsx(frame: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """Render a DataFrame as a one-sheet workbook."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name)
    return buffer.getvalue()


def multi_sheet_xlsx(sheets: dict[str, pd.DataFrame]) -> bytes:
    """Render several named worksheets into one workbook, in order."""
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, index=False, sheet_name=name)
    return buffer.getvalue()


def build_json(header: list[str], rows: list[list[Any]]) -> bytes:
    """Render a header and rows as a JSON array of objects."""
    records = [dict(zip(header, row)) for row in rows]
    return json.dumps(records).encode("utf-8")


def frame_to_json(frame: pd.DataFrame, envelope: str | None = None) -> bytes:
    """Render a DataFrame as JSON records, optionally under an envelope key.

    ``NaN`` becomes ``null``, which is what a JSON exporter writes and what
    the adapter reads back as a missing value.
    """
    records = json.loads(frame.to_json(orient="records"))
    document: Any = records if envelope is None else {envelope: records}
    return json.dumps(document).encode("utf-8")


def csv_as_xlsx(content: bytes) -> bytes:
    """Re-express CSV bytes as an equivalent one-sheet workbook."""
    return frame_to_xlsx(frame_from_csv(content))


def csv_as_json(content: bytes, envelope: str | None = None) -> bytes:
    """Re-express CSV bytes as equivalent JSON records."""
    return frame_to_json(frame_from_csv(content), envelope=envelope)


class FakeUpload:
    """Minimal stand-in for ``UploadFile`` with only an async ``read``."""

    def __init__(self, content: bytes) -> None:
        """Store the content the fake upload will stream."""
        self._buffer = io.BytesIO(content)

    async def read(self, size: int = -1) -> bytes:
        """Return up to ``size`` bytes, or everything when ``size`` is -1."""
        return self._buffer.read(size)
