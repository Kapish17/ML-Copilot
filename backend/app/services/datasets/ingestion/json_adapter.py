"""The JSON adapter.

JSON is not a tabular format, so this adapter has to answer a question the
other two do not: *is this document a table at all?* It answers narrowly and
refuses the rest, because a silent guess about structure is worse than an
error a person can read.

Two shapes are accepted.

**An array of objects** — the shape every ``to_json(orient="records")`` and
every REST list endpoint produces::

    [{"age": 20, "income": 50000}, {"age": 30, "income": 70000}]

**An object holding one array of records** — the same thing under an envelope,
which is how most APIs actually return it::

    {"rows": [{"age": 20}, {"age": 30}]}

When several keys hold record arrays the document is ambiguous, and rather
than pick one it is refused — unless exactly one of the conventional envelope
names is present, in which case that is not a guess.

Everything else is rejected: a scalar, an empty document, an array of numbers,
an array of mixed types, a record whose field holds a list. **Nothing is
executed.** ``json.loads`` builds plain Python values from text; there is no
evaluation step, no object hook and no custom decoder, so a string that reads
like code is a string.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.core.config import Settings
from app.core.errors import EmptyDatasetError, InvalidJSONError
from app.services.datasets.ingestion.base import BaseDatasetAdapter
from app.services.datasets.ingestion.formats import DatasetFormat
from app.services.datasets.ingestion.normalisation import (
    looks_binary,
    validate_columns,
    validate_frame,
)
from app.services.datasets.loader import decode_text

#: Envelope keys that conventionally hold the records. Consulted only to break
#: a tie between several candidate arrays, never to prefer one over a document
#: that has exactly one candidate anyway.
RECORD_KEYS: tuple[str, ...] = ("data", "records", "rows", "items", "results")

#: Separator used when a nested object is flattened into columns.
NESTED_SEPARATOR = "."


def _decode(content: bytes) -> Any:
    """Turn uploaded bytes into plain Python values.

    Args:
        content: The uploaded bytes.

    Returns:
        Any: Whatever the document holds — a list, a dict or a scalar.

    Raises:
        InvalidJSONError: If the bytes are binary, undecodable as text, or
            not valid JSON.
    """
    if looks_binary(content):
        raise InvalidJSONError(
            "The file is not text, so it cannot be read as JSON."
        )
    text = decode_text(
        content, lambda message: InvalidJSONError(f"{message} JSON must be text.")
    )
    if not text.strip():
        raise EmptyDatasetError("The file contains no data to profile.")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidJSONError(
            f"The file is not valid JSON: {exc.msg} at line {exc.lineno}, "
            f"column {exc.colno}."
        ) from exc


def _is_record_list(value: Any) -> bool:
    """Whether a value is a non-empty list of objects."""
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) for item in value)
    )


def _records_from_list(document: list[Any]) -> list[dict[str, Any]]:
    """Interpret a top-level array as a list of records.

    Raises:
        EmptyDatasetError: If the array is empty.
        InvalidJSONError: If any element is not an object.
    """
    if not document:
        raise EmptyDatasetError("The JSON array is empty, so there are no rows.")
    if not all(isinstance(item, dict) for item in document):
        raise InvalidJSONError(
            "A JSON array must hold objects, one per row, for example "
            '[{"age": 20}, {"age": 30}]. This array holds other values.'
        )
    return list(document)


def _records_from_object(document: dict[str, Any]) -> list[dict[str, Any]]:
    """Interpret a top-level object as records, or as one record.

    Raises:
        EmptyDatasetError: If the object is empty.
        InvalidJSONError: If it is neither an envelope around one record
            array nor a single flat record.
    """
    if not document:
        raise EmptyDatasetError("The JSON object is empty, so there are no rows.")

    candidates = [key for key, value in document.items() if _is_record_list(value)]
    if len(candidates) == 1:
        return list(document[candidates[0]])
    if len(candidates) > 1:
        preferred = [key for key in RECORD_KEYS if key in candidates]
        if len(preferred) == 1:
            return list(document[preferred[0]])
        raise InvalidJSONError(
            "The JSON object holds more than one list of records ("
            + ", ".join(sorted(candidates))
            + "), so it is not clear which one is the dataset. Upload the "
            "records as a top-level array instead."
        )

    if all(not isinstance(value, (list, dict)) for value in document.values()):
        return [document]

    raise InvalidJSONError(
        "The JSON object is not tabular. Upload an array of objects, one per "
        'row, or an object holding one such array, for example {"rows": [...]}.'
    )


def extract_records(document: Any) -> list[dict[str, Any]]:
    """Reduce a parsed JSON document to a list of records.

    Args:
        document: Whatever ``json.loads`` produced.

    Returns:
        list[dict]: One dictionary per row.

    Raises:
        EmptyDatasetError: If the document holds no rows.
        InvalidJSONError: If the document cannot reasonably become a table.
    """
    if isinstance(document, list):
        return _records_from_list(document)
    if isinstance(document, dict):
        return _records_from_object(document)
    raise InvalidJSONError(
        "A single JSON value is not a dataset. Upload an array of objects, "
        'one per row, for example [{"age": 20}, {"age": 30}].'
    )


def _reject_nested_values(frame: pd.DataFrame) -> None:
    """Refuse a frame whose cells still hold lists or objects.

    ``json_normalize`` flattens nested *objects* into dotted column names, but
    a nested *array* has no tabular meaning — a cell holding ``[1, 2, 3]`` is
    not a value the profiler, the preprocessing pipeline or a model can use,
    and quietly stringifying it would change the caller's data.

    Raises:
        InvalidJSONError: If any cell holds a list or a dict.
    """
    nested: list[str] = []
    for name in frame.columns:
        column = frame[name]
        if column.dtype != object:
            continue
        if column.map(lambda value: isinstance(value, (list, dict))).any():
            nested.append(str(name))

    if nested:
        raise InvalidJSONError(
            "These fields hold nested arrays or objects that cannot become "
            "table cells: " + ", ".join(nested[:10]) + ". Flatten them before "
            "uploading, or upload the data as CSV.",
            details={"nested_fields": nested[:10]},
        )


class JSONAdapter(BaseDatasetAdapter):
    """Reads tabular JSON into a standardised DataFrame."""

    format = DatasetFormat.JSON

    def load(self, content: bytes, settings: Settings) -> pd.DataFrame:
        """Parse uploaded JSON bytes into a standardised DataFrame.

        Args:
            content: The uploaded bytes. Untrusted, parsed as data, never
                executed.
            settings: Active application settings, supplying the same row and
                column limits every other format is held to.

        Returns:
            pandas.DataFrame: A validated, non-empty dataset.

        Raises:
            InvalidJSONError: If the bytes are not JSON, or are JSON that
                cannot reasonably become a table.
            EmptyDatasetError: If the document holds no rows.
            DuplicateColumnsError: If two records flatten to the same column.
            DatasetTooLargeError: If the result exceeds a configured limit.
        """
        records = extract_records(_decode(content))

        try:
            frame = pd.json_normalize(records, sep=NESTED_SEPARATOR)
        except Exception as exc:  # noqa: BLE001 - normalisation errors are open-ended
            raise InvalidJSONError(
                "The JSON records could not be flattened into a table. "
                "Their fields are not consistent enough to form columns."
            ) from exc

        validate_columns([str(name) for name in frame.columns], settings)
        _reject_nested_values(frame)
        validate_frame(frame, settings)
        return frame


__all__ = ["NESTED_SEPARATOR", "RECORD_KEYS", "JSONAdapter", "extract_records"]
