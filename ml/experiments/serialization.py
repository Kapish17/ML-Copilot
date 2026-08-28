"""Turning ML objects into something JSON can hold.

``json.dumps(obj.__dict__)`` fails on almost everything this project produces:
numpy scalars, enums, timestamps, ``NaN``, nested dataclasses. So conversion is
explicit, and it has opinions.

Two of those opinions matter.

**Some things must never be stored.** A fitted sklearn pipeline or a SHAP
explainer is not experiment history — it is a large binary object whose text
form says nothing useful. Rather than quietly writing ``str(pipeline)`` into a
record, conversion refuses, so the mistake surfaces where it is made instead of
in a bloated file nobody reads.

**Invalid JSON must not be written.** ``NaN`` and infinity are not JSON, even
though Python's encoder emits them by default. Non-finite numbers become
``null`` during conversion, and the writer is configured to raise if one gets
through anyway.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence, Set
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from ml.errors import MalformedExperimentError, SerializationError

#: Objects from these libraries are never stored: they are model artefacts, not
#: experiment history.
FORBIDDEN_MODULE_PREFIXES = ("sklearn", "shap")
#: Largest array or sequence an experiment record may hold. Confusion matrices
#: and feature lists are small; a transformed feature matrix is not.
MAX_SEQUENCE_LENGTH = 10_000
#: Guard against a self-referencing structure.
MAX_DEPTH = 32


def _module_root(value: Any) -> str:
    """Return the top-level module a value's class comes from."""
    return type(value).__module__.split(".")[0]


def _reject_forbidden(value: Any) -> None:
    """Refuse to store a model artefact.

    Raises:
        SerializationError: If the value comes from a blocked library.
    """
    root = _module_root(value)
    if root in FORBIDDEN_MODULE_PREFIXES:
        raise SerializationError(
            f"{type(value).__name__} objects are not stored in experiment "
            "records. Persist the metadata and results, not the fitted "
            "artefact.",
            details={"type": type(value).__name__, "module": root},
        )


def _finite(value: float) -> float | None:
    """Return a finite float, or ``None`` — ``NaN`` is not valid JSON."""
    return value if math.isfinite(value) else None


def _sequence(values: Sequence[Any] | Set[Any], depth: int) -> list[Any]:
    """Convert a sequence, refusing one large enough to be model data."""
    items = list(values)
    if len(items) > MAX_SEQUENCE_LENGTH:
        raise SerializationError(
            f"A sequence of {len(items)} values is too large for an experiment "
            f"record (limit {MAX_SEQUENCE_LENGTH}). Store a summary instead.",
            details={"length": len(items), "limit": MAX_SEQUENCE_LENGTH},
        )
    return [to_jsonable(item, depth=depth + 1) for item in items]


def to_jsonable(value: Any, *, depth: int = 0) -> Any:
    """Convert any value into something ``json.dumps`` can write.

    Args:
        value: The value to convert.
        depth: Recursion depth, used only to catch cyclic structures.

    Returns:
        A structure of dicts, lists, strings, numbers, booleans and ``None``.

    Raises:
        SerializationError: If the value is a model artefact, is too large, or
            nests more deeply than :data:`MAX_DEPTH`.
    """
    if depth > MAX_DEPTH:
        raise SerializationError(
            f"Nesting deeper than {MAX_DEPTH} levels; the value is probably "
            "cyclic."
        )

    if value is None or value is pd.NaT:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return _finite(value)
    if isinstance(value, str):
        return value

    _reject_forbidden(value)

    if isinstance(value, Enum):
        return to_jsonable(value.value, depth=depth + 1)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return to_jsonable(value.item(), depth=depth + 1)
    if isinstance(value, np.ndarray):
        return _sequence(value.tolist(), depth)
    if isinstance(value, pd.DataFrame):
        raise SerializationError(
            "DataFrames are not stored in experiment records. Persist the "
            "shape, the column names and the results instead.",
            details={"shape": list(value.shape)},
        )
    if isinstance(value, (pd.Series, pd.Index)):
        return _sequence(value.tolist(), depth)

    if hasattr(value, "as_dict") and callable(value.as_dict):
        return to_jsonable(value.as_dict(), depth=depth + 1)
    if hasattr(value, "to_dict") and callable(value.to_dict) and is_dataclass(value):
        return to_jsonable(value.to_dict(), depth=depth + 1)
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value), depth=depth + 1)

    if isinstance(value, Mapping):
        return {
            str(key): to_jsonable(item, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return _sequence(value, depth)

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return str(value)


def json_dumps(payload: Any) -> str:
    """Render a converted payload as readable, strictly valid JSON.

    ``allow_nan=False`` is the safety net: if a non-finite number survived
    conversion, writing fails loudly rather than producing a file that no other
    JSON reader will accept.

    Raises:
        SerializationError: If the payload cannot be written as valid JSON.
    """
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SerializationError(f"The record could not be written as JSON: {exc}") from exc


def json_loads(text: str) -> Any:
    """Parse stored JSON, reporting corruption clearly.

    Raises:
        MalformedExperimentError: If the text is not valid JSON.
    """
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise MalformedExperimentError(
            f"The stored record is not valid JSON: {exc}"
        ) from exc
