"""Conversion helpers shared by the dataset service.

pandas and numpy values are not directly JSON-serialisable: ``NaN``/``inf``
are invalid JSON and numpy scalars are not Python scalars. Every value that
leaves the service passes through one of these helpers.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

PERCENTAGE_PRECISION = 4


def safe_float(value: Any) -> float | None:
    """Convert a value to a finite Python float, or ``None``.

    Args:
        value: Any numeric-like value, possibly ``NaN``, ``inf`` or missing.

    Returns:
        float | None: A finite float, or ``None`` when the value cannot be
        represented in JSON.
    """
    if value is None or value is pd.NA:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def safe_int(value: Any) -> int | None:
    """Convert a value to a Python int, or ``None`` when not representable."""
    number = safe_float(value)
    return int(number) if number is not None else None


def percentage(part: float, whole: float) -> float:
    """Return ``part`` as a percentage of ``whole``, rounded and zero-safe.

    Args:
        part: Numerator.
        whole: Denominator; a value of zero yields ``0.0``.

    Returns:
        float: Percentage in the range 0-100.
    """
    if not whole:
        return 0.0
    return round((part / whole) * 100, PERCENTAGE_PRECISION)


def stringify(value: Any) -> str:
    """Render a cell value as a stable, human-readable string.

    Timestamps become ISO 8601, missing values become ``"NaN"`` and everything
    else falls back to ``str``. Used for value distributions, where a uniform
    string keeps the JSON contract simple.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NaN"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)
