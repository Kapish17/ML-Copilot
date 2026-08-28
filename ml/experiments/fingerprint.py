"""Identifying a dataset by its content rather than by where it came from.

An experiment record needs to answer "was this run on the same data?" months
later, when the file that produced it may have been renamed, moved, re-exported
from a database or converted from CSV to Parquet. A filename cannot answer
that; a hash of the content can.

**What goes into the fingerprint**

* the column names, in order;
* each column's dtype, as a string;
* the number of rows and columns;
* a content hash of every column's values, in column order.

**What deliberately does not**

* the filename, path or source format — the same table exported twice under
  different names fingerprints identically, which is the point;
* the time it was read, or anything about the machine;
* the index, which is an artefact of how the frame was built.

Row order *does* count. Two frames holding the same rows in a different order
fingerprint differently, because they produce different train/test splits and
so are not interchangeable for an experiment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

#: Hash algorithm used throughout. Recorded in the fingerprint so a future
#: change can be told apart from a data change.
FINGERPRINT_ALGORITHM = "sha256"
#: Characters of the digest kept. 16 hex characters is 64 bits, far more than
#: enough to tell apart the datasets one project will ever see, and short
#: enough to read in a directory listing.
FINGERPRINT_LENGTH = 16


@dataclass(frozen=True)
class DatasetFingerprint:
    """A content-derived identity for one dataset."""

    value: str
    algorithm: str
    row_count: int
    column_count: int
    columns: tuple[str, ...]
    dtypes: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        """Render the fingerprint as plain, JSON-friendly values."""
        return {
            "value": self.value,
            "algorithm": self.algorithm,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": list(self.columns),
            "dtypes": dict(self.dtypes),
        }


def _schema_bytes(frame: pd.DataFrame) -> bytes:
    """Encode the frame's shape and schema in a stable, ordered form."""
    schema = {
        "columns": [str(name) for name in frame.columns],
        "dtypes": [str(frame[name].dtype) for name in frame.columns],
        "row_count": int(frame.shape[0]),
        "column_count": int(frame.shape[1]),
    }
    return json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _column_bytes(column: pd.Series) -> bytes:
    """Hash one column's values into a compact, order-sensitive digest.

    ``hash_pandas_object`` gives a per-row hash that ignores the index, so a
    frame is fingerprinted by what it holds rather than by how it was indexed.
    """
    hashed = pd.util.hash_pandas_object(column, index=False)
    return hashed.to_numpy().tobytes()


def fingerprint_dataset(frame: pd.DataFrame) -> DatasetFingerprint:
    """Compute a deterministic content fingerprint for a dataset.

    Args:
        frame: The standardised dataset, whatever format it was read from.

    Returns:
        DatasetFingerprint: The digest plus the schema facts behind it.

    Raises:
        TypeError: If given something other than a DataFrame.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            "A dataset fingerprint is computed from a pandas DataFrame, not "
            f"{type(frame).__name__}."
        )

    digest = hashlib.new(FINGERPRINT_ALGORITHM)
    digest.update(_schema_bytes(frame))
    for name in frame.columns:
        digest.update(str(name).encode("utf-8"))
        digest.update(_column_bytes(frame[name]))

    return DatasetFingerprint(
        value=digest.hexdigest()[:FINGERPRINT_LENGTH],
        algorithm=FINGERPRINT_ALGORITHM,
        row_count=int(frame.shape[0]),
        column_count=int(frame.shape[1]),
        columns=tuple(str(name) for name in frame.columns),
        dtypes={str(name): str(frame[name].dtype) for name in frame.columns},
    )
