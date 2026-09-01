"""The CSV adapter.

CSV was the only format for thirteen commits, and its behaviour — the
encodings it accepts, how it detects a missing header, what it calls a
duplicate column, which failures are 413 and which are 422 — is settled and
tested. So this adapter adds nothing to it. It is a wrapper around
:func:`app.services.datasets.loader.load_csv`, which is the one CSV
implementation in the project.

The point of the wrapper is uniformity: once CSV is an adapter like the other
two, the service can hold a registry instead of a special case, and no caller
needs to know which format it is dealing with.
"""

from __future__ import annotations

import pandas as pd

from app.core.config import Settings
from app.services.datasets.ingestion.base import BaseDatasetAdapter
from app.services.datasets.ingestion.formats import DatasetFormat
from app.services.datasets.loader import load_csv


class CSVAdapter(BaseDatasetAdapter):
    """Reads comma-separated text into a standardised DataFrame."""

    format = DatasetFormat.CSV

    def load(self, content: bytes, settings: Settings) -> pd.DataFrame:
        """Parse uploaded CSV bytes.

        Args:
            content: The uploaded bytes. Untrusted, and never executed.
            settings: Active application settings.

        Returns:
            pandas.DataFrame: A validated, non-empty dataset.

        Raises:
            DatasetError: On an undecodable file, a missing or duplicated
                header, inconsistent rows, an empty dataset, or a dataset
                over a configured limit.
        """
        return load_csv(content, settings)


__all__ = ["CSVAdapter"]
