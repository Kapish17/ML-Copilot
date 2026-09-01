"""The contract every format adapter satisfies.

An adapter turns the bytes of one physical format into the **one** thing the
rest of the application understands: a standardised pandas DataFrame, plus a
few facts about where it came from.

::

    bytes + format  ->  adapter  ->  DataFrame + metadata

The interface is a structural :class:`typing.Protocol` rather than a base
class, so an adapter is any object with the right shape and no adapter is
forced to inherit anything. Four responsibilities are kept apart on purpose:

``detection``      which format these bytes claim to be (``detection.py``)
``validation``     is it safe and sane to try parsing them
``parsing``        the format-specific read
``normalisation``  the shared checks every parsed frame must pass

Only the third of those differs between formats. The first and last are shared,
which is what stops three adapters from becoming three subtly different
definitions of "an acceptable dataset".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from app.core.config import Settings
from app.services.datasets.ingestion.formats import DatasetFormat


@dataclass(frozen=True)
class DatasetMetadata:
    """The safe facts about an ingested dataset.

    **No rows and no values.** This describes the dataset; it never carries
    any of it. That matters because this object is rendered into API
    responses and, indirectly, into experiment records, and uploaded content
    must reach neither.

    ``original_filename`` is display text that has already been reduced to a
    bare name. Nothing anywhere resolves it to a location.
    """

    source_format: DatasetFormat
    row_count: int
    column_count: int
    original_filename: str

    def as_dict(self) -> dict[str, Any]:
        """Render the metadata as plain, JSON-safe values."""
        return {
            "source_format": self.source_format.value,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "original_filename": self.original_filename,
        }


@dataclass(frozen=True)
class IngestedDataset:
    """One parsed dataset and the facts about how it arrived."""

    frame: pd.DataFrame
    metadata: DatasetMetadata

    @property
    def source_format(self) -> DatasetFormat:
        """The format the bytes were read as."""
        return self.metadata.source_format


@runtime_checkable
class DatasetAdapter(Protocol):
    """Reads one physical format into a standardised DataFrame."""

    @property
    def format(self) -> DatasetFormat:  # pragma: no cover - protocol
        """The format this adapter reads."""
        ...

    @property
    def format_name(self) -> str:  # pragma: no cover - protocol
        """The format's short name, as reported to clients."""
        ...

    def can_handle(self, dataset_format: DatasetFormat) -> bool:  # pragma: no cover
        """Whether this adapter reads the given format."""
        ...

    def load(
        self, content: bytes, settings: Settings
    ) -> pd.DataFrame:  # pragma: no cover - protocol
        """Parse and validate raw bytes into a standardised DataFrame.

        Args:
            content: The uploaded bytes. Untrusted, and never executed.
            settings: Active application settings, supplying the shared
                row and column limits. An adapter never defines a limit of
                its own.

        Returns:
            pandas.DataFrame: A validated, non-empty dataset.

        Raises:
            DatasetError: If the bytes are not readable as this format, or
                the parsed dataset is empty or over a configured limit.
        """
        ...


class BaseDatasetAdapter:
    """Shared plumbing for the concrete adapters.

    Supplies the identity members so each adapter file contains its parsing
    and nothing else. Subclasses set ``format`` and implement ``load``.
    """

    #: The format this adapter reads. Overridden by every subclass.
    format: DatasetFormat

    @property
    def format_name(self) -> str:
        """The format's short name, as reported to clients."""
        return self.format.value

    def can_handle(self, dataset_format: DatasetFormat) -> bool:
        """Whether this adapter reads the given format."""
        return dataset_format is self.format

    def load(self, content: bytes, settings: Settings) -> pd.DataFrame:
        """Parse and validate raw bytes. Implemented by each adapter."""
        raise NotImplementedError  # pragma: no cover - abstract


__all__ = [
    "BaseDatasetAdapter",
    "DatasetAdapter",
    "DatasetMetadata",
    "IngestedDataset",
]
