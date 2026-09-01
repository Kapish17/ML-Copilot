"""The registry that maps a detected format to the adapter that reads it.

An explicit allowlist, in the same spirit as the agent's tool registry: a
format is readable because an adapter for it was registered here, not because
a filename happened to end in something. There is no dynamic lookup by name,
no import by string and no plugin discovery, so the set of readable formats is
exactly what this file lists and can be read off it.

Adding Parquet, a SQL source or a cloud object store later is one adapter plus
one line here. **None of those are implemented.**
"""

from __future__ import annotations

import pandas as pd

from app.core.config import Settings
from app.core.errors import UnsupportedFileTypeError
from app.services.datasets.ingestion.base import (
    DatasetAdapter,
    DatasetMetadata,
    IngestedDataset,
)
from app.services.datasets.ingestion.csv_adapter import CSVAdapter
from app.services.datasets.ingestion.excel_adapter import ExcelAdapter
from app.services.datasets.ingestion.formats import DatasetFormat
from app.services.datasets.ingestion.json_adapter import JSONAdapter


class DatasetAdapterRegistry:
    """The adapters this application can read a dataset with."""

    def __init__(self, adapters: list[DatasetAdapter] | None = None) -> None:
        """Build a registry over the given adapters.

        Args:
            adapters: Adapters to register, in order. Defaults to none, so a
                caller must register deliberately; :func:`default_registry`
                builds the one the application uses.
        """
        self._adapters: dict[DatasetFormat, DatasetAdapter] = {}
        for adapter in adapters or []:
            self.register(adapter)

    def register(self, adapter: DatasetAdapter) -> DatasetAdapterRegistry:
        """Register one adapter, replacing any adapter for the same format.

        Args:
            adapter: The adapter to add.

        Returns:
            DatasetAdapterRegistry: This registry, for chaining.
        """
        self._adapters[adapter.format] = adapter
        return self

    def formats(self) -> tuple[DatasetFormat, ...]:
        """Every registered format, in registration order."""
        return tuple(self._adapters)

    def format_names(self) -> tuple[str, ...]:
        """Every registered format's short name."""
        return tuple(fmt.value for fmt in self._adapters)

    def extensions(self) -> tuple[str, ...]:
        """Every registered format's canonical file extension."""
        return tuple(fmt.extension for fmt in self._adapters)

    def supports(self, dataset_format: DatasetFormat) -> bool:
        """Whether some registered adapter reads this format."""
        return dataset_format in self._adapters

    def adapter_for(self, dataset_format: DatasetFormat) -> DatasetAdapter:
        """Return the adapter that reads a format.

        Args:
            dataset_format: The detected format.

        Returns:
            DatasetAdapter: The adapter registered for it.

        Raises:
            UnsupportedFileTypeError: If no adapter is registered. Detection
                normally prevents this, so it means the settings allowlist and
                the registry disagree — reported as the same 415 a client
                would get for any unreadable format rather than as a crash.
        """
        adapter = self._adapters.get(dataset_format)
        if adapter is None:
            raise UnsupportedFileTypeError(
                f"No reader is available for {dataset_format.value} files. "
                "Supported types: " + ", ".join(self.extensions()) + ".",
                details={"supported_formats": list(self.format_names())},
            )
        return adapter

    def load(
        self,
        content: bytes,
        dataset_format: DatasetFormat,
        settings: Settings,
        *,
        filename: str = "upload",
    ) -> IngestedDataset:
        """Read bytes of a known format into a standardised dataset.

        Args:
            content: The uploaded bytes.
            dataset_format: The format detection settled on.
            settings: Active application settings.
            filename: Path-free display name, carried into the metadata.

        Returns:
            IngestedDataset: The parsed frame and the safe facts about it.

        Raises:
            DatasetError: If no adapter reads the format, or the adapter
                rejects the bytes.
        """
        frame: pd.DataFrame = self.adapter_for(dataset_format).load(content, settings)
        rows, columns = frame.shape
        return IngestedDataset(
            frame=frame,
            metadata=DatasetMetadata(
                source_format=dataset_format,
                row_count=int(rows),
                column_count=int(columns),
                original_filename=filename,
            ),
        )


def default_registry() -> DatasetAdapterRegistry:
    """Build the registry the application uses.

    Returns:
        DatasetAdapterRegistry: A registry holding the CSV, Excel and JSON
        adapters, and nothing else.
    """
    return DatasetAdapterRegistry([CSVAdapter(), ExcelAdapter(), JSONAdapter()])


__all__ = ["DatasetAdapterRegistry", "default_registry"]
