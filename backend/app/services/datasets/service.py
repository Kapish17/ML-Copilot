"""Orchestration of the dataset ingestion and profiling workflow.

The service is the only entry point route handlers use. It sequences
validation, loading, profiling, quality analysis and target analysis, and
returns a fully built response model. It holds no mutable state: everything it
needs comes from the settings passed at construction.

It is also the single ingestion entry point. The experiment runner and the
agent need the same validated bytes turned into the same standardised
DataFrame, so the two steps are exposed separately — :meth:`load_upload`
produces the DataFrame, :meth:`profile_frame` profiles one — and no caller
anywhere re-implements file validation or parsing::

    upload -> validate -> detect format -> adapter -> DataFrame -> profile
                                                          |
                                                          +-> experiment runner
                                                          +-> agent

Three formats are implemented — CSV, Excel (``.xlsx``) and JSON — and the
service does not read any of them itself. Each is read by an adapter in
:mod:`app.services.datasets.ingestion`, and everything from the DataFrame
rightwards is format-agnostic: :meth:`profile_frame` and every caller of it
cannot tell which adapter produced the data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from app.core.config import Settings
from app.core.errors import DatasetError
from app.schemas.dataset import DatasetProfileResponse, TargetProfile
from app.services.datasets.ingestion import (
    DatasetAdapterRegistry,
    DatasetFormat,
    default_registry,
    detect_format,
    supported_formats as permitted_formats,
)
from app.services.datasets.profiler import profile_columns, summarise_dataset
from app.services.datasets.quality import analyse_quality
from app.services.datasets.target import analyse_target
from app.services.datasets.validation import (
    AsyncReadable,
    declared_content_type,
    read_upload,
    safe_filename,
    validate_size,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedDataset:
    """A validated dataset in its standardised in-memory form.

    This is where file formats stop. Whatever the bytes were, what comes out
    is a DataFrame plus two facts about its origin, and no caller downstream
    can tell CSV from a spreadsheet from JSON except by reading ``format``,
    which exists to be reported and recorded — never to branch modelling
    behaviour on.

    ``filename`` is kept only for display and is already path-free; nothing
    downstream may use it to reach the filesystem.
    """

    frame: pd.DataFrame
    filename: str
    format: DatasetFormat = DatasetFormat.CSV

    @property
    def source_format(self) -> str:
        """The format's short name — ``"csv"``, ``"xlsx"`` or ``"json"``."""
        return self.format.value


def _normalise_target(target_column: str | None) -> str | None:
    """Treat blank or whitespace-only target names as 'not provided'."""
    if target_column is None:
        return None
    stripped = target_column.strip()
    return stripped or None


class DatasetProfilingService:
    """Validate an uploaded dataset and produce its profile."""

    def __init__(
        self, settings: Settings, adapters: DatasetAdapterRegistry | None = None
    ) -> None:
        """Bind the service to a settings instance and a set of adapters.

        Args:
            settings: Active application settings, supplying every limit and
                heuristic threshold used downstream.
            adapters: The formats this service can read. Defaults to the
                application's registry — CSV, Excel and JSON. Injectable so a
                test can narrow or widen the set without touching a global.
        """
        self._settings = settings
        self._adapters = adapters or default_registry()

    @property
    def adapters(self) -> DatasetAdapterRegistry:
        """The adapter registry this service reads formats through."""
        return self._adapters

    def supported_formats(self) -> tuple[str, ...]:
        """The format names this service accepts, e.g. ``("csv", "xlsx")``."""
        return tuple(
            fmt.value
            for fmt in permitted_formats(self._settings)
            if self._adapters.supports(fmt)
        )

    # -- Ingestion ---------------------------------------------------------

    async def load_upload(
        self,
        upload: AsyncReadable,
        filename: str | None,
        content_type: str | None = None,
    ) -> LoadedDataset:
        """Validate an in-flight upload and parse it into a DataFrame.

        The format is settled before any bytes are read, so an unsupported
        file is rejected without being buffered, and the bytes are never
        written anywhere: the upload lives in memory for the length of the
        request only. Detection only chooses *which* adapter tries the bytes;
        whether they really are that format is the adapter's decision, made on
        the content.

        Args:
            upload: The incoming file object.
            filename: Client-supplied filename, used only for its extension.
            content_type: The client's declared media type. Read from the
                upload itself when not given, and consulted only when the
                filename carries no usable extension.

        Returns:
            LoadedDataset: The parsed frame, the path-free filename and the
            format it was read as.

        Raises:
            DatasetError: If the upload or its content fails validation.
        """
        try:
            name = safe_filename(filename)
            detected = detect_format(
                name, self._settings, content_type or declared_content_type(upload)
            )
            content = await read_upload(upload, self._settings)
        except DatasetError as exc:
            logger.info("Rejected an upload before parsing: %s", exc.code)
            raise
        return self._accept(content, detected, name)

    def load_content(
        self,
        filename: str | None,
        content: bytes,
        content_type: str | None = None,
    ) -> LoadedDataset:
        """Validate dataset bytes already in memory and parse them.

        Args:
            filename: Name the content was uploaded under.
            content: Raw file bytes.
            content_type: The client's declared media type, if any.

        Returns:
            LoadedDataset: The parsed frame, the path-free filename and the
            format it was read as.

        Raises:
            DatasetError: If the content fails validation.
        """
        try:
            name = safe_filename(filename)
            detected = detect_format(name, self._settings, content_type)
            validate_size(len(content), self._settings)
        except DatasetError as exc:
            logger.info("Rejected an upload before parsing: %s", exc.code)
            raise
        return self._accept(content, detected, name)

    def _accept(
        self, content: bytes, dataset_format: DatasetFormat, name: str
    ) -> LoadedDataset:
        """Parse validated bytes, recording the outcome either way.

        This is the one place an upload becomes a DataFrame, so it is the right
        place to say in the log whether that worked. Both lines carry the
        format and, on failure, the stable error code — enough to answer "are
        people's Excel files failing?" from a log alone.

        **What is deliberately absent: the filename.** It is chosen by whoever
        made the request, and text a caller chooses does not belong in a line
        an operator reads as though the server wrote it. The shape of the data
        is the useful part, and it is the server's own measurement.

        Args:
            content: The validated bytes.
            dataset_format: The format detection settled on.
            name: Path-free filename, carried into the dataset's metadata.

        Returns:
            LoadedDataset: The parsed frame and the facts about its origin.

        Raises:
            DatasetError: If the adapter rejects the bytes.
        """
        try:
            ingested = self._adapters.load(
                content, dataset_format, self._settings, filename=name
            )
        except DatasetError as exc:
            logger.info(
                "Rejected a %s upload of %d bytes: %s",
                dataset_format.value,
                len(content),
                exc.code,
            )
            raise

        rows, columns = ingested.frame.shape
        logger.info(
            "Ingested a %s dataset: %d rows x %d columns",
            ingested.source_format.value,
            rows,
            columns,
        )
        return LoadedDataset(
            frame=ingested.frame, filename=name, format=ingested.source_format
        )

    # -- Profiling ---------------------------------------------------------

    async def profile_upload(
        self,
        upload: AsyncReadable,
        filename: str | None,
        target_column: str | None = None,
        content_type: str | None = None,
    ) -> DatasetProfileResponse:
        """Profile an in-flight upload.

        Args:
            upload: The incoming file object.
            filename: Client-supplied filename, used only for its extension.
            target_column: Optional target column to analyse.

        Returns:
            DatasetProfileResponse: The complete dataset profile.

        Raises:
            DatasetError: If the upload or its content fails validation.
        """
        loaded = await self.load_upload(upload, filename, content_type)
        return self.profile_frame(
            loaded.frame,
            filename=loaded.filename,
            target_column=_normalise_target(target_column),
            source_format=loaded.source_format,
        )

    def profile_content(
        self,
        filename: str | None,
        content: bytes,
        target_column: str | None = None,
        content_type: str | None = None,
    ) -> DatasetProfileResponse:
        """Profile dataset bytes that are already in memory.

        Args:
            filename: Name the content was uploaded under.
            content: Raw file bytes.
            target_column: Optional target column to analyse.

        Returns:
            DatasetProfileResponse: The complete dataset profile.

        Raises:
            DatasetError: If the content fails validation.
        """
        loaded = self.load_content(filename, content, content_type)
        return self.profile_frame(
            loaded.frame,
            filename=loaded.filename,
            target_column=_normalise_target(target_column),
            source_format=loaded.source_format,
        )

    def profile_frame(
        self,
        frame: pd.DataFrame,
        *,
        filename: str = "dataset",
        target_column: str | None = None,
        source_format: str | None = None,
    ) -> DatasetProfileResponse:
        """Profile an already-standardised DataFrame.

        This is the format-agnostic entry point, and the boundary the whole
        ingestion layer exists to create: it knows nothing about files, takes
        the same argument whether the data came from CSV, a spreadsheet or
        JSON, and behaves identically in every case. The experiment runner
        uses it so that one profiling implementation serves both the profiling
        endpoint and experiment execution.

        Args:
            frame: A standardised dataset.
            filename: Label for the response; never used as a path.
            target_column: Optional target column to analyse.
            source_format: Where the data came from, reported back as
                context. It is carried, never branched on — no statistic,
                threshold or decision below depends on it.

        Returns:
            DatasetProfileResponse: The complete dataset profile.
        """
        columns = profile_columns(frame, self._settings)
        summary = summarise_dataset(frame, columns)

        target: TargetProfile | None = None
        if target_column is not None:
            target = analyse_target(frame, columns, target_column, self._settings)

        quality = analyse_quality(
            frame, columns, self._settings, target_column=target.name if target else None
        )

        return DatasetProfileResponse(
            filename=filename,
            source_format=source_format,
            generated_at=datetime.now(timezone.utc),
            dataset=summary,
            columns=columns,
            quality=quality,
            target=target,
        )
