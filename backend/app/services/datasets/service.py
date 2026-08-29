"""Orchestration of the dataset ingestion and profiling workflow.

The service is the only entry point route handlers use. It sequences
validation, loading, profiling, quality analysis and target analysis, and
returns a fully built response model. It holds no mutable state: everything it
needs comes from the settings passed at construction.

It is also the single ingestion adapter. The experiment runner needs the same
validated bytes turned into the same standardised DataFrame, so the two steps
are exposed separately — :meth:`load_upload` produces the DataFrame,
:meth:`profile_frame` profiles one — and no caller anywhere re-implements file
validation or CSV parsing::

    upload -> validate -> DataFrame -> profile
                             |
                             +-> experiment runner

CSV is the only implemented format. Another format means another loader behind
:meth:`load_content`; nothing downstream of the DataFrame changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from app.core.config import Settings
from app.schemas.dataset import DatasetProfileResponse, TargetProfile
from app.services.datasets.loader import load_csv
from app.services.datasets.profiler import profile_columns, summarise_dataset
from app.services.datasets.quality import analyse_quality
from app.services.datasets.target import analyse_target
from app.services.datasets.validation import (
    AsyncReadable,
    read_upload,
    safe_filename,
    validate_extension,
    validate_size,
)


@dataclass(frozen=True)
class LoadedDataset:
    """A validated dataset in its standardised in-memory form.

    ``filename`` is kept only for display and is already path-free; nothing
    downstream may use it to reach the filesystem.
    """

    frame: pd.DataFrame
    filename: str


def _normalise_target(target_column: str | None) -> str | None:
    """Treat blank or whitespace-only target names as 'not provided'."""
    if target_column is None:
        return None
    stripped = target_column.strip()
    return stripped or None


class DatasetProfilingService:
    """Validate an uploaded dataset and produce its profile."""

    def __init__(self, settings: Settings) -> None:
        """Bind the service to a settings instance.

        Args:
            settings: Active application settings, supplying every limit and
                heuristic threshold used downstream.
        """
        self._settings = settings

    # -- Ingestion ---------------------------------------------------------

    async def load_upload(
        self, upload: AsyncReadable, filename: str | None
    ) -> LoadedDataset:
        """Validate an in-flight upload and parse it into a DataFrame.

        The extension is checked before any bytes are read, so an unsupported
        file is rejected without being buffered, and the bytes are never
        written anywhere: the upload lives in memory for the length of the
        request only.

        Args:
            upload: The incoming file object.
            filename: Client-supplied filename, used only for its extension.

        Returns:
            LoadedDataset: The parsed frame and the path-free filename.

        Raises:
            DatasetError: If the upload or its content fails validation.
        """
        name = safe_filename(filename)
        validate_extension(name, self._settings)
        content = await read_upload(upload, self._settings)
        return LoadedDataset(frame=load_csv(content, self._settings), filename=name)

    def load_content(self, filename: str | None, content: bytes) -> LoadedDataset:
        """Validate dataset bytes already in memory and parse them.

        Args:
            filename: Name the content was uploaded under.
            content: Raw file bytes.

        Returns:
            LoadedDataset: The parsed frame and the path-free filename.

        Raises:
            DatasetError: If the content fails validation.
        """
        name = safe_filename(filename)
        validate_extension(name, self._settings)
        validate_size(len(content), self._settings)
        return LoadedDataset(frame=load_csv(content, self._settings), filename=name)

    # -- Profiling ---------------------------------------------------------

    async def profile_upload(
        self, upload: AsyncReadable, filename: str | None, target_column: str | None = None
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
        loaded = await self.load_upload(upload, filename)
        return self.profile_frame(
            loaded.frame,
            filename=loaded.filename,
            target_column=_normalise_target(target_column),
        )

    def profile_content(
        self, filename: str | None, content: bytes, target_column: str | None = None
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
        loaded = self.load_content(filename, content)
        return self.profile_frame(
            loaded.frame,
            filename=loaded.filename,
            target_column=_normalise_target(target_column),
        )

    def profile_frame(
        self,
        frame: pd.DataFrame,
        *,
        filename: str = "dataset",
        target_column: str | None = None,
    ) -> DatasetProfileResponse:
        """Profile an already-standardised DataFrame.

        This is the format-agnostic entry point: it knows nothing about files.
        The experiment runner uses it so that one profiling implementation
        serves both the profiling endpoint and experiment execution.

        Args:
            frame: A standardised dataset.
            filename: Label for the response; never used as a path.
            target_column: Optional target column to analyse.

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
            generated_at=datetime.now(timezone.utc),
            dataset=summary,
            columns=columns,
            quality=quality,
            target=target,
        )
