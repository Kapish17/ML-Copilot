"""Orchestration of the dataset profiling workflow.

The service is the only entry point route handlers use. It sequences
validation, loading, profiling, quality analysis and target analysis, and
returns a fully built response model. It holds no mutable state: everything it
needs comes from the settings passed at construction.
"""

from __future__ import annotations

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

    async def profile_upload(
        self, upload: AsyncReadable, filename: str | None, target_column: str | None = None
    ) -> DatasetProfileResponse:
        """Profile an in-flight upload.

        The extension is checked before any bytes are read, so an unsupported
        file is rejected without being buffered.

        Args:
            upload: The incoming file object.
            filename: Client-supplied filename, used only for its extension.
            target_column: Optional target column to analyse.

        Returns:
            DatasetProfileResponse: The complete dataset profile.

        Raises:
            DatasetError: If the upload or its content fails validation.
        """
        name = safe_filename(filename)
        validate_extension(name, self._settings)
        content = await read_upload(upload, self._settings)
        return self._build_profile(name, content, _normalise_target(target_column))

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
        name = safe_filename(filename)
        validate_extension(name, self._settings)
        validate_size(len(content), self._settings)
        return self._build_profile(name, content, _normalise_target(target_column))

    def _build_profile(
        self, filename: str, content: bytes, target_column: str | None
    ) -> DatasetProfileResponse:
        """Run the profiling pipeline over validated bytes."""
        frame = load_csv(content, self._settings)
        return self._profile_frame(frame, filename=filename, target_column=target_column)

    def _profile_frame(
        self, frame: pd.DataFrame, *, filename: str, target_column: str | None
    ) -> DatasetProfileResponse:
        """Assemble the response from a validated DataFrame."""
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
