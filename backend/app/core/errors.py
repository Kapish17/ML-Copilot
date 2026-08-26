"""Domain errors raised by the service layer.

Every error carries a stable machine-readable ``code`` and the HTTP status the
API should answer with. Route handlers never build error responses by hand:
they let these propagate and a single exception handler turns them into the
documented error envelope. Messages are written for a human reading a frontend,
never containing stack traces or internal paths.
"""

from __future__ import annotations

from typing import Any


class MLCopilotError(Exception):
    """Base class for expected, client-facing application errors."""

    code = "internal_error"
    status_code = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        """Store the user-facing message and optional structured context.

        Args:
            message: Explanation safe to show to an API consumer.
            details: Extra machine-readable context (limits, column names).
        """
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}


class DatasetError(MLCopilotError):
    """Base class for dataset ingestion and profiling failures."""

    code = "dataset_error"
    status_code = 400


class UnsupportedFileTypeError(DatasetError):
    """The uploaded file is not one of the supported dataset formats."""

    code = "unsupported_file_type"
    status_code = 415


class FileTooLargeError(DatasetError):
    """The upload exceeds the configured size limit."""

    code = "file_too_large"
    status_code = 413


class EmptyFileError(DatasetError):
    """The upload contains no bytes at all."""

    code = "empty_file"
    status_code = 400


class MalformedCSVError(DatasetError):
    """The file could not be parsed as CSV."""

    code = "malformed_csv"
    status_code = 422


class MissingHeaderError(DatasetError):
    """The CSV does not start with a usable header row."""

    code = "missing_header"
    status_code = 422


class DuplicateColumnsError(DatasetError):
    """The CSV header repeats one or more column names."""

    code = "duplicate_columns"
    status_code = 422


class EmptyDatasetError(DatasetError):
    """The file parsed successfully but contains no rows or no columns."""

    code = "empty_dataset"
    status_code = 422


class DatasetTooLargeError(DatasetError):
    """The parsed dataset exceeds the configured row or column limits."""

    code = "dataset_too_large"
    status_code = 413


class TargetColumnNotFoundError(DatasetError):
    """The requested target column is not present in the dataset."""

    code = "target_column_not_found"
    status_code = 422
