"""Errors raised by the machine-learning layer.

These are plain Python exceptions with no HTTP meaning. The API layer is
responsible for translating them into responses when preprocessing is exposed
over HTTP, which keeps this package free of any web dependency.
"""

from __future__ import annotations

from typing import Any


class MLError(Exception):
    """Base class for every error raised by the ML layer."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        """Store a human-readable message and optional structured context.

        Args:
            message: Explanation of what went wrong.
            details: Machine-readable context, such as the offending columns.
        """
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}


class ConfigurationError(MLError):
    """The preprocessing configuration cannot be applied to the dataset."""


class MissingTargetError(ConfigurationError):
    """The configured target column is absent from the dataset."""


class TargetLeakageError(ConfigurationError):
    """The target column was also assigned to a feature group."""


class UnknownColumnError(ConfigurationError):
    """A configured column does not exist in the dataset."""


class DuplicateColumnAssignmentError(ConfigurationError):
    """A column was assigned to more than one feature group."""


class EmptyFeatureSetError(ConfigurationError):
    """No usable feature columns remain after applying the configuration."""


class InsufficientDataError(MLError):
    """There are too few usable rows to build a train/test split."""
