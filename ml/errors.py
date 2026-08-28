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


class ModelError(MLError):
    """Base class for failures in the model training and evaluation layer."""


class UnknownModelError(ModelError):
    """The requested model is not in the registry."""


class IncompatibleTaskError(ModelError):
    """The requested model does not support the dataset's task type."""


class InvalidHyperparameterError(ModelError):
    """A hyperparameter is not accepted by the requested estimator."""


class InvalidMetricError(ModelError):
    """The requested metric does not exist for the task type."""


class InvalidFoldCountError(ModelError):
    """The requested number of cross-validation folds cannot be used.

    Raised when the fold count is below two, exceeds the number of rows, or —
    for classification — exceeds the size of the smallest class, since a class
    with fewer members than folds cannot appear in every validation fold.
    """


class ModelTrainingError(ModelError):
    """An estimator failed while being fitted or while predicting."""


class NoSuccessfulModelError(ModelError):
    """No model in a comparison run finished successfully."""
