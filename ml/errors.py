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


class ExplainabilityError(MLError):
    """Base class for failures in the explainability layer.

    An estimator that no explainer supports is *not* an error: it produces a
    structured result with a reason, so a caller can see what happened rather
    than catching an exception.
    """


class InvalidTrainedModelError(ExplainabilityError):
    """The object handed in is not a usable, fitted trained model."""


class MissingFeatureColumnsError(ExplainabilityError):
    """The data to explain is missing columns the model was fitted on."""


class EmptyExplanationDataError(ExplainabilityError):
    """There are no rows to explain."""


class InvalidExplanationRowError(ExplainabilityError):
    """A local explanation was asked for something other than a single row."""


class ExperimentError(MLError):
    """Base class for failures in the experiment-tracking layer."""


class SerializationError(ExperimentError):
    """A value cannot be stored in an experiment record.

    Raised rather than quietly writing something useless — a fitted pipeline or
    a SHAP explainer has no place in an experiment record, and a silent
    ``str()`` of one would bloat the file while telling nobody anything.
    """


class InvalidExperimentIdError(ExperimentError):
    """An experiment identifier is malformed or unsafe to use as a path."""


class ExperimentNotFoundError(ExperimentError):
    """No experiment is stored under the requested identifier."""


class MalformedExperimentError(ExperimentError):
    """A stored experiment record could not be read."""


class UnsupportedSchemaVersionError(ExperimentError):
    """A stored record uses a schema version this code cannot read."""


class InvalidExperimentRecordError(ExperimentError):
    """A stored record is missing a required field or has the wrong type."""


class IncomparableExperimentsError(ExperimentError):
    """Runs cannot be ranked together because their metrics differ.

    Ranking an F1 against an RMSE is meaningless, so it is refused rather than
    producing a confidently wrong ordering.
    """
