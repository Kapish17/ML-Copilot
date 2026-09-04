"""Translation of ML-layer exceptions into the API error contract.

The ML and experiment packages raise plain Python exceptions with no HTTP
meaning — that independence is deliberate, and this module is the single place
where it is bridged. Each known error type is mapped to a stable API code and
an HTTP status, so a preprocessing failure and a dataset failure reach the
client in exactly the same envelope.

Two things are scrubbed on the way out.

**Filesystem paths.** ``ExperimentNotFoundError`` carries the store directory
in its details, which is useful in a log and has no place in an HTTP response.
Any detail that names a path is removed rather than reworded.

**Unknown errors.** An exception with no mapping becomes a generic 500. It is
the caller's business that the request failed, not which internal class raised.
"""

from __future__ import annotations

import re
from typing import Any

from ml.errors import (
    ConfigurationError,
    DuplicateColumnAssignmentError,
    EmptyExplanationDataError,
    EmptyFeatureSetError,
    ExperimentError,
    ExperimentNotFoundError,
    ExplainabilityError,
    IncomparableExperimentsError,
    IncompatibleTaskError,
    InsufficientDataError,
    InvalidExperimentIdError,
    InvalidExperimentRecordError,
    InvalidExplanationRowError,
    InvalidFoldCountError,
    InvalidHyperparameterError,
    InvalidMetricError,
    InvalidTrainedModelError,
    MalformedExperimentError,
    MissingFeatureColumnsError,
    MissingTargetError,
    MLError,
    ModelArtifactError,
    ModelArtifactNotFoundError,
    ModelArtifactUnreadableError,
    ModelError,
    ModelTrainingError,
    NoSuccessfulModelError,
    PredictionError,
    PredictionInputError,
    SerializationError,
    TargetLeakageError,
    UnknownColumnError,
    UnknownModelError,
    UnsupportedSchemaVersionError,
)

#: Detail keys that hold, or could hold, a filesystem location.
_PATH_DETAIL_KEYS = frozenset({"root", "path", "directory", "store_root", "location"})

#: A value that looks like an absolute POSIX or Windows path.
_PATH_VALUE = re.compile(r"^(/|[A-Za-z]:[\\/]|\\\\)")

_GENERIC_MESSAGE = "The request could not be completed."

#: ``exception type -> (api code, http status)``, most specific first. Order
#: matters: the first class an exception is an instance of wins, so subclasses
#: must precede the bases they refine.
_ERROR_MAPPING: tuple[tuple[type[Exception], str, int], ...] = (
    # Configuration and preprocessing.
    (MissingTargetError, "missing_target", 422),
    (TargetLeakageError, "target_leakage", 409),
    (DuplicateColumnAssignmentError, "duplicate_column_assignment", 409),
    (UnknownColumnError, "unknown_column", 422),
    (EmptyFeatureSetError, "empty_feature_set", 422),
    (ConfigurationError, "invalid_configuration", 400),
    (InsufficientDataError, "insufficient_data", 422),
    # Model selection.
    (UnknownModelError, "unknown_model", 400),
    (IncompatibleTaskError, "incompatible_model_task", 409),
    (InvalidHyperparameterError, "invalid_hyperparameter", 400),
    (InvalidMetricError, "invalid_metric", 400),
    (InvalidFoldCountError, "invalid_fold_count", 400),
    (NoSuccessfulModelError, "no_successful_model", 422),
    (ModelTrainingError, "model_training_failed", 500),
    (ModelError, "model_error", 500),
    # Explainability.
    (InvalidTrainedModelError, "invalid_trained_model", 500),
    (MissingFeatureColumnsError, "missing_feature_columns", 422),
    (EmptyExplanationDataError, "empty_explanation_data", 422),
    (InvalidExplanationRowError, "invalid_explanation_row", 400),
    (ExplainabilityError, "explanation_failed", 500),
    # Persisted models and prediction.
    #
    # `model_not_available` is a 409 rather than a 404 on purpose: the
    # experiment exists and is perfectly valid, it simply has no model to
    # predict from — a run recorded before Commit 22, or one whose artifact was
    # removed. A 404 here would say the run was gone, which is a different
    # problem with a different fix.
    (ModelArtifactNotFoundError, "model_not_available", 409),
    (ModelArtifactUnreadableError, "model_artifact_unreadable", 500),
    (ModelArtifactError, "model_artifact_error", 500),
    (PredictionInputError, "invalid_prediction_input", 422),
    (PredictionError, "prediction_failed", 500),
    # Experiment tracking.
    (InvalidExperimentIdError, "invalid_experiment_id", 400),
    (ExperimentNotFoundError, "experiment_not_found", 404),
    (MalformedExperimentError, "experiment_data_integrity_error", 500),
    (UnsupportedSchemaVersionError, "unsupported_schema_version", 500),
    (InvalidExperimentRecordError, "experiment_data_integrity_error", 500),
    (SerializationError, "experiment_serialization_error", 500),
    (IncomparableExperimentsError, "incomparable_experiments", 409),
    (ExperimentError, "experiment_error", 500),
)

#: Statuses whose message is authored for a client. Anything mapped to 5xx
#: describes an internal failure, so its message is replaced.
_CLIENT_FACING_MAX_STATUS = 499


def sanitise_details(details: Any) -> dict[str, Any]:
    """Return error details with anything path-like removed.

    Args:
        details: The ``details`` mapping carried by an ML error.

    Returns:
        dict: The same information minus keys that name a filesystem location
        and minus values that look like absolute paths.
    """
    if not isinstance(details, dict):
        return {}
    clean: dict[str, Any] = {}
    for key, value in details.items():
        if str(key).lower() in _PATH_DETAIL_KEYS:
            continue
        if isinstance(value, str) and _PATH_VALUE.match(value):
            continue
        clean[str(key)] = value
    return clean


def translate_ml_error(exc: Exception) -> tuple[str, int, str, dict[str, Any]]:
    """Map an ML-layer exception onto the API error contract.

    Args:
        exc: The exception raised by ``ml``.

    Returns:
        tuple: ``(code, status_code, message, details)`` ready for the envelope.
        For a 5xx the message is generic and the details are dropped, because
        an internal failure is not something a client can act on.
    """
    for error_type, code, status_code in _ERROR_MAPPING:
        if isinstance(exc, error_type):
            break
    else:
        code, status_code = "ml_error", 500

    if status_code > _CLIENT_FACING_MAX_STATUS:
        return code, status_code, _GENERIC_MESSAGE, {}

    message = str(exc) or _GENERIC_MESSAGE
    details = sanitise_details(getattr(exc, "details", None))
    return code, status_code, message, details


def is_client_error(exc: Exception) -> bool:
    """Return whether an ML error maps to a 4xx status.

    Useful where the caller wants to log server-side failures loudly and leave
    client mistakes quiet.
    """
    return translate_ml_error(exc)[1] <= _CLIENT_FACING_MAX_STATUS


__all__ = ["MLError", "is_client_error", "sanitise_details", "translate_ml_error"]
