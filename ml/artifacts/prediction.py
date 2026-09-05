"""Turning submitted records into predictions from a persisted model.

Two jobs, in this order, and the first is the larger one.

**Validate.** A prediction frame has to match what the pipeline was fitted on
exactly, because a ``ColumnTransformer`` selects columns by name with
``remainder="drop"``. A missing column raises inside sklearn with a message
about the transformer; a *misspelt* one does not raise at all — it is dropped,
the intended column is missing, and the caller gets a confident prediction made
without the feature they thought they supplied. So every record is checked
against the manifest before anything reaches the model, and both a missing and
an unexpected column are refused by name.

**Predict.** The loaded pipeline is used exactly as it came off disk. Its
preprocessing was fitted on the training rows during the experiment and is
applied here unchanged — there is no ``fit`` call anywhere in this module, and
`ml/tests/test_artifacts.py` makes one impossible by monkeypatching it to
explode.

---------------------------------------------------------------------------
Why unexpected columns are refused rather than ignored
---------------------------------------------------------------------------
Dropping them silently is what sklearn already does, and it is the failure mode
that produces a wrong answer instead of an error. A caller who sends
``tenure_month`` for ``tenure_months`` has made a mistake worth being told
about; the alternative is a prediction computed from an imputed value for a
column they believe they provided. This project has refused to silently change
a user's values since Commit 15, and this is the same rule applied to a
different boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ml.artifacts.schema import ModelArtifactMetadata, _jsonable
from ml.artifacts.store import LoadedModel
from ml.errors import PredictionInputError
from ml.features.types import FeatureType, TaskType

#: Most records one request may predict on. Prediction is cheap per row, but a
#: request is still synchronous and holds a worker, so it is bounded like every
#: other expensive path in this project.
DEFAULT_MAX_RECORDS = 500


def _is_missing(value: Any) -> bool:
    """Whether a submitted value means "not supplied".

    Three spellings, and each is here for a reason. ``None`` is what JSON null
    parses to. The empty string is what an HTML form sends for a blank field,
    and reading it as the *string* ``""`` would make a numeric column fail with
    a type error instead of being imputed the way training handled a missing
    value. ``NaN`` is what a caller inside the process may pass.

    Written as explicit type checks rather than ``value in (None, "")``, which
    compares with ``==`` and therefore does something surprising for a value
    that overrides it — a NumPy array answers elementwise and raises
    "truth value is ambiguous", turning a bad input into a 500. Nothing
    reaching this function from an HTTP request can be an array today; the
    function is public API of ``ml`` and should not depend on that.
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, np.floating):
        return bool(np.isnan(value))
    return False


@dataclass(frozen=True)
class Prediction:
    """One row's result."""

    #: Index of the record in the submitted batch, so a caller pairing results
    #: with inputs never has to rely on ordering alone.
    index: int
    #: The predicted class label, or the predicted number.
    prediction: Any
    #: ``{class label: probability}``, for a classifier that provides them.
    #: ``None`` for regression, and for a classifier without `predict_proba`.
    probabilities: dict[str, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Render as plain JSON-safe values."""
        payload: dict[str, Any] = {
            "index": self.index,
            "prediction": _jsonable(self.prediction),
        }
        if self.probabilities is not None:
            payload["probabilities"] = self.probabilities
        return payload


@dataclass(frozen=True)
class PredictionResult:
    """Every row's result, and what produced them."""

    predictions: tuple[Prediction, ...]
    metadata: ModelArtifactMetadata

    def as_dict(self) -> dict[str, Any]:
        """Render the whole result, model description included.

        The model description comes from ``public_summary``, which carries no
        filesystem location and no host detail.
        """
        return {
            "predictions": [item.as_dict() for item in self.predictions],
            "prediction_count": len(self.predictions),
            "model": self.metadata.public_summary(),
        }


def _coerce(value: Any, feature_name: str, kind: FeatureType) -> Any:
    """Return one submitted value as the kind of thing its column expects.

    JSON has four scalar types and pandas has many, so something has to bridge
    them. The bridge is deliberately narrow: a numeric column takes a number or
    a string that is unambiguously one, never a string like ``"high"``; a
    boolean takes a boolean or a recognised spelling of one; a datetime takes
    something pandas can parse. Anything else is refused by name rather than
    coerced into a value the caller did not write.

    Raises:
        PredictionInputError: If the value cannot be read as that kind.
    """
    if _is_missing(value):
        # Missing is legitimate: the pipeline's imputers were fitted for it,
        # and refusing here would be stricter than training was.
        return np.nan

    # Refused once, here, for every kind. A feature value is a scalar: JSON
    # nests, and a list or an object reaching a coercion written for scalars
    # produces something unhelpful rather than an error — `pd.to_datetime` on a
    # list returns an index whose truthiness raises, which would turn a
    # malformed record into a 500 instead of the 422 it is.
    if isinstance(value, (Mapping, list, tuple, set)):
        raise PredictionInputError(
            f"'{feature_name}' expects a single value, not a list or an object.",
            details={"feature": feature_name, "expected": "a single value"},
        )

    if kind is FeatureType.NUMERIC:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                pass
        raise PredictionInputError(
            f"'{feature_name}' expects a number.",
            details={"feature": feature_name, "expected": "number"},
        )

    if kind is FeatureType.BOOLEAN:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "1"):
                return True
            if lowered in ("false", "no", "0"):
                return False
        raise PredictionInputError(
            f"'{feature_name}' expects true or false.",
            details={"feature": feature_name, "expected": "boolean"},
        )

    if kind is FeatureType.DATETIME:
        parsed = pd.to_datetime(value, errors="coerce")
        if pd.isna(parsed):
            raise PredictionInputError(
                f"'{feature_name}' expects a date or timestamp.",
                details={"feature": feature_name, "expected": "date"},
            )
        return parsed

    # Categorical. A number is accepted and rendered as text, because a
    # category that looked numeric in the source data was encoded as its string
    # form and refusing `3` for a column trained on "3" would be pedantry.
    return value if isinstance(value, str) else str(value)


def build_frame(
    records: Sequence[Mapping[str, Any]],
    metadata: ModelArtifactMetadata,
    *,
    max_records: int = DEFAULT_MAX_RECORDS,
) -> pd.DataFrame:
    """Validate submitted records and build the frame the pipeline expects.

    The result has exactly the manifest's columns, in exactly the manifest's
    order — the order the pipeline was fitted on — so nothing downstream has to
    reason about column alignment.

    Args:
        records: The submitted records, one per row.
        metadata: The manifest describing the model's feature schema.
        max_records: Most rows one request may carry.

    Returns:
        pandas.DataFrame: Ready for ``pipeline.predict``.

    Raises:
        PredictionInputError: If the batch is empty or too large, if a record
            is not an object, or if any record's columns do not match the
            schema.
    """
    if not records:
        raise PredictionInputError(
            "At least one record is required to make a prediction.",
            details={"record_count": 0},
        )
    if len(records) > max_records:
        raise PredictionInputError(
            f"At most {max_records} records may be predicted in one request.",
            details={"record_count": len(records), "maximum": max_records},
        )

    expected = metadata.feature_names
    expected_set = set(expected)
    rows: list[dict[str, Any]] = []

    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise PredictionInputError(
                f"Record {position} must be an object of feature values.",
                details={"index": position, "received_type": type(record).__name__},
            )

        supplied = set(record)
        missing = [name for name in expected if name not in supplied]
        unexpected = sorted(supplied - expected_set)

        if missing:
            raise PredictionInputError(
                f"Record {position} is missing "
                f"{len(missing)} required feature(s): {', '.join(missing[:10])}.",
                details={
                    "index": position,
                    "missing_features": missing,
                    "expected_features": list(expected),
                },
            )
        if unexpected:
            raise PredictionInputError(
                f"Record {position} carries {len(unexpected)} feature(s) the "
                f"model was not trained on: {', '.join(unexpected[:10])}. A "
                "column the model does not know is dropped rather than used, "
                "so it is refused here instead of silently ignored.",
                details={
                    "index": position,
                    "unexpected_features": unexpected,
                    "expected_features": list(expected),
                },
            )

        try:
            rows.append(
                {
                    feature.name: _coerce(
                        record[feature.name], feature.name, feature.kind
                    )
                    for feature in metadata.features
                }
            )
        except PredictionInputError as exc:
            # Re-raised with the record's position, so a batch failure says
            # which row is at fault rather than only which column.
            raise PredictionInputError(
                f"Record {position}: {exc.message}",
                details={**exc.details, "index": position},
            ) from None

    return pd.DataFrame(rows, columns=list(expected))


def _probabilities(
    model: LoadedModel, frame: pd.DataFrame
) -> np.ndarray | None:
    """Return per-class probabilities, or ``None`` when unavailable.

    Availability is asked of the loaded estimator rather than assumed from the
    model's name: the registry's models differ, and a future one might not
    provide them.
    """
    if model.metadata.task_type is not TaskType.CLASSIFICATION:
        return None
    if not hasattr(model.pipeline, "predict_proba"):
        return None
    try:
        return np.asarray(model.pipeline.predict_proba(frame))
    except (AttributeError, NotImplementedError):  # pragma: no cover
        return None


def predict(model: LoadedModel, frame: pd.DataFrame) -> PredictionResult:
    """Predict from raw feature rows, using the pipeline exactly as stored.

    **Nothing here fits anything.** The preprocessing inside this pipeline was
    fitted once, during the experiment, on the training rows alone. It is
    applied to these rows the same way it was applied to the test set, which is
    what makes a prediction made now comparable to the score the experiment
    reported.

    Args:
        model: The loaded pipeline and its manifest.
        frame: Rows built by :func:`build_frame`.

    Returns:
        PredictionResult: One result per row, plus the model's description.

    Raises:
        PredictionInputError: If the model rejects the frame — which, after
            :func:`build_frame`, means a value that is the right kind and still
            unusable.
    """
    try:
        raw = np.asarray(model.pipeline.predict(frame))
        probabilities = _probabilities(model, frame)
    except PredictionInputError:  # pragma: no cover - not raised by sklearn
        raise
    except Exception as exc:  # noqa: BLE001 - sklearn raises many types
        raise PredictionInputError(
            "The model could not make a prediction from these values.",
            details={"reason": type(exc).__name__},
        ) from None

    classes = [str(_jsonable(value)) for value in model.metadata.classes]
    predictions: list[Prediction] = []

    for position, value in enumerate(raw):
        row: dict[str, float] | None = None
        if probabilities is not None and position < len(probabilities):
            scores = probabilities[position]
            if len(classes) == len(scores):
                row = {
                    label: round(float(score), 6)
                    for label, score in zip(classes, scores)
                }
        predictions.append(
            Prediction(index=position, prediction=value, probabilities=row)
        )

    return PredictionResult(
        predictions=tuple(predictions), metadata=model.metadata
    )


__all__ = [
    "DEFAULT_MAX_RECORDS",
    "Prediction",
    "PredictionResult",
    "build_frame",
    "predict",
]
