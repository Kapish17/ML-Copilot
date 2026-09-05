"""What a persisted model needs to remember about itself.

A fitted ``Pipeline`` is enough to *make* a prediction and not enough to
*accept* one. The pipeline's ``ColumnTransformer`` selects columns by name with
``remainder="drop"``, so a caller who misspells a feature does not get an
error — they get a prediction computed without it. Recovering the schema from
the fitted object afterwards is possible and fragile; recording it at training
time is neither.

So every artifact carries a manifest alongside the model: the columns the
pipeline was fitted on, in order, each with the branch of preprocessing that
handles it; the target and, for a classifier, its classes; and enough training
metadata to say what this model is and where it came from.

**The manifest is data, never instructions.** Nothing in it is used to build a
filesystem path, choose a module to import, or decide what to deserialise. The
model file's name is a constant in :mod:`ml.artifacts.store`; the manifest
records it only so a human reading the directory can see what should be there.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd

from ml.errors import ModelArtifactUnreadableError
from ml.features.types import FeatureType, TaskType
from ml.models.selection import ModelSelectionResult
from ml.pipelines.result import PreparedDataset

#: Bumped when the manifest's shape changes incompatibly. An artifact written
#: under an unknown version is refused rather than guessed at.
ARTIFACT_SCHEMA_VERSION = "1.0"
SUPPORTED_ARTIFACT_VERSIONS = frozenset({ARTIFACT_SCHEMA_VERSION})


def _jsonable(value: Any) -> Any:
    """Return a plain Python value for something that may be a NumPy scalar.

    Class labels come back from an estimator as NumPy types, which
    ``json.dump`` refuses. ``.item()`` is the documented way to get the Python
    equivalent, and anything without it is rendered as a string rather than
    silently dropped.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (ValueError, TypeError):  # pragma: no cover - defensive
            pass
    return str(value)


@dataclass(frozen=True)
class FeatureSpec:
    """One column the model expects, and how it will be treated."""

    #: The column's name in the source dataset, exactly as trained on.
    name: str
    #: Which branch of the preprocessing pipeline handles it.
    kind: FeatureType
    #: The pandas dtype the column had at training time, recorded so a
    #: prediction frame can be built with the same one. Informational for a
    #: reader; load-bearing for the coercion in :mod:`ml.artifacts.prediction`.
    dtype: str

    def as_dict(self) -> dict[str, Any]:
        """Render as plain JSON-safe values."""
        return {"name": self.name, "kind": self.kind.value, "dtype": self.dtype}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FeatureSpec:
        """Rebuild from a manifest entry.

        Raises:
            ModelArtifactUnreadableError: If the entry is malformed.
        """
        try:
            return cls(
                name=str(payload["name"]),
                kind=FeatureType(str(payload["kind"])),
                dtype=str(payload["dtype"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelArtifactUnreadableError(
                "A feature entry in the model manifest is malformed.",
                details={"reason": type(exc).__name__},
            ) from exc


@dataclass(frozen=True)
class ModelArtifactMetadata:
    """Everything about a persisted model except the model itself."""

    experiment_id: str
    created_at: datetime
    model_name: str
    display_name: str
    task_type: TaskType
    target_column: str
    #: Class labels, for a classifier. Empty for regression. Taken from the
    #: fitted estimator, so they are the labels a prediction will actually
    #: return rather than the ones the data happened to contain.
    classes: tuple[Any, ...]
    #: The columns the pipeline was fitted on, **in fit order**. A prediction
    #: frame is built with exactly these, in exactly this order.
    features: tuple[FeatureSpec, ...]
    train_row_count: int
    test_row_count: int
    primary_metric: str
    primary_metric_value: float | None
    random_state: int | None
    #: The interpreter and library versions that wrote the artifact. A pickle
    #: is not portable across arbitrary versions, so this is what a failed load
    #: is diagnosed with.
    environment: dict[str, str] = field(default_factory=dict)
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    @property
    def feature_names(self) -> tuple[str, ...]:
        """The expected columns, in fit order."""
        return tuple(feature.name for feature in self.features)

    @property
    def supports_probabilities(self) -> bool:
        """Whether a classification prediction can carry class probabilities.

        A property of the task and the estimator together: only a classifier
        has classes, and only some classifiers expose ``predict_proba``. This
        answers the first half; :mod:`ml.artifacts.prediction` asks the loaded
        estimator about the second.
        """
        return self.task_type is TaskType.CLASSIFICATION and bool(self.classes)

    def as_dict(self) -> dict[str, Any]:
        """Render the manifest as JSON-safe values."""
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "created_at": self.created_at.isoformat(),
            "model": {
                "name": self.model_name,
                "display_name": self.display_name,
                "task_type": self.task_type.value,
            },
            "target": {
                "column": self.target_column,
                "classes": [_jsonable(value) for value in self.classes],
            },
            "features": [feature.as_dict() for feature in self.features],
            "training": {
                "train_row_count": self.train_row_count,
                "test_row_count": self.test_row_count,
                "primary_metric": self.primary_metric,
                "primary_metric_value": self.primary_metric_value,
                "random_state": self.random_state,
            },
            "environment": dict(self.environment),
        }

    def public_summary(self) -> dict[str, Any]:
        """The part of the manifest an API may return.

        Deliberately narrower than :meth:`as_dict`: **no filesystem location
        appears in either**, but this one also drops the environment, which
        names interpreter and library versions that a caller has no use for and
        that describe the host rather than the model. It also drops the random
        seed, which belongs to reproducing the run rather than to using it.

        What is kept is what a caller needs to *act*: what the model predicts,
        what it wants to be given, how well it did on data it never saw, and
        how large the two halves of that measurement were. ``test_row_count``
        earns its place beside the metric because "0.87 f1" and "0.87 f1 on 60
        rows" are different claims, and a client showing the first without the
        second is overstating what it knows.
        """
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at.isoformat(),
            "model_name": self.model_name,
            "display_name": self.display_name,
            "task_type": self.task_type.value,
            "target_column": self.target_column,
            "classes": [_jsonable(value) for value in self.classes],
            "features": [feature.as_dict() for feature in self.features],
            "train_row_count": self.train_row_count,
            "test_row_count": self.test_row_count,
            "primary_metric": self.primary_metric,
            "primary_metric_value": self.primary_metric_value,
            # Whether a probability breakdown is worth rendering, answered
            # before a prediction is made rather than discovered from a null
            # afterwards. See the property: it is the task-and-classes half of
            # the question, and the estimator answers the rest at predict time.
            "supports_probabilities": self.supports_probabilities,
            # Which manifest shape this is. A client that meets an artifact
            # from a newer ML Copilot should say "upgrade", not "re-run".
            "artifact_schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> ModelArtifactMetadata:
        """Rebuild a manifest, validating as it goes.

        Raises:
            ModelArtifactUnreadableError: If the manifest is not an object, is
                of an unknown version, or is missing something required.
        """
        if not isinstance(payload, Mapping):
            raise ModelArtifactUnreadableError(
                "A model manifest must be an object, not "
                f"{type(payload).__name__}."
            )

        version = payload.get("schema_version")
        if version not in SUPPORTED_ARTIFACT_VERSIONS:
            raise ModelArtifactUnreadableError(
                f"Model artifact schema version {version!r} cannot be read by "
                "this version of ML Copilot.",
                details={
                    "found": version,
                    "supported": sorted(SUPPORTED_ARTIFACT_VERSIONS),
                },
            )

        try:
            model = payload["model"]
            target = payload["target"]
            training = payload["training"]
            return cls(
                experiment_id=str(payload["experiment_id"]),
                created_at=datetime.fromisoformat(str(payload["created_at"])),
                model_name=str(model["name"]),
                display_name=str(model["display_name"]),
                task_type=TaskType(str(model["task_type"])),
                target_column=str(target["column"]),
                classes=tuple(target.get("classes") or ()),
                features=tuple(
                    FeatureSpec.from_dict(entry) for entry in payload["features"]
                ),
                train_row_count=int(training["train_row_count"]),
                test_row_count=int(training["test_row_count"]),
                primary_metric=str(training["primary_metric"]),
                primary_metric_value=(
                    None
                    if training.get("primary_metric_value") is None
                    else float(training["primary_metric_value"])
                ),
                random_state=(
                    None
                    if training.get("random_state") is None
                    else int(training["random_state"])
                ),
                environment=dict(payload.get("environment") or {}),
                schema_version=str(version),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelArtifactUnreadableError(
                "The model manifest is missing a required field or holds an "
                "unusable value.",
                details={"reason": type(exc).__name__},
            ) from exc


def _environment() -> dict[str, str]:
    """Describe the interpreter and libraries that are writing this artifact."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "pandas": pd.__version__,
    }
    try:  # pragma: no cover - both are hard requirements of this package
        import joblib
        import sklearn

        versions["scikit_learn"] = sklearn.__version__
        versions["joblib"] = joblib.__version__
    except ImportError:  # pragma: no cover - defensive
        pass
    return versions


def build_metadata(
    *,
    experiment_id: str,
    prepared: PreparedDataset,
    selection: ModelSelectionResult,
    created_at: datetime | None = None,
) -> ModelArtifactMetadata:
    """Describe the model an experiment just produced.

    The feature schema comes from the *fitted* preprocessing configuration —
    ``config.feature_columns`` is the same tuple, in the same order, that the
    pipeline was fitted on — so the manifest cannot drift from the model beside
    it. The dtypes come from the training frame for the same reason.

    Args:
        experiment_id: The run this model belongs to.
        prepared: The dataset the pipeline was fitted on.
        selection: The finished selection, carrying the winning model.
        created_at: When the artifact was written; now, in UTC, when omitted.

    Returns:
        ModelArtifactMetadata: The manifest to store beside the model.
    """
    config = prepared.config
    trained = selection.final_model
    dtypes = prepared.X_train_raw.dtypes

    features = tuple(
        FeatureSpec(
            name=column,
            kind=config.feature_type_of(column) or FeatureType.CATEGORICAL,
            dtype=str(dtypes.get(column, "object")),
        )
        for column in config.feature_columns
    )

    # Asked of the fitted estimator rather than of the training labels: these
    # are the values `predict` will actually return, and their order is the
    # order `predict_proba` puts its columns in.
    #
    # `classes_` is a NumPy array, so it is converted before anything tests it
    # — `array or ()` raises rather than falling back, which is the sort of
    # thing that only shows up on the first classification run.
    raw_classes = getattr(trained.estimator, "classes_", None)
    classes = tuple(raw_classes) if raw_classes is not None else ()

    return ModelArtifactMetadata(
        experiment_id=experiment_id,
        created_at=created_at or datetime.now(timezone.utc),
        model_name=trained.model_name,
        display_name=trained.display_name,
        task_type=trained.task_type,
        target_column=config.target_column,
        classes=classes,
        features=features,
        train_row_count=prepared.train_row_count,
        test_row_count=prepared.test_row_count,
        primary_metric=trained.primary_metric.key,
        primary_metric_value=trained.primary_metric_value,
        random_state=config.random_state,
        environment=_environment(),
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "SUPPORTED_ARTIFACT_VERSIONS",
    "FeatureSpec",
    "ModelArtifactMetadata",
    "build_metadata",
]
