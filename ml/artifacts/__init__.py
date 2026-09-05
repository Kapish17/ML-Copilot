"""Persisted models, and predicting from them.

An experiment produces a fitted ``Pipeline(preprocessing, estimator)``. Until
this package existed it lived only in memory: the run record described what had
been trained and held no model, so there was nothing to predict with afterwards.

This package closes that gap, and the whole of it rests on one property the
training layer already had — **the trained artefact is a full pipeline that
takes raw feature rows**. There is nothing to reassemble and no preprocessing to
rebuild: the object saved here is the object the experiment fitted, and the
object a prediction runs through later is the same one, unchanged.

    ml.artifacts.schema      what a model must remember about itself
    ml.artifacts.store       where it is kept, the rules for loading it, and
                             the one check that decides whether it is usable
    ml.artifacts.prediction  validating records, and predicting from them

Two things worth reading before changing anything here: the trust boundary at
the top of :mod:`ml.artifacts.store` — joblib is pickle, and only files this
application wrote are ever loaded — and the note in
:mod:`ml.artifacts.prediction` on why a column the model does not know is
refused instead of dropped.

**Raw uploaded datasets are still never persisted.** What is stored is a fitted
model and a manifest of column *names*, kinds and dtypes. No cell value, no
row, and no copy of the uploaded file is written by anything in this package.
"""

from ml.artifacts.prediction import (
    DEFAULT_MAX_RECORDS,
    Prediction,
    PredictionResult,
    build_frame,
    predict,
)
from ml.artifacts.schema import (
    ARTIFACT_SCHEMA_VERSION,
    FeatureSpec,
    ModelArtifactMetadata,
    build_metadata,
)
from ml.artifacts.store import (
    MANIFEST_FILENAME,
    MODEL_FILENAME,
    STATE_AVAILABLE,
    STATE_CORRUPTED,
    STATE_NOT_AVAILABLE,
    ArtifactStatus,
    LoadedModel,
    LocalModelArtifactStore,
    ModelArtifactStore,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DEFAULT_MAX_RECORDS",
    "MANIFEST_FILENAME",
    "MODEL_FILENAME",
    "STATE_AVAILABLE",
    "STATE_CORRUPTED",
    "STATE_NOT_AVAILABLE",
    "ArtifactStatus",
    "FeatureSpec",
    "LoadedModel",
    "LocalModelArtifactStore",
    "ModelArtifactMetadata",
    "ModelArtifactStore",
    "Prediction",
    "PredictionResult",
    "build_frame",
    "build_metadata",
    "predict",
]
