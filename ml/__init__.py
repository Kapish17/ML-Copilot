"""ML Copilot machine-learning layer.

This package is deliberately independent of the HTTP backend. It receives a
standardised ``pandas.DataFrame`` and never sees files, uploads, request
objects or any format-specific type, so a future Excel, JSON, Parquet, SQL or
API ingestion adapter can feed exactly the same pipeline.

The intended flow is::

    ingestion -> DataFrame -> profiling -> configuration -> preprocessing -> training

Typical use::

    from ml import infer_configuration, prepare_dataset

    inferred = infer_configuration(profile, target_column="churn")
    prepared = prepare_dataset(frame, inferred.config, decisions=inferred.decisions)

Only configuration and preprocessing live here today. Model training arrives in
a later commit.
"""

from ml.features.config import PreprocessingConfig, validate_config
from ml.features.inference import InferredConfiguration, infer_configuration
from ml.pipelines.preparation import prepare_dataset
from ml.pipelines.preprocessing import build_preprocessor
from ml.pipelines.result import PreparedDataset

__all__ = [
    "InferredConfiguration",
    "PreparedDataset",
    "PreprocessingConfig",
    "build_preprocessor",
    "infer_configuration",
    "prepare_dataset",
    "validate_config",
]
