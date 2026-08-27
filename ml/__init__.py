"""ML Copilot machine-learning layer.

This package is deliberately independent of the HTTP backend. It receives a
standardised ``pandas.DataFrame`` and never sees files, uploads, request
objects or any format-specific type, so a future Excel, JSON, Parquet, SQL or
API ingestion adapter can feed exactly the same pipeline.

The intended flow is::

    ingestion -> DataFrame -> profiling -> configuration -> preprocessing
              -> training -> evaluation -> comparison -> best model

Typical use::

    from ml import compare_models, infer_configuration, prepare_dataset, select_best_model

    inferred = infer_configuration(profile, target_column="churn")
    prepared = prepare_dataset(frame, inferred.config, decisions=inferred.decisions)
    comparison = compare_models(prepared)
    best = select_best_model(comparison)

Hyperparameter optimisation, explainability and experiment tracking are not
implemented.
"""

from ml.evaluation.metrics import EvaluationMetrics, MetricDefinition, MetricDirection
from ml.features.config import PreprocessingConfig, validate_config
from ml.features.inference import InferredConfiguration, infer_configuration
from ml.models.baselines import BaselineResult, evaluate_baseline
from ml.models.comparison import (
    ComparisonEntry,
    ModelComparison,
    compare_models,
    select_best_model,
)
from ml.models.registry import ModelRegistry, default_registry, list_available_models
from ml.models.result import TrainedModel
from ml.models.spec import ModelSpec, get_model_spec
from ml.models.training import train_model
from ml.pipelines.preparation import prepare_dataset
from ml.pipelines.preprocessing import build_preprocessor
from ml.pipelines.result import PreparedDataset

__all__ = [
    "BaselineResult",
    "ComparisonEntry",
    "EvaluationMetrics",
    "InferredConfiguration",
    "MetricDefinition",
    "MetricDirection",
    "ModelComparison",
    "ModelRegistry",
    "ModelSpec",
    "PreparedDataset",
    "PreprocessingConfig",
    "TrainedModel",
    "build_preprocessor",
    "compare_models",
    "default_registry",
    "evaluate_baseline",
    "get_model_spec",
    "infer_configuration",
    "list_available_models",
    "prepare_dataset",
    "select_best_model",
    "train_model",
    "validate_config",
]
