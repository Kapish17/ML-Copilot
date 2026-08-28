"""ML Copilot machine-learning layer.

This package is deliberately independent of the HTTP backend. It receives a
standardised ``pandas.DataFrame`` and never sees files, uploads, request
objects or any format-specific type, so a future Excel, JSON, Parquet, SQL or
API ingestion adapter can feed exactly the same pipeline.

The intended flow is::

    ingestion -> DataFrame -> profiling -> configuration -> preprocessing
              -> cross-validation -> model selection -> final training
              -> untouched test evaluation -> explanation

Typical use::

    from ml import infer_configuration, prepare_dataset, select_and_evaluate_best_model

    inferred = infer_configuration(profile, target_column="churn")
    prepared = prepare_dataset(frame, inferred.config, decisions=inferred.decisions)
    outcome = select_and_evaluate_best_model(prepared, folds=5)

    outcome.selected_model_name   # chosen by cross-validation alone
    outcome.final_test_score      # the single untouched-test measurement

    explain_global(outcome.final_model, prepared.X_train_raw)
    explain_prediction(outcome.final_model, prepared.X_test_raw.iloc[[0]])

Cross-validation selects the model; the held-out test set is reserved for the
final evaluation.

Explanations describe model behaviour and associations; they do not establish
causal relationships.

Hyperparameter optimisation and experiment tracking are not implemented.
"""

from ml.evaluation.metrics import EvaluationMetrics, MetricDefinition, MetricDirection
from ml.features.config import PreprocessingConfig, validate_config
from ml.features.inference import InferredConfiguration, infer_configuration
from ml.models.baselines import BaselineResult, evaluate_baseline
from ml.models.comparison import (
    ComparisonEntry,
    ModelComparison,
    SelectionStrategy,
    compare_models,
    format_comparison_table,
    select_best_model,
)
from ml.models.registry import ModelRegistry, default_registry, list_available_models
from ml.models.result import TrainedModel
from ml.models.selection import ModelSelectionResult, select_and_evaluate_best_model
from ml.models.spec import ModelSpec, get_model_spec
from ml.models.training import train_model
from ml.pipelines.preparation import prepare_dataset
from ml.pipelines.preprocessing import build_preprocessor
from ml.pipelines.result import PreparedDataset

# Imported after ``ml.models`` so the cross-validation module is already loaded
# as one of its dependencies; see the note in ``ml.models.comparison``.
from ml.evaluation.cross_validation import (  # noqa: E402
    DEFAULT_FOLDS,
    CrossValidationResult,
    FoldResult,
    cross_validate_model,
)
from ml.explainability import (  # noqa: E402
    ExplanationConfig,
    GlobalExplanation,
    LocalExplanation,
    explain_global,
    explain_prediction,
    get_feature_importance,
)

__all__ = [
    "DEFAULT_FOLDS",
    "BaselineResult",
    "ComparisonEntry",
    "CrossValidationResult",
    "EvaluationMetrics",
    "ExplanationConfig",
    "FoldResult",
    "GlobalExplanation",
    "LocalExplanation",
    "InferredConfiguration",
    "MetricDefinition",
    "MetricDirection",
    "ModelComparison",
    "ModelRegistry",
    "ModelSelectionResult",
    "ModelSpec",
    "PreparedDataset",
    "PreprocessingConfig",
    "SelectionStrategy",
    "TrainedModel",
    "build_preprocessor",
    "compare_models",
    "cross_validate_model",
    "default_registry",
    "evaluate_baseline",
    "explain_global",
    "explain_prediction",
    "format_comparison_table",
    "get_feature_importance",
    "get_model_spec",
    "infer_configuration",
    "list_available_models",
    "prepare_dataset",
    "select_and_evaluate_best_model",
    "select_best_model",
    "train_model",
    "validate_config",
]
