"""Model registry, training, baselines, comparison and selection.

``registry``    which estimators exist, and how to build them
``spec``        the validated request describing one training run
``baselines``   the naive reference a real model is measured against
``training``    fitting ``Pipeline(preprocessing, estimator)`` and scoring it
``comparison``  running several models and ranking them
``selection``   choosing a winner, then measuring it once on the test set
``result``      the structured outcome of one training run

Cross-validation itself lives in :mod:`ml.evaluation.cross_validation`, which
builds on ``training``.

The public functions — ``list_available_models``, ``get_model_spec``,
``train_model``, ``compare_models``, ``select_best_model``,
``select_and_evaluate_best_model`` — are deterministic and take and return
plain structures, so a later agent can call them as tools. No agent exists yet.
"""

from ml.models.baselines import (
    BaselineComparison,
    BaselineResult,
    compare_to_baseline,
    evaluate_baseline,
)
from ml.models.comparison import (
    ComparisonEntry,
    ModelComparison,
    ModelStatus,
    SelectionStrategy,
    compare_models,
    format_comparison_table,
    select_best_model,
)
from ml.models.registry import (
    ModelDefinition,
    ModelRegistry,
    default_registry,
    list_available_models,
)
from ml.models.result import DatasetInfo, TrainedModel
from ml.models.selection import (
    DEFAULT_SELECTION_STRATEGY,
    ModelSelectionResult,
    select_and_evaluate_best_model,
)
from ml.models.spec import ModelSpec, get_model_spec, validate_spec
from ml.models.training import train_model

__all__ = [
    "DEFAULT_SELECTION_STRATEGY",
    "BaselineComparison",
    "BaselineResult",
    "ComparisonEntry",
    "DatasetInfo",
    "ModelComparison",
    "ModelDefinition",
    "ModelRegistry",
    "ModelSelectionResult",
    "ModelSpec",
    "ModelStatus",
    "SelectionStrategy",
    "TrainedModel",
    "compare_models",
    "compare_to_baseline",
    "default_registry",
    "evaluate_baseline",
    "format_comparison_table",
    "get_model_spec",
    "list_available_models",
    "select_and_evaluate_best_model",
    "select_best_model",
    "train_model",
    "validate_spec",
]
