"""Explainability for trained models.

``types``         the vocabulary: method, status, direction
``config``        row limits and the seed that makes sampling deterministic
``results``       the structured, JSON-safe explanation objects
``strategy``      which SHAP explainer suits which estimator
``shap_backend``  running SHAP and normalising what it returns
``permutation``   the global-only fallback when SHAP cannot help
``service``       the public operations

The layer reads a trained model and never changes it: no refitting, no target
values, no mutation of the fitted pipeline.

Explanations describe model behaviour and associations; they do not establish
causal relationships.
"""

from ml.explainability.config import ExplanationConfig
from ml.explainability.results import (
    CAUSALITY_DISCLAIMER,
    FeatureContribution,
    FeatureImportance,
    GlobalExplanation,
    LocalExplanation,
)
from ml.explainability.service import (
    explain_global,
    explain_prediction,
    get_feature_importance,
)
from ml.explainability.strategy import ExplainerPlan, select_explainer
from ml.explainability.types import (
    ContributionDirection,
    ExplainerKind,
    ExplanationMethod,
    ExplanationStatus,
)

__all__ = [
    "CAUSALITY_DISCLAIMER",
    "ContributionDirection",
    "ExplainerKind",
    "ExplainerPlan",
    "ExplanationConfig",
    "ExplanationMethod",
    "ExplanationStatus",
    "FeatureContribution",
    "FeatureImportance",
    "GlobalExplanation",
    "LocalExplanation",
    "explain_global",
    "explain_prediction",
    "get_feature_importance",
    "select_explainer",
]
