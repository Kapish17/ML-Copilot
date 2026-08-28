"""Choosing an explainer for an estimator.

There is no single SHAP explainer that suits every model. A tree ensemble can
be explained exactly and quickly by walking its trees; a linear model can be
explained from its coefficients and the background distribution; a model that
is neither would need the general-purpose kernel explainer, which is slow
enough on real data to be the wrong default.

So the estimator's family decides, and anything unrecognised is reported as
unsupported — with a reason — rather than being forced through an explainer
that would be wrong, slow, or both. The permutation fallback then covers global
importance for those models.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from ml.explainability.types import ExplainerKind

TREE_EXPLAINER = "TreeExplainer"
LINEAR_EXPLAINER = "LinearExplainer"

#: Estimator classes SHAP's tree explainer handles exactly.
TREE_ESTIMATORS: tuple[type, ...] = (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

#: Estimator classes SHAP's linear explainer handles from their coefficients.
LINEAR_ESTIMATORS: tuple[type, ...] = (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)

#: Third-party gradient-boosting libraries SHAP's tree explainer also supports.
#: Listed by module so adding one of these later needs no change here.
TREE_MODULE_PREFIXES = ("xgboost", "lightgbm", "catboost")


@dataclass(frozen=True)
class ExplainerPlan:
    """Which explainer to use for an estimator, or why none applies."""

    kind: ExplainerKind
    explainer_name: str | None = None
    reason: str | None = None

    @property
    def supported(self) -> bool:
        """True when SHAP can explain this estimator."""
        return self.kind is not ExplainerKind.UNSUPPORTED


def _module_root(estimator: BaseEstimator) -> str:
    """Return the top-level module an estimator's class comes from."""
    return type(estimator).__module__.split(".")[0]


def select_explainer(estimator: BaseEstimator) -> ExplainerPlan:
    """Decide how to explain an estimator.

    Recognised tree and linear families are matched by class first, then a
    couple of structural checks catch models from libraries this package has
    never heard of: a gradient-boosting library by module name, and anything
    exposing ``coef_`` as linear.

    Args:
        estimator: The fitted estimator inside a trained pipeline.

    Returns:
        ExplainerPlan: The explainer to build, or an explained refusal.
    """
    if isinstance(estimator, TREE_ESTIMATORS):
        return ExplainerPlan(kind=ExplainerKind.TREE, explainer_name=TREE_EXPLAINER)
    if isinstance(estimator, LINEAR_ESTIMATORS):
        return ExplainerPlan(kind=ExplainerKind.LINEAR, explainer_name=LINEAR_EXPLAINER)

    if _module_root(estimator) in TREE_MODULE_PREFIXES:
        return ExplainerPlan(kind=ExplainerKind.TREE, explainer_name=TREE_EXPLAINER)
    if hasattr(estimator, "coef_"):
        return ExplainerPlan(kind=ExplainerKind.LINEAR, explainer_name=LINEAR_EXPLAINER)

    return ExplainerPlan(
        kind=ExplainerKind.UNSUPPORTED,
        reason=(
            f"No SHAP explainer is configured for {type(estimator).__name__}. "
            "Tree ensembles and linear models are supported; anything else "
            "would need the kernel explainer, which is too slow to run by "
            "default."
        ),
    )
