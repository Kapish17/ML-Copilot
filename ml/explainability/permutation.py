"""The fallback: permutation importance.

When no SHAP explainer suits a model, global importance can still be measured
by a cruder but model-agnostic method: shuffle one feature's values and see how
far the model's score falls. A feature the model leans on will hurt when it is
scrambled; one it ignores will not.

Two honest limits come with it.

**It needs the labels.** Permutation importance is defined as a drop in *score*,
and a score needs the right answers. Nothing about the target enters the model
— it is already trained and is not touched — but without ``y`` there is nothing
to measure the drop against, so the fallback is simply unavailable.

**It is global only.** It says how much a feature matters across a set of rows.
It cannot say why one particular row got its prediction, and this package never
pretends otherwise: a local explanation that SHAP cannot produce is reported as
unavailable rather than filled in from here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.inspection import permutation_importance

from ml.explainability.config import ExplanationConfig

PERMUTATION_AGGREGATION = "mean score drop when the feature is shuffled"


def permutation_global_importance(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    config: ExplanationConfig,
) -> np.ndarray:
    """Measure how much each feature's shuffling costs the model's score.

    The estimator is used read-only — ``permutation_importance`` scores copies
    of the data and never refits anything.

    Args:
        estimator: The already-fitted estimator.
        features: Transformed features, with Commit 3's column names.
        target: The true values for those rows.
        config: Repeat count and seed, so the result is reproducible.

    Returns:
        numpy.ndarray: Mean score drop per feature, in column order.
    """
    result = permutation_importance(
        estimator,
        features,
        target,
        n_repeats=config.permutation_repeats,
        random_state=config.random_state,
        n_jobs=None,
    )
    return np.asarray(result.importances_mean, dtype="float64")
