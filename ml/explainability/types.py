"""Vocabulary for the explainability layer.

The string values are stable and serialisable, because a future agent will read
them rather than parse prose.
"""

from __future__ import annotations

from enum import Enum


class ExplanationMethod(str, Enum):
    """How an explanation was produced.

    The method is always reported, so a reader never has to guess whether a
    number came from SHAP or from the coarser permutation fallback.
    """

    SHAP = "shap"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    NONE = "none"


class ExplanationStatus(str, Enum):
    """Whether an explanation could be produced at all."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ContributionDirection(str, Enum):
    """Which way a feature moved this particular prediction.

    These describe an association learned by the model, not a cause in the
    world: "increases_prediction" means the model's output was higher with this
    value than without it, nothing more.
    """

    INCREASES = "increases_prediction"
    DECREASES = "decreases_prediction"
    NEUTRAL = "no_effect"


class ExplainerKind(str, Enum):
    """Which family of SHAP explainer suits an estimator."""

    TREE = "tree"
    LINEAR = "linear"
    UNSUPPORTED = "unsupported"


def direction_of(contribution: float) -> ContributionDirection:
    """Classify a signed contribution."""
    if contribution > 0:
        return ContributionDirection.INCREASES
    if contribution < 0:
        return ContributionDirection.DECREASES
    return ContributionDirection.NEUTRAL
