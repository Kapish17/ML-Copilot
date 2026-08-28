"""Structured results of an explanation.

Everything a caller receives is a plain dataclass of Python scalars. No SHAP
explainer, numpy array, sklearn estimator or DataFrame appears in a result, and
``summary()`` is JSON-safe by construction — which is what lets a future agent
or API hand these straight on.

The wording throughout is deliberately about *association*: a contribution says
how the model's output moved, not what would happen in the world if the value
were different.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ml.explainability.types import (
    ContributionDirection,
    ExplanationMethod,
    ExplanationStatus,
    direction_of,
)

#: Wording attached to every result, so a consumer cannot miss the caveat.
CAUSALITY_DISCLAIMER = (
    "Explanations describe how this model's output varies with these features. "
    "They are associations learned from the training data, not causal effects."
)


def to_float(value: Any) -> float | None:
    """Convert any numeric-like value to a finite Python float, or ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def to_plain(value: Any) -> Any:
    """Render a cell value as a JSON-safe scalar.

    Numbers stay numbers where they can; anything else becomes a string, so a
    category or a timestamp is still readable in a serialised result.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return bool(value)
    number = to_float(value)
    if number is not None and not isinstance(value, (str, bytes)):
        return number
    return str(value)


@dataclass(frozen=True)
class FeatureImportance:
    """One feature's overall influence on a model, and where it ranks."""

    feature: str
    importance: float
    rank: int

    def as_dict(self) -> dict[str, Any]:
        """Render the entry as plain, JSON-friendly values."""
        return {
            "feature": self.feature,
            "importance": self.importance,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class FeatureContribution:
    """One feature's effect on a single prediction."""

    feature: str
    feature_value: float | None
    contribution: float
    direction: ContributionDirection
    rank: int
    raw_value: Any = None

    def as_dict(self) -> dict[str, Any]:
        """Render the contribution as plain, JSON-friendly values."""
        return {
            "feature": self.feature,
            "feature_value": self.feature_value,
            "raw_value": to_plain(self.raw_value),
            "contribution": self.contribution,
            "direction": self.direction.value,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class GlobalExplanation:
    """What generally drives a model's predictions.

    Importances are ranked descending. Under SHAP they are the mean absolute
    SHAP value per feature — how far, on average, each feature moved the
    output in either direction. Under the permutation fallback they are the
    mean drop in the model's score when that feature is shuffled.
    """

    status: ExplanationStatus
    method: ExplanationMethod
    model_name: str
    task_type: str
    feature_importances: tuple[FeatureImportance, ...] = ()
    sample_count: int = 0
    feature_count: int = 0
    explainer: str | None = None
    aggregation: str | None = None
    explained_output: str | None = None
    reason: str | None = None
    warnings: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        """True when importances were produced."""
        return self.status is ExplanationStatus.AVAILABLE

    def top(self, count: int) -> tuple[FeatureImportance, ...]:
        """Return the ``count`` most influential features."""
        return self.feature_importances[:count]

    def summary(self, top_n: int | None = None) -> dict[str, Any]:
        """Return a serialisable description of the explanation."""
        entries = self.top(top_n) if top_n is not None else self.feature_importances
        return {
            "scope": "global",
            "status": self.status.value,
            "method": self.method.value,
            "explainer": self.explainer,
            "model_name": self.model_name,
            "task_type": self.task_type,
            "aggregation": self.aggregation,
            "explained_output": self.explained_output,
            "sample_count": self.sample_count,
            "feature_count": self.feature_count,
            "feature_importances": [entry.as_dict() for entry in entries],
            "reason": self.reason,
            "warnings": list(self.warnings),
            "disclaimer": CAUSALITY_DISCLAIMER,
        }


@dataclass(frozen=True)
class LocalExplanation:
    """Why one particular row received the prediction it did.

    ``base_value`` is the model's average output over the background data; the
    contributions are what moved this row away from it. For a classifier they
    are expressed on the model's own output scale — log-odds for a linear
    model, the tree ensemble's output for a forest — not as probabilities.
    """

    status: ExplanationStatus
    method: ExplanationMethod
    model_name: str
    task_type: str
    prediction: Any = None
    probability: float | None = None
    probabilities: dict[str, float] | None = None
    predicted_class: str | None = None
    explained_class: str | None = None
    positive_class: str | None = None
    base_value: float | None = None
    feature_contributions: tuple[FeatureContribution, ...] = ()
    sample_count: int = 0
    feature_count: int = 0
    explainer: str | None = None
    reason: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def available(self) -> bool:
        """True when contributions were produced."""
        return self.status is ExplanationStatus.AVAILABLE

    def top(self, count: int) -> tuple[FeatureContribution, ...]:
        """Return the ``count`` contributions with the largest magnitude."""
        return self.feature_contributions[:count]

    def summary(self, top_n: int | None = None) -> dict[str, Any]:
        """Return a serialisable description of the explanation."""
        entries = (
            self.top(top_n) if top_n is not None else self.feature_contributions
        )
        return {
            "scope": "local",
            "status": self.status.value,
            "method": self.method.value,
            "explainer": self.explainer,
            "model_name": self.model_name,
            "task_type": self.task_type,
            "prediction": to_plain(self.prediction),
            "probability": self.probability,
            "probabilities": dict(self.probabilities) if self.probabilities else None,
            "predicted_class": self.predicted_class,
            "explained_class": self.explained_class,
            "positive_class": self.positive_class,
            "base_value": self.base_value,
            "sample_count": self.sample_count,
            "feature_count": self.feature_count,
            "feature_contributions": [entry.as_dict() for entry in entries],
            "reason": self.reason,
            "warnings": list(self.warnings),
            "disclaimer": CAUSALITY_DISCLAIMER,
        }


def rank_importances(
    names: list[str], values: list[float]
) -> tuple[FeatureImportance, ...]:
    """Rank features by importance, largest first.

    Ties are broken by feature name so the ordering is stable across runs.
    """
    pairs = sorted(
        zip(names, values, strict=True), key=lambda item: (-item[1], item[0])
    )
    return tuple(
        FeatureImportance(feature=name, importance=value, rank=position)
        for position, (name, value) in enumerate(pairs, start=1)
    )


def rank_contributions(
    names: list[str],
    contributions: list[float],
    feature_values: list[float | None],
    raw_values: list[Any],
) -> tuple[FeatureContribution, ...]:
    """Rank contributions by magnitude, largest absolute effect first.

    Sign is preserved — the ranking is by how much a feature mattered, not by
    which direction it pushed. Ties are broken by feature name.
    """
    rows = sorted(
        zip(names, contributions, feature_values, raw_values, strict=True),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    return tuple(
        FeatureContribution(
            feature=name,
            feature_value=value,
            contribution=contribution,
            direction=direction_of(contribution),
            rank=position,
            raw_value=raw,
        )
        for position, (name, contribution, value, raw) in enumerate(rows, start=1)
    )
