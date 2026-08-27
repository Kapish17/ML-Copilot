"""The estimator registry.

A registry is an immutable collection of :class:`ModelDefinition` records. It
holds no module-level mutable state: :func:`default_registry` builds a fresh
one on every call, and :meth:`ModelRegistry.extend` returns a new registry
rather than mutating the existing one, so one caller can never change what
another sees.

The registry is the only place that knows *which* estimators exist. Training,
evaluation and comparison work from a definition, so adding XGBoost or LightGBM
later means appending one definition — no change to ``train_model``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression

from ml.errors import UnknownModelError
from ml.features.types import TaskType

#: Trees used by the forest models. Enough for a stable score, small enough to
#: keep a comparison run quick.
DEFAULT_FOREST_SIZE = 200
#: Iterations allowed for the logistic solver before it gives up converging.
DEFAULT_LOGISTIC_MAX_ITER = 1000


@dataclass(frozen=True)
class ModelDefinition:
    """Everything the training layer needs to know about one estimator."""

    identifier: str
    display_name: str
    task_type: TaskType
    factory: Callable[..., BaseEstimator]
    default_parameters: Mapping[str, Any] = field(default_factory=dict)
    supports_random_state: bool = False
    supports_probabilities: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        """Freeze the default parameters so a caller cannot edit them in place."""
        object.__setattr__(
            self, "default_parameters", MappingProxyType(dict(self.default_parameters))
        )

    def as_dict(self) -> dict[str, Any]:
        """Render the definition as plain, JSON-friendly values.

        Shaped for a future agent tool: everything needed to choose a model,
        and nothing that cannot be serialised.
        """
        return {
            "identifier": self.identifier,
            "display_name": self.display_name,
            "task_type": self.task_type.value,
            "default_parameters": dict(self.default_parameters),
            "supports_random_state": self.supports_random_state,
            "supports_probabilities": self.supports_probabilities,
            "description": self.description,
        }


class ModelRegistry:
    """An immutable lookup of model identifiers to estimator definitions."""

    def __init__(self, definitions: Iterable[ModelDefinition]) -> None:
        """Build a registry from a set of definitions.

        Args:
            definitions: The models the registry should expose.

        Raises:
            ValueError: If two definitions share an identifier.
        """
        items = tuple(definitions)
        identifiers = [item.identifier for item in items]
        duplicates = sorted({name for name in identifiers if identifiers.count(name) > 1})
        if duplicates:
            raise ValueError(
                "Duplicate model identifier(s): " + ", ".join(duplicates) + "."
            )
        self._definitions = items

    @property
    def definitions(self) -> tuple[ModelDefinition, ...]:
        """Every definition in the registry, in declaration order."""
        return self._definitions

    def list_models(self, task_type: TaskType | None = None) -> tuple[ModelDefinition, ...]:
        """Return the definitions for a task, or all of them.

        Args:
            task_type: Restrict to models solving this task, or ``None`` for all.

        Returns:
            tuple[ModelDefinition, ...]: The matching definitions.
        """
        if task_type is None:
            return self._definitions
        return tuple(item for item in self._definitions if item.task_type is task_type)

    def identifiers(self, task_type: TaskType | None = None) -> tuple[str, ...]:
        """Return the identifiers for a task, or all of them."""
        return tuple(item.identifier for item in self.list_models(task_type))

    def contains(self, identifier: str) -> bool:
        """Return True when the registry knows this identifier."""
        return any(item.identifier == identifier for item in self._definitions)

    def get(self, identifier: str) -> ModelDefinition:
        """Look up one definition.

        Args:
            identifier: The model's stable identifier.

        Returns:
            ModelDefinition: The matching definition.

        Raises:
            UnknownModelError: If no model has that identifier.
        """
        for item in self._definitions:
            if item.identifier == identifier:
                return item
        available = list(self.identifiers())
        raise UnknownModelError(
            f"Unknown model '{identifier}'. Available models: "
            + ", ".join(available)
            + ".",
            details={"model_name": identifier, "available_models": available},
        )

    def extend(self, *definitions: ModelDefinition) -> ModelRegistry:
        """Return a new registry with extra models added.

        This is how a later commit adds XGBoost or LightGBM: append a
        definition, and every existing training and comparison path picks it up
        unchanged.
        """
        return ModelRegistry((*self._definitions, *definitions))


CLASSIFICATION_DEFINITIONS: tuple[ModelDefinition, ...] = (
    ModelDefinition(
        identifier="logistic_regression",
        display_name="Logistic Regression",
        task_type=TaskType.CLASSIFICATION,
        factory=LogisticRegression,
        default_parameters={"max_iter": DEFAULT_LOGISTIC_MAX_ITER},
        supports_random_state=True,
        supports_probabilities=True,
        description="Linear baseline. Fast, and its coefficients are readable.",
    ),
    ModelDefinition(
        identifier="random_forest_classifier",
        display_name="Random Forest Classifier",
        task_type=TaskType.CLASSIFICATION,
        factory=RandomForestClassifier,
        default_parameters={"n_estimators": DEFAULT_FOREST_SIZE},
        supports_random_state=True,
        supports_probabilities=True,
        description="Bagged decision trees. Robust with little tuning.",
    ),
    ModelDefinition(
        identifier="hist_gradient_boosting_classifier",
        display_name="Histogram Gradient Boosting Classifier",
        task_type=TaskType.CLASSIFICATION,
        factory=HistGradientBoostingClassifier,
        default_parameters={},
        supports_random_state=True,
        supports_probabilities=True,
        description="Boosted trees. Usually the strongest of the three on tabular data.",
    ),
)

REGRESSION_DEFINITIONS: tuple[ModelDefinition, ...] = (
    ModelDefinition(
        identifier="linear_regression",
        display_name="Linear Regression",
        task_type=TaskType.REGRESSION,
        factory=LinearRegression,
        default_parameters={},
        supports_random_state=False,
        description="Ordinary least squares. The reference point for regression.",
    ),
    ModelDefinition(
        identifier="random_forest_regressor",
        display_name="Random Forest Regressor",
        task_type=TaskType.REGRESSION,
        factory=RandomForestRegressor,
        default_parameters={"n_estimators": DEFAULT_FOREST_SIZE},
        supports_random_state=True,
        description="Bagged decision trees for continuous targets.",
    ),
    ModelDefinition(
        identifier="hist_gradient_boosting_regressor",
        display_name="Histogram Gradient Boosting Regressor",
        task_type=TaskType.REGRESSION,
        factory=HistGradientBoostingRegressor,
        default_parameters={},
        supports_random_state=True,
        description="Boosted trees for continuous targets.",
    ),
)


def default_registry() -> ModelRegistry:
    """Build the registry shipped with ML Copilot.

    A fresh instance is returned each call, so nothing shared can be mutated.
    """
    return ModelRegistry((*CLASSIFICATION_DEFINITIONS, *REGRESSION_DEFINITIONS))


def list_available_models(
    task_type: TaskType | str | None = None, *, registry: ModelRegistry | None = None
) -> tuple[dict[str, Any], ...]:
    """List the models available, as plain serialisable records.

    Deterministic and free of sklearn objects, so a future agent can call it as
    a tool and read the result directly.

    Args:
        task_type: Restrict to one task, as an enum member or its string value.
        registry: Registry to read; the default registry when omitted.

    Returns:
        tuple[dict, ...]: One record per model.
    """
    resolved = TaskType(task_type) if task_type is not None else None
    active = registry or default_registry()
    return tuple(item.as_dict() for item in active.list_models(resolved))
