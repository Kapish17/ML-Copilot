"""The model specification: what to train, and how.

A ``ModelSpec`` is a small, serialisable request. It is validated against a
registry before anything is built, so a bad model name, an incompatible task or
a misspelled hyperparameter fails immediately with a message naming the valid
options — rather than deep inside scikit-learn, or worse, silently.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from sklearn.base import BaseEstimator

from ml.errors import (
    IncompatibleTaskError,
    InvalidHyperparameterError,
)
from ml.evaluation.metrics import resolve_primary_metric
from ml.features.types import TaskType
from ml.models.registry import ModelDefinition, ModelRegistry, default_registry

RANDOM_STATE_PARAMETER = "random_state"


@dataclass(frozen=True)
class ModelSpec:
    """A request to train one model.

    Attributes:
        model_name: Registry identifier, e.g. ``"random_forest_classifier"``.
        task_type: The task this model is expected to solve. Optional; when
            omitted it is taken from the registry definition, and when given it
            is checked against it.
        hyperparameters: Overrides applied on top of the model's defaults.
        random_state: Seed for estimators that accept one. Omitted means "use
            the seed the dataset was split with", which keeps a whole run
            reproducible from a single number.
        primary_metric: Metric used to rank this model against others. Omitted
            means the task's default.
    """

    model_name: str
    task_type: TaskType | None = None
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    random_state: int | None = None
    primary_metric: str | None = None

    def __post_init__(self) -> None:
        """Normalise the task type and freeze the hyperparameter mapping."""
        if self.task_type is not None:
            object.__setattr__(self, "task_type", TaskType(self.task_type))
        object.__setattr__(
            self, "hyperparameters", MappingProxyType(dict(self.hyperparameters))
        )

    def as_dict(self) -> dict[str, Any]:
        """Render the specification as plain, JSON-friendly values."""
        return {
            "model_name": self.model_name,
            "task_type": self.task_type.value if self.task_type else None,
            "hyperparameters": dict(self.hyperparameters),
            "random_state": self.random_state,
            "primary_metric": self.primary_metric,
        }


def validate_spec(
    spec: ModelSpec,
    task_type: TaskType | None = None,
    *,
    registry: ModelRegistry | None = None,
) -> ModelDefinition:
    """Check that a specification can actually be built and run.

    Args:
        spec: The requested model.
        task_type: The dataset's task, when known. Checked against both the
            specification and the registry definition.
        registry: Registry to look the model up in; the default when omitted.

    Returns:
        ModelDefinition: The definition the specification resolves to.

    Raises:
        UnknownModelError: The model is not in the registry.
        IncompatibleTaskError: The model does not solve the requested task.
        InvalidHyperparameterError: A hyperparameter is not accepted.
        InvalidMetricError: The requested primary metric does not exist.
    """
    active = registry or default_registry()
    definition = active.get(spec.model_name)

    for requested in (spec.task_type, task_type):
        if requested is not None and requested is not definition.task_type:
            compatible = list(active.identifiers(requested))
            raise IncompatibleTaskError(
                f"Model '{definition.identifier}' solves "
                f"{definition.task_type.value} problems, but a "
                f"{requested.value} problem was requested. Models for "
                f"{requested.value}: " + ", ".join(compatible) + ".",
                details={
                    "model_name": definition.identifier,
                    "model_task_type": definition.task_type.value,
                    "requested_task_type": requested.value,
                    "compatible_models": compatible,
                },
            )

    validate_hyperparameters(definition, spec.hyperparameters)
    resolve_primary_metric(definition.task_type, spec.primary_metric)
    return definition


def validate_hyperparameters(
    definition: ModelDefinition, hyperparameters: Mapping[str, Any]
) -> None:
    """Check that every hyperparameter name is accepted by the estimator.

    Names are checked against the estimator's own parameter list. Values are
    left to scikit-learn, which validates them properly when the estimator is
    constructed and fitted.

    Raises:
        InvalidHyperparameterError: If any name is not accepted.
    """
    if not hyperparameters:
        return
    accepted = set(definition.factory().get_params())
    unknown = sorted(set(hyperparameters) - accepted)
    if unknown:
        raise InvalidHyperparameterError(
            f"Model '{definition.identifier}' does not accept: "
            + ", ".join(unknown)
            + ".",
            details={
                "model_name": definition.identifier,
                "unknown_parameters": unknown,
                "accepted_parameters": sorted(accepted),
            },
        )


def build_estimator(
    definition: ModelDefinition,
    spec: ModelSpec,
    *,
    fallback_random_state: int | None = None,
) -> BaseEstimator:
    """Construct the estimator a specification describes.

    Defaults come from the registry, the specification's hyperparameters are
    layered on top, and a random seed is supplied to estimators that accept one
    unless the caller already set it explicitly.

    Args:
        definition: The resolved registry definition.
        spec: The requested model.
        fallback_random_state: Seed to use when the specification has none —
            normally the seed the dataset was split with.

    Returns:
        BaseEstimator: An unfitted estimator.

    Raises:
        InvalidHyperparameterError: If scikit-learn rejects the parameters.
    """
    parameters: dict[str, Any] = {
        **dict(definition.default_parameters),
        **dict(spec.hyperparameters),
    }
    if definition.supports_random_state and RANDOM_STATE_PARAMETER not in parameters:
        seed = spec.random_state if spec.random_state is not None else fallback_random_state
        if seed is not None:
            parameters[RANDOM_STATE_PARAMETER] = seed

    try:
        return definition.factory(**parameters)
    except TypeError as exc:
        raise InvalidHyperparameterError(
            f"Could not build '{definition.identifier}' with the given "
            f"hyperparameters: {exc}",
            details={
                "model_name": definition.identifier,
                "parameters": {key: str(value) for key, value in parameters.items()},
            },
        ) from exc


def get_model_spec(
    model_name: str,
    *,
    hyperparameters: Mapping[str, Any] | None = None,
    random_state: int | None = None,
    primary_metric: str | None = None,
    registry: ModelRegistry | None = None,
) -> ModelSpec:
    """Build and validate a specification for a registered model.

    A convenience entry point — and the shape a future agent tool would call:
    it takes plain values, validates them, and returns a structured request.

    Args:
        model_name: Registry identifier of the model.
        hyperparameters: Overrides for the model's defaults.
        random_state: Seed for estimators that accept one.
        primary_metric: Metric used to rank this model.
        registry: Registry to validate against; the default when omitted.

    Returns:
        ModelSpec: A validated specification, with its task type filled in.

    Raises:
        UnknownModelError: The model is not in the registry.
        InvalidHyperparameterError: A hyperparameter is not accepted.
        InvalidMetricError: The requested primary metric does not exist.
    """
    active = registry or default_registry()
    definition = active.get(model_name)
    spec = ModelSpec(
        model_name=model_name,
        task_type=definition.task_type,
        hyperparameters=hyperparameters or {},
        random_state=random_state,
        primary_metric=primary_metric,
    )
    validate_spec(spec, registry=active)
    return spec
