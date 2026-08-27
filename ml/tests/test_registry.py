"""Tests for the model registry and the model specification."""

from __future__ import annotations

import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from ml.errors import (
    IncompatibleTaskError,
    InvalidHyperparameterError,
    InvalidMetricError,
    UnknownModelError,
)
from ml.features.types import TaskType
from ml.models.registry import (
    ModelDefinition,
    ModelRegistry,
    default_registry,
    list_available_models,
)
from ml.models.spec import ModelSpec, build_estimator, get_model_spec, validate_spec

EXPECTED_CLASSIFIERS = {
    "logistic_regression",
    "random_forest_classifier",
    "hist_gradient_boosting_classifier",
}
EXPECTED_REGRESSORS = {
    "linear_regression",
    "random_forest_regressor",
    "hist_gradient_boosting_regressor",
}


def test_default_registry_holds_the_expected_models() -> None:
    """The shipped registry is a small, deliberate model suite."""
    registry = default_registry()

    assert set(registry.identifiers(TaskType.CLASSIFICATION)) == EXPECTED_CLASSIFIERS
    assert set(registry.identifiers(TaskType.REGRESSION)) == EXPECTED_REGRESSORS
    assert len(registry.identifiers()) == 6


def test_registry_instances_are_independent() -> None:
    """No shared mutable registry: each call builds its own."""
    first, second = default_registry(), default_registry()
    assert first is not second
    assert first.identifiers() == second.identifiers()


def test_definitions_carry_what_training_needs() -> None:
    """Each definition has a stable id, a name, a task and a factory."""
    definition = default_registry().get("random_forest_classifier")

    assert definition.identifier == "random_forest_classifier"
    assert definition.display_name == "Random Forest Classifier"
    assert definition.task_type is TaskType.CLASSIFICATION
    assert definition.supports_random_state is True
    assert definition.supports_probabilities is True
    assert definition.default_parameters["n_estimators"] > 0
    assert isinstance(definition.factory(), BaseEstimator)


def test_default_parameters_cannot_be_mutated() -> None:
    """A caller cannot edit the defaults another caller will read."""
    definition = default_registry().get("logistic_regression")
    with pytest.raises(TypeError):
        definition.default_parameters["max_iter"] = 1  # type: ignore[index]


def test_linear_regression_declares_no_random_state() -> None:
    """Deterministic estimators are not handed a seed they cannot accept."""
    assert default_registry().get("linear_regression").supports_random_state is False


def test_unknown_model_lists_the_alternatives() -> None:
    """A wrong identifier fails with the available options."""
    with pytest.raises(UnknownModelError) as exc_info:
        default_registry().get("xgboost")
    assert "logistic_regression" in exc_info.value.details["available_models"]


def test_registry_can_be_extended_without_mutation() -> None:
    """Adding a model returns a new registry; the original is untouched.

    This is the path a later commit takes to add XGBoost or LightGBM: append a
    definition, and training needs no change.
    """
    registry = default_registry()
    extra = ModelDefinition(
        identifier="future_boosted_trees",
        display_name="Future Boosted Trees",
        task_type=TaskType.CLASSIFICATION,
        factory=LogisticRegression,
    )
    extended = registry.extend(extra)

    assert extended.contains("future_boosted_trees")
    assert not registry.contains("future_boosted_trees")
    assert len(extended.identifiers()) == len(registry.identifiers()) + 1


def test_duplicate_identifiers_are_refused() -> None:
    """Two models cannot share an identifier."""
    definition = default_registry().get("logistic_regression")
    with pytest.raises(ValueError, match="Duplicate"):
        ModelRegistry((definition, definition))


def test_list_available_models_is_serialisable() -> None:
    """The listing is plain data, suitable for a future agent tool."""
    records = list_available_models()

    assert {record["identifier"] for record in records} == (
        EXPECTED_CLASSIFIERS | EXPECTED_REGRESSORS
    )
    assert all(isinstance(record["default_parameters"], dict) for record in records)
    assert all(record["description"] for record in records)


def test_list_available_models_filters_by_task() -> None:
    """The task may be given as an enum member or as its string value."""
    from_string = list_available_models("classification")
    from_enum = list_available_models(TaskType.CLASSIFICATION)

    assert {record["identifier"] for record in from_string} == EXPECTED_CLASSIFIERS
    assert from_string == from_enum


def test_get_model_spec_fills_in_the_task() -> None:
    """A specification knows which task its model solves."""
    spec = get_model_spec("linear_regression")

    assert spec.model_name == "linear_regression"
    assert spec.task_type is TaskType.REGRESSION


def test_spec_hyperparameters_are_frozen() -> None:
    """A specification cannot be edited after it has been validated."""
    spec = get_model_spec("random_forest_classifier", hyperparameters={"max_depth": 3})
    with pytest.raises(TypeError):
        spec.hyperparameters["max_depth"] = 9  # type: ignore[index]


def test_spec_is_serialisable() -> None:
    """A specification renders as plain values."""
    payload = get_model_spec(
        "logistic_regression", hyperparameters={"C": 0.5}, random_state=7
    ).as_dict()

    assert payload == {
        "model_name": "logistic_regression",
        "task_type": "classification",
        "hyperparameters": {"C": 0.5},
        "random_state": 7,
        "primary_metric": None,
    }


def test_unknown_model_in_a_spec_is_rejected() -> None:
    """Validation refuses a model the registry does not know."""
    with pytest.raises(UnknownModelError):
        validate_spec(ModelSpec(model_name="not_a_model"))


def test_incompatible_task_is_rejected() -> None:
    """A regressor cannot be asked to solve a classification problem."""
    with pytest.raises(IncompatibleTaskError) as exc_info:
        validate_spec(ModelSpec(model_name="linear_regression"), TaskType.CLASSIFICATION)

    details = exc_info.value.details
    assert details["model_task_type"] == "regression"
    assert details["requested_task_type"] == "classification"
    assert "logistic_regression" in details["compatible_models"]


def test_task_declared_on_the_spec_is_also_checked() -> None:
    """A specification that contradicts the registry is refused."""
    with pytest.raises(IncompatibleTaskError):
        validate_spec(
            ModelSpec(model_name="linear_regression", task_type=TaskType.CLASSIFICATION)
        )


def test_unknown_hyperparameter_is_rejected() -> None:
    """A misspelled hyperparameter fails early, listing what is accepted."""
    with pytest.raises(InvalidHyperparameterError) as exc_info:
        validate_spec(
            ModelSpec(
                model_name="random_forest_classifier",
                hyperparameters={"n_estimatorz": 10},
            )
        )

    assert exc_info.value.details["unknown_parameters"] == ["n_estimatorz"]
    assert "n_estimators" in exc_info.value.details["accepted_parameters"]


def test_valid_hyperparameters_pass() -> None:
    """A real hyperparameter validates without complaint."""
    definition = validate_spec(
        ModelSpec(model_name="random_forest_classifier", hyperparameters={"max_depth": 4})
    )
    assert definition.identifier == "random_forest_classifier"


def test_invalid_primary_metric_is_rejected() -> None:
    """A primary metric from the wrong task is refused."""
    with pytest.raises(InvalidMetricError):
        validate_spec(ModelSpec(model_name="logistic_regression", primary_metric="rmse"))


def test_build_estimator_layers_defaults_then_overrides() -> None:
    """Registry defaults apply unless the specification replaces them."""
    definition = default_registry().get("random_forest_classifier")
    estimator = build_estimator(
        definition,
        ModelSpec(model_name=definition.identifier, hyperparameters={"n_estimators": 7}),
        fallback_random_state=42,
    )
    params = estimator.get_params()

    assert params["n_estimators"] == 7
    assert params["random_state"] == 42


def test_build_estimator_prefers_the_spec_seed() -> None:
    """An explicit seed beats the dataset's."""
    definition = default_registry().get("random_forest_classifier")
    estimator = build_estimator(
        definition,
        ModelSpec(model_name=definition.identifier, random_state=99),
        fallback_random_state=42,
    )
    assert estimator.get_params()["random_state"] == 99


def test_build_estimator_skips_seeds_for_deterministic_models() -> None:
    """Linear regression is never handed a random_state it cannot take."""
    definition = default_registry().get("linear_regression")
    estimator = build_estimator(
        definition,
        ModelSpec(model_name=definition.identifier),
        fallback_random_state=42,
    )
    assert "random_state" not in estimator.get_params()
