"""Tests for explainer selection, row limits and result helpers.

These exercise the parts that decide *how* an explanation will be produced,
without running SHAP itself.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor

from ml.explainability.config import ExplanationConfig, limit_rows
from ml.explainability.results import (
    CAUSALITY_DISCLAIMER,
    GlobalExplanation,
    rank_contributions,
    rank_importances,
    to_plain,
)
from ml.explainability.shap_backend import (
    ShapUnavailable,
    normalise_values,
    select_output,
)
from ml.explainability.strategy import (
    LINEAR_EXPLAINER,
    TREE_EXPLAINER,
    select_explainer,
)
from ml.explainability.types import (
    ContributionDirection,
    ExplainerKind,
    ExplanationMethod,
    ExplanationStatus,
    direction_of,
)


# --------------------------------------------------------------------------
# Explainer selection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "estimator",
    [
        RandomForestClassifier(),
        HistGradientBoostingRegressor(),
        DecisionTreeRegressor(),
    ],
)
def test_tree_models_use_the_tree_explainer(estimator) -> None:
    """Tree ensembles are explained exactly by walking their trees."""
    plan = select_explainer(estimator)

    assert plan.kind is ExplainerKind.TREE
    assert plan.explainer_name == TREE_EXPLAINER
    assert plan.supported is True


@pytest.mark.parametrize(
    "estimator", [LogisticRegression(), LinearRegression(), Ridge()]
)
def test_linear_models_use_the_linear_explainer(estimator) -> None:
    """Linear models are explained from their coefficients."""
    plan = select_explainer(estimator)

    assert plan.kind is ExplainerKind.LINEAR
    assert plan.explainer_name == LINEAR_EXPLAINER


def test_an_unrecognised_model_is_unsupported_with_a_reason() -> None:
    """No explainer is forced onto a model it does not suit."""
    plan = select_explainer(KNeighborsClassifier())

    assert plan.kind is ExplainerKind.UNSUPPORTED
    assert plan.supported is False
    assert "KNeighborsClassifier" in (plan.reason or "")
    assert "kernel explainer" in (plan.reason or "")


def test_a_model_exposing_coefficients_is_treated_as_linear() -> None:
    """Structural checks catch linear models this package has not listed."""

    class _CustomLinear(LinearRegression):
        """A linear model the strategy has never heard of by name."""

    plan = select_explainer(_CustomLinear())
    assert plan.kind is ExplainerKind.LINEAR


def test_a_model_without_structure_is_unsupported() -> None:
    """A dummy estimator has neither trees nor coefficients."""
    assert select_explainer(DummyClassifier()).kind is ExplainerKind.UNSUPPORTED


# --------------------------------------------------------------------------
# Row limits and determinism
# --------------------------------------------------------------------------


def _frame(rows: int) -> pd.DataFrame:
    """A trivial frame of the requested length."""
    return pd.DataFrame({"value": range(rows)})


def test_rows_under_the_limit_are_left_alone() -> None:
    """Nothing is sampled when the data already fits."""
    frame = _frame(10)
    capped, sampled = limit_rows(frame, 50, random_state=42)

    assert sampled is False
    assert capped is frame


def test_rows_over_the_limit_are_sampled_and_reported() -> None:
    """Sampling happens, and the caller is told it happened."""
    capped, sampled = limit_rows(_frame(100), 20, random_state=42)

    assert sampled is True
    assert capped.shape[0] == 20


def test_sampling_is_deterministic_and_ordered() -> None:
    """The same seed gives the same rows in the same order."""
    first, _ = limit_rows(_frame(100), 20, random_state=42)
    second, _ = limit_rows(_frame(100), 20, random_state=42)

    assert list(first.index) == list(second.index)
    assert list(first.index) == sorted(first.index), "original order is kept"


def test_a_different_seed_samples_differently() -> None:
    """The seed genuinely drives the sample."""
    first, _ = limit_rows(_frame(100), 20, random_state=1)
    second, _ = limit_rows(_frame(100), 20, random_state=2)

    assert list(first.index) != list(second.index)


def test_configuration_defaults_are_serialisable() -> None:
    """The configuration renders as plain values."""
    payload = ExplanationConfig().as_dict()

    json.dumps(payload)
    assert payload["max_reference_rows"] > 0
    assert payload["max_explanation_rows"] > 0
    assert payload["random_state"] == 42


# --------------------------------------------------------------------------
# Normalising SHAP output
# --------------------------------------------------------------------------


def test_two_dimensional_values_become_one_output() -> None:
    """A regressor's values are reshaped to a single output."""
    values = normalise_values(
        np.array([[1.0, 2.0], [3.0, 4.0]]), 0.5, feature_count=2
    )

    assert values.values.shape == (2, 2, 1)
    assert values.output_count == 1
    assert values.base_values.tolist() == [0.5]


def test_a_list_of_arrays_becomes_one_output_per_class() -> None:
    """Older SHAP versions return a list; the shape is normalised the same."""
    values = normalise_values(
        [np.array([[1.0, 2.0]]), np.array([[-1.0, -2.0]])],
        np.array([0.4, 0.6]),
        feature_count=2,
    )

    assert values.values.shape == (1, 2, 2)
    assert values.output_count == 2
    assert values.base_values.tolist() == [0.4, 0.6]


def test_a_single_base_value_is_repeated_across_outputs() -> None:
    """A scalar expected value is broadcast to every output."""
    values = normalise_values(np.zeros((1, 2, 3)), 0.25, feature_count=2)

    assert values.base_values.tolist() == [0.25, 0.25, 0.25]


def test_a_wrong_feature_count_is_refused() -> None:
    """Values that do not line up with the features are not guessed at."""
    with pytest.raises(ShapUnavailable, match="feature values"):
        normalise_values(np.zeros((2, 5)), 0.0, feature_count=3)


def test_regression_selects_the_single_output() -> None:
    """A regressor has one output and no explained class."""
    values = normalise_values(np.zeros((2, 3)), 1.0, feature_count=3)
    selection = select_output(
        values, is_classification=False, classes=None, target_class=None
    )

    assert selection.index == 0
    assert selection.negated is False
    assert selection.explained_class is None


def test_one_output_per_class_selects_by_class() -> None:
    """A multiclass model is indexed by the class being explained."""
    values = normalise_values(np.zeros((1, 3, 3)), np.zeros(3), feature_count=3)
    selection = select_output(
        values, is_classification=True, classes=["a", "b", "c"], target_class="b"
    )

    assert selection.index == 1
    assert selection.negated is False
    assert selection.explained_class == "b"


def test_a_single_binary_output_explains_the_positive_class() -> None:
    """One output means the positive class, and it is reported as such."""
    values = normalise_values(np.zeros((1, 3)), 0.2, feature_count=3)
    selection = select_output(
        values, is_classification=True, classes=["no", "yes"], target_class="yes"
    )

    assert selection.index == 0
    assert selection.negated is False
    assert selection.explained_class == "yes"
    assert selection.note is None


def test_the_negative_class_negates_a_single_binary_output() -> None:
    """The two binary margins are mirror images, and the note says so."""
    values = normalise_values(np.zeros((1, 3)), 0.2, feature_count=3)
    selection = select_output(
        values, is_classification=True, classes=["no", "yes"], target_class="no"
    )

    assert selection.negated is True
    assert selection.explained_class == "no"
    assert "negation" in (selection.note or "")


def test_outputs_that_cannot_be_matched_to_classes_are_refused() -> None:
    """A shape that cannot be mapped is reported, not guessed at."""
    values = normalise_values(np.zeros((1, 3)), 0.0, feature_count=3)
    with pytest.raises(ShapUnavailable, match="per-class explanation"):
        select_output(
            values,
            is_classification=True,
            classes=["a", "b", "c"],
            target_class="a",
        )


# --------------------------------------------------------------------------
# Ranking and rendering
# --------------------------------------------------------------------------


def test_importances_are_ranked_descending() -> None:
    """The most influential feature comes first and is rank 1."""
    ranked = rank_importances(["a", "b", "c"], [0.1, 0.9, 0.5])

    assert [entry.feature for entry in ranked] == ["b", "c", "a"]
    assert [entry.rank for entry in ranked] == [1, 2, 3]


def test_importance_ties_are_broken_by_name() -> None:
    """Equal importances still produce a stable order."""
    ranked = rank_importances(["z", "a"], [0.5, 0.5])
    assert [entry.feature for entry in ranked] == ["a", "z"]


def test_contributions_rank_by_magnitude_and_keep_their_sign() -> None:
    """A strong negative contribution outranks a weak positive one."""
    ranked = rank_contributions(
        ["a", "b", "c"], [0.2, -0.9, 0.0], [1.0, 2.0, 3.0], [None, None, None]
    )

    assert [entry.feature for entry in ranked] == ["b", "a", "c"]
    assert ranked[0].contribution == pytest.approx(-0.9)
    assert ranked[0].direction is ContributionDirection.DECREASES
    assert ranked[1].direction is ContributionDirection.INCREASES
    assert ranked[2].direction is ContributionDirection.NEUTRAL


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.5, ContributionDirection.INCREASES),
        (-0.5, ContributionDirection.DECREASES),
        (0.0, ContributionDirection.NEUTRAL),
    ],
)
def test_direction_of(value: float, expected: ContributionDirection) -> None:
    """Direction follows the sign of the contribution."""
    assert direction_of(value) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1.5, 1.5), (True, True), ("basic", "basic"), (None, None), (np.float64(2), 2.0)],
)
def test_to_plain_renders_json_safe_scalars(value, expected) -> None:
    """Numpy scalars become floats; anything else becomes a readable string."""
    assert to_plain(value) == expected


def test_an_unavailable_result_still_summarises() -> None:
    """A result with no numbers still says what it is and why."""
    explanation = GlobalExplanation(
        status=ExplanationStatus.UNAVAILABLE,
        method=ExplanationMethod.NONE,
        model_name="some_model",
        task_type="classification",
        reason="nothing worked",
    )
    summary = explanation.summary()

    json.dumps(summary)
    assert explanation.available is False
    assert summary["status"] == "unavailable"
    assert summary["method"] == "none"
    assert summary["reason"] == "nothing worked"
    assert summary["feature_importances"] == []
    assert summary["disclaimer"] == CAUSALITY_DISCLAIMER
