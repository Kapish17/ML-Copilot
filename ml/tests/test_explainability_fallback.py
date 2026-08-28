"""Tests for the permutation fallback and for leaving the model untouched."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

from ml.explainability import explain_global, explain_prediction, get_feature_importance
from ml.explainability.config import ExplanationConfig
from ml.explainability.types import ExplanationMethod, ExplanationStatus
from ml.features.types import TaskType
from ml.models.registry import ModelDefinition, ModelRegistry, default_registry
from ml.models.result import TrainedModel
from ml.models.training import train_model
from ml.pipelines.result import PreparedDataset

UNSUPPORTED_MODEL = "k_nearest_neighbours"


def _registry_with_unsupported_model() -> ModelRegistry:
    """The default registry plus a model no SHAP explainer covers."""
    return default_registry().extend(
        ModelDefinition(
            identifier=UNSUPPORTED_MODEL,
            display_name="K Nearest Neighbours",
            task_type=TaskType.CLASSIFICATION,
            factory=KNeighborsClassifier,
            supports_probabilities=True,
        )
    )


@pytest.fixture(scope="module")
def unsupported_model(classification_prepared: PreparedDataset) -> TrainedModel:
    """A fitted model that neither SHAP explainer can handle."""
    return train_model(
        classification_prepared,
        UNSUPPORTED_MODEL,
        registry=_registry_with_unsupported_model(),
    )


# --------------------------------------------------------------------------
# Global fallback
# --------------------------------------------------------------------------


def test_an_unsupported_model_falls_back_to_permutation_importance(
    unsupported_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """SHAP cannot help, so the coarser method runs — and says it did."""
    explanation = explain_global(
        unsupported_model,
        classification_prepared.X_train_raw,
        classification_prepared.y_train,
    )

    assert explanation.status is ExplanationStatus.AVAILABLE
    assert explanation.method is ExplanationMethod.PERMUTATION_IMPORTANCE
    assert explanation.explainer == "permutation_importance"
    assert "shuffled" in (explanation.aggregation or "")


def test_the_fallback_records_why_shap_was_not_used(
    unsupported_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """The reason is kept, so a fallback is never silent."""
    explanation = explain_global(
        unsupported_model,
        classification_prepared.X_train_raw,
        classification_prepared.y_train,
    )
    reasons = " ".join(explanation.warnings)

    assert "SHAP was unavailable" in reasons
    assert "KNeighborsClassifier" in reasons


def test_the_fallback_covers_and_ranks_every_feature(
    unsupported_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Permutation importance uses the same feature names and ranking rules."""
    explanation = explain_global(
        unsupported_model,
        classification_prepared.X_train_raw,
        classification_prepared.y_train,
    )
    values = [entry.importance for entry in explanation.feature_importances]

    assert {entry.feature for entry in explanation.feature_importances} == set(
        unsupported_model.feature_names
    )
    assert values == sorted(values, reverse=True)
    assert explanation.sample_count == classification_prepared.train_row_count


def test_the_fallback_finds_the_informative_features(
    unsupported_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Shuffling a feature the label depends on must cost the model score."""
    explanation = explain_global(
        unsupported_model,
        classification_prepared.X_train_raw,
        classification_prepared.y_train,
    )
    ranked = {entry.feature: entry.rank for entry in explanation.feature_importances}

    assert ranked["income"] <= 2
    assert ranked["tenure_months"] <= 2


def test_the_fallback_is_deterministic(
    unsupported_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A fixed seed makes the shuffling reproducible."""
    config = ExplanationConfig(permutation_repeats=3)
    first = explain_global(
        unsupported_model,
        classification_prepared.X_train_raw,
        classification_prepared.y_train,
        config=config,
    )
    second = explain_global(
        unsupported_model,
        classification_prepared.X_train_raw,
        classification_prepared.y_train,
        config=config,
    )

    assert [entry.as_dict() for entry in first.feature_importances] == [
        entry.as_dict() for entry in second.feature_importances
    ]


def test_without_targets_the_fallback_is_unavailable_not_invented(
    unsupported_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Permutation importance needs labels; without them, nothing is made up."""
    explanation = explain_global(
        unsupported_model, classification_prepared.X_train_raw
    )

    assert explanation.status is ExplanationStatus.UNAVAILABLE
    assert explanation.method is ExplanationMethod.NONE
    assert explanation.feature_importances == ()
    assert "y_reference" in (explanation.reason or "")


def test_get_feature_importance_is_empty_when_unavailable(
    unsupported_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """The convenience helper returns no facts rather than fabricated ones."""
    assert (
        get_feature_importance(unsupported_model, classification_prepared.X_train_raw)
        == ()
    )


def test_the_fallback_summary_is_json_safe(
    unsupported_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A fallback result serialises like any other."""
    summary = explain_global(
        unsupported_model,
        classification_prepared.X_train_raw,
        classification_prepared.y_train,
        top_n=2,
    ).summary()

    json.dumps(summary)
    assert summary["method"] == "permutation_importance"
    assert len(summary["feature_importances"]) == 2


# --------------------------------------------------------------------------
# Local explanations are never faked
# --------------------------------------------------------------------------


def test_a_local_explanation_is_unavailable_for_an_unsupported_model(
    unsupported_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Permutation importance is global; it is not passed off as a local one."""
    row = classification_prepared.X_test_raw.iloc[[0]]
    explanation = explain_prediction(unsupported_model, row)

    assert explanation.status is ExplanationStatus.UNAVAILABLE
    assert explanation.method is ExplanationMethod.NONE
    assert explanation.feature_contributions == ()
    assert explanation.base_value is None
    assert "global measure" in (explanation.reason or "")


def test_an_unavailable_local_result_still_reports_the_prediction(
    unsupported_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """The prediction is a fact about the model, so it is still given."""
    row = classification_prepared.X_test_raw.iloc[[0]]
    explanation = explain_prediction(unsupported_model, row)

    assert explanation.prediction == unsupported_model.predict(row)[0]
    assert explanation.predicted_class == str(explanation.prediction)
    assert explanation.positive_class == "yes"
    assert explanation.probabilities is not None
    assert sum(explanation.probabilities.values()) == pytest.approx(1.0)
    json.dumps(explanation.summary())


# --------------------------------------------------------------------------
# Explaining changes nothing
# --------------------------------------------------------------------------


def _model_state(model: TrainedModel, prepared: PreparedDataset) -> dict:
    """Snapshot everything an explanation must leave alone."""
    branch = model.preprocessor.named_transformers_["numeric"]
    imputer = dict(branch.transformer_list)["values"].named_steps["impute"]
    scaler = dict(branch.transformer_list)["values"].named_steps["scale"]
    return {
        "predictions": model.predict(prepared.X_test_raw).tolist(),
        "probabilities": model.predict_proba(prepared.X_test_raw).tolist(),
        "imputer_statistics": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "estimator_params": {
            key: str(value) for key, value in model.estimator.get_params().items()
        },
        "feature_names": list(model.feature_names),
        "transformed": model.preprocessor.transform(prepared.X_test_raw).to_numpy(),
    }


def _assert_unchanged(before: dict, after: dict) -> None:
    """Compare two snapshots, arrays included."""
    for key in (
        "predictions",
        "probabilities",
        "imputer_statistics",
        "scaler_mean",
        "scaler_scale",
        "estimator_params",
        "feature_names",
    ):
        assert before[key] == after[key], f"{key} changed"
    assert np.array_equal(before["transformed"], after["transformed"])


def test_the_model_is_unchanged_by_a_global_explanation(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Explaining reads the model; it never refits or mutates it."""
    before = _model_state(forest_model, classification_prepared)
    pipeline, preprocessor, estimator = (
        forest_model.pipeline,
        forest_model.preprocessor,
        forest_model.estimator,
    )

    explain_global(forest_model, classification_prepared.X_train_raw)

    _assert_unchanged(before, _model_state(forest_model, classification_prepared))
    assert forest_model.pipeline is pipeline
    assert forest_model.preprocessor is preprocessor
    assert forest_model.estimator is estimator


def test_the_model_is_unchanged_by_a_local_explanation(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """The same holds for explaining a single row."""
    before = _model_state(forest_model, classification_prepared)

    explain_prediction(forest_model, classification_prepared.X_test_raw.iloc[[0]])

    _assert_unchanged(before, _model_state(forest_model, classification_prepared))


def test_the_preprocessing_is_not_refitted_on_explanation_data(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Explaining unfamiliar rows must not move the fitted statistics.

    The rows handed in are deliberately extreme. If the preprocessing were
    refitted on them, the imputer and scaler would shift; they do not.
    """
    branch = forest_model.preprocessor.named_transformers_["numeric"]
    values = dict(branch.transformer_list)["values"]
    before = (
        values.named_steps["impute"].statistics_.copy(),
        values.named_steps["scale"].mean_.copy(),
    )

    extreme = classification_prepared.X_test_raw.copy()
    extreme["income"] = 10_000_000.0
    extreme["tenure_months"] = np.nan
    explain_global(forest_model, extreme)

    after = (
        values.named_steps["impute"].statistics_,
        values.named_steps["scale"].mean_,
    )
    assert np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])


def test_the_fallback_does_not_change_the_model(
    unsupported_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Permutation importance shuffles copies, not the fitted estimator."""
    before = _model_state(unsupported_model, classification_prepared)

    explain_global(
        unsupported_model,
        classification_prepared.X_train_raw,
        classification_prepared.y_train,
        config=ExplanationConfig(permutation_repeats=2),
    )

    _assert_unchanged(before, _model_state(unsupported_model, classification_prepared))
