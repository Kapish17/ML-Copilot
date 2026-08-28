"""Tests for global feature importance."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from ml.errors import (
    EmptyExplanationDataError,
    InvalidTrainedModelError,
    MissingFeatureColumnsError,
)
from ml.explainability import explain_global, get_feature_importance
from ml.explainability.config import ExplanationConfig
from ml.explainability.results import CAUSALITY_DISCLAIMER
from ml.explainability.types import ExplanationMethod, ExplanationStatus
from ml.models.result import TrainedModel
from ml.pipelines.result import PreparedDataset


def test_global_explanation_for_binary_classification(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A tree classifier is summarised by SHAP over the training rows."""
    explanation = explain_global(forest_model, classification_prepared.X_train_raw)

    assert explanation.status is ExplanationStatus.AVAILABLE
    assert explanation.method is ExplanationMethod.SHAP
    assert explanation.explainer == "TreeExplainer"
    assert explanation.model_name == "random_forest_classifier"
    assert explanation.task_type == "classification"
    assert explanation.sample_count == classification_prepared.train_row_count
    assert explanation.feature_count == len(forest_model.feature_names)


def test_global_explanation_for_regression(
    linear_regression_model: TrainedModel, regression_prepared: PreparedDataset
) -> None:
    """A linear regressor is summarised the same way, via its own explainer."""
    explanation = explain_global(
        linear_regression_model, regression_prepared.X_train_raw
    )

    assert explanation.status is ExplanationStatus.AVAILABLE
    assert explanation.method is ExplanationMethod.SHAP
    assert explanation.explainer == "LinearExplainer"
    assert explanation.task_type == "regression"
    assert explanation.sample_count == regression_prepared.train_row_count


def test_every_transformed_feature_is_reported(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """The result covers exactly the model's features, by their real names."""
    explanation = explain_global(forest_model, classification_prepared.X_train_raw)
    reported = [entry.feature for entry in explanation.feature_importances]

    assert set(reported) == set(forest_model.feature_names)
    assert len(reported) == len(set(reported)), "no feature is listed twice"


def test_feature_names_are_the_ones_preprocessing_produced(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """One-hot and pass-through names survive; nothing is called x0."""
    explanation = explain_global(forest_model, classification_prepared.X_train_raw)
    reported = {entry.feature for entry in explanation.feature_importances}

    assert "income" in reported
    assert "segment_retail" in reported
    assert not any(name.startswith("x") and name[1:].isdigit() for name in reported)


def test_importances_are_ranked_descending(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Ranks run 1..n and the values never increase down the list."""
    entries = explain_global(
        forest_model, classification_prepared.X_train_raw
    ).feature_importances
    values = [entry.importance for entry in entries]

    assert [entry.rank for entry in entries] == list(range(1, len(entries) + 1))
    assert values == sorted(values, reverse=True)
    assert all(value >= 0 for value in values), "mean absolute values are non-negative"


def test_the_informative_features_outrank_the_noise(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """The ranking reflects the signal the synthetic data actually carries.

    The label is built from income and tenure, so those must matter more than
    any single one-hot segment column.
    """
    entries = explain_global(forest_model, classification_prepared.X_train_raw)
    ranked = {entry.feature: entry.rank for entry in entries.feature_importances}

    assert ranked["income"] <= 2
    assert ranked["tenure_months"] <= 2


def test_top_n_truncates_the_ranking(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A caller can ask for only the most influential features."""
    explanation = explain_global(
        forest_model, classification_prepared.X_train_raw, top_n=3
    )

    assert len(explanation.feature_importances) == 3
    assert explanation.feature_count == len(forest_model.feature_names)
    assert len(explanation.top(2)) == 2


def test_binary_importances_report_which_output_was_summarised(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A two-output model says it summarised the positive class."""
    explanation = explain_global(forest_model, classification_prepared.X_train_raw)

    assert "positive class" in (explanation.explained_output or "")
    assert any("positive class" in note for note in explanation.warnings)


def test_multiclass_importances_are_averaged_over_classes(
    multiclass_model: TrainedModel, multiclass_prepared: PreparedDataset
) -> None:
    """Three classes are averaged, and the result says so rather than hiding it."""
    explanation = explain_global(multiclass_model, multiclass_prepared.X_train_raw)

    assert explanation.status is ExplanationStatus.AVAILABLE
    assert "3 classes" in (explanation.explained_output or "")
    assert any("averaged over the 3 classes" in note for note in explanation.warnings)


def test_boosting_model_reports_a_single_output(
    boosting_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A model that emits one margin is summarised on that margin."""
    explanation = explain_global(boosting_model, classification_prepared.X_train_raw)

    assert explanation.status is ExplanationStatus.AVAILABLE
    assert explanation.explained_output == "the model's single output"


def test_explanations_are_deterministic(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """The same model and rows give the same importances twice."""
    first = explain_global(forest_model, classification_prepared.X_train_raw)
    second = explain_global(forest_model, classification_prepared.X_train_raw)

    assert [entry.as_dict() for entry in first.feature_importances] == [
        entry.as_dict() for entry in second.feature_importances
    ]


def test_sampling_is_deterministic_and_reported(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A row cap is honoured, announced, and reproducible."""
    config = ExplanationConfig(max_explanation_rows=40)
    first = explain_global(
        forest_model, classification_prepared.X_train_raw, config=config
    )
    second = explain_global(
        forest_model, classification_prepared.X_train_raw, config=config
    )

    assert first.sample_count == 40
    assert any("Sampled 40 of 240" in note for note in first.warnings)
    assert [entry.as_dict() for entry in first.feature_importances] == [
        entry.as_dict() for entry in second.feature_importances
    ]


def test_no_sampling_warning_when_the_data_fits(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Nothing is said about sampling when none happened."""
    explanation = explain_global(
        forest_model,
        classification_prepared.X_train_raw,
        config=ExplanationConfig(max_explanation_rows=10_000),
    )
    assert not any("Sampled" in note for note in explanation.warnings)


def test_reference_row_limit_applies_to_linear_models(
    logistic_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A small background is still enough to produce an explanation."""
    explanation = explain_global(
        logistic_model,
        classification_prepared.X_train_raw,
        config=ExplanationConfig(max_reference_rows=25),
    )

    assert explanation.status is ExplanationStatus.AVAILABLE
    assert explanation.explainer == "LinearExplainer"
    assert len(explanation.feature_importances) == len(logistic_model.feature_names)


def test_test_rows_can_also_be_summarised(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Explaining test rows is allowed: no target is used and nothing is fitted."""
    explanation = explain_global(forest_model, classification_prepared.X_test_raw)

    assert explanation.status is ExplanationStatus.AVAILABLE
    assert explanation.sample_count == classification_prepared.test_row_count


def test_summary_is_json_safe(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """The summary is plain data and carries the non-causality caveat."""
    explanation = explain_global(forest_model, classification_prepared.X_train_raw)
    summary = explanation.summary(top_n=3)

    json.dumps(summary)
    assert summary["scope"] == "global"
    assert summary["method"] == "shap"
    assert summary["explainer"] == "TreeExplainer"
    assert len(summary["feature_importances"]) == 3
    assert summary["feature_importances"][0]["rank"] == 1
    assert isinstance(summary["feature_importances"][0]["importance"], float)
    assert summary["disclaimer"] == CAUSALITY_DISCLAIMER


def test_get_feature_importance_returns_plain_records(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """The agent-shaped entry point returns facts, not objects."""
    records = get_feature_importance(
        forest_model, classification_prepared.X_train_raw, top_n=2
    )

    json.dumps(records)
    assert len(records) == 2
    assert set(records[0]) == {"feature", "importance", "rank"}
    assert records[0]["rank"] == 1


def test_an_unfitted_object_is_rejected(
    classification_prepared: PreparedDataset,
) -> None:
    """Explanations need a real trained model."""
    with pytest.raises(InvalidTrainedModelError) as exc_info:
        explain_global("not a model", classification_prepared.X_train_raw)
    assert exc_info.value.details["received_type"] == "str"


def test_missing_feature_columns_are_named(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A missing column fails with the column named."""
    incomplete = classification_prepared.X_train_raw.drop(columns=["income"])

    with pytest.raises(MissingFeatureColumnsError) as exc_info:
        explain_global(forest_model, incomplete)
    assert exc_info.value.details["missing_columns"] == ["income"]


def test_extra_columns_and_a_different_order_are_tolerated(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Only the fitted columns are used; the rest are ignored."""
    frame = classification_prepared.X_train_raw.copy()
    frame["unrelated"] = 1.0
    reordered = frame[list(reversed(frame.columns))]

    explanation = explain_global(forest_model, reordered)

    assert explanation.status is ExplanationStatus.AVAILABLE
    assert "unrelated" not in {
        entry.feature for entry in explanation.feature_importances
    }


def test_empty_data_is_rejected(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """There is nothing to explain about no rows."""
    with pytest.raises(EmptyExplanationDataError):
        explain_global(forest_model, classification_prepared.X_train_raw.iloc[:0])


def test_non_tabular_input_is_rejected(forest_model: TrainedModel) -> None:
    """The layer takes a DataFrame, not a dictionary."""
    with pytest.raises(Exception, match="DataFrame"):
        explain_global(forest_model, {"income": [1.0]})


def test_a_single_row_frame_is_acceptable_globally(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """One row is a small reference set, not an error."""
    explanation = explain_global(
        forest_model, classification_prepared.X_train_raw.iloc[[0]]
    )

    assert explanation.status is ExplanationStatus.AVAILABLE
    assert explanation.sample_count == 1


def test_a_series_is_accepted_as_one_row(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A Series is treated as a single row of features."""
    row: pd.Series = classification_prepared.X_train_raw.iloc[0]
    explanation = explain_global(forest_model, row)

    assert explanation.status is ExplanationStatus.AVAILABLE
    assert explanation.sample_count == 1
