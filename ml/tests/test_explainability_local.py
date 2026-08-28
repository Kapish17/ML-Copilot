"""Tests for explaining a single prediction."""

from __future__ import annotations

import json

import pytest

from ml.errors import ExplainabilityError, InvalidExplanationRowError
from ml.explainability import explain_prediction
from ml.explainability.results import CAUSALITY_DISCLAIMER
from ml.explainability.types import (
    ContributionDirection,
    ExplanationMethod,
    ExplanationStatus,
)
from ml.models.result import TrainedModel
from ml.pipelines.result import PreparedDataset

TOLERANCE = 1e-4


def _row(prepared: PreparedDataset, position: int = 0):
    """Return one raw test row as a one-row frame."""
    return prepared.X_test_raw.iloc[[position]]


def _transformed(model: TrainedModel, prepared: PreparedDataset, position: int = 0):
    """Return the same row after the model's own preprocessing."""
    return model.preprocessor.transform(_row(prepared, position))


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------


def test_local_explanation_for_regression(
    forest_regression_model: TrainedModel, regression_prepared: PreparedDataset
) -> None:
    """A regressor's prediction is explained on its own numeric scale."""
    explanation = explain_prediction(
        forest_regression_model, _row(regression_prepared)
    )

    assert explanation.status is ExplanationStatus.AVAILABLE
    assert explanation.method is ExplanationMethod.SHAP
    assert explanation.task_type == "regression"
    assert explanation.predicted_class is None
    assert explanation.explained_class is None
    assert explanation.probability is None
    assert explanation.sample_count == 1


def test_regression_contributions_add_up_to_the_prediction(
    forest_regression_model: TrainedModel, regression_prepared: PreparedDataset
) -> None:
    """SHAP's defining property: base value plus contributions is the output.

    This is the strongest available check that the values are real rather than
    merely present.
    """
    explanation = explain_prediction(
        forest_regression_model, _row(regression_prepared)
    )
    total = explanation.base_value + sum(
        entry.contribution for entry in explanation.feature_contributions
    )

    assert total == pytest.approx(explanation.prediction, rel=TOLERANCE)


def test_linear_regression_contributions_add_up(
    linear_regression_model: TrainedModel, regression_prepared: PreparedDataset
) -> None:
    """The same additivity holds through the linear explainer."""
    explanation = explain_prediction(
        linear_regression_model,
        _row(regression_prepared),
        background=regression_prepared.X_train_raw,
    )
    total = explanation.base_value + sum(
        entry.contribution for entry in explanation.feature_contributions
    )

    assert explanation.explainer == "LinearExplainer"
    assert total == pytest.approx(explanation.prediction, rel=TOLERANCE)


def test_the_prediction_matches_the_trained_model(
    forest_regression_model: TrainedModel, regression_prepared: PreparedDataset
) -> None:
    """The explanation reports the model's own answer, not a recomputation."""
    row = _row(regression_prepared)
    explanation = explain_prediction(forest_regression_model, row)

    assert explanation.prediction == pytest.approx(
        forest_regression_model.predict(row)[0]
    )


# --------------------------------------------------------------------------
# Binary classification
# --------------------------------------------------------------------------


def test_local_explanation_for_binary_classification(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Predicted, explained and positive classes are each named."""
    row = _row(classification_prepared)
    explanation = explain_prediction(forest_model, row)

    assert explanation.status is ExplanationStatus.AVAILABLE
    assert explanation.prediction == forest_model.predict(row)[0]
    assert explanation.predicted_class == str(explanation.prediction)
    assert explanation.explained_class == explanation.predicted_class
    assert explanation.positive_class == "yes"


def test_the_positive_class_follows_the_training_convention(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """The positive class is the estimator's last sorted class, as in Commit 4."""
    explanation = explain_prediction(forest_model, _row(classification_prepared))

    assert explanation.positive_class == str(forest_model.estimator.classes_[-1])


def test_probabilities_are_reported_and_consistent(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Every class gets a probability, and they match the model's own."""
    row = _row(classification_prepared)
    explanation = explain_prediction(forest_model, row)
    expected = forest_model.predict_proba(row)[0]

    assert explanation.probabilities is not None
    assert set(explanation.probabilities) == {"no", "yes"}
    assert sum(explanation.probabilities.values()) == pytest.approx(1.0)
    assert explanation.probabilities["yes"] == pytest.approx(expected[1])
    assert explanation.probability == pytest.approx(
        explanation.probabilities[explanation.explained_class]
    )


def test_forest_contributions_add_up_to_the_class_probability(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """For this explainer the values live in probability space, and add up."""
    explanation = explain_prediction(forest_model, _row(classification_prepared))
    total = explanation.base_value + sum(
        entry.contribution for entry in explanation.feature_contributions
    )

    assert total == pytest.approx(explanation.probability, rel=TOLERANCE)


def test_logistic_contributions_add_up_to_the_decision_margin(
    logistic_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A single-output model's values add up to its log-odds margin.

    The margin from ``decision_function`` is for the positive class, so
    explaining the negative class must give exactly its negation.
    """
    row = _row(classification_prepared)
    margin = float(
        logistic_model.estimator.decision_function(
            _transformed(logistic_model, classification_prepared)
        )[0]
    )

    for target, sign in (("yes", 1.0), ("no", -1.0)):
        explanation = explain_prediction(
            logistic_model,
            row,
            background=classification_prepared.X_train_raw,
            target_class=target,
        )
        total = explanation.base_value + sum(
            entry.contribution for entry in explanation.feature_contributions
        )
        assert total == pytest.approx(sign * margin, rel=TOLERANCE, abs=TOLERANCE)
        assert explanation.explained_class == target


def test_explaining_the_negative_class_records_the_convention(
    boosting_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A negated single-output explanation says that is what happened."""
    explanation = explain_prediction(
        boosting_model, _row(classification_prepared), target_class="no"
    )

    assert explanation.explained_class == "no"
    assert explanation.positive_class == "yes"
    assert any("negation" in note for note in explanation.warnings)


def test_explaining_a_chosen_class_overrides_the_predicted_one(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Predicted and explained classes are separate, and both are reported."""
    row = _row(classification_prepared)
    predicted = str(forest_model.predict(row)[0])
    other = "yes" if predicted == "no" else "no"

    explanation = explain_prediction(forest_model, row, target_class=other)

    assert explanation.predicted_class == predicted
    assert explanation.explained_class == other
    assert explanation.probability == pytest.approx(
        explanation.probabilities[other]
    )


def test_binary_class_contributions_are_mirror_images(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """With one output per class, the two explanations are opposites."""
    row = _row(classification_prepared)
    positive = explain_prediction(forest_model, row, target_class="yes")
    negative = explain_prediction(forest_model, row, target_class="no")

    by_feature = {entry.feature: entry.contribution for entry in negative.feature_contributions}
    for entry in positive.feature_contributions:
        assert by_feature[entry.feature] == pytest.approx(-entry.contribution, abs=1e-9)


# --------------------------------------------------------------------------
# Multiclass
# --------------------------------------------------------------------------


def test_multiclass_explains_the_predicted_class(
    multiclass_model: TrainedModel, multiclass_prepared: PreparedDataset
) -> None:
    """A three-class problem is not collapsed into a binary one."""
    row = _row(multiclass_prepared)
    explanation = explain_prediction(multiclass_model, row)

    assert explanation.status is ExplanationStatus.AVAILABLE
    assert explanation.explained_class == str(multiclass_model.predict(row)[0])
    assert explanation.probabilities is not None
    assert len(explanation.probabilities) == 3
    assert sum(explanation.probabilities.values()) == pytest.approx(1.0)


def test_multiclass_can_explain_another_class(
    multiclass_model: TrainedModel, multiclass_prepared: PreparedDataset
) -> None:
    """Each class has its own contributions, and they differ."""
    row = _row(multiclass_prepared)
    first = explain_prediction(multiclass_model, row, target_class="low")
    second = explain_prediction(multiclass_model, row, target_class="high")

    assert first.explained_class == "low"
    assert second.explained_class == "high"
    assert [entry.contribution for entry in first.feature_contributions] != [
        entry.contribution for entry in second.feature_contributions
    ]


def test_multiclass_contributions_add_up_to_the_class_probability(
    multiclass_model: TrainedModel, multiclass_prepared: PreparedDataset
) -> None:
    """Additivity holds per class for a multiclass forest."""
    explanation = explain_prediction(multiclass_model, _row(multiclass_prepared))
    total = explanation.base_value + sum(
        entry.contribution for entry in explanation.feature_contributions
    )

    assert total == pytest.approx(explanation.probability, rel=TOLERANCE)


# --------------------------------------------------------------------------
# Contributions themselves
# --------------------------------------------------------------------------


def test_contributions_cover_every_feature_by_name(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Names are the transformed ones, and every feature is present once."""
    explanation = explain_prediction(forest_model, _row(classification_prepared))
    reported = [entry.feature for entry in explanation.feature_contributions]

    assert set(reported) == set(forest_model.feature_names)
    assert len(reported) == len(set(reported))
    assert "segment_business" in reported


def test_contributions_are_ranked_by_magnitude(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """The feature that mattered most comes first, whichever way it pushed."""
    entries = explain_prediction(
        forest_model, _row(classification_prepared)
    ).feature_contributions
    magnitudes = [abs(entry.contribution) for entry in entries]

    assert [entry.rank for entry in entries] == list(range(1, len(entries) + 1))
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_direction_matches_the_sign_of_every_contribution(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Direction is derived from the number, never asserted separately."""
    for entry in explain_prediction(
        forest_model, _row(classification_prepared)
    ).feature_contributions:
        if entry.contribution > 0:
            assert entry.direction is ContributionDirection.INCREASES
        elif entry.contribution < 0:
            assert entry.direction is ContributionDirection.DECREASES
        else:
            assert entry.direction is ContributionDirection.NEUTRAL


def test_feature_values_are_what_the_model_saw(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Each contribution carries the transformed value it belongs to."""
    transformed = _transformed(forest_model, classification_prepared).iloc[0]
    explanation = explain_prediction(forest_model, _row(classification_prepared))

    for entry in explanation.feature_contributions:
        assert entry.feature_value == pytest.approx(float(transformed[entry.feature]))


def test_raw_values_are_carried_for_pass_through_columns(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A scaled number is easier to read next to its original value."""
    raw = _row(classification_prepared).iloc[0]
    explanation = explain_prediction(forest_model, _row(classification_prepared))
    by_feature = {entry.feature: entry for entry in explanation.feature_contributions}

    assert by_feature["income"].raw_value == pytest.approx(raw["income"])
    assert by_feature["segment_retail"].raw_value is None, "one-hot has no raw column"


def test_top_n_truncates_the_contributions(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A caller can ask for only the largest effects."""
    explanation = explain_prediction(
        forest_model, _row(classification_prepared), top_n=2
    )

    assert len(explanation.feature_contributions) == 2
    assert explanation.feature_count == len(forest_model.feature_names)


def test_local_explanations_are_deterministic(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """The same row explained twice gives the same numbers."""
    first = explain_prediction(forest_model, _row(classification_prepared))
    second = explain_prediction(forest_model, _row(classification_prepared))

    assert first.summary() == second.summary()


def test_summary_is_json_safe(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """The summary is plain data and carries the non-causality caveat."""
    summary = explain_prediction(
        forest_model, _row(classification_prepared)
    ).summary(top_n=3)

    json.dumps(summary)
    assert summary["scope"] == "local"
    assert summary["method"] == "shap"
    assert summary["predicted_class"] in {"yes", "no"}
    assert summary["positive_class"] == "yes"
    assert isinstance(summary["base_value"], float)
    assert len(summary["feature_contributions"]) == 3
    entry = summary["feature_contributions"][0]
    assert set(entry) == {
        "feature",
        "feature_value",
        "raw_value",
        "contribution",
        "direction",
        "rank",
    }
    assert entry["direction"] in {
        "increases_prediction",
        "decreases_prediction",
        "no_effect",
    }
    assert summary["disclaimer"] == CAUSALITY_DISCLAIMER


# --------------------------------------------------------------------------
# Invalid requests
# --------------------------------------------------------------------------


def test_more_than_one_row_is_rejected(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A local explanation covers exactly one row."""
    with pytest.raises(InvalidExplanationRowError) as exc_info:
        explain_prediction(forest_model, classification_prepared.X_test_raw.iloc[:3])
    assert exc_info.value.details["row_count"] == 3


def test_a_series_row_is_accepted(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """One row may be given as a Series."""
    explanation = explain_prediction(
        forest_model, classification_prepared.X_test_raw.iloc[0]
    )
    assert explanation.status is ExplanationStatus.AVAILABLE


def test_a_linear_model_needs_background_rows(
    logistic_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Without a reference distribution the request is refused, with advice."""
    with pytest.raises(ExplainabilityError, match="background"):
        explain_prediction(logistic_model, _row(classification_prepared))


def test_a_tree_model_needs_no_background(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """Tree explainers read the trees, so no reference data is required."""
    explanation = explain_prediction(forest_model, _row(classification_prepared))
    assert explanation.status is ExplanationStatus.AVAILABLE


def test_an_unknown_target_class_is_rejected(
    forest_model: TrainedModel, classification_prepared: PreparedDataset
) -> None:
    """A class the model has never seen cannot be explained."""
    with pytest.raises(ExplainabilityError) as exc_info:
        explain_prediction(
            forest_model, _row(classification_prepared), target_class="maybe"
        )
    assert exc_info.value.details["available_classes"] == ["no", "yes"]
