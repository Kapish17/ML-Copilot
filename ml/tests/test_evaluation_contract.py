"""The evaluation contract, stated as tests.

The pipeline has always kept selection and final evaluation apart —
:mod:`ml.models.selection` cross-validates on the training rows, retrains the
winner on the full training portion and measures it once on rows nothing has
touched, and :mod:`ml.tests.test_leakage` proves the preprocessor never learns
from the test split. Those tests check the machinery.

This module checks the *contract as a reader of a finished record sees it*:
that the record says which number chose the model and which number measured
it, that it explains the choice from its own numbers, and that it carries what
a reproduction attempt would need. A guarantee nobody can see in the output is
one that quietly stops holding.
"""

from __future__ import annotations

import pytest

from ml.experiments import create_experiment_run
from ml.experiments.run import ExperimentRun, SelectionSection, selection_rationale
from ml.models.selection import select_and_evaluate_best_model
from ml.pipelines.result import PreparedDataset
from ml.tests.factories import learnable_classification_frame

CANDIDATES = ["logistic_regression", "random_forest_classifier"]
FOLDS = 3


@pytest.fixture(scope="module")
def run(classification_prepared: PreparedDataset) -> ExperimentRun:
    """One complete cross-validated run, recorded."""
    outcome = select_and_evaluate_best_model(
        classification_prepared, models=CANDIDATES, folds=FOLDS
    )
    return create_experiment_run(
        learnable_classification_frame(),
        classification_prepared,
        outcome,
        name="contract run",
        source_format="csv",
    )


@pytest.fixture(scope="module")
def holdout_run(classification_prepared: PreparedDataset) -> ExperimentRun:
    """The same pipeline under the strategy that does use the test set."""
    outcome = select_and_evaluate_best_model(
        classification_prepared, models=CANDIDATES, strategy="holdout"
    )
    return create_experiment_run(
        learnable_classification_frame(),
        classification_prepared,
        outcome,
        name="holdout run",
    )


# --------------------------------------------------------------------------
# Which number chose the model, and which number measured it
# --------------------------------------------------------------------------


def test_the_record_names_the_data_each_score_came_from(run: ExperimentRun) -> None:
    """Two scores, two provenances, both written down."""
    assert run.selection.scored_on == "training_folds"
    assert run.selection.uses_test_data is False
    assert run.evaluation.is_unbiased is True


def test_the_selection_score_is_not_the_held_out_score(run: ExperimentRun) -> None:
    """They are separate measurements and the record keeps them separate."""
    assert run.selection.selection_score is not None
    assert run.evaluation.primary_metric_value is not None
    assert (
        run.selection.selection_score is not run.evaluation.primary_metric_value
    ), "the two scores must not be the same object read twice"


def test_the_held_out_rows_are_the_split_the_preprocessor_never_saw(
    run: ExperimentRun, classification_prepared: PreparedDataset
) -> None:
    """The measured rows are the test split, whole and unmixed.

    Read from the record rather than from the pipeline object: this is the
    number a reader is given, and it should be the number that was measured.
    """
    assert run.evaluation.test_row_count == classification_prepared.test_row_count
    assert run.preprocessing.train_row_count == classification_prepared.train_row_count
    assert (
        run.evaluation.test_row_count + run.preprocessing.train_row_count
        == run.dataset.row_count - run.preprocessing.rows_dropped_missing_target
    )


def test_a_holdout_run_says_its_final_score_is_not_independent(
    holdout_run: ExperimentRun,
) -> None:
    """The strategy that spends the test set says so in three places."""
    assert holdout_run.selection.uses_test_data is True
    assert holdout_run.evaluation.is_unbiased is False
    assert "selection_used_test_data" in {
        item["code"] for item in holdout_run.evaluation.diagnostics
    }


# --------------------------------------------------------------------------
# Why the winner won
# --------------------------------------------------------------------------


def test_the_record_says_why_the_winner_won(run: ExperimentRun) -> None:
    """The sentence the spec asked for, from the run's own numbers."""
    rationale = run.selection.rationale

    assert rationale
    assert "cross-validation" in rationale
    assert "best" in rationale
    assert "held-out score is an independent measurement" in rationale


def test_the_rationale_names_the_selection_basis_not_the_test_score() -> None:
    """The held-out score is never offered as the reason for the choice."""
    sentence = selection_rationale(
        strategy="cross_validation",
        selected_model="Random Forest",
        primary_metric="f1",
        selection_score=0.8421,
        selection_score_std=0.031,
        folds=5,
        candidate_count=3,
    )

    assert sentence.startswith(
        "Random Forest selected because it achieved the best cross-validation F1"
    )
    assert "over 5 folds" in sentence
    assert "0.8421 ± 0.0310" in sentence
    assert "among 3 candidate models" in sentence


def test_a_holdout_rationale_admits_the_test_set_chose_the_model() -> None:
    """Same sentence, opposite provenance, and it says so."""
    sentence = selection_rationale(
        strategy="holdout",
        selected_model="Ridge",
        primary_metric="rmse",
        selection_score=5276.43,
        uses_test_data=True,
        candidate_count=2,
    )

    assert "best held-out Root mean squared error" in sentence
    assert "not independent of this choice" in sentence


def test_a_single_candidate_is_not_called_the_best_of_anything() -> None:
    """"Best of one" would overstate a comparison that never happened."""
    sentence = selection_rationale(
        strategy="cross_validation",
        selected_model="Linear Regression",
        primary_metric="r2",
        selection_score=0.71,
        folds=5,
        candidate_count=1,
    )

    assert "was the only candidate model" in sentence
    assert "best" not in sentence


def test_the_rationale_makes_no_claim_about_quality() -> None:
    """It explains a choice; it does not endorse the result."""
    sentence = selection_rationale(
        strategy="cross_validation",
        selected_model="Random Forest",
        primary_metric="f1",
        selection_score=0.51,
        folds=5,
        candidate_count=4,
    ).lower()

    for phrase in ("good", "strong", "accurate", "reliable", "excellent", "poor"):
        assert phrase not in sentence


def test_a_record_written_before_rationales_still_explains_itself() -> None:
    """The field is additive; the property recomposes what is missing."""
    section = SelectionSection(
        strategy="cross_validation",
        primary_metric="f1",
        selected_model="random_forest",
        folds=5,
        candidate_models=("logistic_regression", "random_forest"),
        selection_score=0.84,
    )

    assert section.rationale is None
    assert "cross-validation F1" in section.selection_rationale
    assert section.as_dict()["rationale"] == section.selection_rationale


def test_the_rationale_survives_a_round_trip(run: ExperimentRun) -> None:
    """What was stored is what comes back."""
    restored = ExperimentRun.from_dict(run.to_dict())

    assert restored.selection.rationale == run.selection.rationale
    assert restored.evaluation.diagnostics == run.evaluation.diagnostics


# --------------------------------------------------------------------------
# What a reproduction attempt would need
# --------------------------------------------------------------------------


def test_the_record_carries_everything_a_rerun_would_need(
    run: ExperimentRun, classification_prepared: PreparedDataset
) -> None:
    """Seed, data identity, feature schema and configuration, in one record."""
    assert run.environment.random_state == classification_prepared.config.random_state
    assert run.preprocessing.random_state == classification_prepared.config.random_state
    assert run.dataset.fingerprint and run.dataset.fingerprint_algorithm == "sha256"
    assert run.dataset.columns and run.dataset.dtypes
    assert run.preprocessing.transformed_feature_names
    assert run.feature_count == len(run.preprocessing.transformed_feature_names)
    assert run.preprocessing.config
    assert run.configuration_hash in run.experiment_id


def test_the_same_configuration_hashes_the_same_across_two_runs(
    classification_prepared: PreparedDataset,
) -> None:
    """The hash covers the inputs, so a repeat is findable as a repeat."""
    frame = learnable_classification_frame()
    first, second = (
        create_experiment_run(
            frame,
            classification_prepared,
            select_and_evaluate_best_model(
                classification_prepared, models=CANDIDATES, folds=FOLDS
            ),
            name=label,
        )
        for label in ("first", "second")
    )

    assert first.configuration_hash == second.configuration_hash
    assert first.dataset.fingerprint == second.dataset.fingerprint
    assert first.experiment_id != second.experiment_id


def test_the_record_still_holds_no_data(run: ExperimentRun) -> None:
    """Diagnostics read the recorded numbers; they add no rows to the record."""
    text = str(run.to_dict())

    for column in run.dataset.columns:
        assert f"'{column}': [" not in text, "no column of values may appear"
    assert "DataFrame" not in text
