"""Tests for the signals raised about a finished run.

Two things are checked throughout, and the second matters as much as the first.

**Does the check fire when it should, and stay quiet when it should not?** A
diagnostic that fires on ordinary fold noise is one people learn to ignore, so
every threshold is tested from both sides.

**Does it say the right kind of thing?** These messages are the project's most
direct claims about whether a result can be trusted, and the difference between
"potential overfitting signal" and "the model is overfit" is the difference
between prompting a check and asserting something the numbers do not support.
:func:`test_no_diagnostic_states_a_verdict` holds that line for every message
the module can produce.
"""

from __future__ import annotations

import pytest

from ml.evaluation.diagnostics import (
    BASELINE_NOT_BEATEN,
    CLASS_IMBALANCE,
    CLASS_IMBALANCE_RATIO,
    GENERALISATION_GAP,
    GENERALISATION_GAP_RATIO,
    HIGH_CV_VARIABILITY,
    HIGH_CV_VARIABILITY_RATIO,
    MISSING_CLASS_IN_TEST,
    SELECTION_USED_TEST_DATA,
    SMALL_DATASET,
    SMALL_DATASET_ROWS,
    SMALL_TEST_SET,
    SMALL_TEST_ROWS,
    UNDEFINED_METRIC,
    Diagnostic,
    Severity,
    baseline_diagnostics,
    class_diagnostics,
    diagnose_run,
    generalisation_diagnostics,
    metric_diagnostics,
    selection_diagnostics,
    size_diagnostics,
    summarise_diagnostics,
    variability_diagnostics,
)
from ml.evaluation.metrics import MetricDirection

HIGHER = MetricDirection.HIGHER_IS_BETTER
LOWER = MetricDirection.LOWER_IS_BETTER


def codes(diagnostics) -> set[str]:
    """The codes raised, as a set."""
    return {item.code for item in diagnostics}


def healthy_run() -> dict:
    """Arguments describing a run with nothing to flag."""
    return {
        "metric": "f1",
        "direction": HIGHER,
        "selection_score": 0.880,
        "selection_score_std": 0.012,
        "held_out_score": 0.869,
        "folds": 5,
        "uses_test_data": False,
        "strategy": "cross_validation",
        "row_count": 4_000,
        "train_row_count": 3_200,
        "test_row_count": 800,
        "classification_details": {
            "class_labels": ["no", "yes"],
            "class_distribution": {"no": 430, "yes": 370},
        },
        "train_class_labels": ("no", "yes"),
        "unavailable_metrics": {},
        "baseline_comparison": {
            "metric": "f1",
            "model_value": 0.869,
            "baseline_value": 0.612,
            "beats_baseline": True,
        },
    }


# --------------------------------------------------------------------------
# The gap between what chose the model and what measured it
# --------------------------------------------------------------------------


def test_a_large_shortfall_on_held_out_data_is_flagged() -> None:
    """The two numbers this project keeps apart, compared."""
    found = generalisation_diagnostics(
        selection_score=0.92, held_out_score=0.61, direction=HIGHER, metric="f1"
    )

    assert codes(found) == {GENERALISATION_GAP}
    assert found[0].severity is Severity.WARNING
    assert found[0].details["relative_shortfall"] == pytest.approx(0.3370, abs=1e-4)


def test_a_small_shortfall_is_not_flagged() -> None:
    """Fold-to-fold noise must not produce a warning on every run."""
    just_inside = 0.90 * (1 - GENERALISATION_GAP_RATIO * 0.9)
    found = generalisation_diagnostics(
        selection_score=0.90, held_out_score=just_inside, direction=HIGHER, metric="f1"
    )

    assert found == []


def test_a_held_out_score_that_is_better_is_not_flagged() -> None:
    """Doing better than expected is not a warning."""
    found = generalisation_diagnostics(
        selection_score=0.70, held_out_score=0.95, direction=HIGHER, metric="f1"
    )

    assert found == []


def test_the_gap_follows_the_metric_direction() -> None:
    """For RMSE, a *larger* held-out number is the worse one."""
    worse = generalisation_diagnostics(
        selection_score=100.0, held_out_score=180.0, direction=LOWER, metric="rmse"
    )
    better = generalisation_diagnostics(
        selection_score=100.0, held_out_score=60.0, direction=LOWER, metric="rmse"
    )

    assert codes(worse) == {GENERALISATION_GAP}
    assert better == []


def test_a_direction_stored_as_a_string_is_understood() -> None:
    """Records store the direction's value, not the enum."""
    found = generalisation_diagnostics(
        selection_score=100.0,
        held_out_score=180.0,
        direction="lower_is_better",
        metric="rmse",
    )

    assert codes(found) == {GENERALISATION_GAP}


def test_a_missing_score_produces_nothing() -> None:
    """No comparison is possible, so nothing is claimed."""
    assert (
        generalisation_diagnostics(
            selection_score=None, held_out_score=0.6, direction=HIGHER, metric="f1"
        )
        == []
    )


def test_scores_near_zero_do_not_produce_a_ratio() -> None:
    """A relative shortfall between two tiny numbers means nothing."""
    found = generalisation_diagnostics(
        selection_score=1e-12, held_out_score=5e-12, direction=LOWER, metric="rmse"
    )

    assert found == []


# --------------------------------------------------------------------------
# Fold disagreement
# --------------------------------------------------------------------------


def test_widely_varying_folds_are_flagged() -> None:
    """A mean over folds that disagree is a weak summary of them."""
    found = variability_diagnostics(
        selection_score=0.60, selection_score_std=0.22, metric="f1", folds=5
    )

    assert codes(found) == {HIGH_CV_VARIABILITY}
    assert "across 5 folds" in found[0].message


def test_consistent_folds_are_not_flagged() -> None:
    """The common case must stay silent."""
    found = variability_diagnostics(
        selection_score=0.90,
        selection_score_std=0.90 * HIGH_CV_VARIABILITY_RATIO * 0.5,
        metric="f1",
        folds=5,
    )

    assert found == []


def test_the_spread_is_never_called_a_confidence_interval() -> None:
    """The one interpretation the spec forbids, checked in the wording."""
    found = variability_diagnostics(
        selection_score=0.60, selection_score_std=0.22, metric="f1", folds=5
    )

    assert "not a confidence interval" in found[0].message
    assert "confident" not in found[0].message.lower()


# --------------------------------------------------------------------------
# How much data there was
# --------------------------------------------------------------------------


def test_a_small_dataset_is_flagged() -> None:
    """Every score on a tiny dataset is a small-sample estimate."""
    found = size_diagnostics(
        row_count=SMALL_DATASET_ROWS - 1, train_row_count=120, test_row_count=40
    )

    assert codes(found) == {SMALL_DATASET, SMALL_TEST_SET}


def test_a_large_dataset_with_a_large_split_is_not_flagged() -> None:
    """Nothing is said about a run with plenty of data."""
    assert (
        size_diagnostics(row_count=5_000, train_row_count=4_000, test_row_count=1_000)
        == []
    )


def test_a_small_test_split_is_flagged_on_its_own() -> None:
    """A big dataset can still be measured on too few rows."""
    found = size_diagnostics(
        row_count=10_000, train_row_count=9_960, test_row_count=SMALL_TEST_ROWS - 1
    )

    assert codes(found) == {SMALL_TEST_SET}


def test_an_empty_test_split_is_not_reported_as_small() -> None:
    """Zero is a different problem, and not this module's to describe."""
    found = size_diagnostics(row_count=5_000, train_row_count=5_000, test_row_count=0)

    assert codes(found) == set()


# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------


def test_a_dominant_class_is_flagged() -> None:
    """Accuracy flatters a model on data like this, and the message says so."""
    found = class_diagnostics({"class_distribution": {"no": 940, "yes": 60}})

    assert codes(found) == {CLASS_IMBALANCE}
    assert found[0].details["majority_share"] == pytest.approx(0.94)
    assert "Extreme" not in found[0].message


def test_a_severe_imbalance_is_worded_more_strongly() -> None:
    """The wording escalates; the severity and code do not change."""
    found = class_diagnostics({"class_distribution": {"no": 990, "yes": 10}})

    assert found[0].message.startswith("Extreme class imbalance")
    assert found[0].code == CLASS_IMBALANCE


def test_a_balanced_split_is_not_flagged() -> None:
    """The ordinary case stays silent."""
    below = int(CLASS_IMBALANCE_RATIO * 100) - 5
    found = class_diagnostics(
        {"class_distribution": {"no": below, "yes": 100 - below}}
    )

    assert found == []


def test_a_class_missing_from_the_test_split_is_flagged() -> None:
    """Its per-class metrics were computed from no rows at all."""
    found = class_diagnostics(
        {"class_distribution": {"a": 60, "b": 40}},
        train_class_labels=("a", "b", "c"),
    )

    assert codes(found) == {MISSING_CLASS_IN_TEST}
    assert found[0].details["missing_classes"] == ["c"]


def test_regression_produces_no_class_diagnostics() -> None:
    """There are no classes to describe."""
    assert class_diagnostics(None) == []
    assert class_diagnostics({}) == []


# --------------------------------------------------------------------------
# Metrics that could not be computed, and the baseline
# --------------------------------------------------------------------------


def test_undefined_metrics_are_reported_once_as_a_note() -> None:
    """One diagnostic for all of them, and only a note."""
    found = metric_diagnostics(
        {"roc_auc": "the model has no predict_proba", "r2": "the target never varies"}
    )

    assert codes(found) == {UNDEFINED_METRIC}
    assert found[0].severity is Severity.INFO
    assert "r2, roc_auc" in found[0].message


def test_no_undefined_metrics_produces_nothing() -> None:
    """A complete metric set says nothing."""
    assert metric_diagnostics({}) == []
    assert metric_diagnostics(None) == []


def test_failing_to_beat_the_baseline_is_flagged() -> None:
    """The plainest warning in the module."""
    found = baseline_diagnostics(
        {
            "metric": "f1",
            "model_value": 0.41,
            "baseline_value": 0.52,
            "beats_baseline": False,
        }
    )

    assert codes(found) == {BASELINE_NOT_BEATEN}
    assert "0.4100 against 0.5200" in found[0].message


def test_beating_the_baseline_is_not_flagged() -> None:
    """Good news is not a diagnostic."""
    assert baseline_diagnostics({"beats_baseline": True}) == []
    assert baseline_diagnostics({}) == []


def test_a_holdout_selection_is_flagged_as_not_independent() -> None:
    """The score that chose the model also measured it."""
    found = selection_diagnostics(uses_test_data=True, strategy="holdout")

    assert codes(found) == {SELECTION_USED_TEST_DATA}
    assert "'holdout'" in found[0].message


def test_cross_validated_selection_is_not_flagged() -> None:
    """The default strategy keeps the two apart, so there is nothing to say."""
    assert selection_diagnostics(uses_test_data=False, strategy="cross_validation") == []


# --------------------------------------------------------------------------
# The whole run
# --------------------------------------------------------------------------


def test_a_healthy_run_raises_nothing() -> None:
    """The default outcome of a good run is an empty list.

    Worth stating as a test: a diagnostics feature that always finds something
    trains its readers to skip it.
    """
    assert diagnose_run(**healthy_run()) == ()


def test_a_troubled_run_raises_every_relevant_signal() -> None:
    """One deliberately bad run, and the full set it should produce."""
    found = diagnose_run(
        metric="f1",
        direction=HIGHER,
        selection_score=0.90,
        selection_score_std=0.30,
        held_out_score=0.50,
        folds=3,
        uses_test_data=False,
        strategy="cross_validation",
        row_count=120,
        train_row_count=96,
        test_row_count=24,
        classification_details={"class_distribution": {"no": 22, "yes": 2}},
        train_class_labels=("no", "yes", "maybe"),
        unavailable_metrics={"roc_auc": "the model has no predict_proba"},
        baseline_comparison={
            "metric": "f1",
            "model_value": 0.50,
            "baseline_value": 0.66,
            "beats_baseline": False,
        },
    )

    assert codes(found) == {
        GENERALISATION_GAP,
        HIGH_CV_VARIABILITY,
        BASELINE_NOT_BEATEN,
        CLASS_IMBALANCE,
        MISSING_CLASS_IN_TEST,
        SMALL_DATASET,
        SMALL_TEST_SET,
        UNDEFINED_METRIC,
    }


def test_warnings_are_ordered_before_notes() -> None:
    """A reader should meet the things that matter first."""
    found = diagnose_run(
        metric="f1",
        direction=HIGHER,
        selection_score=0.90,
        held_out_score=0.40,
        unavailable_metrics={"roc_auc": "no probabilities"},
    )
    severities = [item.severity for item in found]

    assert severities == sorted(severities, key=lambda item: 0 if item.is_warning else 1)
    assert found[0].code == GENERALISATION_GAP


def test_the_same_run_diagnoses_the_same_way_twice() -> None:
    """Stable output, so a rendered list does not reshuffle between views."""
    arguments = healthy_run() | {"held_out_score": 0.40, "test_row_count": 20}

    assert diagnose_run(**arguments) == diagnose_run(**arguments)


def test_diagnosing_a_run_computes_no_new_numbers() -> None:
    """Every number in a diagnostic came from the arguments it was given."""
    found = diagnose_run(
        metric="rmse",
        direction=LOWER,
        selection_score=100.0,
        held_out_score=180.0,
    )
    details = found[0].details

    assert details["cross_validation_score"] == 100.0
    assert details["held_out_score"] == 180.0


def test_a_summary_counts_what_was_found() -> None:
    """For a listing with no room for the sentences."""
    found = diagnose_run(
        metric="f1",
        direction=HIGHER,
        selection_score=0.90,
        held_out_score=0.40,
        unavailable_metrics={"roc_auc": "no probabilities"},
    )
    summary = summarise_diagnostics(found)

    assert summary["count"] == 2
    assert summary["warning_count"] == 1
    assert GENERALISATION_GAP in summary["codes"]


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------


def test_a_diagnostic_round_trips_through_a_record() -> None:
    """Stored as plain values, read back the same."""
    original = Diagnostic(
        code=SMALL_TEST_SET,
        severity=Severity.WARNING,
        message="Small held-out set.",
        details={"test_row_count": 12},
    )

    assert Diagnostic.from_dict(original.as_dict()) == original


def test_an_unknown_severity_reads_back_as_a_note() -> None:
    """A record from a later version stays readable."""
    restored = Diagnostic.from_dict(
        {"code": "future_signal", "severity": "catastrophic", "message": "?"}
    )

    assert restored.severity is Severity.INFO


# --------------------------------------------------------------------------
# The wording rule
# --------------------------------------------------------------------------

#: Phrasings that turn a signal into a verdict. A diagnostic is a prompt to
#: look at something; none of these leaves the reader anything to look at.
VERDICT_PHRASES = (
    "is overfit",
    "is overfitted",
    "the model is bad",
    "this model is unusable",
    "do not use",
    "proves",
    "guarantees",
    "definitely",
    "certainly",
)


def every_message() -> list[str]:
    """One message from every check the module can raise."""
    produced = diagnose_run(
        metric="f1",
        direction=HIGHER,
        selection_score=0.90,
        selection_score_std=0.30,
        held_out_score=0.45,
        folds=3,
        uses_test_data=True,
        strategy="holdout",
        row_count=120,
        train_row_count=96,
        test_row_count=24,
        classification_details={"class_distribution": {"no": 23, "yes": 1}},
        train_class_labels=("no", "yes", "maybe"),
        unavailable_metrics={"roc_auc": "the model has no predict_proba"},
        baseline_comparison={
            "metric": "f1",
            "model_value": 0.45,
            "baseline_value": 0.70,
            "beats_baseline": False,
        },
    )
    return [item.message for item in produced]


def test_every_check_is_covered_by_the_wording_test() -> None:
    """The wording test is only worth anything if it sees every message."""
    assert len(every_message()) == 9


@pytest.mark.parametrize("phrase", VERDICT_PHRASES)
def test_no_diagnostic_states_a_verdict(phrase: str) -> None:
    """Signals, not conclusions — checked in the words themselves.

    The spec is explicit: "potential overfitting signal: held-out performance
    is materially below cross-validation performance" rather than "the model is
    overfit". This test is what stops that from drifting.
    """
    for message in every_message():
        assert phrase not in message.lower(), message


def test_the_overfitting_signal_is_worded_as_a_signal() -> None:
    """The specific sentence the spec asked for."""
    found = generalisation_diagnostics(
        selection_score=0.92, held_out_score=0.55, direction=HIGHER, metric="f1"
    )
    message = found[0].message

    assert message.startswith("Potential overfitting signal")
    assert "worth checking, not a conclusion" in message
