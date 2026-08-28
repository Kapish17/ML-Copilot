"""Tests for building experiment records and comparing history.

The end-to-end test at the bottom is the one that matters most: it runs the
whole pipeline, saves the result, throws the store away, opens a new one and
checks that everything worth remembering came back.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ml.errors import IncomparableExperimentsError
from ml.experiments import (
    LocalExperimentStore,
    compare_experiments,
    create_experiment_run,
    fingerprint_dataset,
)
from ml.experiments.builder import configuration_components
from ml.experiments.identity import configuration_hash
from ml.experiments.run import EXPERIMENT_SCHEMA_VERSION
from ml.experiments.serialization import json_dumps
from ml.experiments.store import ExperimentQuery
from ml.explainability import explain_global
from ml.models.selection import select_and_evaluate_best_model
from ml.pipelines.result import PreparedDataset
from ml.tests.factories import (
    FakeProfile,
    FakeProfiledColumn,
    FakeQualityIssue,
    FakeQualityReport,
    experiment_run,
    learnable_classification_frame,
)

CANDIDATES = ["logistic_regression", "random_forest_classifier"]
FOLDS = 3


@pytest.fixture(scope="module")
def pipeline_outputs(classification_prepared: PreparedDataset):
    """Run selection and explanation once for the whole module."""
    outcome = select_and_evaluate_best_model(
        classification_prepared, models=CANDIDATES, folds=FOLDS
    )
    explanation = explain_global(
        outcome.final_model, classification_prepared.X_train_raw
    )
    return outcome, explanation


@pytest.fixture(scope="module")
def built_run(classification_prepared: PreparedDataset, pipeline_outputs):
    """An experiment record composed from real pipeline results."""
    outcome, explanation = pipeline_outputs
    return create_experiment_run(
        learnable_classification_frame(),
        classification_prepared,
        outcome,
        name="renewal baseline",
        description="first tracked run",
        explanation=explanation,
        tags=("baseline", "commit-7"),
        source_format="csv",
    )


# --------------------------------------------------------------------------
# Building a record from real results
# --------------------------------------------------------------------------


def test_the_record_identifies_itself(built_run) -> None:
    """Identity, schema version and labels are all present."""
    assert built_run.experiment_id.startswith(f"exp_{built_run.configuration_hash}_")
    assert built_run.schema_version == EXPERIMENT_SCHEMA_VERSION
    assert built_run.name == "renewal baseline"
    assert built_run.description == "first tracked run"
    assert built_run.tags == ("baseline", "commit-7")
    assert built_run.created_at.tzinfo is not None


def test_the_dataset_is_recorded_by_content(built_run) -> None:
    """The record names the data by fingerprint, not by any file."""
    expected = fingerprint_dataset(learnable_classification_frame())

    assert built_run.dataset.fingerprint == expected.value
    assert built_run.dataset.row_count == 300
    assert built_run.dataset.column_count == 4
    assert built_run.dataset.target_column == "renewed"
    assert built_run.dataset.task_type == "classification"
    assert built_run.dataset.source_format == "csv"


def test_the_preprocessing_decisions_are_recorded(
    built_run, classification_prepared: PreparedDataset
) -> None:
    """What became a feature, and what did not, is part of the history."""
    section = built_run.preprocessing

    assert section.selected_columns == classification_prepared.config.feature_columns
    assert section.transformed_feature_names == classification_prepared.feature_names
    assert section.train_row_count == classification_prepared.train_row_count
    assert section.test_row_count == classification_prepared.test_row_count
    assert section.random_state == classification_prepared.config.random_state
    assert section.stratified is True
    assert section.config["scaling_strategy"] == "standard"
    assert len(section.column_decisions) == 4


def test_the_selection_is_recorded_with_its_provenance(built_run) -> None:
    """Which models were tried, which won, and on what data it was judged."""
    section = built_run.selection

    assert section.strategy == "cross_validation"
    assert section.folds == FOLDS
    assert set(section.candidate_models) == set(CANDIDATES)
    assert section.selected_model in CANDIDATES
    assert section.primary_metric == "f1"
    assert section.primary_metric_direction == "higher_is_better"
    assert section.selection_score is not None
    assert section.selection_score_std is not None
    assert section.scored_on == "training_folds"
    assert section.uses_test_data is False


def test_the_final_evaluation_is_recorded(built_run, pipeline_outputs) -> None:
    """The one untouched-test measurement, with its baseline."""
    outcome, _ = pipeline_outputs
    section = built_run.evaluation

    assert section.primary_metric == "f1"
    assert section.primary_metric_value == pytest.approx(outcome.final_test_score)
    assert set(section.metrics) >= {"accuracy", "precision", "recall", "f1"}
    assert section.baseline_identifier == "majority_class_baseline"
    assert section.baseline_metrics["f1"] > 0
    assert section.baseline_comparison["beats_baseline"] is True
    assert section.is_unbiased is True


def test_the_explanation_is_recorded(built_run, pipeline_outputs) -> None:
    """The SHAP findings survive as facts, not as an explainer object."""
    _, explanation = pipeline_outputs
    section = built_run.explainability

    assert section is not None
    assert section.status == "available"
    assert section.method == "shap"
    assert section.explainer in {"TreeExplainer", "LinearExplainer"}
    assert section.sample_count == explanation.sample_count
    assert section.feature_count == explanation.feature_count
    assert [item["feature"] for item in section.feature_importances] == [
        item.feature for item in explanation.feature_importances
    ]
    assert section.feature_importances[0]["rank"] == 1


def test_the_environment_is_recorded(built_run, classification_prepared) -> None:
    """Enough to reproduce the run, and nothing about the machine's owner."""
    environment = built_run.environment

    assert environment.python_version
    assert environment.random_state == classification_prepared.config.random_state
    assert {"pandas", "numpy", "scikit-learn", "shap"} <= set(environment.packages)


def test_a_run_without_an_explanation_is_still_a_record(
    classification_prepared: PreparedDataset, pipeline_outputs
) -> None:
    """Explanations are optional context, not a requirement."""
    outcome, _ = pipeline_outputs
    run = create_experiment_run(
        learnable_classification_frame(),
        classification_prepared,
        outcome,
        name="no explanation",
    )

    assert run.explainability is None
    assert run.evaluation.primary_metric_value is not None


def test_profile_findings_are_carried_into_the_record(
    classification_prepared: PreparedDataset, pipeline_outputs
) -> None:
    """A profile's quality findings become searchable history.

    The profile is read structurally, so the experiment layer does not depend
    on the profiling implementation.
    """
    outcome, _ = pipeline_outputs
    profile = FakeProfile(
        columns=[FakeProfiledColumn("income", "float")],
        quality=FakeQualityReport(
            issues=[FakeQualityIssue("high_missing_values", ["income"])]
        ),
    )
    run = create_experiment_run(
        learnable_classification_frame(),
        classification_prepared,
        outcome,
        name="with profile",
        profile=profile,
    )

    assert len(run.dataset.data_quality_issues) == 1
    assert run.dataset.data_quality_issues[0]["code"] == "high_missing_values"
    assert run.dataset.data_quality_issues[0]["columns"] == ["income"]


# --------------------------------------------------------------------------
# Identity of a configuration
# --------------------------------------------------------------------------


def test_repeating_a_run_keeps_the_configuration_hash(
    classification_prepared: PreparedDataset, pipeline_outputs
) -> None:
    """Same inputs, same configuration hash — but a distinct execution."""
    outcome, _ = pipeline_outputs
    frame = learnable_classification_frame()
    first = create_experiment_run(frame, classification_prepared, outcome, name="a")
    second = create_experiment_run(frame, classification_prepared, outcome, name="b")

    assert first.configuration_hash == second.configuration_hash
    assert first.experiment_id != second.experiment_id


def test_a_different_candidate_set_changes_the_configuration_hash(
    classification_prepared: PreparedDataset, built_run
) -> None:
    """Offering different models is a different experiment."""
    components = configuration_components(
        fingerprint=built_run.dataset.fingerprint,
        prepared=classification_prepared,
        selection=built_run.selection,
    )
    changed = {**components, "candidate_models": ["logistic_regression"]}

    assert configuration_hash(changed) != built_run.configuration_hash


def test_the_configuration_hash_ignores_the_outcome(
    built_run, classification_prepared: PreparedDataset
) -> None:
    """The hash describes the inputs, so a different winner would not move it."""
    components = configuration_components(
        fingerprint=built_run.dataset.fingerprint,
        prepared=classification_prepared,
        selection=built_run.selection,
    )

    # The setup is hashed; nothing the run concluded is.
    assert configuration_hash(components) == built_run.configuration_hash
    for outcome_key in (
        "selected_model",
        "selection_score",
        "selection_score_std",
        "test_score",
        "baseline_comparison",
    ):
        assert outcome_key not in components
    assert components["candidate_models"] == sorted(CANDIDATES)
    assert built_run.selection.selected_model in CANDIDATES


# --------------------------------------------------------------------------
# What must not be stored
# --------------------------------------------------------------------------


def test_the_record_holds_no_fitted_artefacts(built_run) -> None:
    """No pipeline, no estimator, no explainer — only what they produced."""
    text = json_dumps(built_run.to_dict())

    for artefact in (
        "Pipeline(",
        "ColumnTransformer(",
        "LogisticRegression(",
        "RandomForestClassifier(",
        "TreeExplainer",
        "LinearExplainer(",
        "SimpleImputer(",
    ):
        assert artefact not in text


def test_the_record_holds_no_dataset(built_run) -> None:
    """Row values never reach the record, only counts and column names."""
    frame = learnable_classification_frame()
    text = json_dumps(built_run.to_dict())

    assert str(frame.loc[0, "income"]) not in text
    assert built_run.dataset.row_count == 300


def test_the_record_stays_small(built_run) -> None:
    """History should stay readable and cheap to keep."""
    assert len(json_dumps(built_run.to_dict())) < 32_000


# --------------------------------------------------------------------------
# The property that matters: experiment memory
# --------------------------------------------------------------------------


def test_an_experiment_survives_the_process_that_made_it(
    built_run, tmp_path: Path
) -> None:
    """Save, discard the store, open a new one, and find everything again.

    This is what experiment tracking is for: before this commit, every result
    vanished when the process ended.
    """
    LocalExperimentStore(tmp_path).save(built_run)

    reopened = LocalExperimentStore(tmp_path)
    loaded = reopened.get(built_run.experiment_id)

    assert loaded.to_dict() == built_run.to_dict()
    assert loaded.dataset.fingerprint == built_run.dataset.fingerprint
    assert loaded.selection.selected_model == built_run.selected_model
    assert loaded.selection.selection_score == pytest.approx(
        built_run.selection.selection_score
    )
    assert loaded.evaluation.primary_metric_value == pytest.approx(
        built_run.evaluation.primary_metric_value
    )
    assert loaded.evaluation.baseline_comparison == built_run.evaluation.baseline_comparison
    assert loaded.explainability is not None
    assert loaded.explainability.feature_importances == (
        built_run.explainability.feature_importances
    )
    assert loaded.preprocessing.transformed_feature_names == (
        built_run.preprocessing.transformed_feature_names
    )
    assert loaded.environment.packages == built_run.environment.packages


def test_a_stored_experiment_is_findable_by_its_dataset(
    built_run, tmp_path: Path
) -> None:
    """History is queried by what was run on, not by what it was called."""
    store = LocalExperimentStore(tmp_path)
    store.save(built_run)

    found = store.list(
        ExperimentQuery(dataset_fingerprint=built_run.dataset.fingerprint)
    )

    assert [run.experiment_id for run in found] == [built_run.experiment_id]


def test_the_stored_file_is_readable_json(built_run, tmp_path: Path) -> None:
    """A person should be able to open a record and understand it."""
    store = LocalExperimentStore(tmp_path)
    store.save(built_run)
    payload = json.loads(
        store.path_for(built_run.experiment_id).read_text(encoding="utf-8")
    )

    assert set(payload) >= {
        "schema_version",
        "experiment_id",
        "configuration_hash",
        "created_at",
        "dataset",
        "preprocessing",
        "selection",
        "evaluation",
        "explainability",
        "environment",
    }


# --------------------------------------------------------------------------
# Comparing history
# --------------------------------------------------------------------------


def test_runs_are_ranked_by_their_final_test_score() -> None:
    """For F1, the best historical run is the highest-scoring one."""
    comparison = compare_experiments(
        [
            experiment_run(experiment_id="exp_low", test_score=0.74),
            experiment_run(experiment_id="exp_high", test_score=0.88),
            experiment_run(experiment_id="exp_mid", test_score=0.81),
        ]
    )

    assert [row.experiment_id for row in comparison.rows] == [
        "exp_high",
        "exp_mid",
        "exp_low",
    ]
    assert comparison.best().experiment_id == "exp_high"
    assert comparison.higher_is_better is True


def test_error_metrics_rank_the_other_way() -> None:
    """For RMSE, the best historical run is the lowest-scoring one."""
    comparison = compare_experiments(
        [
            experiment_run(
                experiment_id="exp_worse",
                task_type="regression",
                primary_metric="rmse",
                test_score=4200.0,
                baseline_score=9000.0,
            ),
            experiment_run(
                experiment_id="exp_better",
                task_type="regression",
                primary_metric="rmse",
                test_score=1100.0,
                baseline_score=9000.0,
            ),
        ]
    )

    assert comparison.higher_is_better is False
    assert comparison.best().experiment_id == "exp_better"


def test_the_comparison_carries_the_baseline_and_improvement() -> None:
    """A score without its baseline says little; both are in the table."""
    row = compare_experiments([experiment_run(test_score=0.86, baseline_score=0.71)]).rows[0]

    assert row.baseline_score == pytest.approx(0.71)
    assert row.improvement == pytest.approx(0.15)
    assert row.selection_score == pytest.approx(0.81)


def test_unscored_runs_rank_last_in_a_comparison() -> None:
    """A run with no final score cannot be the best one."""
    comparison = compare_experiments(
        [
            experiment_run(experiment_id="exp_none", test_score=None),
            experiment_run(experiment_id="exp_scored", test_score=0.5),
        ]
    )

    assert [row.experiment_id for row in comparison.rows] == ["exp_scored", "exp_none"]
    assert comparison.best().experiment_id == "exp_scored"


def test_mixing_metrics_is_refused() -> None:
    """An F1 and an RMSE cannot be ranked together, so they are not."""
    with pytest.raises(IncomparableExperimentsError) as exc_info:
        compare_experiments(
            [
                experiment_run(experiment_id="exp_f1"),
                experiment_run(
                    experiment_id="exp_rmse",
                    task_type="regression",
                    primary_metric="rmse",
                    test_score=900.0,
                ),
            ]
        )

    assert set(exc_info.value.details["primary_metrics"]) == {"f1", "rmse"}


def test_comparing_nothing_is_refused() -> None:
    """An empty comparison is a mistake, not an empty table."""
    with pytest.raises(IncomparableExperimentsError, match="no runs"):
        compare_experiments([])


def test_the_comparison_is_serialisable() -> None:
    """The table is plain data, ready for a report or an API."""
    summary = compare_experiments(
        [experiment_run(experiment_id="exp_a"), experiment_run(experiment_id="exp_b", test_score=0.9)]
    ).summary()

    json.dumps(summary)
    assert summary["primary_metric"] == "f1"
    assert summary["direction"] == "higher_is_better"
    assert summary["best_experiment_id"] == "exp_b"
    assert len(summary["runs"]) == 2


def test_the_comparison_renders_a_readable_table() -> None:
    """Columns say which metric, and where each number came from."""
    text = compare_experiments(
        [experiment_run(experiment_id="exp_a"), experiment_run(experiment_id="exp_b", test_score=0.9)]
    ).as_text()

    assert "CV F1" in text
    assert "Test F1" in text
    assert "Baseline" in text
    assert "Improvement" in text
    assert "higher is better" in text
    assert "exp_b" in text.splitlines()[2], "the best run leads the table"


def test_history_can_be_compared_after_reloading(
    built_run, tmp_path: Path
) -> None:
    """Comparison works on records read back from disk, not just fresh ones."""
    store = LocalExperimentStore(tmp_path)
    store.save(built_run)
    store.save(
        experiment_run(
            experiment_id="exp_earlier",
            fingerprint=built_run.dataset.fingerprint,
            test_score=0.5,
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        )
    )

    comparison = compare_experiments(store.list())

    assert len(comparison.rows) == 2
    assert comparison.best().experiment_id == built_run.experiment_id
