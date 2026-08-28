"""Tests for the local experiment store and for querying history."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sklearn.linear_model import LogisticRegression

from ml.errors import (
    ExperimentNotFoundError,
    IncomparableExperimentsError,
    InvalidExperimentIdError,
    InvalidExperimentRecordError,
    MalformedExperimentError,
    SerializationError,
    UnsupportedSchemaVersionError,
)
from ml.experiments.local_store import RUN_FILENAME, LocalExperimentStore
from ml.experiments.store import (
    ExperimentQuery,
    ExperimentSortKey,
    ExperimentStore,
)
from ml.tests.factories import experiment_run


@pytest.fixture
def store(tmp_path: Path) -> LocalExperimentStore:
    """A store rooted in a temporary directory, never in the repository."""
    return LocalExperimentStore(tmp_path / "runs")


def _write_raw(store: LocalExperimentStore, experiment_id: str, text: str) -> Path:
    """Put arbitrary text where a record should be, to simulate corruption."""
    path = store.path_for(experiment_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------


def test_the_local_store_satisfies_the_interface(store: LocalExperimentStore) -> None:
    """Callers depend on the protocol, so the backend must implement it."""
    assert isinstance(store, ExperimentStore)


def test_constructing_a_store_creates_nothing(tmp_path: Path) -> None:
    """Pointing at a directory is not a side effect; writing is."""
    root = tmp_path / "not_yet"
    LocalExperimentStore(root)

    assert not root.exists()


# --------------------------------------------------------------------------
# Saving and loading
# --------------------------------------------------------------------------


def test_a_saved_run_can_be_read_back(store: LocalExperimentStore) -> None:
    """The basic promise: what goes in comes out."""
    run = experiment_run()
    returned = store.save(run)
    loaded = store.get(run.experiment_id)

    assert returned == run.experiment_id
    assert loaded.to_dict() == run.to_dict()


def test_a_run_survives_a_new_store_instance(tmp_path: Path) -> None:
    """History outlives the object that wrote it — the point of persistence."""
    run = experiment_run()
    LocalExperimentStore(tmp_path).save(run)

    loaded = LocalExperimentStore(tmp_path).get(run.experiment_id)

    assert loaded.experiment_id == run.experiment_id
    assert loaded.evaluation.primary_metric_value == pytest.approx(0.80)


def test_the_file_layout_is_predictable(store: LocalExperimentStore) -> None:
    """One directory per run, one readable file inside it."""
    run = experiment_run()
    store.save(run)
    path = store.path_for(run.experiment_id)

    assert path == store.root / run.experiment_id / RUN_FILENAME
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8"))["experiment_id"] == (
        run.experiment_id
    )


def test_saving_again_replaces_the_record(store: LocalExperimentStore) -> None:
    """Re-saving one id leaves one complete record, not two half ones."""
    run = experiment_run(name="first")
    store.save(run)
    store.save(experiment_run(name="second"))

    assert store.get(run.experiment_id).name == "second"
    assert len(list((store.root / run.experiment_id).iterdir())) == 1


def test_exists_reports_what_is_stored(store: LocalExperimentStore) -> None:
    """A cheap check that does not load the record."""
    run = experiment_run()

    assert store.exists(run.experiment_id) is False
    store.save(run)
    assert store.exists(run.experiment_id) is True


def test_a_missing_run_is_reported_clearly(store: LocalExperimentStore) -> None:
    """Asking for something absent is an error, not an empty result."""
    with pytest.raises(ExperimentNotFoundError, match="exp_missing"):
        store.get("exp_missing")


def test_a_run_can_be_deleted(store: LocalExperimentStore) -> None:
    """Deleting removes the whole directory and says whether it did."""
    run = experiment_run()
    store.save(run)

    assert store.delete(run.experiment_id) is True
    assert store.exists(run.experiment_id) is False
    assert store.delete(run.experiment_id) is False


# --------------------------------------------------------------------------
# Robust writes
# --------------------------------------------------------------------------


def test_no_temporary_files_are_left_behind(store: LocalExperimentStore) -> None:
    """A completed write leaves exactly the record and nothing else."""
    run = experiment_run()
    store.save(run)

    entries = list((store.root / run.experiment_id).iterdir())
    assert [entry.name for entry in entries] == [RUN_FILENAME]


def test_an_interrupted_write_leaves_the_previous_record_intact(
    store: LocalExperimentStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The decisive property: a crash mid-write cannot corrupt history.

    The rename is made to fail, standing in for a process dying between the
    write and the move. The record already on disk must be untouched, and no
    fragment may be left lying around.
    """
    run = experiment_run(name="original")
    store.save(run)

    def explode(*args: object, **kwargs: object) -> None:
        """Fail exactly where a crash would hurt most."""
        raise OSError("interrupted")

    monkeypatch.setattr(os, "replace", explode)
    with pytest.raises(OSError, match="interrupted"):
        store.save(experiment_run(name="replacement"))

    assert store.get(run.experiment_id).name == "original"
    assert [entry.name for entry in (store.root / run.experiment_id).iterdir()] == [
        RUN_FILENAME
    ]


def test_an_unwritable_run_is_not_half_saved(store: LocalExperimentStore) -> None:
    """A record holding a model artefact is refused, and writes nothing.

    Serialisation happens before the filesystem is touched, so a mistake like
    stuffing an estimator into the configuration cannot leave a stray
    directory behind either.
    """
    run = experiment_run()
    broken = replace(
        run,
        preprocessing=replace(
            run.preprocessing, config={"model": LogisticRegression()}
        ),
    )

    with pytest.raises(SerializationError):
        store.save(broken)
    assert not (store.root / run.experiment_id).exists()


# --------------------------------------------------------------------------
# Safety
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "experiment_id", ["../escape", "../../etc/passwd", "/etc/passwd", "..", "a/b"]
)
def test_unsafe_identifiers_cannot_reach_the_filesystem(
    store: LocalExperimentStore, experiment_id: str
) -> None:
    """No caller-supplied string escapes the store directory."""
    with pytest.raises(InvalidExperimentIdError):
        store.get(experiment_id)
    with pytest.raises(InvalidExperimentIdError):
        store.path_for(experiment_id)
    with pytest.raises(InvalidExperimentIdError):
        store.delete(experiment_id)


def test_exists_is_false_for_an_unsafe_identifier(
    store: LocalExperimentStore,
) -> None:
    """A membership check answers the question rather than raising."""
    assert store.exists("../../etc/passwd") is False


def test_a_traversal_attempt_writes_nothing_outside_the_root(
    tmp_path: Path,
) -> None:
    """The neighbouring directory stays empty however the id is crafted."""
    outside = tmp_path / "outside"
    outside.mkdir()
    store = LocalExperimentStore(tmp_path / "runs")

    with pytest.raises(InvalidExperimentIdError):
        store.save(experiment_run(experiment_id="../outside/stolen"))

    assert list(outside.iterdir()) == []


# --------------------------------------------------------------------------
# Corrupted records
# --------------------------------------------------------------------------


def test_reading_a_corrupted_record_raises(store: LocalExperimentStore) -> None:
    """A caller who names a run is told when it is broken."""
    _write_raw(store, "exp_broken", "{ this is not json")

    with pytest.raises(MalformedExperimentError):
        store.get("exp_broken")


def test_reading_a_future_schema_raises(store: LocalExperimentStore) -> None:
    """A record from a newer format is refused, not partially read."""
    payload = experiment_run().to_dict()
    payload["schema_version"] = "42.0"
    _write_raw(store, "exp_future", json.dumps(payload))

    with pytest.raises(UnsupportedSchemaVersionError):
        store.get("exp_future")


def test_reading_an_incomplete_record_raises(store: LocalExperimentStore) -> None:
    """A record missing a required section is refused."""
    payload = experiment_run().to_dict()
    del payload["selection"]
    _write_raw(store, "exp_partial", json.dumps(payload))

    with pytest.raises(InvalidExperimentRecordError):
        store.get("exp_partial")


def test_listing_skips_corrupted_records(store: LocalExperimentStore) -> None:
    """One bad file must not hide an entire history."""
    store.save(experiment_run(experiment_id="exp_good"))
    _write_raw(store, "exp_broken", "{ not json")

    listed = store.list()

    assert [run.experiment_id for run in listed] == ["exp_good"]


def test_verify_reports_what_listing_skipped(store: LocalExperimentStore) -> None:
    """Skipping is not hiding: the problems remain inspectable."""
    store.save(experiment_run(experiment_id="exp_good"))
    _write_raw(store, "exp_broken", "{ not json")

    problems = store.verify()

    assert [experiment_id for experiment_id, _ in problems] == ["exp_broken"]
    assert "JSON" in problems[0][1]


def test_verify_is_empty_for_a_healthy_store(store: LocalExperimentStore) -> None:
    """Nothing to report when everything reads."""
    store.save(experiment_run())
    assert store.verify() == ()


# --------------------------------------------------------------------------
# Querying
# --------------------------------------------------------------------------


def _populate(store: LocalExperimentStore) -> None:
    """Store a small, deliberately varied history."""
    store.save(
        experiment_run(
            experiment_id="exp_a",
            fingerprint="dataset_one",
            model_name="logistic_regression",
            strategy="cross_validation",
            test_score=0.80,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            tags=("baseline",),
        )
    )
    store.save(
        experiment_run(
            experiment_id="exp_b",
            fingerprint="dataset_one",
            model_name="random_forest_classifier",
            strategy="holdout",
            test_score=0.86,
            created_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
            tags=("baseline", "tuned"),
        )
    )
    store.save(
        experiment_run(
            experiment_id="exp_c",
            fingerprint="dataset_two",
            target_column="price",
            task_type="regression",
            primary_metric="rmse",
            model_name="linear_regression",
            strategy="cross_validation",
            test_score=1200.0,
            baseline_score=5000.0,
            created_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        )
    )


def test_listing_returns_everything_newest_first(
    store: LocalExperimentStore,
) -> None:
    """The default order is the one a person usually wants."""
    _populate(store)

    assert [run.experiment_id for run in store.list()] == ["exp_c", "exp_b", "exp_a"]


def test_filtering_by_dataset(store: LocalExperimentStore) -> None:
    """Runs are found by what they ran on, not by any filename."""
    _populate(store)
    found = store.list(ExperimentQuery(dataset_fingerprint="dataset_one"))

    assert {run.experiment_id for run in found} == {"exp_a", "exp_b"}


def test_filtering_by_task_and_target(store: LocalExperimentStore) -> None:
    """The problem being solved is a first-class filter."""
    _populate(store)

    assert [run.experiment_id for run in store.list(ExperimentQuery(task_type="regression"))] == [
        "exp_c"
    ]
    assert [
        run.experiment_id for run in store.list(ExperimentQuery(target_column="price"))
    ] == ["exp_c"]


def test_filtering_by_model(store: LocalExperimentStore) -> None:
    """Which model won is a filter, so a family can be tracked over time."""
    _populate(store)
    found = store.list(ExperimentQuery(model_name="random_forest_classifier"))

    assert [run.experiment_id for run in found] == ["exp_b"]


def test_filtering_by_selection_strategy(store: LocalExperimentStore) -> None:
    """Cross-validated runs can be separated from holdout ones."""
    _populate(store)
    found = store.list(ExperimentQuery(selection_strategy="cross_validation"))

    assert {run.experiment_id for run in found} == {"exp_a", "exp_c"}


def test_filtering_by_tags(store: LocalExperimentStore) -> None:
    """Tags narrow a history to a line of work."""
    _populate(store)

    assert len(store.list(ExperimentQuery(tags=("baseline",)))) == 2
    assert len(store.list(ExperimentQuery(tags=("baseline", "tuned")))) == 1


def test_filters_combine(store: LocalExperimentStore) -> None:
    """Several filters narrow together, not separately."""
    _populate(store)
    found = store.list(
        ExperimentQuery(dataset_fingerprint="dataset_one", selection_strategy="holdout")
    )

    assert [run.experiment_id for run in found] == ["exp_b"]


def test_a_limit_truncates_the_result(store: LocalExperimentStore) -> None:
    """Only the first few runs are needed for a dashboard."""
    _populate(store)
    assert len(store.list(ExperimentQuery(limit=2))) == 2


def test_sorting_oldest_first(store: LocalExperimentStore) -> None:
    """The order is a choice, not a fixed behaviour."""
    _populate(store)
    found = store.list(ExperimentQuery(descending=False))

    assert [run.experiment_id for run in found] == ["exp_a", "exp_b", "exp_c"]


def test_sorting_by_model_name(store: LocalExperimentStore) -> None:
    """A history can be grouped by which model won."""
    _populate(store)
    found = store.list(
        ExperimentQuery(sort_by=ExperimentSortKey.MODEL_NAME, descending=False)
    )

    assert [run.selected_model for run in found] == [
        "linear_regression",
        "logistic_regression",
        "random_forest_classifier",
    ]


def test_sorting_by_a_score_metric_puts_the_largest_first(
    store: LocalExperimentStore,
) -> None:
    """F1 is a score, so best means highest."""
    _populate(store)
    found = store.list(
        ExperimentQuery(task_type="classification", sort_by=ExperimentSortKey.PRIMARY_METRIC)
    )

    assert [run.experiment_id for run in found] == ["exp_b", "exp_a"]


def test_sorting_by_an_error_metric_puts_the_smallest_first(
    store: LocalExperimentStore,
) -> None:
    """RMSE is an error, so best means lowest — read from the metric itself."""
    store.save(
        experiment_run(
            experiment_id="exp_low",
            task_type="regression",
            primary_metric="rmse",
            test_score=900.0,
        )
    )
    store.save(
        experiment_run(
            experiment_id="exp_high",
            task_type="regression",
            primary_metric="rmse",
            test_score=4000.0,
        )
    )
    found = store.list(
        ExperimentQuery(task_type="regression", sort_by=ExperimentSortKey.PRIMARY_METRIC)
    )

    assert [run.experiment_id for run in found] == ["exp_low", "exp_high"]


def test_sorting_a_mixed_history_by_metric_is_refused(
    store: LocalExperimentStore,
) -> None:
    """Ranking an F1 against an RMSE is meaningless, so it is not done."""
    _populate(store)

    with pytest.raises(IncomparableExperimentsError) as exc_info:
        store.list(ExperimentQuery(sort_by=ExperimentSortKey.PRIMARY_METRIC))

    assert set(exc_info.value.details["primary_metrics"]) == {"f1", "rmse"}
    assert "filter by task_type" in str(exc_info.value).lower()


def test_unscored_runs_sort_last(store: LocalExperimentStore) -> None:
    """A run with no score cannot outrank one that has a score."""
    store.save(experiment_run(experiment_id="exp_scored", test_score=0.5))
    store.save(experiment_run(experiment_id="exp_unscored", test_score=None))

    found = store.list(ExperimentQuery(sort_by=ExperimentSortKey.PRIMARY_METRIC))

    assert [run.experiment_id for run in found] == ["exp_scored", "exp_unscored"]


def test_listing_an_empty_store_is_empty(store: LocalExperimentStore) -> None:
    """No history is not an error."""
    assert store.list() == ()
