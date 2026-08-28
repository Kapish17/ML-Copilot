"""Tests for dataset fingerprints and experiment identifiers."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ml.errors import InvalidExperimentIdError
from ml.experiments.fingerprint import (
    FINGERPRINT_LENGTH,
    fingerprint_dataset,
)
from ml.experiments.identity import (
    EXPERIMENT_ID_PREFIX,
    canonical_json,
    configuration_hash,
    generate_experiment_id,
    validate_experiment_id,
)
from ml.tests.factories import learnable_classification_frame


# --------------------------------------------------------------------------
# Dataset fingerprints
# --------------------------------------------------------------------------


def test_the_same_data_fingerprints_the_same() -> None:
    """Two identical frames are recognised as the same dataset."""
    first = fingerprint_dataset(learnable_classification_frame())
    second = fingerprint_dataset(learnable_classification_frame())

    assert first.value == second.value
    assert len(first.value) == FINGERPRINT_LENGTH


def test_a_changed_value_changes_the_fingerprint() -> None:
    """One edited cell is enough to make it a different dataset."""
    frame = learnable_classification_frame()
    edited = frame.copy()
    edited.loc[0, "income"] = edited.loc[0, "income"] + 1.0

    assert fingerprint_dataset(frame).value != fingerprint_dataset(edited).value


def test_added_rows_change_the_fingerprint() -> None:
    """A dataset that grew is not the dataset it grew from."""
    frame = learnable_classification_frame()
    longer = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    assert fingerprint_dataset(frame).value != fingerprint_dataset(longer).value


def test_renamed_columns_change_the_fingerprint() -> None:
    """The schema is part of the identity, not just the values."""
    frame = learnable_classification_frame()
    renamed = frame.rename(columns={"income": "salary"})

    assert fingerprint_dataset(frame).value != fingerprint_dataset(renamed).value


def test_reordered_rows_change_the_fingerprint() -> None:
    """Row order matters: it changes the splits, so it is a different run."""
    frame = learnable_classification_frame()
    shuffled = frame.iloc[::-1].reset_index(drop=True)

    assert fingerprint_dataset(frame).value != fingerprint_dataset(shuffled).value


def test_the_index_does_not_affect_the_fingerprint() -> None:
    """How a frame was indexed is an artefact, not part of the data."""
    frame = learnable_classification_frame()
    reindexed = frame.copy()
    reindexed.index = range(1000, 1000 + len(frame))

    assert fingerprint_dataset(frame).value == fingerprint_dataset(reindexed).value


def test_the_fingerprint_does_not_depend_on_the_file_path(tmp_path) -> None:
    """The same table read from two different files is the same dataset.

    This is the property that makes experiment history survive a file being
    renamed, moved or re-exported from another format.
    """
    frame = learnable_classification_frame()
    first_path = tmp_path / "customers.csv"
    second_path = tmp_path / "nested" / "renamed_export.csv"
    second_path.parent.mkdir()
    frame.to_csv(first_path, index=False)
    frame.to_csv(second_path, index=False)

    first = fingerprint_dataset(pd.read_csv(first_path))
    second = fingerprint_dataset(pd.read_csv(second_path))

    assert first.value == second.value


def test_the_fingerprint_records_the_schema() -> None:
    """The facts behind the digest travel with it."""
    frame = learnable_classification_frame()
    fingerprint = fingerprint_dataset(frame)

    assert fingerprint.row_count == len(frame)
    assert fingerprint.column_count == frame.shape[1]
    assert fingerprint.columns == tuple(frame.columns)
    assert set(fingerprint.dtypes) == set(frame.columns)
    assert fingerprint.algorithm == "sha256"


def test_the_fingerprint_is_serialisable() -> None:
    """The fingerprint renders as plain values."""
    payload = fingerprint_dataset(learnable_classification_frame()).as_dict()

    assert payload["value"]
    assert isinstance(payload["columns"], list)


def test_only_a_dataframe_can_be_fingerprinted() -> None:
    """The layer works on standardised frames, whatever the source format."""
    with pytest.raises(TypeError, match="DataFrame"):
        fingerprint_dataset({"income": [1, 2, 3]})


# --------------------------------------------------------------------------
# Configuration hashes and experiment ids
# --------------------------------------------------------------------------


def test_the_same_configuration_hashes_the_same() -> None:
    """Re-running one setup produces the same configuration hash."""
    components = {"dataset": "abc", "folds": 5, "models": ["a", "b"]}

    assert configuration_hash(components) == configuration_hash(dict(components))


def test_key_order_does_not_change_the_hash() -> None:
    """Two equal configurations hash the same however they were built."""
    first = {"dataset": "abc", "folds": 5}
    second = {"folds": 5, "dataset": "abc"}

    assert configuration_hash(first) == configuration_hash(second)


def test_a_changed_configuration_changes_the_hash() -> None:
    """A different seed is a different configuration."""
    base = {"dataset": "abc", "random_state": 42}
    changed = {"dataset": "abc", "random_state": 7}

    assert configuration_hash(base) != configuration_hash(changed)


def test_canonical_json_is_stable() -> None:
    """The text that gets hashed is deterministic and compact."""
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_an_experiment_id_carries_its_configuration_hash() -> None:
    """The id links an execution back to the setup that produced it."""
    config_hash = configuration_hash({"dataset": "abc"})
    experiment_id = generate_experiment_id(config_hash)

    assert experiment_id.startswith(f"{EXPERIMENT_ID_PREFIX}_{config_hash}_")
    validate_experiment_id(experiment_id)


def test_an_experiment_id_carries_the_moment_it_ran() -> None:
    """The timestamp is in the id, in UTC, so runs sort readably."""
    moment = datetime(2026, 8, 26, 13, 45, 0, tzinfo=timezone.utc)
    experiment_id = generate_experiment_id("abc123", created_at=moment)

    assert "_20260826T134500Z_" in experiment_id


def test_repeating_a_run_produces_a_distinct_id() -> None:
    """Two executions of one configuration do not overwrite each other."""
    moment = datetime(2026, 8, 26, 13, 45, 0, tzinfo=timezone.utc)
    first = generate_experiment_id("abc123", created_at=moment)
    second = generate_experiment_id("abc123", created_at=moment)

    assert first != second, "even within the same second"
    assert first.rsplit("_", 1)[0] == second.rsplit("_", 1)[0]


def test_a_naive_timestamp_is_treated_as_utc() -> None:
    """An id never depends on the local timezone of the machine."""
    naive = datetime(2026, 8, 26, 13, 45, 0)
    assert "_20260826T134500Z_" in generate_experiment_id("abc123", created_at=naive)


@pytest.mark.parametrize(
    "experiment_id",
    [
        "../../etc/passwd",
        "/etc/passwd",
        "runs/../secret",
        "..",
        ".",
        "exp/../..",
        "exp\\windows",
        "",
        "exp id with spaces",
        "_leading_underscore",
        "a" * 129,
    ],
)
def test_unsafe_identifiers_are_refused(experiment_id: str) -> None:
    """Nothing that could climb out of a storage directory is accepted."""
    with pytest.raises(InvalidExperimentIdError):
        validate_experiment_id(experiment_id)


def test_a_non_string_identifier_is_refused() -> None:
    """An id must be text before it can be a directory name."""
    with pytest.raises(InvalidExperimentIdError) as exc_info:
        validate_experiment_id(42)
    assert exc_info.value.details["received_type"] == "int"


@pytest.mark.parametrize(
    "experiment_id",
    ["exp_abc123_20260101T000000Z_0f3a", "run-1", "A1", "0", "a" * 128],
)
def test_well_formed_identifiers_are_accepted(experiment_id: str) -> None:
    """Letters, digits, underscores and hyphens are all that is needed."""
    assert validate_experiment_id(experiment_id) == experiment_id


def test_numpy_values_do_not_break_the_hash() -> None:
    """Configuration values arriving from numpy still hash deterministically."""
    components = {"threshold": np.float64(0.5), "folds": np.int64(5)}
    assert configuration_hash(components) == configuration_hash(dict(components))
