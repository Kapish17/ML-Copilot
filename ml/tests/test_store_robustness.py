"""What the experiment store does when a record on disk is wrong.

The store's own docstring makes a promise: ``get`` raises for a named record,
``list`` skips a bad one with a warning so it cannot hide the rest, and
``verify`` reports exactly which records are unreadable and why.

That promise held for the failures the code anticipated — invalid JSON, a
missing field, an unknown schema version — and not for the ones it did not.
``list`` and ``verify`` both catch ``ExperimentError``; a file that is not
UTF-8, or that cannot be read at all, raises something else entirely, escapes
both, and turns a listing that nine good records could have answered into a
500. These tests are that class of failure, made explicit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ml.errors import (
    ExperimentError,
    ExperimentNotFoundError,
    IncomparableExperimentsError,
    MalformedExperimentError,
)
from ml.experiments import LocalExperimentStore
from ml.experiments.local_store import RUN_FILENAME
from ml.experiments.store import ExperimentQuery, ExperimentSortKey
from ml.tests.factories import experiment_run

GOOD_ID = "exp_0123456789ab_20260101T000000Z_0001"
BAD_ID = "exp_ffffffffffff_20260101T000000Z_0002"


@pytest.fixture
def store(tmp_path: Path) -> LocalExperimentStore:
    """A store holding one perfectly good record."""
    store = LocalExperimentStore(tmp_path)
    store.save(experiment_run(experiment_id=GOOD_ID))
    return store


def write_record(store: LocalExperimentStore, experiment_id: str, content: bytes) -> Path:
    """Put arbitrary bytes where a record file belongs."""
    directory = store.root / experiment_id
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / RUN_FILENAME
    path.write_bytes(content)
    return path


# ---------------------------------------------------------------------------
# One bad record must not hide the good ones
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("not json", b"{ this is not json"),
        ("empty", b""),
        ("a json list", b"[]"),
        ("a json string", b'"a record"'),
        ("missing sections", b'{"schema_version": "1.0"}'),
        ("unknown schema version", b'{"schema_version": "99.0"}'),
        # The one that escaped: valid JSON bytes are irrelevant if the file
        # cannot be decoded as UTF-8 in the first place.
        ("not utf-8", b"\xff\xfe\x00broken"),
    ],
)
def test_a_bad_record_is_skipped_by_the_listing(
    store: LocalExperimentStore, label: str, content: bytes
) -> None:
    """The good record still comes back, whatever the bad one contains."""
    write_record(store, BAD_ID, content)

    listed = store.list()

    assert [run.experiment_id for run in listed] == [GOOD_ID], label


@pytest.mark.parametrize(
    ("label", "content"),
    [
        ("not json", b"{ this is not json"),
        ("not utf-8", b"\xff\xfe\x00broken"),
    ],
)
def test_a_bad_record_is_reported_by_verify(
    store: LocalExperimentStore, label: str, content: bytes
) -> None:
    """`verify` is the tool for finding these, so it must survive them.

    It caught the same exceptions `list` did, which meant the one failure that
    could break a listing also broke the diagnostic for it.
    """
    write_record(store, BAD_ID, content)

    problems = dict(store.verify())

    assert BAD_ID in problems, label
    assert problems[BAD_ID]


def test_getting_a_named_bad_record_raises_a_typed_error(
    store: LocalExperimentStore,
) -> None:
    """A caller who names a broken record is told it is broken.

    Typed, so the API layer maps it to a status rather than a 500 — and the
    message carries the id and the failure's type, never the file's path.
    """
    write_record(store, BAD_ID, b"\xff\xfe\x00broken")

    with pytest.raises(MalformedExperimentError) as failure:
        store.get(BAD_ID)

    assert isinstance(failure.value, ExperimentError)
    assert BAD_ID in str(failure.value)
    assert str(store.root) not in str(failure.value)
    assert failure.value.details["reason"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read anything")
def test_an_unreadable_file_does_not_break_the_listing(
    store: LocalExperimentStore,
) -> None:
    """Permissions change under a running service more often than anyone plans."""
    path = write_record(store, BAD_ID, b'{"schema_version": "1.0"}')
    path.chmod(0o000)

    try:
        assert [run.experiment_id for run in store.list()] == [GOOD_ID]
        assert BAD_ID in dict(store.verify())
    finally:
        path.chmod(0o600)


def test_a_missing_record_is_still_a_clean_not_found(
    store: LocalExperimentStore,
) -> None:
    """The ordinary case, asserted so the new guard did not swallow it."""
    with pytest.raises(ExperimentNotFoundError):
        store.get(BAD_ID)


# ---------------------------------------------------------------------------
# A record this version no longer understands
# ---------------------------------------------------------------------------


def test_a_record_naming_an_unknown_task_cannot_break_a_sorted_listing(
    store: LocalExperimentStore, tmp_path: Path
) -> None:
    """Sorting by metric had to resolve every record's task and metric.

    A record whose ``task_type`` is not one this version defines made
    ``TaskType(...)`` raise a bare ``ValueError`` — not an ``ExperimentError``,
    so it escaped the listing's own guard and became a 500 on a page that had
    nine perfectly readable runs on it.
    """
    payload = json.loads(
        (store.root / GOOD_ID / RUN_FILENAME).read_text(encoding="utf-8")
    )
    payload["experiment_id"] = BAD_ID
    payload["dataset"]["task_type"] = "clustering"
    write_record(store, BAD_ID, json.dumps(payload).encode("utf-8"))

    unsorted = store.list()
    assert len(unsorted) == 2, "an unknown task is still a readable record"

    with pytest.raises(IncomparableExperimentsError):
        store.list(ExperimentQuery(sort_by=ExperimentSortKey.PRIMARY_METRIC))


def test_an_old_record_without_the_newest_fields_still_reads(
    store: LocalExperimentStore,
) -> None:
    """Fields added after a record was written must read back as absent.

    The additive-section pattern is what lets this project add to a record
    without a schema-version bump, and it is only true while something checks.
    """
    payload = json.loads(
        (store.root / GOOD_ID / RUN_FILENAME).read_text(encoding="utf-8")
    )
    payload["experiment_id"] = BAD_ID
    payload.pop("explainability", None)
    payload.pop("model_artifact", None)
    payload["selection"].pop("rationale", None)
    payload["evaluation"].pop("diagnostics", None)
    payload["evaluation"].pop("warning_count", None)
    write_record(store, BAD_ID, json.dumps(payload).encode("utf-8"))

    restored = store.get(BAD_ID)

    assert restored.explainability is None
    assert restored.model_artifact is None
    assert restored.evaluation.diagnostics == ()
    # The rationale is recomposed from the numbers rather than left blank.
    assert restored.selection.rationale is None
    assert restored.selection.selection_rationale


# ---------------------------------------------------------------------------
# Partial writes and odd shapes on disk
# ---------------------------------------------------------------------------


def test_a_leftover_temporary_file_is_not_mistaken_for_a_record(
    store: LocalExperimentStore,
) -> None:
    """A crash mid-write leaves a dotfile, not a half-written record."""
    directory = store.root / GOOD_ID
    (directory / f".{RUN_FILENAME}.abcd1234.tmp").write_text("{", encoding="utf-8")

    assert [run.experiment_id for run in store.list()] == [GOOD_ID]
    assert store.verify() == ()


def test_a_directory_where_a_record_belongs_is_ignored(
    store: LocalExperimentStore,
) -> None:
    """Not a record, not an error, and above all not a crash."""
    (store.root / BAD_ID / RUN_FILENAME).mkdir(parents=True)

    assert [run.experiment_id for run in store.list()] == [GOOD_ID]
    assert store.exists(BAD_ID) is False


def test_saving_twice_leaves_exactly_one_record(
    store: LocalExperimentStore,
) -> None:
    """The write is atomic, so a repeat replaces rather than accumulates."""
    store.save(experiment_run(experiment_id=GOOD_ID, name="second"))

    files = list((store.root / GOOD_ID).iterdir())

    assert [path.name for path in files] == [RUN_FILENAME]
    assert store.get(GOOD_ID).name == "second"
