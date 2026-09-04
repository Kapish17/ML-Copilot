"""Tests for persisting a trained model and predicting from it.

The claim this commit makes is narrow and load-bearing: **a prediction made
later runs through the same fitted preprocessing as the training that produced
the score.** Everything else here supports that claim or guards the fact that
loading a model means unpickling a file.

Four groups:

**The artifact round-trips.** What comes back is the pipeline that went in,
with a manifest describing exactly the columns it was fitted on.

**Nothing is refitted.** Proved rather than asserted: `fit` is replaced with a
function that raises, and a prediction is made anyway.

**Validation is honest.** A missing column is refused, an unexpected one is
refused rather than dropped, and a value of the wrong kind is refused by name.

**The store loads only what it wrote.** No path comes from a caller, a
traversal attempt is refused, and a substituted file fails its digest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ml.artifacts import (
    LocalModelArtifactStore,
    build_frame,
    build_metadata,
    predict,
)
from ml.artifacts.store import MANIFEST_FILENAME, MODEL_FILENAME
from ml.errors import (
    InvalidExperimentIdError,
    ModelArtifactNotFoundError,
    ModelArtifactUnreadableError,
    PredictionInputError,
)
from ml.features.types import FeatureType, TaskType
from ml.models.selection import select_and_evaluate_best_model
from ml.pipelines.result import PreparedDataset

EXPERIMENT_ID = "exp_test_0001"


@pytest.fixture(scope="module")
def classification_selection(classification_prepared: PreparedDataset):
    """A finished selection over the binary dataset."""
    return select_and_evaluate_best_model(
        classification_prepared, models=["logistic_regression"]
    )


@pytest.fixture(scope="module")
def regression_selection(regression_prepared: PreparedDataset):
    """A finished selection over the housing dataset."""
    return select_and_evaluate_best_model(
        regression_prepared, models=["linear_regression"]
    )


@pytest.fixture
def store(tmp_path: Path) -> LocalModelArtifactStore:
    """An empty artifact store in a directory of this test's own."""
    return LocalModelArtifactStore(tmp_path / "models")


@pytest.fixture
def saved(store, classification_prepared, classification_selection):
    """A stored classification model, and the store holding it."""
    metadata = build_metadata(
        experiment_id=EXPERIMENT_ID,
        prepared=classification_prepared,
        selection=classification_selection,
    )
    store.save(
        EXPERIMENT_ID, classification_selection.final_model.pipeline, metadata
    )
    return store


def a_record(prepared: PreparedDataset) -> dict:
    """One valid record, taken from the training frame's first row."""
    row = prepared.X_train_raw.iloc[0]
    return {column: row[column] for column in prepared.config.feature_columns}


# ---------------------------------------------------------------------------
# The manifest describes the model that is beside it
# ---------------------------------------------------------------------------


def test_the_manifest_names_the_columns_the_pipeline_was_fitted_on(
    classification_prepared, classification_selection
) -> None:
    """In fit order, which is what makes a prediction frame unambiguous.

    Taken from the same configuration the preprocessing was built from, so the
    manifest cannot drift from the model beside it.
    """
    metadata = build_metadata(
        experiment_id=EXPERIMENT_ID,
        prepared=classification_prepared,
        selection=classification_selection,
    )

    assert metadata.feature_names == classification_prepared.config.feature_columns
    assert metadata.feature_names == tuple(classification_prepared.X_train_raw.columns)
    assert metadata.target_column == "renewed"
    assert metadata.task_type is TaskType.CLASSIFICATION


def test_each_column_records_the_branch_that_handles_it(
    classification_prepared, classification_selection
) -> None:
    """Numeric and categorical are treated differently, so the kind is stored."""
    metadata = build_metadata(
        experiment_id=EXPERIMENT_ID,
        prepared=classification_prepared,
        selection=classification_selection,
    )
    kinds = {feature.name: feature.kind for feature in metadata.features}

    assert kinds["income"] is FeatureType.NUMERIC
    assert kinds["segment"] is FeatureType.CATEGORICAL


def test_a_classifier_records_the_labels_it_can_return(
    classification_prepared, classification_selection
) -> None:
    """Asked of the fitted estimator, so they are the labels `predict` returns."""
    metadata = build_metadata(
        experiment_id=EXPERIMENT_ID,
        prepared=classification_prepared,
        selection=classification_selection,
    )

    assert len(metadata.classes) == 2
    assert metadata.supports_probabilities


def test_a_regressor_records_no_classes(
    regression_prepared, regression_selection
) -> None:
    """There are none, and inventing an empty set would imply otherwise."""
    metadata = build_metadata(
        experiment_id=EXPERIMENT_ID,
        prepared=regression_prepared,
        selection=regression_selection,
    )

    assert metadata.classes == ()
    assert not metadata.supports_probabilities


def test_the_manifest_round_trips_through_json(
    classification_prepared, classification_selection
) -> None:
    """It is stored as JSON, so it has to survive being one."""
    from ml.artifacts.schema import ModelArtifactMetadata

    original = build_metadata(
        experiment_id=EXPERIMENT_ID,
        prepared=classification_prepared,
        selection=classification_selection,
    )
    restored = ModelArtifactMetadata.from_dict(
        json.loads(json.dumps(original.as_dict()))
    )

    assert restored.feature_names == original.feature_names
    assert restored.task_type is original.task_type
    assert restored.model_name == original.model_name


def test_no_dataset_value_reaches_the_manifest(
    classification_prepared, classification_selection
) -> None:
    """Column names, kinds and counts — never a cell.

    The uploaded dataset is not persisted, and neither is any part of it. A
    manifest that recorded example values or category lists would be a copy of
    the data by another name.
    """
    metadata = build_metadata(
        experiment_id=EXPERIMENT_ID,
        prepared=classification_prepared,
        selection=classification_selection,
    )
    rendered = json.dumps(metadata.as_dict())

    for value in classification_prepared.X_train_raw["income"].head(20):
        assert str(value) not in rendered


# ---------------------------------------------------------------------------
# The store round-trips, and only loads what it wrote
# ---------------------------------------------------------------------------


def test_saving_writes_a_model_and_a_manifest(saved) -> None:
    """Two files, both named by this code rather than by any input."""
    directory = saved.directory_for(EXPERIMENT_ID)

    assert (directory / MODEL_FILENAME).is_file()
    assert (directory / MANIFEST_FILENAME).is_file()
    assert saved.exists(EXPERIMENT_ID)
    assert saved.stored_ids() == [EXPERIMENT_ID]


def test_loading_returns_the_pipeline_that_was_saved(saved) -> None:
    """A full `Pipeline(preprocessing, estimator)`, fitted."""
    loaded = saved.load(EXPERIMENT_ID)

    assert isinstance(loaded.pipeline, Pipeline)
    assert isinstance(loaded.pipeline.named_steps["preprocessing"], ColumnTransformer)
    # Fitted: an unfitted ColumnTransformer has no `transformers_`.
    assert hasattr(loaded.pipeline.named_steps["preprocessing"], "transformers_")


def test_the_manifest_can_be_read_without_loading_the_model(saved) -> None:
    """So asking "what does this model want?" costs a small JSON read."""
    metadata = saved.metadata_for(EXPERIMENT_ID)

    assert metadata.experiment_id == EXPERIMENT_ID
    assert metadata.feature_names


def test_an_experiment_with_no_artifact_reports_so(store) -> None:
    """Absence is a normal answer, not an error, until someone asks to load."""
    assert store.exists("exp_nothing_here") is False

    with pytest.raises(ModelArtifactNotFoundError):
        store.metadata_for("exp_nothing_here")


@pytest.mark.parametrize(
    "hostile",
    [
        "../escape",
        "../../etc/passwd",
        "/etc/passwd",
        "a/b",
        "..",
        ".",
        "with space",
        "",
    ],
)
def test_an_unsafe_identifier_never_becomes_a_path(store, hostile: str) -> None:
    """The id is validated before it is joined to anything.

    This is the first of the four barriers in the store: an id that could climb
    out of the root is refused as an id, rather than sanitised into something
    that looks acceptable.
    """
    with pytest.raises((InvalidExperimentIdError, ModelArtifactUnreadableError)):
        store.directory_for(hostile)

    # And the convenience path answers "no artifact" rather than raising.
    assert store.exists(hostile) is False


def test_every_resolved_path_stays_inside_the_root(store) -> None:
    """The second barrier, for the case the first cannot see: a symlink."""
    root = store.root.resolve()

    for identifier in ("exp_a", "exp_b_1", "A1"):
        resolved = store.directory_for(identifier).resolve()
        assert root in resolved.parents


def test_a_substituted_model_file_fails_its_digest(saved) -> None:
    """An integrity check, not an authenticity one — and it catches this.

    Anyone who can write to the artifact directory can rewrite the manifest
    too, so this does not stop an attacker with that access. What it does stop
    is a truncated write and a file swapped by accident, either of which would
    otherwise be unpickled.
    """
    directory = saved.directory_for(EXPERIMENT_ID)
    (directory / MODEL_FILENAME).write_bytes(b"not a model at all")

    with pytest.raises(ModelArtifactUnreadableError, match="digest"):
        saved.load(EXPERIMENT_ID)


def test_a_manifest_of_an_unknown_version_is_refused(saved) -> None:
    """Guessed-at compatibility is worse than a clear refusal."""
    manifest = saved.directory_for(EXPERIMENT_ID) / MANIFEST_FILENAME
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = "99.0"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelArtifactUnreadableError):
        saved.metadata_for(EXPERIMENT_ID)


def test_the_filename_is_never_taken_from_the_manifest(saved) -> None:
    """Opening a name that came out of a file would be the whole vulnerability.

    The manifest records the model file's name for a human reading the
    directory. Changing it must therefore change nothing: the store opens a
    constant, and a load still works.
    """
    manifest = saved.directory_for(EXPERIMENT_ID) / MANIFEST_FILENAME
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["model_file"]["name"] = "../../../etc/passwd"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert saved.load(EXPERIMENT_ID).pipeline is not None


def test_deleting_removes_the_artifact(saved) -> None:
    """And says whether there was one."""
    assert saved.delete(EXPERIMENT_ID) is True
    assert saved.exists(EXPERIMENT_ID) is False
    assert saved.delete(EXPERIMENT_ID) is False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_a_valid_record_becomes_a_frame_in_fit_order(
    saved, classification_prepared
) -> None:
    """Exactly the manifest's columns, in exactly its order."""
    metadata = saved.metadata_for(EXPERIMENT_ID)
    frame = build_frame([a_record(classification_prepared)], metadata)

    assert list(frame.columns) == list(metadata.feature_names)
    assert len(frame) == 1


def test_a_missing_feature_is_refused_by_name(saved, classification_prepared) -> None:
    """With the list, so a caller can fix it in one go."""
    record = a_record(classification_prepared)
    record.pop("income")

    with pytest.raises(PredictionInputError) as raised:
        build_frame([record], saved.metadata_for(EXPERIMENT_ID))

    assert "income" in raised.value.details["missing_features"]


def test_an_unexpected_feature_is_refused_rather_than_dropped(
    saved, classification_prepared
) -> None:
    """The rule that matters most, and the one sklearn does not enforce.

    A `ColumnTransformer` with `remainder="drop"` ignores a column it does not
    know. So `tenure_month` for `tenure_months` would not raise — it would be
    dropped, the real column would be missing, and the caller would get a
    confident prediction made without the value they thought they supplied.
    """
    record = a_record(classification_prepared)
    record["tenure_month"] = 12  # a plausible misspelling

    with pytest.raises(PredictionInputError) as raised:
        build_frame([record], saved.metadata_for(EXPERIMENT_ID))

    assert raised.value.details["unexpected_features"] == ["tenure_month"]


def test_a_value_of_the_wrong_kind_is_refused_by_name(
    saved, classification_prepared
) -> None:
    """A numeric column will not take "high"."""
    record = a_record(classification_prepared)
    record["income"] = "high"

    with pytest.raises(PredictionInputError) as raised:
        build_frame([record], saved.metadata_for(EXPERIMENT_ID))

    assert raised.value.details["feature"] == "income"
    assert raised.value.details["index"] == 0


def test_a_numeric_string_is_accepted(saved, classification_prepared) -> None:
    """`"42"` from an HTML form is a number, and refusing it would be pedantry."""
    record = a_record(classification_prepared)
    record["income"] = "42000"

    frame = build_frame([record], saved.metadata_for(EXPERIMENT_ID))

    assert frame["income"].iloc[0] == 42000.0


def test_null_is_accepted_because_imputation_was_fitted_for_it(
    saved, classification_prepared
) -> None:
    """Training handled missing values; refusing them here would be stricter."""
    record = a_record(classification_prepared)
    record["income"] = None

    frame = build_frame([record], saved.metadata_for(EXPERIMENT_ID))

    assert pd.isna(frame["income"].iloc[0])


def test_an_empty_batch_is_refused(saved) -> None:
    """There is nothing to predict, and an empty result would look like one."""
    with pytest.raises(PredictionInputError):
        build_frame([], saved.metadata_for(EXPERIMENT_ID))


def test_a_batch_beyond_the_ceiling_is_refused(
    saved, classification_prepared
) -> None:
    """Prediction is cheap per row and still synchronous."""
    records = [a_record(classification_prepared)] * 5

    with pytest.raises(PredictionInputError, match="At most 2"):
        build_frame(records, saved.metadata_for(EXPERIMENT_ID), max_records=2)


def test_a_record_that_is_not_an_object_is_refused(saved) -> None:
    """With its position, so a batch failure says which row."""
    with pytest.raises(PredictionInputError) as raised:
        build_frame([["not", "an", "object"]], saved.metadata_for(EXPERIMENT_ID))

    assert raised.value.details["index"] == 0


# ---------------------------------------------------------------------------
# Prediction, through the pipeline the experiment fitted
# ---------------------------------------------------------------------------


def test_a_classification_prediction_carries_a_label_and_probabilities(
    saved, classification_prepared
) -> None:
    """Both, because a class without a confidence is half an answer."""
    model = saved.load(EXPERIMENT_ID)
    frame = build_frame([a_record(classification_prepared)], model.metadata)

    result = predict(model, frame)
    first = result.predictions[0]

    assert first.index == 0
    assert first.prediction in list(model.metadata.classes)
    assert first.probabilities is not None
    assert set(first.probabilities) == {str(c) for c in model.metadata.classes}
    assert 0.99 < sum(first.probabilities.values()) < 1.01


def test_a_regression_prediction_is_a_number_with_no_probabilities(
    store, regression_prepared, regression_selection
) -> None:
    """There are no classes, so there is nothing to be confident about."""
    metadata = build_metadata(
        experiment_id="exp_regression_1",
        prepared=regression_prepared,
        selection=regression_selection,
    )
    store.save(
        "exp_regression_1", regression_selection.final_model.pipeline, metadata
    )

    model = store.load("exp_regression_1")
    frame = build_frame([a_record(regression_prepared)], model.metadata)
    result = predict(model, frame)

    assert isinstance(float(result.predictions[0].prediction), float)
    assert result.predictions[0].probabilities is None


def test_a_batch_returns_one_result_per_record_in_order(
    saved, classification_prepared
) -> None:
    """And each result says which record it came from."""
    records = [
        a_record(classification_prepared),
        {
            **a_record(classification_prepared),
            "income": float(classification_prepared.X_train_raw["income"].max()),
        },
        a_record(classification_prepared),
    ]
    model = saved.load(EXPERIMENT_ID)

    result = predict(model, build_frame(records, model.metadata))

    assert len(result.predictions) == 3
    assert [item.index for item in result.predictions] == [0, 1, 2]


def test_preprocessing_is_not_refitted_during_prediction(
    saved, classification_prepared, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The claim this whole commit rests on, proved rather than asserted.

    `fit` and `fit_transform` are replaced with functions that raise. If any
    part of the prediction path refitted the preprocessing — or the estimator —
    this test would fail loudly rather than silently returning a number
    computed from statistics learned on the prediction rows.
    """

    def explode(*_args, **_kwargs):
        raise AssertionError("prediction refitted the pipeline")

    monkeypatch.setattr(ColumnTransformer, "fit", explode)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", explode)
    monkeypatch.setattr(Pipeline, "fit", explode)

    model = saved.load(EXPERIMENT_ID)
    frame = build_frame([a_record(classification_prepared)], model.metadata)

    result = predict(model, frame)

    assert len(result.predictions) == 1


def test_the_loaded_preprocessing_holds_the_training_statistics(
    saved, classification_prepared
) -> None:
    """Not merely "not refitted" — fitted on the *training* rows, as trained.

    The imputer's learned statistic is compared against the one computed from
    the training frame the experiment used. If prediction had refitted on the
    submitted rows, this would differ.
    """
    model = saved.load(EXPERIMENT_ID)
    preprocessing = model.pipeline.named_steps["preprocessing"]

    # numeric branch -> FeatureUnion -> the "values" pipeline -> the imputer.
    branches = {name: transformer for name, transformer, _ in preprocessing.transformers_}
    columns = {name: cols for name, _, cols in preprocessing.transformers_}
    values = dict(branches["numeric"].transformer_list)["values"]
    imputer = values.named_steps["impute"]

    learned = dict(zip(columns["numeric"], imputer.statistics_))
    expected = classification_prepared.X_train_raw["income"].median()

    assert learned["income"] == pytest.approx(expected)


def test_the_result_carries_the_model_and_no_filesystem_location(
    saved, classification_prepared
) -> None:
    """A caller learns what predicted, never where it lives."""
    model = saved.load(EXPERIMENT_ID)
    frame = build_frame([a_record(classification_prepared)], model.metadata)

    rendered = json.dumps(predict(model, frame).as_dict())

    assert "model_name" in rendered
    assert str(saved.root) not in rendered
    for leak in ("/tmp", "model.joblib", "artifact.json", "sha256"):
        assert leak not in rendered
