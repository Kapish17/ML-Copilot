"""Tests for JSON-safe serialization and the experiment record schema."""

from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.errors import (
    InvalidExperimentRecordError,
    MalformedExperimentError,
    SerializationError,
    UnsupportedSchemaVersionError,
)
from ml.experiments.run import (
    EXPERIMENT_SCHEMA_VERSION,
    ExperimentRun,
    capture_environment,
)
from ml.experiments.serialization import (
    MAX_SEQUENCE_LENGTH,
    json_dumps,
    json_loads,
    to_jsonable,
)
from ml.explainability.types import ExplanationMethod
from ml.tests.factories import experiment_run


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (True, True),
        (7, 7),
        (1.5, 1.5),
        ("text", "text"),
        (float("nan"), None),
        (float("inf"), None),
    ],
)
def test_scalars_convert_predictably(value, expected) -> None:
    """Non-finite numbers become null, because they are not valid JSON."""
    assert to_jsonable(value) == expected


def test_numpy_scalars_become_python_numbers() -> None:
    """numpy types would break json.dumps; they are unwrapped first."""
    assert to_jsonable(np.float64(2.5)) == 2.5
    assert to_jsonable(np.int64(7)) == 7
    assert to_jsonable(np.bool_(True)) is True
    assert to_jsonable(np.float64("nan")) is None


def test_numpy_arrays_become_lists() -> None:
    """Small arrays such as a confusion matrix survive as nested lists."""
    assert to_jsonable(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]


def test_pandas_values_convert() -> None:
    """Series, Index and missing markers all have a JSON form."""
    assert to_jsonable(pd.Series([1, 2, 3])) == [1, 2, 3]
    assert to_jsonable(pd.Index(["a", "b"])) == ["a", "b"]
    assert to_jsonable(pd.NaT) is None
    assert to_jsonable(pd.Timestamp("2026-08-26T13:45:00")) == "2026-08-26T13:45:00"


def test_datetimes_become_iso_strings() -> None:
    """A timestamp round-trips through ISO 8601."""
    moment = datetime(2026, 8, 26, 13, 45, tzinfo=timezone.utc)

    assert to_jsonable(moment) == "2026-08-26T13:45:00+00:00"
    assert to_jsonable(date(2026, 8, 26)) == "2026-08-26"
    assert datetime.fromisoformat(to_jsonable(moment)) == moment


def test_enums_become_their_values() -> None:
    """An enum is stored as the string it stands for."""
    assert to_jsonable(ExplanationMethod.SHAP) == "shap"


def test_nested_structures_convert_all_the_way_down() -> None:
    """Conversion is recursive through dicts, lists and tuples."""
    payload = to_jsonable(
        {"scores": [np.float64(1.0), (np.int64(2), float("nan"))], 7: "int key"}
    )

    assert payload == {"scores": [1.0, [2, None]], "7": "int key"}


def test_objects_with_as_dict_are_used() -> None:
    """Project result objects serialise through their own summaries."""
    run = experiment_run()
    assert to_jsonable(run.dataset)["fingerprint"] == run.dataset.fingerprint


def test_a_fitted_pipeline_is_refused() -> None:
    """A model artefact is not experiment history and is never written."""
    pipeline = Pipeline([("model", LogisticRegression())])

    with pytest.raises(SerializationError, match="not stored"):
        to_jsonable(pipeline)


def test_an_estimator_is_refused() -> None:
    """The same applies to a bare estimator."""
    with pytest.raises(SerializationError) as exc_info:
        to_jsonable({"model": LogisticRegression()})
    assert exc_info.value.details["module"] == "sklearn"


def test_a_shap_explainer_is_refused() -> None:
    """And to anything from SHAP."""
    import shap

    explainer = shap.TreeExplainer.__new__(shap.TreeExplainer)
    with pytest.raises(SerializationError) as exc_info:
        to_jsonable(explainer)
    assert exc_info.value.details["module"] == "shap"


def test_a_dataframe_is_refused() -> None:
    """Records hold metadata, not datasets."""
    with pytest.raises(SerializationError, match="DataFrames are not stored"):
        to_jsonable(pd.DataFrame({"a": [1, 2]}))


def test_an_oversized_sequence_is_refused() -> None:
    """A transformed feature matrix has no place in a record."""
    with pytest.raises(SerializationError, match="too large"):
        to_jsonable(list(range(MAX_SEQUENCE_LENGTH + 1)))


def test_a_cyclic_structure_is_refused() -> None:
    """Runaway nesting fails rather than recursing forever."""
    payload: dict = {}
    payload["self"] = payload

    with pytest.raises(SerializationError, match="cyclic"):
        to_jsonable(payload)


def test_writing_rejects_non_finite_numbers() -> None:
    """The writer is the safety net if a NaN escaped conversion."""
    with pytest.raises(SerializationError):
        json_dumps({"value": float("nan")})


def test_reading_reports_corruption() -> None:
    """Malformed text is named as such, not raised as a bare ValueError."""
    with pytest.raises(MalformedExperimentError, match="not valid JSON"):
        json_loads("{not json")


# --------------------------------------------------------------------------
# The record itself
# --------------------------------------------------------------------------


def test_a_run_renders_as_valid_json() -> None:
    """A whole record is writable without a custom encoder."""
    payload = experiment_run().to_dict()
    text = json_dumps(payload)

    assert json.loads(text) == payload
    assert payload["schema_version"] == EXPERIMENT_SCHEMA_VERSION


def test_a_run_survives_a_round_trip() -> None:
    """What is written is what comes back."""
    run = experiment_run()
    restored = ExperimentRun.from_dict(json.loads(json_dumps(run.to_dict())))

    assert restored.to_dict() == run.to_dict()
    assert restored.experiment_id == run.experiment_id
    assert restored.created_at == run.created_at
    assert restored.selected_model == run.selected_model
    assert restored.evaluation.primary_metric_value == pytest.approx(
        run.evaluation.primary_metric_value
    )


def test_the_headline_is_a_one_line_view() -> None:
    """Listings need a compact, serialisable summary of each run."""
    headline = experiment_run().headline()

    json.dumps(headline)
    assert headline["selected_model"] == "logistic_regression"
    assert headline["primary_metric"] == "f1"
    assert headline["test_score"] == pytest.approx(0.80)


def test_explainability_survives_the_round_trip() -> None:
    """The SHAP findings are part of the history, not a side note."""
    restored = ExperimentRun.from_dict(experiment_run().to_dict())

    assert restored.explainability is not None
    assert restored.explainability.method == "shap"
    assert restored.explainability.feature_importances[0]["feature"] == "income"


def test_a_run_without_explanations_round_trips() -> None:
    """Explanations are optional; their absence is recorded as absence."""
    run = experiment_run()
    payload = run.to_dict()
    payload["explainability"] = None

    assert ExperimentRun.from_dict(payload).explainability is None


def test_an_unsupported_schema_version_is_refused() -> None:
    """A record from a future format is not half-read."""
    payload = experiment_run().to_dict()
    payload["schema_version"] = "99.0"

    with pytest.raises(UnsupportedSchemaVersionError) as exc_info:
        ExperimentRun.from_dict(payload)
    assert exc_info.value.details["found"] == "99.0"
    assert EXPERIMENT_SCHEMA_VERSION in exc_info.value.details["supported"]


def test_a_missing_schema_version_is_refused() -> None:
    """An unversioned record cannot be trusted."""
    payload = experiment_run().to_dict()
    del payload["schema_version"]

    with pytest.raises(UnsupportedSchemaVersionError):
        ExperimentRun.from_dict(payload)


@pytest.mark.parametrize(
    "field", ["experiment_id", "created_at", "name", "dataset", "evaluation"]
)
def test_a_missing_required_field_is_named(field: str) -> None:
    """The error says which field is missing, not merely that one is."""
    payload = experiment_run().to_dict()
    del payload[field]

    with pytest.raises(InvalidExperimentRecordError, match=field):
        ExperimentRun.from_dict(payload)


def test_a_field_of_the_wrong_type_is_refused() -> None:
    """Types are checked where it is practical to check them."""
    payload = experiment_run().to_dict()
    payload["dataset"]["row_count"] = "three hundred"

    with pytest.raises(InvalidExperimentRecordError, match="row_count"):
        ExperimentRun.from_dict(payload)


def test_a_broken_timestamp_is_refused() -> None:
    """A record whose creation time cannot be read is not loaded."""
    payload = experiment_run().to_dict()
    payload["created_at"] = "last Tuesday"

    with pytest.raises(InvalidExperimentRecordError, match="ISO 8601"):
        ExperimentRun.from_dict(payload)


def test_a_record_that_is_not_an_object_is_refused() -> None:
    """A list is not an experiment."""
    with pytest.raises(InvalidExperimentRecordError, match="must be an object"):
        ExperimentRun.from_dict([1, 2, 3])


# --------------------------------------------------------------------------
# Reproducibility metadata
# --------------------------------------------------------------------------


def test_the_environment_records_what_reproduction_needs() -> None:
    """Interpreter, platform, library versions and the seed."""
    environment = capture_environment(random_state=42)

    assert environment.python_version.count(".") == 2
    assert environment.platform
    assert environment.random_state == 42
    assert "pandas" in environment.packages
    assert "scikit-learn" in environment.packages
    assert "shap" in environment.packages


def test_the_environment_records_nothing_identifying() -> None:
    """No hostname, user, path or environment variable is captured."""
    payload = capture_environment().as_dict()

    assert set(payload) == {"python_version", "platform", "packages", "random_state"}
    text = json.dumps(payload).lower()
    for secret in ("token", "key", "password", "secret", "/home/", "http"):
        assert secret not in text


def test_the_record_holds_no_model_or_explainer_text() -> None:
    """A stored record cannot be padded out with an artefact's repr."""
    text = json_dumps(experiment_run().to_dict())

    for artefact in ("Pipeline(", "LogisticRegression(", "ColumnTransformer(", "Explainer object"):
        assert artefact not in text


def test_a_record_stays_small() -> None:
    """Experiment history is metadata; it should not grow like data."""
    text = json_dumps(experiment_run().to_dict())
    assert len(text) < 16_000
    assert not math.isnan(len(text))
