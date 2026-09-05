"""Tests for the prediction endpoints, over the real HTTP contract.

`ml/tests/test_artifacts.py` covers the store and the validation rules directly.
This module asks the questions only the API can answer: does a finished run
leave a model behind, does the endpoint find it, and does everything the store
refuses stay refused when the request comes over the wire.

The distinctions worth having are the ones a caller has to act on:

    unknown experiment              404  experiment_not_found
    known experiment, no model      409  model_not_available
    known experiment, damaged model 500  model_artifact_unreadable
    model, bad records              422  invalid_prediction_input
    model, good records             200
    body larger than the ceiling    413  request_body_too_large

Collapsing the first two into one error would tell someone their run had
vanished when it is sitting in the history, so each is asserted separately.

The model endpoint answers the same question without failing: `available`,
`not_available` and `corrupted` all arrive as **200**, because "this run
cannot be predicted from" is an answer a client needs in order to render the
right thing, and each of the three implies a different next step.

**No path reaches this API and none leaves it.** A prediction request carries
feature values and nothing else; the model is chosen by the id in the URL. The
last group of tests tries every way of smuggling one in and then checks that no
response mentions where anything is stored.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.schemas.prediction import MAX_RECORDS_HARD_LIMIT
from ml.artifacts import LocalModelArtifactStore
from ml.artifacts.store import MANIFEST_FILENAME, MODEL_FILENAME

#: A small learnable table: two numeric columns, one categorical, a binary
#: target. Deterministic, and large enough to cross-validate.
CLASSIFICATION_CSV = "\n".join(
    ["income,tenure_months,segment,renewed"]
    + [
        f"{30000 + index * 900},{1 + index % 40},{'a' if index % 2 else 'b'},"
        f"{1 if index % 3 else 0}"
        for index in range(90)
    ]
).encode()

REGRESSION_CSV = "\n".join(
    ["size_sqm,rooms,price"]
    + [f"{40 + index},{1 + index % 5},{100000 + index * 2500}" for index in range(90)]
).encode()


@pytest.fixture(scope="module")
def artifact_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Where this module's models are written."""
    return tmp_path_factory.mktemp("model-artifacts")


@pytest.fixture(scope="module")
def prediction_settings(
    tmp_path_factory: pytest.TempPathFactory, artifact_dir: Path
) -> Settings:
    """Settings whose record store and model store are both temporary."""
    return Settings(
        experiment_store_dir=tmp_path_factory.mktemp("prediction-runs"),
        model_artifact_dir=artifact_dir,
    )


@pytest.fixture(scope="module")
def predict_client(prediction_settings: Settings) -> TestClient:
    """A client for an application that persists the models it trains."""
    with TestClient(create_app(prediction_settings)) as client:
        yield client


def run(client: TestClient, content: bytes, target: str, **extra: Any) -> dict:
    """Run an experiment and return its record."""
    response = client.post(
        "/api/v1/experiments/run",
        files={"file": ("data.csv", io.BytesIO(content), "text/csv")},
        data={"target_column": target, "explain": "false", **extra},
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def classification_run(predict_client: TestClient) -> dict:
    """One finished classification experiment."""
    return run(
        predict_client,
        CLASSIFICATION_CSV,
        "renewed",
        models=["logistic_regression"],
    )


@pytest.fixture(scope="module")
def regression_run(predict_client: TestClient) -> dict:
    """One finished regression experiment."""
    return run(
        predict_client, REGRESSION_CSV, "price", models=["linear_regression"]
    )


def features_of(client: TestClient, experiment_id: str) -> list[dict]:
    """The feature schema the model expects."""
    response = client.get(f"/api/v1/experiments/{experiment_id}/model")
    assert response.status_code == 200, response.text
    return response.json()["features"]


def a_record(client: TestClient, experiment_id: str) -> dict[str, Any]:
    """One valid record built from the model's own declared schema."""
    return {
        feature["name"]: (5 if feature["kind"] == "numeric" else "a")
        for feature in features_of(client, experiment_id)
    }


# ---------------------------------------------------------------------------
# A finished run leaves a model behind
# ---------------------------------------------------------------------------


def test_a_successful_run_persists_a_model(
    classification_run: dict, artifact_dir: Path
) -> None:
    """Two files, under a directory named for the experiment."""
    experiment_id = classification_run["experiment_id"]
    directory = artifact_dir / experiment_id

    assert (directory / MODEL_FILENAME).is_file()
    assert (directory / MANIFEST_FILENAME).is_file()


def test_the_record_says_a_model_was_stored(classification_run: dict) -> None:
    """With the schema a prediction has to satisfy, and no cell of the data."""
    artifact = classification_run["model_artifact"]

    assert artifact is not None
    assert artifact["stored"] is True
    assert artifact["model_name"] == "logistic_regression"
    assert artifact["target_column"] == "renewed"
    assert set(artifact["feature_names"]) == {"income", "tenure_months", "segment"}
    assert artifact["feature_count"] == 3
    assert sorted(artifact["class_labels"]) == ["0", "1"]


def test_the_artifact_belongs_to_the_experiment_that_made_it(
    predict_client: TestClient, classification_run: dict, regression_run: dict
) -> None:
    """Two runs, two models, neither answering for the other."""
    first = predict_client.get(
        f"/api/v1/experiments/{classification_run['experiment_id']}/model"
    ).json()
    second = predict_client.get(
        f"/api/v1/experiments/{regression_run['experiment_id']}/model"
    ).json()

    assert first["experiment_id"] == classification_run["experiment_id"]
    assert second["experiment_id"] == regression_run["experiment_id"]
    assert first["task_type"] == "classification"
    assert second["task_type"] == "regression"
    assert first["target_column"] != second["target_column"]


def test_a_run_with_nowhere_to_store_a_model_still_succeeds(
    tmp_path: Path,
) -> None:
    """Persistence is a bonus on top of an experiment, not a precondition.

    An application built without an artifact store — which is every run made
    before Commit 22 — records the experiment exactly as before and simply has
    no model section.
    """
    from app.api.dependencies import get_model_artifact_store

    settings = Settings(experiment_store_dir=tmp_path / "runs")
    application = create_app(settings)
    application.dependency_overrides[get_model_artifact_store] = lambda: None

    with TestClient(application) as client:
        # The runner takes `None` for "nowhere to write", which is what the
        # override supplies.
        record = run(client, CLASSIFICATION_CSV, "renewed", models=["logistic_regression"])

    assert record["model_artifact"] is None
    assert record["evaluation"]["primary_metric_value"] is not None


def test_a_failed_run_leaves_no_model(
    predict_client: TestClient, artifact_dir: Path
) -> None:
    """Persistence happens after evaluation, so there is nothing to leave.

    The run is refused for a target column that does not exist, which fails
    long before a model is fitted. Nothing new appears in the store.
    """
    before = {path.name for path in artifact_dir.iterdir()}

    response = predict_client.post(
        "/api/v1/experiments/run",
        files={"file": ("data.csv", io.BytesIO(CLASSIFICATION_CSV), "text/csv")},
        data={"target_column": "not_a_column"},
    )

    assert response.status_code >= 400
    assert {path.name for path in artifact_dir.iterdir()} == before


def test_no_uploaded_dataset_is_written_anywhere(
    predict_client: TestClient, classification_run: dict, artifact_dir: Path
) -> None:
    """The standing guarantee, re-checked now that something *is* written.

    Every byte under the artifact directory is searched for distinctive values
    from the uploaded table. A fitted model holds learned coefficients, not
    rows, and the manifest holds column names — so none of the data may appear.
    """
    distinctive = [b"30000", b"38100", b"56100"]

    for path in artifact_dir.rglob("*"):
        if not path.is_file():
            continue
        blob = path.read_bytes()
        for value in distinctive:
            assert value not in blob, f"{path.name} holds a value from the upload"


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


def test_a_run_with_a_model_reports_its_schema(
    predict_client: TestClient, classification_run: dict
) -> None:
    """Enough for a client to build a form without guessing."""
    body = predict_client.get(
        f"/api/v1/experiments/{classification_run['experiment_id']}/model"
    ).json()

    assert body["available"] is True
    assert body["reason"] is None
    assert body["max_records"] > 0
    kinds = {feature["name"]: feature["kind"] for feature in body["features"]}
    assert kinds == {
        "income": "numeric",
        "tenure_months": "numeric",
        "segment": "categorical",
    }


def test_availability_is_answered_by_the_store_not_by_the_record(
    predict_client: TestClient, artifact_dir: Path
) -> None:
    """Delete the artifact and the answer changes; the record does not.

    The record notes what happened when the run finished. Whether a prediction
    can be made *now* is a different question, and this is what makes the
    difference visible.
    """
    record = run(
        predict_client, CLASSIFICATION_CSV, "renewed", models=["logistic_regression"]
    )
    experiment_id = record["experiment_id"]
    assert record["model_artifact"]["stored"] is True

    LocalModelArtifactStore(artifact_dir).delete(experiment_id)

    body = predict_client.get(f"/api/v1/experiments/{experiment_id}/model").json()
    assert body["available"] is False
    assert "no stored model" in body["reason"]
    assert body["features"] == []

    # And the record still says a model was written, which is true.
    stored = predict_client.get(f"/api/v1/experiments/{experiment_id}").json()
    assert stored["model_artifact"]["stored"] is True


def test_asking_about_an_unknown_experiment_is_a_404(
    predict_client: TestClient,
) -> None:
    """The run is gone — a different problem from "the run has no model"."""
    response = predict_client.get("/api/v1/experiments/exp_does_not_exist/model")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "experiment_not_found"


# ---------------------------------------------------------------------------
# The model's lifecycle, over HTTP
#
# The endpoint reports three states and each one means a different next step.
# What these tests pin is that the state is answered from the artifact as it is
# *now* — every one of them changes something on disk and asks again.
# ---------------------------------------------------------------------------


def test_a_usable_model_reports_available_with_its_lifecycle_metadata(
    predict_client: TestClient, classification_run: dict
) -> None:
    """Everything a client needs to describe the model, and nothing more."""
    body = predict_client.get(
        f"/api/v1/experiments/{classification_run['experiment_id']}/model"
    ).json()

    assert body["status"] == "available"
    assert body["available"] is True
    assert body["reason_code"] is None
    assert body["reason"] is None

    # What it is, what it predicts, and how well — with the sample size, so
    # the score is not read as a claim of unknown weight.
    assert body["model_name"] == "logistic_regression"
    assert body["display_name"]
    assert body["task_type"] == "classification"
    assert body["target_column"] == "renewed"
    assert body["train_row_count"] > 0
    assert body["test_row_count"] > 0
    assert body["primary_metric"]
    assert body["primary_metric_value"] is not None
    assert body["supports_probabilities"] is True
    assert body["artifact_schema_version"] == "1.0"
    assert body["created_at"]

    # And nothing about the host or how the artifact is kept.
    assert "environment" not in body
    assert "random_state" not in body


def test_a_regression_model_reports_no_classes_and_no_probabilities(
    predict_client: TestClient, regression_run: dict
) -> None:
    """So a client knows not to render a probability section at all."""
    body = predict_client.get(
        f"/api/v1/experiments/{regression_run['experiment_id']}/model"
    ).json()

    assert body["status"] == "available"
    assert body["classes"] == []
    assert body["supports_probabilities"] is False


def test_a_removed_artifact_reports_not_available_with_a_stable_code(
    predict_client: TestClient, artifact_dir: Path
) -> None:
    """Absence is normal, and says so in a code as well as a sentence."""
    record = run(
        predict_client, CLASSIFICATION_CSV, "renewed", models=["logistic_regression"]
    )
    experiment_id = record["experiment_id"]
    LocalModelArtifactStore(artifact_dir).delete(experiment_id)

    body = predict_client.get(f"/api/v1/experiments/{experiment_id}/model").json()

    assert body["status"] == "not_available"
    assert body["available"] is False
    assert body["reason_code"] == "no_artifact"
    assert "re-run" in body["reason"].lower()
    assert body["features"] == []
    assert body["supports_probabilities"] is False


def test_a_damaged_artifact_reports_corrupted_rather_than_missing(
    predict_client: TestClient, artifact_dir: Path
) -> None:
    """The state a boolean could not express, and the reason it was added.

    "No model" and "a broken model" have different fixes, and a dashboard that
    tells someone to re-run an experiment when the real answer is that a file
    is damaged has sent them to do the wrong thing.
    """
    record = run(
        predict_client, CLASSIFICATION_CSV, "renewed", models=["logistic_regression"]
    )
    experiment_id = record["experiment_id"]
    model = artifact_dir / experiment_id / MODEL_FILENAME
    model.write_bytes(model.read_bytes()[:32])

    body = predict_client.get(f"/api/v1/experiments/{experiment_id}/model").json()

    assert body["status"] == "corrupted"
    assert body["available"] is False
    assert body["reason_code"] == "model_file_truncated"
    # No schema is published for a model that cannot answer: a form built from
    # one would have every submission fail.
    assert body["features"] == []


def test_an_artifact_from_a_newer_version_says_so_specifically(
    predict_client: TestClient, artifact_dir: Path
) -> None:
    """Because the fix is to upgrade, not to re-run."""
    record = run(
        predict_client, CLASSIFICATION_CSV, "renewed", models=["logistic_regression"]
    )
    experiment_id = record["experiment_id"]
    manifest = artifact_dir / experiment_id / MANIFEST_FILENAME
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = "99.0"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    body = predict_client.get(f"/api/v1/experiments/{experiment_id}/model").json()

    assert body["status"] == "corrupted"
    assert body["reason_code"] == "unsupported_schema_version"
    assert "newer version" in body["reason"]


def test_an_unreadable_manifest_reports_corrupted_and_names_no_file(
    predict_client: TestClient, artifact_dir: Path
) -> None:
    """A damaged artifact is described by its condition, never by its bytes."""
    record = run(
        predict_client, CLASSIFICATION_CSV, "renewed", models=["logistic_regression"]
    )
    experiment_id = record["experiment_id"]
    (artifact_dir / experiment_id / MANIFEST_FILENAME).write_text("{ oops", "utf-8")

    body = predict_client.get(f"/api/v1/experiments/{experiment_id}/model").json()

    assert body["status"] == "corrupted"
    assert body["reason_code"] == "manifest_unreadable"
    rendered = json.dumps(body)
    for leak in (MANIFEST_FILENAME, MODEL_FILENAME, "joblib", str(artifact_dir)):
        assert leak not in rendered


def test_a_corrupted_artifact_is_refused_by_predict_without_being_opened(
    predict_client: TestClient, artifact_dir: Path
) -> None:
    """The two endpoints agree, because they ask the same question.

    A prediction against a damaged artifact fails deterministically and with a
    generic message — and it fails at the status check, before anything is
    handed to the deserialiser.
    """
    record = run(
        predict_client, CLASSIFICATION_CSV, "renewed", models=["logistic_regression"]
    )
    experiment_id = record["experiment_id"]
    model = artifact_dir / experiment_id / MODEL_FILENAME
    model.write_bytes(model.read_bytes()[:32])

    status = predict_client.get(
        f"/api/v1/experiments/{experiment_id}/model"
    ).json()["status"]
    response = predict_client.post(
        f"/api/v1/experiments/{experiment_id}/predict",
        json={"records": [{"income": 5, "tenure_months": 5, "segment": "a"}]},
    )

    assert status == "corrupted"
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "model_artifact_unreadable"
    assert body["error"]["details"] == {}


def test_a_model_file_missing_beside_its_manifest_is_corrupted(
    predict_client: TestClient, artifact_dir: Path
) -> None:
    """Not "never had one": the manifest is written second, so this is damage."""
    record = run(
        predict_client, CLASSIFICATION_CSV, "renewed", models=["logistic_regression"]
    )
    experiment_id = record["experiment_id"]
    (artifact_dir / experiment_id / MODEL_FILENAME).unlink()

    body = predict_client.get(f"/api/v1/experiments/{experiment_id}/model").json()

    assert body["status"] == "corrupted"
    assert body["reason_code"] == "model_file_missing"


# ---------------------------------------------------------------------------
# Predicting
# ---------------------------------------------------------------------------


def test_a_classification_prediction_returns_a_class_and_probabilities(
    predict_client: TestClient, classification_run: dict
) -> None:
    """And says which model produced them."""
    experiment_id = classification_run["experiment_id"]
    record = a_record(predict_client, experiment_id)

    response = predict_client.post(
        f"/api/v1/experiments/{experiment_id}/predict", json={"records": [record]}
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prediction_count"] == 1
    first = body["predictions"][0]
    assert first["index"] == 0
    assert str(first["prediction"]) in {"0", "1"}
    assert set(first["probabilities"]) == {"0", "1"}
    assert body["model"]["model_name"] == "logistic_regression"
    assert body["model"]["task_type"] == "classification"


def test_a_regression_prediction_returns_a_number_and_no_probabilities(
    predict_client: TestClient, regression_run: dict
) -> None:
    """There are no classes to be confident about."""
    experiment_id = regression_run["experiment_id"]
    record = a_record(predict_client, experiment_id)

    body = predict_client.post(
        f"/api/v1/experiments/{experiment_id}/predict", json={"records": [record]}
    ).json()

    assert isinstance(body["predictions"][0]["prediction"], (int, float))
    assert body["predictions"][0]["probabilities"] is None
    assert body["model"]["classes"] == []


def test_a_batch_returns_one_result_per_record_in_order(
    predict_client: TestClient, classification_run: dict
) -> None:
    """One shape for one record and for many."""
    experiment_id = classification_run["experiment_id"]
    base = a_record(predict_client, experiment_id)
    records = [base, {**base, "income": 90000}, {**base, "income": 1000}]

    body = predict_client.post(
        f"/api/v1/experiments/{experiment_id}/predict", json={"records": records}
    ).json()

    assert body["prediction_count"] == 3
    assert [item["index"] for item in body["predictions"]] == [0, 1, 2]


def test_the_response_reports_the_model_score_not_the_prediction_confidence(
    predict_client: TestClient, classification_run: dict
) -> None:
    """`primary_metric_value` is the held-out measurement of the model.

    Carried so a caller can see how much to trust the model, and named on the
    schema so it cannot be misread as a property of this prediction.
    """
    experiment_id = classification_run["experiment_id"]
    body = predict_client.post(
        f"/api/v1/experiments/{experiment_id}/predict",
        json={"records": [a_record(predict_client, experiment_id)]},
    ).json()

    assert body["model"]["primary_metric"]
    assert (
        body["model"]["primary_metric_value"]
        == classification_run["evaluation"]["primary_metric_value"]
    )


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda record: record.pop("income"), "missing_features"),
        (lambda record: record.update({"incomes": 5}), "unexpected_features"),
        (lambda record: record.update({"income": "a lot"}), "feature"),
    ],
    ids=["missing", "unexpected", "wrong type"],
)
def test_a_record_that_does_not_match_the_schema_is_refused(
    predict_client: TestClient, classification_run: dict, mutate, expected: str
) -> None:
    """422, with the details naming what is wrong."""
    experiment_id = classification_run["experiment_id"]
    record = a_record(predict_client, experiment_id)
    mutate(record)

    response = predict_client.post(
        f"/api/v1/experiments/{experiment_id}/predict", json={"records": [record]}
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "invalid_prediction_input"
    assert expected in body["error"]["details"]


def test_a_malformed_request_is_refused_by_the_schema(
    predict_client: TestClient, classification_run: dict
) -> None:
    """No `records`, an empty list, or a field the contract does not have."""
    experiment_id = classification_run["experiment_id"]
    url = f"/api/v1/experiments/{experiment_id}/predict"

    for payload in ({}, {"records": []}, {"records": [{}], "model_path": "/etc"}):
        response = predict_client.post(url, json=payload)
        assert response.status_code == 422, payload
        assert "error" in response.json()


def test_an_empty_record_is_refused_as_a_missing_schema_not_an_empty_row(
    predict_client: TestClient, classification_run: dict
) -> None:
    """`{}` is not "predict with defaults" — it is every feature missing.

    Worth its own test because the alternative reading is tempting and wrong:
    imputing every column would produce a confident prediction about nothing
    in particular, which is exactly the outcome this endpoint refuses.
    """
    experiment_id = classification_run["experiment_id"]

    response = predict_client.post(
        f"/api/v1/experiments/{experiment_id}/predict", json={"records": [{}]}
    )

    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "invalid_prediction_input"
    assert set(body["error"]["details"]["missing_features"]) == {
        "income",
        "tenure_months",
        "segment",
    }


@pytest.mark.parametrize(
    "value",
    [{"nested": 1}, [1, 2, 3], True],
    ids=["object", "list", "boolean for a numeric column"],
)
def test_a_malformed_feature_value_is_a_422_and_never_a_crash(
    predict_client: TestClient, classification_run: dict, value: Any
) -> None:
    """JSON nests and a feature value does not.

    A list reaching a coercion written for scalars is the sort of input that
    produces an ambiguous-truth-value `ValueError` deep in pandas and a 500 at
    the edge. It is refused by name instead. (A boolean *is* accepted for a
    numeric column — `True` is 1 — so that case asserts a 200; it is here to
    keep the parametrisation honest about which values are actually refused.)
    """
    experiment_id = classification_run["experiment_id"]
    record = {**a_record(predict_client, experiment_id), "income": value}

    response = predict_client.post(
        f"/api/v1/experiments/{experiment_id}/predict", json={"records": [record]}
    )

    if isinstance(value, bool):
        assert response.status_code == 200
        return
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_prediction_input"


# ---------------------------------------------------------------------------
# Resource protection
#
# Two ceilings in two dimensions, because one of them alone is not a bound.
# The record limit stops a long batch; the body limit stops a short one made
# of very large values. No rate limiting is implemented — see
# `docs/PRODUCTION_READINESS.md`, which says so rather than implying otherwise.
# ---------------------------------------------------------------------------


def test_a_batch_beyond_the_configured_ceiling_is_refused(
    tmp_path: Path,
) -> None:
    """The limit is the configured one, not a constant in the route.

    Set low deliberately: a test that had to build five hundred records to
    prove a five-hundred-record limit would be proving something about the
    default rather than about the mechanism.
    """
    settings = Settings(
        experiment_store_dir=tmp_path / "runs",
        model_artifact_dir=tmp_path / "models",
        max_prediction_records=3,
    )

    with TestClient(create_app(settings)) as client:
        record = run(
            client, CLASSIFICATION_CSV, "renewed", models=["logistic_regression"]
        )
        experiment_id = record["experiment_id"]
        url = f"/api/v1/experiments/{experiment_id}/predict"
        one = a_record(client, experiment_id)

        assert client.get(f"/api/v1/experiments/{experiment_id}/model").json()[
            "max_records"
        ] == 3

        within = client.post(url, json={"records": [one] * 3})
        beyond = client.post(url, json={"records": [one] * 4})

    assert within.status_code == 200
    assert within.json()["prediction_count"] == 3

    assert beyond.status_code == 422
    body = beyond.json()
    assert body["error"]["code"] == "invalid_prediction_input"
    assert body["error"]["details"]["maximum"] == 3
    assert body["error"]["details"]["record_count"] == 4


def test_the_schema_refuses_an_absurd_batch_before_a_list_is_built(
    predict_client: TestClient, classification_run: dict
) -> None:
    """A second ceiling, in the request contract itself.

    The configured limit is checked after the body has been parsed into
    dictionaries. This one is checked by the schema, so a caller sending a
    million records is refused without a million dictionaries existing first.
    """
    experiment_id = classification_run["experiment_id"]

    response = predict_client.post(
        f"/api/v1/experiments/{experiment_id}/predict",
        json={"records": [{"income": 1}] * (MAX_RECORDS_HARD_LIMIT + 1)},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_a_body_larger_than_the_limit_is_refused_before_it_is_parsed(
    tmp_path: Path,
) -> None:
    """The bound the record ceiling does not provide.

    Five hundred records is a limit on rows and says nothing about their size:
    a handful of records carrying very long strings is legal under it and
    unbounded under any other measure. Bodies are bounded too, and the refusal
    uses the same envelope as every other failure so a client parses it the
    same way.
    """
    settings = Settings(
        experiment_store_dir=tmp_path / "runs",
        model_artifact_dir=tmp_path / "models",
        max_request_body_bytes=4096,
    )

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/experiments/exp_anything/predict",
            json={"records": [{"income": "x" * 20_000}]},
        )
        # A small body on the same client still reaches the application, so
        # the limit is a limit and not a closed door.
        small = client.post(
            "/api/v1/experiments/exp_anything/predict",
            json={"records": [{"income": 1}]},
        )

    assert response.status_code == 413
    body = response.json()
    assert body["error"]["code"] == "request_body_too_large"
    assert body["error"]["details"]["max_bytes"] == 4096
    assert response.headers.get("X-Request-ID")

    assert small.status_code == 404


def test_the_body_limit_leaves_dataset_uploads_alone(tmp_path: Path) -> None:
    """Uploads have their own, larger limit and their own streaming reader.

    A second ceiling over the top of them would be a confusing way to change
    `MAX_UPLOAD_MB`, so multipart requests are passed straight through.
    """
    settings = Settings(
        experiment_store_dir=tmp_path / "runs",
        model_artifact_dir=tmp_path / "models",
        max_request_body_bytes=1024,
    )

    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/datasets/profile",
            files={"file": ("data.csv", io.BytesIO(CLASSIFICATION_CSV), "text/csv")},
        )

    assert len(CLASSIFICATION_CSV) > 1024
    assert response.status_code == 200


def test_predicting_from_an_experiment_with_no_model_is_a_409(
    predict_client: TestClient, artifact_dir: Path
) -> None:
    """`model_not_available` — the run is fine, it just has no model.

    Deliberately not a 404: that would say the experiment was gone, which is a
    different problem with a different fix.
    """
    record = run(
        predict_client, CLASSIFICATION_CSV, "renewed", models=["logistic_regression"]
    )
    experiment_id = record["experiment_id"]
    LocalModelArtifactStore(artifact_dir).delete(experiment_id)

    response = predict_client.post(
        f"/api/v1/experiments/{experiment_id}/predict",
        json={"records": [{"income": 5, "tenure_months": 5, "segment": "a"}]},
    )

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "model_not_available"


def test_predicting_from_an_unknown_experiment_is_a_404(
    predict_client: TestClient,
) -> None:
    """Distinguished from the case above, because the fixes differ."""
    response = predict_client.post(
        "/api/v1/experiments/exp_not_here/predict",
        json={"records": [{"income": 1}]},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "experiment_not_found"


def test_preprocessing_is_not_refitted_by_an_http_prediction(
    predict_client: TestClient, classification_run: dict, monkeypatch
) -> None:
    """The same proof as the unit test, through the whole HTTP path.

    Anything in the request path that refitted the pipeline — a helper, a
    middleware, a re-validation step — would raise here rather than quietly
    returning a number computed from the submitted rows.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    def explode(*_args, **_kwargs):
        raise AssertionError("an HTTP prediction refitted the pipeline")

    monkeypatch.setattr(ColumnTransformer, "fit", explode)
    monkeypatch.setattr(ColumnTransformer, "fit_transform", explode)
    monkeypatch.setattr(Pipeline, "fit", explode)

    experiment_id = classification_run["experiment_id"]
    response = predict_client.post(
        f"/api/v1/experiments/{experiment_id}/predict",
        json={"records": [a_record(predict_client, experiment_id)]},
    )

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hostile_id",
    [
        "..%2F..%2Fetc%2Fpasswd",
        "..",
        "exp_a%2F..%2F..%2Fetc",
        "a b",
        "exp_$(whoami)",
    ],
)
def test_a_hostile_experiment_id_never_reaches_the_filesystem(
    predict_client: TestClient, hostile_id: str
) -> None:
    """Refused as an identifier, long before it could address anything.

    Both routes take the id from the URL and nowhere else, and it is validated
    by the record store and again by the artifact store. Whatever the status,
    it is never a 200 and never a traceback.
    """
    for url in (
        f"/api/v1/experiments/{hostile_id}/model",
        f"/api/v1/experiments/{hostile_id}/predict",
    ):
        response = (
            predict_client.get(url)
            if url.endswith("/model")
            else predict_client.post(url, json={"records": [{"a": 1}]})
        )
        assert response.status_code in (400, 404, 422), (url, response.status_code)
        assert "Traceback" not in response.text


def test_a_request_cannot_name_a_file_to_load(
    predict_client: TestClient, classification_run: dict
) -> None:
    """There is no field for it, and adding one is refused by the contract.

    The request model forbids extra fields, so every attempt to smuggle a path
    fails request validation rather than reaching any code that could act on
    it. This is the property the whole trust boundary rests on.
    """
    experiment_id = classification_run["experiment_id"]

    for smuggled in (
        {"model_path": "/etc/passwd"},
        {"artifact": "../../model.joblib"},
        {"pickle": "http://elsewhere/evil.joblib"},
        {"model_file": MODEL_FILENAME},
    ):
        response = predict_client.post(
            f"/api/v1/experiments/{experiment_id}/predict",
            json={"records": [a_record(predict_client, experiment_id)], **smuggled},
        )
        assert response.status_code == 422, smuggled


def test_a_feature_named_like_a_path_is_just_a_rejected_feature(
    predict_client: TestClient, classification_run: dict
) -> None:
    """Record keys are feature names, and are only ever compared to the schema.

    Nothing joins one to a directory, so the worst a hostile key can do is be
    an unexpected feature.
    """
    experiment_id = classification_run["experiment_id"]
    record = a_record(predict_client, experiment_id)
    record["../../../etc/passwd"] = 1

    response = predict_client.post(
        f"/api/v1/experiments/{experiment_id}/predict", json={"records": [record]}
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_prediction_input"


def test_no_response_reveals_where_anything_is_stored(
    predict_client: TestClient, classification_run: dict, artifact_dir: Path
) -> None:
    """Not on success, and not on any of the failure paths."""
    experiment_id = classification_run["experiment_id"]
    record = a_record(predict_client, experiment_id)

    responses = [
        predict_client.get(f"/api/v1/experiments/{experiment_id}/model"),
        predict_client.post(
            f"/api/v1/experiments/{experiment_id}/predict", json={"records": [record]}
        ),
        predict_client.post(
            f"/api/v1/experiments/{experiment_id}/predict",
            json={"records": [{"income": "bad"}]},
        ),
        predict_client.get("/api/v1/experiments/exp_missing/model"),
        predict_client.get(f"/api/v1/experiments/{experiment_id}"),
    ]

    # The digest recorded in the manifest, by value rather than by the word
    # "sha256" — which legitimately appears in every record as the name of the
    # *dataset* fingerprint algorithm and has nothing to do with the artifact.
    manifest = json.loads(
        (artifact_dir / experiment_id / MANIFEST_FILENAME).read_text("utf-8")
    )
    digest = manifest["model_file"]["sha256"]

    for response in responses:
        text = response.text
        assert str(artifact_dir) not in text
        assert digest not in text
        for leak in (MODEL_FILENAME, MANIFEST_FILENAME, "/tmp/", "joblib"):
            assert leak not in text, (leak, response.request.url)


def test_a_corrupt_artifact_is_a_server_error_with_a_generic_message(
    predict_client: TestClient, artifact_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The caller learns the request failed; the real cause goes to the log."""
    record = run(
        predict_client, CLASSIFICATION_CSV, "renewed", models=["logistic_regression"]
    )
    experiment_id = record["experiment_id"]
    (artifact_dir / experiment_id / MODEL_FILENAME).write_bytes(b"corrupted")

    with caplog.at_level(logging.WARNING):
        response = predict_client.post(
            f"/api/v1/experiments/{experiment_id}/predict",
            json={"records": [{"income": 5, "tenure_months": 5, "segment": "a"}]},
        )

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "model_artifact_unreadable"
    assert body["error"]["details"] == {}
    assert "digest" not in body["error"]["message"]


def test_predicting_needs_the_api_key_when_authentication_is_enabled(
    tmp_path: Path,
) -> None:
    """Both new routes are protected, like every other expensive one."""
    # A different literal from the one `test_authentication.py` uses: that
    # module asserts its key appears nowhere else in the repository, and
    # sharing the string would make the assertion false for no benefit.
    key = "prediction-test-key-not-a-real-secret-01"
    settings = Settings(
        api_auth_enabled=True,
        api_auth_key=key,
        experiment_store_dir=tmp_path / "runs",
        model_artifact_dir=tmp_path / "models",
    )

    with TestClient(create_app(settings)) as client:
        for method, url in (
            ("get", "/api/v1/experiments/exp_x/model"),
            ("post", "/api/v1/experiments/exp_x/predict"),
        ):
            call = getattr(client, method)
            unauthenticated = (
                call(url) if method == "get" else call(url, json={"records": [{"a": 1}]})
            )
            assert unauthenticated.status_code == 401
            assert (
                unauthenticated.json()["error"]["code"] == "authentication_required"
            )

        # And with the key, the request gets as far as the real answer.
        with_key = client.get(
            "/api/v1/experiments/exp_x/model",
            headers={"Authorization": f"Bearer {key}"},
        )
        assert with_key.status_code == 404


def test_the_prediction_log_records_counts_and_never_values(
    predict_client: TestClient,
    classification_run: dict,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Submitted records are somebody's data, exactly like an uploaded file."""
    experiment_id = classification_run["experiment_id"]
    record = {**a_record(predict_client, experiment_id), "income": 987654}

    with caplog.at_level(logging.INFO):
        predict_client.post(
            f"/api/v1/experiments/{experiment_id}/predict", json={"records": [record]}
        )

    assert "Predicted 1 record(s)" in caplog.text
    assert "987654" not in caplog.text


def test_the_openapi_schema_documents_the_prediction_contract(
    predict_client: TestClient,
) -> None:
    """Both routes, their statuses, and the security they require."""
    schema = predict_client.get("/openapi.json").json()

    predict = schema["paths"]["/api/v1/experiments/{experiment_id}/predict"]["post"]
    model = schema["paths"]["/api/v1/experiments/{experiment_id}/model"]["get"]

    assert predict["security"] and model["security"]
    assert {"200", "401", "404", "409", "422", "500"} <= set(predict["responses"])
    # And nothing in the schema hints that a path could be supplied.
    rendered = json.dumps(schema)
    assert "model_path" not in rendered
    assert "joblib" not in rendered
