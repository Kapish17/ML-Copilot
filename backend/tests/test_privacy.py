"""Proof that a caller's data does not survive their request.

The method is the same throughout: put a string in the data that could not
occur anywhere else — ``ZZQPRIVATEMARKER7419`` — run a real request end to
end, and then look for that string in every place the project promises it will
not be. A test that searched for "the value 42" would pass by luck; this one
can only pass if the value is genuinely absent.

Three documented exceptions are asserted *as* exceptions rather than quietly
tolerated, because each is a deliberate trade and a reader deserves to see it
named:

* **Column names.** A dataset's schema is what the profile, the record and the
  prediction contract are all about. Names travel; cells do not.
* **Class labels.** The values of the target column are the model's output
  vocabulary — you cannot read a prediction without them. They appear in the
  model manifest and in prediction responses by design.
* **One-hot feature names.** ``segment_business`` embeds a category value,
  because a feature importance the reader cannot map back to a column is
  useless. This is the widest of the exceptions and the one worth stating
  precisely: for a column that *became a feature*, up to
  ``max_categorical_cardinality`` (50) of its distinct values appear in the
  record as feature names. A column with more distinct values than the cap is
  excluded from the feature set entirely, so nothing of it appears at all —
  which means the leak is bounded at fifty values per encoded column and never
  extends to a free-text or identifier column. ``test_the_encoded_feature_names_
  are_bounded_by_the_cardinality_cap`` holds that bound.
* **The uploaded filename**, when the caller does not name the run. It becomes
  the run's default label — ``customers.csv · renewed`` — which is what makes a
  history readable. It is metadata the caller typed, not data from inside the
  file, and a caller who does not want it there passes ``name``.

Everything else — cells of feature columns, the distribution of the target,
the file's name, the request's records — is checked to be gone.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app

#: A string that cannot occur by accident anywhere in this repository.
MARKER = "ZZQPRIVATEMARKER7419"
#: A second one, for the value of a *target* class, whose visibility rules
#: differ from an ordinary cell's.
CLASS_MARKER = "ZZQCLASSMARKER8520"


def _strings_containing(node: Any, needle: str, path: str = ""):
    """Every ``(path, value)`` in a payload whose string holds ``needle``.

    A path-carrying walk rather than a substring check on the whole JSON, so a
    test can say *where* a value survived and not merely *that* it did.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _strings_containing(value, needle, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _strings_containing(value, needle, f"{path}[{index}]")
    elif isinstance(node, str) and needle in node:
        yield path, node


def marked_csv(rows: int = 80) -> bytes:
    """A learnable dataset whose free-text column carries the marker.

    The marker is a cell value in a feature column, which is the category the
    project promises never to store, log, index or return.
    """
    lines = ["note,measure,score,outcome\n"]
    for index in range(rows):
        positive = index % 2 == 0
        note = f"{MARKER}-row-{index}"
        measure = 10 + index if positive else 100 + index
        score = 1.5 if positive else 9.5
        outcome = CLASS_MARKER if positive else "other"
        lines.append(f"{note},{measure},{score},{outcome}\n")
    return "".join(lines).encode("utf-8")


@pytest.fixture(scope="module")
def private_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An isolated store, so nothing here touches the repository's own."""
    return tmp_path_factory.mktemp("privacy-store")


@pytest.fixture(scope="module")
def private_client(private_store: Path) -> TestClient:
    """A client whose records and artifacts are written under the temp store."""
    settings = Settings(
        experiment_store_dir=private_store / "runs",
        model_artifact_dir=private_store / "models",
    )
    return TestClient(create_app(settings))


@pytest.fixture(scope="module")
def marked_run(private_client: TestClient) -> dict[str, Any]:
    """One complete experiment on the marked dataset."""
    response = private_client.post(
        "/api/v1/experiments/run",
        files={"file": ("dataset.csv", marked_csv(), "text/csv")},
        data={"target_column": "outcome", "models": "logistic_regression"},
    )
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# The stored record
# ---------------------------------------------------------------------------


def test_no_cell_of_the_dataset_reaches_the_experiment_record(
    marked_run: dict[str, Any]
) -> None:
    """The record holds the schema and the scores, never the data.

    This is the promise the whole experiment layer rests on: history is
    metadata, so a run can be kept indefinitely without keeping the data it
    ran on.
    """
    assert MARKER not in json.dumps(marked_run)


def test_no_cell_reaches_the_stored_file_on_disk(
    marked_run: dict[str, Any], private_store: Path
) -> None:
    """Checked on disk as well as in the response, because they can differ."""
    written = list((private_store / "runs").rglob("*.json"))

    assert written, "the run should have been stored"
    for path in written:
        assert MARKER not in path.read_text(encoding="utf-8")


def test_the_uploaded_filename_becomes_the_run_label_and_nothing_else(
    private_client: TestClient,
) -> None:
    """A documented exception, and the limit of it.

    The filename is what makes a history readable, so an unnamed run is
    labelled after the file it ran on. That is metadata the caller typed
    rather than data from inside the file — but it *is* retained and indexed,
    and filenames carry client names and case numbers more often than anyone
    intends, so the behaviour is asserted here and stated in the API
    documentation. A caller who does not want it passes ``name``.
    """
    labelled = private_client.post(
        "/api/v1/experiments/run",
        files={"file": (f"{MARKER}-file.csv", marked_csv(), "text/csv")},
        data={"target_column": "outcome", "models": "logistic_regression"},
    ).json()

    assert MARKER in labelled["name"]
    # Everywhere else, it is gone: not the fingerprint, not the columns, not
    # the preprocessing decisions.
    without_name = dict(labelled)
    without_name.pop("name")
    assert MARKER not in json.dumps(without_name)

    named = private_client.post(
        "/api/v1/experiments/run",
        files={"file": (f"{MARKER}-file.csv", marked_csv(), "text/csv")},
        data={
            "target_column": "outcome",
            "models": "logistic_regression",
            "name": "quarterly refresh",
        },
    ).json()

    assert named["name"] == "quarterly refresh"
    assert MARKER not in json.dumps(named)


def test_the_encoded_feature_names_are_bounded_by_the_cardinality_cap(
    private_client: TestClient,
) -> None:
    """The widest exception, and the bound that makes it acceptable.

    A categorical column that becomes a feature is one-hot encoded, and each
    of its values becomes a feature name — which is what makes an importance
    chart readable and is also the one route by which a cell value enters a
    record. The bound is the cardinality cap: a column with more distinct
    values than ``max_categorical_cardinality`` is not encoded at all, so a
    free-text or identifier column contributes nothing.

    Both sides are asserted here, on the same column, either side of the cap.
    """
    def dataset(distinct: int, rows: int) -> bytes:
        """Rows whose ``note`` column has exactly ``distinct`` values."""
        lines = ["note,measure,outcome\n"]
        for index in range(rows):
            positive = index % 2 == 0
            lines.append(
                f"{MARKER}-{index % distinct},"
                f"{10 + index if positive else 100 + index},"
                f"{'yes' if positive else 'no'}\n"
            )
        return "".join(lines).encode("utf-8")

    under = private_client.post(
        "/api/v1/experiments/run",
        files={"file": ("under.csv", dataset(distinct=6, rows=80), "text/csv")},
        data={"target_column": "outcome", "models": "logistic_regression"},
    ).json()
    encoded = [
        name
        for name in under["preprocessing"]["transformed_feature_names"]
        if MARKER in name
    ]

    assert encoded, "a categorical feature is encoded by value, by design"
    assert len(encoded) <= 50, "and never more than the cardinality cap"

    # And *only* in the two places a feature name belongs. This is the part
    # worth pinning: a third field carrying a cell value would be a new leak
    # wearing the same clothes as an old exception.
    carrying = {
        path.split("[")[0] for path, _ in _strings_containing(under, MARKER)
    }
    assert carrying == {
        ".preprocessing.transformed_feature_names",
        ".explainability.feature_importances",
    }, carrying

    over = private_client.post(
        "/api/v1/experiments/run",
        files={"file": ("over.csv", dataset(distinct=60, rows=120), "text/csv")},
        data={"target_column": "outcome", "models": "logistic_regression"},
    ).json()

    assert MARKER not in json.dumps(over), (
        "a column above the cap is excluded from the feature set, so none of "
        "its values reach the record"
    )
    assert "note" in over["preprocessing"]["excluded_columns"]


def test_the_record_keeps_the_schema_it_is_supposed_to_keep(
    marked_run: dict[str, Any]
) -> None:
    """The counterpart to the tests above: absence is not the same as broken."""
    dataset = marked_run["dataset"]

    assert set(dataset["columns"]) == {"note", "measure", "score", "outcome"}
    assert dataset["target_column"] == "outcome"
    assert dataset["row_count"] == 80


def test_the_target_labels_are_kept_and_that_is_deliberate(
    marked_run: dict[str, Any]
) -> None:
    """A documented exception, asserted so it stays a decision.

    A prediction is a class label. Hiding the vocabulary would leave every
    prediction uninterpretable, so labels travel — and this test is here so
    that if the rule ever changes, it changes on purpose.
    """
    details = marked_run["evaluation"]["classification_details"]

    assert CLASS_MARKER in [str(label) for label in details["class_labels"]]


# ---------------------------------------------------------------------------
# The retrieval index
# ---------------------------------------------------------------------------


def test_no_cell_reaches_the_document_a_run_is_indexed_as(
    marked_run: dict[str, Any], private_store: Path
) -> None:
    """A record becomes a retrievable document; the document is the record.

    Worth its own test because the rendering step is where a value could be
    reintroduced — an error message, a candidate's failure, a warning quoting
    something the estimator said.
    """
    from ml.experiments import LocalExperimentStore
    from rag.ingestion.experiments import render_experiment

    store = LocalExperimentStore(private_store / "runs")
    rendered = render_experiment(store.get(marked_run["experiment_id"]))

    assert MARKER not in rendered
    # And it is a real document, not an empty one.
    assert marked_run["experiment_id"] in rendered


# ---------------------------------------------------------------------------
# The logs
# ---------------------------------------------------------------------------


def test_no_cell_reaches_the_logs(
    private_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Counts and durations are logged; content is not.

    The log is the place where "just this once, for debugging" is most
    tempting and least visible.
    """
    with caplog.at_level(logging.DEBUG):
        response = private_client.post(
            "/api/v1/datasets/profile",
            files={"file": (f"{MARKER}.csv", marked_csv(rows=20), "text/csv")},
        )

    assert response.status_code == 200
    assert MARKER not in caplog.text


def test_a_failing_request_does_not_log_the_data_that_failed(
    private_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The tempting case: something went wrong, so surely we log the input.

    A dataset whose target has one class per row cannot be split for
    cross-validation. The refusal must say so without quoting the column.
    """
    rows = "\n".join(f"{MARKER}-{index},{index}" for index in range(6))
    content = f"label,measure\n{rows}\n".encode("utf-8")

    with caplog.at_level(logging.DEBUG):
        response = private_client.post(
            "/api/v1/experiments/run",
            files={"file": ("tiny.csv", content, "text/csv")},
            data={"target_column": "label"},
        )

    assert response.status_code >= 400
    assert MARKER not in caplog.text
    assert MARKER not in response.text


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_a_rare_class_refusal_does_not_report_the_class_distribution(
    private_client: TestClient,
) -> None:
    """This error used to return every distinct value of the target column.

    Not a summary of it — the whole distribution, label by label, in the
    ``details`` of a 4xx. For a target that is a customer segment or a
    diagnosis, that is the column, returned to whoever asked for five folds.
    """
    lines = ["outcome,measure\n"]
    for index in range(40):
        label = f"{MARKER}-rare" if index == 0 else "common"
        lines.append(f"{label},{index}\n")
    content = "".join(lines).encode("utf-8")

    response = private_client.post(
        "/api/v1/experiments/run",
        files={"file": ("rare.csv", content, "text/csv")},
        data={"target_column": "outcome", "folds": "5"},
    )

    assert response.status_code >= 400
    assert MARKER not in response.text
    details = response.json()["error"].get("details", {})
    assert "class_counts" not in json.dumps(details)


def test_a_prediction_request_is_not_persisted_anywhere(
    private_client: TestClient, marked_run: dict[str, Any], private_store: Path
) -> None:
    """A prediction is answered and forgotten.

    There is no prediction history in this project, which is a limitation
    documented as one — and the other half of that limitation is that a
    submitted record is not lying somewhere unnamed either.
    """
    experiment_id = marked_run["experiment_id"]
    submitted = f"{MARKER}-at-prediction-time"

    # Built from the model's own declared schema, so this test is about
    # persistence rather than about which columns became features.
    model = private_client.get(f"/api/v1/experiments/{experiment_id}/model").json()
    record: dict[str, Any] = {}
    for feature in model["features"]:
        name = feature["name"]
        record[name] = submitted if feature["kind"] == "categorical" else 12

    response = private_client.post(
        f"/api/v1/experiments/{experiment_id}/predict",
        json={"records": [record]},
    )

    assert response.status_code == 200, response.text
    assert submitted not in response.text

    for path in private_store.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".log", ".txt"}:
            assert submitted not in path.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_the_api_key_never_appears_in_a_response_or_a_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Asserted here as well as in the authentication suite.

    Not duplication: that suite proves the key is checked correctly, this one
    proves it does not travel. They fail for different reasons.
    """
    key = f"{MARKER}-secret-key-long-enough-to-be-valid"
    settings = Settings(api_auth_enabled=True, api_auth_key=key)
    client = TestClient(create_app(settings))

    with caplog.at_level(logging.DEBUG):
        unauthorised = client.get("/api/v1/experiments")
        wrong = client.get(
            "/api/v1/experiments", headers={"Authorization": "Bearer wrong-key"}
        )
        authorised = client.get(
            "/api/v1/experiments", headers={"Authorization": f"Bearer {key}"}
        )

    assert unauthorised.status_code == 401
    assert wrong.status_code == 401
    assert authorised.status_code == 200
    for response in (unauthorised, wrong, authorised):
        assert key not in response.text
    assert key not in caplog.text
    assert client.get("/openapi.json").text.count(key) == 0
