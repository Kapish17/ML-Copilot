"""Tests for the experiment endpoints.

The expensive part of this suite is one classification run and one regression
run, so both are module-scoped fixtures and every test that only needs to look
at a completed result reuses them.

The tests at the end are the ones that matter most: the end-to-end flow from
upload to stored record, and the response-contract checks that no sklearn
object, numpy array, DataFrame, SHAP explainer, traceback or filesystem path
can reach a client.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.factories import (
    build_csv,
    experiment_form,
    high_cardinality_csv,
    learnable_classification_csv,
    regression_csv,
    upload_payload,
)

RUN_URL = "/api/v1/experiments/run"
LIST_URL = "/api/v1/experiments"
COMPARE_URL = "/api/v1/experiments/compare"

CLASSIFIERS = ["logistic_regression", "random_forest_classifier"]
FOLDS = 3


def run_experiment(
    client: TestClient,
    content: bytes | None = None,
    *,
    filename: str = "customers.csv",
    **options: Any,
):
    """POST one experiment, returning the raw response."""
    payload = content if content is not None else learnable_classification_csv()
    return client.post(
        RUN_URL,
        files=upload_payload(payload, filename),
        data=experiment_form(**options),
    )


# --------------------------------------------------------------------------
# Completed runs, reused across the suite
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def classification_run(experiment_client: TestClient) -> dict[str, Any]:
    """One completed classification experiment."""
    response = run_experiment(
        experiment_client,
        target_column="renewed",
        models=CLASSIFIERS,
        folds=FOLDS,
        name="renewal baseline",
        description="the first tracked run",
        tags=["baseline", "api"],
    )
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def regression_run(experiment_client: TestClient) -> dict[str, Any]:
    """One completed regression experiment."""
    response = run_experiment(
        experiment_client,
        regression_csv(),
        filename="housing.csv",
        target_column="price",
        models=["linear_regression"],
        folds=FOLDS,
        name="price baseline",
    )
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------
# Existing endpoints are unchanged
# --------------------------------------------------------------------------


def test_the_service_root_still_answers(experiment_client: TestClient) -> None:
    """Commit 1's root endpoint is untouched by the new routes."""
    payload = experiment_client.get("/").json()

    assert payload["name"]
    assert payload["docs_url"] == "/docs"


def test_the_health_endpoint_still_answers(experiment_client: TestClient) -> None:
    """So is the liveness check."""
    assert experiment_client.get("/health").json()["status"] == "ok"


def test_the_profile_endpoint_still_answers(experiment_client: TestClient) -> None:
    """Commit 2's profiling endpoint keeps its contract after the refactor.

    Loading and validation now live behind methods the experiment runner also
    uses; this asserts that sharing them changed nothing observable.
    """
    response = experiment_client.post(
        "/api/v1/datasets/profile",
        files=upload_payload(learnable_classification_csv(), "customers.csv"),
        data={"target_column": "renewed"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "customers.csv"
    assert payload["dataset"]["row_count"] == 240
    assert payload["target"]["name"] == "renewed"
    assert payload["target"]["task_suggestion"] == "classification"


# --------------------------------------------------------------------------
# Running experiments
# --------------------------------------------------------------------------


def test_a_classification_experiment_reports_what_it_did(
    classification_run: dict[str, Any]
) -> None:
    """Identity, dataset, selection and final evaluation all come back."""
    assert classification_run["experiment_id"].startswith("exp_")
    assert classification_run["schema_version"] == "1.0"
    assert classification_run["name"] == "renewal baseline"
    assert classification_run["tags"] == ["baseline", "api"]
    assert classification_run["dataset"]["task_type"] == "classification"
    assert classification_run["dataset"]["target_column"] == "renewed"
    assert classification_run["dataset"]["row_count"] == 240
    assert classification_run["dataset"]["source_format"] == "csv"
    assert classification_run["selection"]["selected_model"] in CLASSIFIERS
    assert classification_run["evaluation"]["primary_metric_value"] > 0.5


def test_a_regression_experiment_uses_a_regression_metric(
    regression_run: dict[str, Any]
) -> None:
    """The task decides the metric and its direction, not the caller."""
    assert regression_run["dataset"]["task_type"] == "regression"
    assert regression_run["selection"]["selected_model"] == "linear_regression"
    assert regression_run["selection"]["primary_metric_direction"] == "lower_is_better"
    assert regression_run["evaluation"]["primary_metric_value"] >= 0


def test_cross_validation_never_reads_the_test_set(
    classification_run: dict[str, Any]
) -> None:
    """The default strategy keeps the final measurement unbiased."""
    selection = classification_run["selection"]

    assert selection["strategy"] == "cross_validation"
    assert selection["folds"] == FOLDS
    assert selection["scored_on"] == "training_folds"
    assert selection["uses_test_data"] is False
    assert classification_run["evaluation"]["is_unbiased"] is True


def test_the_holdout_strategy_declares_itself_optimistic(
    experiment_client: TestClient,
) -> None:
    """Choosing on the test set is allowed, but never reported as unbiased."""
    payload = run_experiment(
        experiment_client,
        target_column="renewed",
        models=CLASSIFIERS,
        strategy="holdout",
        name="holdout run",
    ).json()

    assert payload["selection"]["uses_test_data"] is True
    assert payload["evaluation"]["is_unbiased"] is False


def test_the_result_is_measured_against_a_baseline(
    classification_run: dict[str, Any]
) -> None:
    """A score means nothing without something naive to beat."""
    evaluation = classification_run["evaluation"]

    assert evaluation["baseline_identifier"]
    assert evaluation["baseline_metrics"]
    assert "improvement" in json.dumps(evaluation["baseline_comparison"])


def test_the_winner_is_explained(classification_run: dict[str, Any]) -> None:
    """SHAP importances are named features, ranked, not placeholders."""
    explainability = classification_run["explainability"]
    features = [item["feature"] for item in explainability["feature_importances"]]

    assert explainability["status"] == "available"
    assert explainability["method"] == "shap"
    assert features, "expected ranked importances"
    assert not any(re.fullmatch(r"x\d+", name) for name in features)
    assert "income" in features and "tenure_months" in features


def test_explanation_can_be_skipped(experiment_client: TestClient) -> None:
    """Opting out is recorded as a warning, not silently ignored."""
    payload = run_experiment(
        experiment_client,
        target_column="renewed",
        models=["logistic_regression"],
        folds=FOLDS,
        explain=False,
        name="unexplained run",
    ).json()

    assert payload["explainability"] is None
    assert any("skipped" in warning for warning in payload["warnings"])


def test_an_omitted_target_falls_back_to_the_last_column(
    experiment_client: TestClient,
) -> None:
    """The convention is applied and reported, never applied silently.

    Automatic target detection is not implemented; the last column is a naming
    convention, and the response says so.
    """
    payload = run_experiment(
        experiment_client, models=["logistic_regression"], folds=FOLDS
    ).json()

    assert payload["dataset"]["target_column"] == "renewed"
    assert any("last column" in warning for warning in payload["warnings"])
    assert any("not implemented" in warning for warning in payload["warnings"])


def test_an_omitted_model_list_uses_every_model_for_the_task(
    experiment_client: TestClient,
) -> None:
    """The default candidate set follows the detected task."""
    payload = run_experiment(
        experiment_client,
        target_column="renewed",
        folds=FOLDS,
        name="all classifiers",
    ).json()
    candidates = payload["selection"]["candidate_models"]

    assert len(candidates) >= 3
    assert all("regress" not in name or "logistic" in name for name in candidates)


def test_preprocessing_overrides_reach_the_pipeline(
    experiment_client: TestClient,
) -> None:
    """An explicit choice wins over anything inferred from the profile."""
    payload = run_experiment(
        experiment_client,
        target_column="renewed",
        models=["logistic_regression"],
        folds=FOLDS,
        scaling_strategy="none",
        test_size=0.3,
        random_state=7,
        name="override run",
    ).json()
    preprocessing = payload["preprocessing"]

    assert preprocessing["config"]["scaling_strategy"] == "none"
    assert preprocessing["test_size"] == 0.3
    assert preprocessing["random_state"] == 7
    assert payload["environment"]["random_state"] == 7


def test_excluded_columns_are_kept_out_of_the_features(
    experiment_client: TestClient,
) -> None:
    """A column the caller excludes is recorded as excluded, with a reason."""
    payload = run_experiment(
        experiment_client,
        target_column="renewed",
        models=["logistic_regression"],
        folds=FOLDS,
        excluded_columns=["segment"],
        name="no segment",
    ).json()
    preprocessing = payload["preprocessing"]

    assert "segment" in preprocessing["excluded_columns"]
    assert "segment" not in preprocessing["selected_columns"]
    assert not any(
        name.startswith("segment") for name in preprocessing["transformed_feature_names"]
    )


# --------------------------------------------------------------------------
# Invalid requests
# --------------------------------------------------------------------------


def assert_envelope(response, *, status_code: int, code: str) -> dict[str, Any]:
    """Assert a failure uses the one documented error envelope."""
    assert response.status_code == status_code, response.text
    payload = response.json()
    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message", "details"}
    assert payload["error"]["code"] == code
    assert payload["error"]["message"]
    return payload["error"]


def test_an_unknown_model_is_rejected(experiment_client: TestClient) -> None:
    """The error names the model and lists what is available."""
    error = assert_envelope(
        run_experiment(
            experiment_client, target_column="renewed", models=["magic_forest"]
        ),
        status_code=400,
        code="unknown_model",
    )

    assert error["details"]["model_name"] == "magic_forest"
    assert error["details"]["available_models"]


def test_a_model_for_the_wrong_task_is_rejected(experiment_client: TestClient) -> None:
    """A regressor on a classification target is a request error, not a skip.

    Model comparison tolerates a candidate that fails; asking for a model that
    cannot possibly solve the problem is a different thing, and is answered.
    """
    error = assert_envelope(
        run_experiment(
            experiment_client, target_column="renewed", models=["linear_regression"]
        ),
        status_code=409,
        code="incompatible_model_task",
    )

    assert error["details"]["requested_task_type"] == "classification"
    assert error["details"]["compatible_models"]


def test_a_classifier_for_a_regression_target_is_rejected(
    experiment_client: TestClient,
) -> None:
    """And the same holds in the other direction."""
    assert_envelope(
        run_experiment(
            experiment_client,
            regression_csv(),
            filename="housing.csv",
            target_column="price",
            models=["logistic_regression"],
        ),
        status_code=409,
        code="incompatible_model_task",
    )


def test_a_metric_that_does_not_suit_the_task_is_rejected(
    experiment_client: TestClient,
) -> None:
    """RMSE is not a classification metric, and the error says which are."""
    error = assert_envelope(
        run_experiment(
            experiment_client, target_column="renewed", primary_metric="rmse"
        ),
        status_code=400,
        code="invalid_metric",
    )

    assert error["details"]["task_type"] == "classification"
    assert "f1" in error["details"]["available"]


@pytest.mark.parametrize("folds", [1, 99])
def test_an_unusable_fold_count_is_rejected(
    experiment_client: TestClient, folds: int
) -> None:
    """Below two folds is meaningless; far above the limit is unbounded work."""
    response = run_experiment(
        experiment_client, target_column="renewed", folds=folds
    )

    assert response.status_code in (400, 422)
    assert set(response.json()) == {"error"}


def test_an_unknown_strategy_is_rejected(experiment_client: TestClient) -> None:
    """The error lists the strategies that do exist."""
    error = assert_envelope(
        run_experiment(
            experiment_client, target_column="renewed", strategy="bootstrap"
        ),
        status_code=400,
        code="invalid_configuration",
    )

    assert "cross_validation" in error["details"]["available_strategies"]


def test_a_target_that_is_not_in_the_dataset_is_rejected(
    experiment_client: TestClient,
) -> None:
    """The error names the columns that are there.

    Profiling reaches the missing column first, so this is the dataset
    service's error rather than the ML layer's — but it arrives in the same
    envelope with the same status, which is the point of one error contract.
    """
    error = assert_envelope(
        run_experiment(experiment_client, target_column="churn"),
        status_code=422,
        code="target_column_not_found",
    )

    assert "renewed" in error["details"]["available_columns"]


def test_a_dataset_with_no_usable_features_is_rejected(
    experiment_client: TestClient,
) -> None:
    """Excluding everything leaves nothing to learn from, and says so."""
    assert_envelope(
        run_experiment(
            experiment_client,
            target_column="renewed",
            excluded_columns=["income", "tenure_months", "segment"],
        ),
        status_code=422,
        code="empty_feature_set",
    )


def test_a_malformed_csv_is_rejected(experiment_client: TestClient) -> None:
    """A broken file fails at ingestion, before any model is considered."""
    response = run_experiment(experiment_client, b'a,b\n"unclosed,1\n2,3,4,5\n')

    assert response.status_code in (400, 422)
    assert set(response.json()) == {"error"}


def test_an_unsupported_file_type_is_rejected(experiment_client: TestClient) -> None:
    """CSV, Excel and JSON are implemented; the error says which.

    Parquet is the honest example of a format that is *not* implemented: it
    is named in the roadmap, so a client could reasonably try it, and the
    refusal has to be unambiguous rather than a parse failure later on.
    """
    error = assert_envelope(
        run_experiment(experiment_client, filename="data.parquet"),
        status_code=415,
        code="unsupported_file_type",
    )

    assert error["details"]["supported_extensions"] == [".csv", ".xlsx", ".json"]


def test_an_empty_file_is_rejected(experiment_client: TestClient) -> None:
    """Nothing to parse is a client error with a plain message."""
    assert_envelope(
        run_experiment(experiment_client, b""), status_code=400, code="empty_file"
    )


def test_an_oversized_upload_is_rejected(experiment_store_dir: Path) -> None:
    """The upload limit is enforced before the bytes are fully buffered."""
    from app.core.config import Settings
    from app.main import create_app

    tiny = Settings(max_upload_bytes=512, experiment_store_dir=experiment_store_dir)
    with TestClient(create_app(tiny)) as client:
        error = assert_envelope(
            run_experiment(client, target_column="renewed"),
            status_code=413,
            code="file_too_large",
        )

    assert error["details"]["max_upload_bytes"] == 512


def test_a_dataset_beyond_the_experiment_row_limit_is_rejected(
    experiment_store_dir: Path,
) -> None:
    """Experiment execution has its own, tighter size limit than profiling."""
    from app.core.config import Settings
    from app.main import create_app

    small = Settings(max_experiment_rows=10, experiment_store_dir=experiment_store_dir)
    with TestClient(create_app(small)) as client:
        error = assert_envelope(
            run_experiment(client, target_column="renewed"),
            status_code=413,
            code="dataset_too_large",
        )

    assert error["details"]["max_rows"] == 10


def test_too_few_rows_to_split_is_rejected(experiment_client: TestClient) -> None:
    """A handful of rows cannot be split into train and test halves."""
    tiny = build_csv(["a", "b"], [[1, "yes"], [2, "no"], [3, "yes"]])
    response = run_experiment(experiment_client, tiny, target_column="b")

    assert response.status_code in (409, 422)
    assert set(response.json()) == {"error"}


def test_a_field_of_the_wrong_type_is_rejected(experiment_client: TestClient) -> None:
    """A fold count that is not a number fails before anything is run."""
    response = experiment_client.post(
        RUN_URL,
        files=upload_payload(learnable_classification_csv()),
        data={"target_column": "renewed", "folds": "several"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_a_missing_file_is_rejected(experiment_client: TestClient) -> None:
    """The dataset is required; configuration alone is not a request."""
    response = experiment_client.post(RUN_URL, data={"target_column": "renewed"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


# --------------------------------------------------------------------------
# Persistence and history
# --------------------------------------------------------------------------


def test_a_run_is_persisted_as_one_readable_record(
    classification_run: dict[str, Any], experiment_store_dir: Path
) -> None:
    """The record lands on disk as JSON under the experiment's own id."""
    path = (
        experiment_store_dir
        / classification_run["experiment_id"]
        / "experiment.json"
    )

    assert path.is_file()
    assert classification_run["execution"]["stored"] is True
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["experiment_id"] == classification_run["experiment_id"]


def test_a_stored_run_is_fetchable_by_id(
    experiment_client: TestClient, classification_run: dict[str, Any]
) -> None:
    """What comes back is the record the run returned."""
    experiment_id = classification_run["experiment_id"]
    payload = experiment_client.get(f"{LIST_URL}/{experiment_id}").json()

    assert payload["experiment_id"] == experiment_id
    assert payload["dataset"]["fingerprint"] == classification_run["dataset"]["fingerprint"]
    assert payload["selection"] == classification_run["selection"]
    assert payload["evaluation"] == classification_run["evaluation"]
    assert "execution" not in payload


def test_an_unknown_experiment_is_a_clean_404(experiment_client: TestClient) -> None:
    """A well-formed id that is not stored is not found, not a crash."""
    error = assert_envelope(
        experiment_client.get(f"{LIST_URL}/exp_missing_20260101T000000Z_0000"),
        status_code=404,
        code="experiment_not_found",
    )

    assert "root" not in error["details"], "the store directory must not be exposed"


def test_a_malformed_experiment_id_is_refused(experiment_client: TestClient) -> None:
    """An id that could climb out of the store never reaches the filesystem."""
    response = experiment_client.get(f"{LIST_URL}/..%2F..%2Fetc%2Fpasswd")

    assert response.status_code in (400, 404)
    assert set(response.json()) == {"error"}


def test_the_history_lists_stored_runs(
    experiment_client: TestClient,
    classification_run: dict[str, Any],
    regression_run: dict[str, Any],
) -> None:
    """Both completed runs appear, newest first, as one-line summaries."""
    payload = experiment_client.get(LIST_URL).json()
    identifiers = [item["experiment_id"] for item in payload["experiments"]]

    assert classification_run["experiment_id"] in identifiers
    assert regression_run["experiment_id"] in identifiers
    assert payload["count"] == len(payload["experiments"])
    assert set(payload["experiments"][0]) >= {
        "experiment_id",
        "selected_model",
        "primary_metric",
        "test_score",
    }


def test_history_filters_by_task(
    experiment_client: TestClient, regression_run: dict[str, Any]
) -> None:
    """Filtering narrows the listing rather than reordering it."""
    payload = experiment_client.get(LIST_URL, params={"task_type": "regression"}).json()

    assert payload["count"] >= 1
    assert all(
        item["experiment_id"] == regression_run["experiment_id"]
        for item in payload["experiments"]
    )


def test_history_filters_by_dataset_fingerprint(
    experiment_client: TestClient, classification_run: dict[str, Any]
) -> None:
    """Runs are found by what the data *was*, not by what the file was named."""
    fingerprint = classification_run["dataset"]["fingerprint"]
    payload = experiment_client.get(
        LIST_URL, params={"dataset_fingerprint": fingerprint}
    ).json()

    assert payload["count"] >= 1
    assert all(
        item["dataset_fingerprint"] == fingerprint for item in payload["experiments"]
    )


def test_history_filters_by_tag_and_model(
    experiment_client: TestClient, classification_run: dict[str, Any]
) -> None:
    """Tags and the winning model are both queryable."""
    tagged = experiment_client.get(LIST_URL, params={"tags": ["baseline", "api"]}).json()

    assert classification_run["experiment_id"] in [
        item["experiment_id"] for item in tagged["experiments"]
    ]

    by_model = experiment_client.get(
        LIST_URL, params={"model_name": "linear_regression"}
    ).json()
    assert all(
        item["selected_model"] == "linear_regression"
        for item in by_model["experiments"]
    )


def test_history_sorts_by_score_in_the_metrics_own_direction(
    experiment_client: TestClient,
) -> None:
    """Best first means the largest F1 — the metric decides, not the sort."""
    payload = experiment_client.get(
        LIST_URL,
        params={
            "task_type": "classification",
            "sort_by": "primary_metric",
            "order": "desc",
        },
    ).json()
    scores = [
        item["test_score"]
        for item in payload["experiments"]
        if item["test_score"] is not None
    ]

    assert scores == sorted(scores, reverse=True)


def test_the_history_limit_is_honoured(experiment_client: TestClient) -> None:
    """A page size caps the listing."""
    payload = experiment_client.get(LIST_URL, params={"limit": 1}).json()

    assert payload["count"] == 1
    assert payload["limit"] == 1


@pytest.mark.parametrize(
    ("params", "code"),
    [
        ({"sort_by": "luck"}, "invalid_configuration"),
        ({"order": "sideways"}, "invalid_configuration"),
        ({"limit": 0}, "invalid_configuration"),
    ],
)
def test_invalid_history_queries_are_rejected(
    experiment_client: TestClient, params: dict[str, Any], code: str
) -> None:
    """A bad query parameter is a 400 in the standard envelope."""
    assert_envelope(
        experiment_client.get(LIST_URL, params=params), status_code=400, code=code
    )


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def test_two_classification_runs_can_be_compared(
    experiment_client: TestClient, classification_run: dict[str, Any]
) -> None:
    """Runs sharing a task and metric are ranked, best first."""
    second = run_experiment(
        experiment_client,
        target_column="renewed",
        models=["random_forest_classifier"],
        folds=FOLDS,
        name="forest only",
    ).json()

    payload = experiment_client.post(
        COMPARE_URL,
        json={
            "experiment_ids": [
                classification_run["experiment_id"],
                second["experiment_id"],
            ]
        },
    ).json()

    assert payload["task_type"] == "classification"
    assert payload["primary_metric"] == "f1"
    assert payload["higher_is_better"] is True
    assert payload["run_count"] == 2
    assert payload["best_experiment_id"] in {
        classification_run["experiment_id"],
        second["experiment_id"],
    }
    scores = [row["test_score"] for row in payload["runs"]]
    assert scores == sorted(scores, reverse=True)
    assert "Test F1" in payload["table"]


def test_runs_judged_by_different_metrics_are_not_ranked(
    experiment_client: TestClient,
    classification_run: dict[str, Any],
    regression_run: dict[str, Any],
) -> None:
    """An RMSE never appears in the same ranking as an F1."""
    error = assert_envelope(
        experiment_client.post(
            COMPARE_URL,
            json={
                "experiment_ids": [
                    classification_run["experiment_id"],
                    regression_run["experiment_id"],
                ]
            },
        ),
        status_code=409,
        code="incomparable_experiments",
    )

    assert sorted(error["details"]["primary_metrics"]) == ["f1", "rmse"]


def test_comparing_a_missing_experiment_is_a_404(
    experiment_client: TestClient, classification_run: dict[str, Any]
) -> None:
    """One unknown id fails the whole comparison, cleanly."""
    assert_envelope(
        experiment_client.post(
            COMPARE_URL,
            json={
                "experiment_ids": [
                    classification_run["experiment_id"],
                    "exp_absent_20260101T000000Z_0000",
                ]
            },
        ),
        status_code=404,
        code="experiment_not_found",
    )


def test_comparing_fewer_than_two_experiments_is_rejected(
    experiment_client: TestClient, classification_run: dict[str, Any]
) -> None:
    """A comparison of one thing is not a comparison."""
    response = experiment_client.post(
        COMPARE_URL, json={"experiment_ids": [classification_run["experiment_id"]]}
    )

    assert response.status_code == 422
    assert set(response.json()) == {"error"}


def test_comparing_one_id_twice_is_rejected(
    experiment_client: TestClient, classification_run: dict[str, Any]
) -> None:
    """Duplicates collapse, so the request still names only one run."""
    experiment_id = classification_run["experiment_id"]
    assert_envelope(
        experiment_client.post(
            COMPARE_URL, json={"experiment_ids": [experiment_id, experiment_id]}
        ),
        status_code=400,
        code="invalid_configuration",
    )


# --------------------------------------------------------------------------
# Capabilities and documentation
# --------------------------------------------------------------------------


def test_capabilities_describe_what_a_request_may_contain(
    experiment_client: TestClient,
) -> None:
    """A client can read the models, metrics and limits off the API."""
    payload = experiment_client.get(f"{LIST_URL}/capabilities").json()
    identifiers = [item["identifier"] for item in payload["models"]]

    assert "logistic_regression" in identifiers
    assert "f1" in payload["metrics"]["classification"]
    assert "rmse" in payload["metrics"]["regression"]
    assert payload["strategies"] == ["cross_validation", "holdout"]
    assert payload["supported_dataset_extensions"] == [".csv", ".xlsx", ".json"]
    assert payload["limits"]["max_cv_folds"] >= 2


def test_the_openapi_schema_documents_the_new_endpoints(
    experiment_client: TestClient,
) -> None:
    """Every endpoint, its form fields and its error responses are described."""
    spec = experiment_client.get("/openapi.json").json()
    paths = spec["paths"]

    assert RUN_URL in paths and LIST_URL in paths and COMPARE_URL in paths
    assert f"{LIST_URL}/{{experiment_id}}" in paths

    run = paths[RUN_URL]["post"]
    assert run["summary"]
    assert set(run["responses"]) >= {"200", "400", "409", "413", "415", "422"}

    form = run["requestBody"]["content"]["multipart/form-data"]["schema"]
    properties = spec["components"]["schemas"][form["$ref"].rsplit("/", 1)[-1]][
        "properties"
    ]
    assert {"file", "target_column", "models", "folds", "strategy"} <= set(properties)
    assert properties["target_column"]["description"]


def test_the_docs_page_is_served(experiment_client: TestClient) -> None:
    """The interactive documentation still renders."""
    assert experiment_client.get("/docs").status_code == 200


# --------------------------------------------------------------------------
# Response contract: nothing but JSON-safe values
# --------------------------------------------------------------------------


def _walk(value: Any) -> list[Any]:
    """Flatten a decoded JSON payload into every scalar it contains."""
    if isinstance(value, dict):
        return [item for child in value.values() for item in _walk(child)]
    if isinstance(value, list):
        return [item for child in value for item in _walk(child)]
    return [value]


FORBIDDEN_OBJECT_TEXT = (
    "Pipeline(",
    "ColumnTransformer(",
    "LogisticRegression(",
    "RandomForestClassifier(",
    "LinearRegression(",
    "TreeExplainer(",
    "LinearExplainer(",
    "shap.",
    "sklearn",
    "numpy.ndarray",
    "DataFrame",
    "object at 0x",
)


def test_the_response_holds_only_json_safe_values(
    classification_run: dict[str, Any]
) -> None:
    """Every leaf is a string, number, boolean or null.

    A sklearn estimator, a numpy array or a DataFrame cannot pass through the
    response model, and this asserts what actually arrived over HTTP.
    """
    for leaf in _walk(classification_run):
        assert leaf is None or isinstance(leaf, (str, int, float, bool)), leaf


def test_no_estimator_or_explainer_object_appears_in_the_response(
    classification_run: dict[str, Any], regression_run: dict[str, Any]
) -> None:
    """Model artefacts are results, not payload.

    ``explainer`` naming a strategy is fine; a repr of one is not.
    """
    for payload in (classification_run, regression_run):
        text = json.dumps(payload)
        for artefact in FORBIDDEN_OBJECT_TEXT:
            assert artefact not in text, artefact
    assert classification_run["explainability"]["explainer"] in {
        "TreeExplainer",
        "LinearExplainer",
    }


def test_no_dataset_values_appear_in_the_response(
    classification_run: dict[str, Any]
) -> None:
    """The record describes the data; it does not contain it."""
    text = json.dumps(classification_run)

    assert "30000" not in text and "42000" not in text
    assert classification_run["dataset"]["row_count"] == 240


def test_the_response_leaks_no_filesystem_path(
    classification_run: dict[str, Any], experiment_client: TestClient
) -> None:
    """Neither a success nor a failure reveals where anything is stored."""
    missing = experiment_client.get(f"{LIST_URL}/exp_absent_20260101T000000Z_0000")
    for text in (json.dumps(classification_run), missing.text):
        assert "/home/" not in text
        assert not re.search(r"[A-Za-z]:\\\\", text)
        assert "experiments/runs" not in text
        assert "site-packages" not in text


def test_no_failure_leaks_a_traceback(experiment_client: TestClient) -> None:
    """Every expected failure is a message, never a stack trace."""
    responses = [
        run_experiment(experiment_client, target_column="churn"),
        run_experiment(experiment_client, filename="notes.txt"),
        run_experiment(experiment_client, target_column="renewed", models=["nope"]),
        experiment_client.get(f"{LIST_URL}/exp_absent_20260101T000000Z_0000"),
        experiment_client.get(LIST_URL, params={"sort_by": "luck"}),
    ]

    for response in responses:
        assert response.status_code >= 400
        assert set(response.json()) == {"error"}
        for marker in ("Traceback", "File \"", "line ", "raise ", ".py"):
            assert marker not in response.text, marker


def test_the_environment_section_carries_no_secret(
    classification_run: dict[str, Any]
) -> None:
    """Reproducibility metadata, and nothing that identifies the machine."""
    environment = classification_run["environment"]

    assert set(environment) == {
        "python_version",
        "platform",
        "packages",
        "random_state",
    }
    text = json.dumps(environment).lower()
    for secret in ("token", "api_key", "password", "secret", "/home/", "http"):
        assert secret not in text


def test_uploaded_datasets_are_not_kept(
    experiment_store_dir: Path, classification_run: dict[str, Any]
) -> None:
    """Only records are written; the upload itself is never stored."""
    written = sorted(path.name for path in experiment_store_dir.rglob("*") if path.is_file())

    assert written, "expected at least one record"
    assert set(written) == {"experiment.json"}


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_an_experiment_survives_the_request_that_made_it(
    experiment_client: TestClient, experiment_store_dir: Path
) -> None:
    """The whole path, from upload to a record fetched back over HTTP.

    POST /experiments/run -> profiling -> preprocessing -> cross-validation ->
    winner -> final untouched-test evaluation -> SHAP -> ExperimentRun ->
    LocalExperimentStore, then GET /experiments/{id} and compare the two.
    """
    run = run_experiment(
        experiment_client,
        target_column="renewed",
        models=CLASSIFIERS,
        folds=FOLDS,
        name="end to end",
        tags=["e2e"],
    ).json()
    experiment_id = run["experiment_id"]

    on_disk = json.loads(
        (experiment_store_dir / experiment_id / "experiment.json").read_text(
            encoding="utf-8"
        )
    )
    fetched = experiment_client.get(f"{LIST_URL}/{experiment_id}").json()

    # The record returned by the run, the bytes on disk and the record served
    # back by the API are one and the same thing.
    for section in (
        "dataset",
        "preprocessing",
        "selection",
        "evaluation",
        "explainability",
        "environment",
    ):
        assert fetched[section] == on_disk[section] == run[section], section

    assert fetched["configuration_hash"] == run["configuration_hash"]
    assert fetched["experiment_id"] == on_disk["experiment_id"] == experiment_id

    listed = experiment_client.get(
        LIST_URL, params={"tags": ["e2e"], "limit": 5}
    ).json()
    assert [item["experiment_id"] for item in listed["experiments"]] == [experiment_id]
    assert listed["experiments"][0]["test_score"] == pytest.approx(
        run["evaluation"]["primary_metric_value"]
    )


def test_the_same_configuration_produces_the_same_hash_over_http(
    experiment_client: TestClient,
) -> None:
    """Two identical requests are recognised as the same configuration.

    The ids differ — they are two executions — but the configuration hash and
    the dataset fingerprint match, which is what makes a repeat findable.
    """
    options = dict(
        target_column="renewed",
        models=["logistic_regression"],
        folds=FOLDS,
        random_state=11,
        name="repeatable",
    )
    first = run_experiment(experiment_client, **options).json()
    second = run_experiment(experiment_client, **options).json()

    assert first["configuration_hash"] == second["configuration_hash"]
    assert first["dataset"]["fingerprint"] == second["dataset"]["fingerprint"]
    assert first["experiment_id"] != second["experiment_id"]


def test_a_renamed_file_is_recognised_as_the_same_dataset(
    experiment_client: TestClient,
) -> None:
    """Identity is content, not filename — the property the API inherits."""
    content = learnable_classification_csv()
    first = run_experiment(
        experiment_client, content, filename="customers.csv", target_column="renewed",
        models=["logistic_regression"], folds=FOLDS,
    ).json()
    second = run_experiment(
        experiment_client, content, filename="export_final_v2.csv",
        target_column="renewed", models=["logistic_regression"], folds=FOLDS,
    ).json()

    assert first["dataset"]["fingerprint"] == second["dataset"]["fingerprint"]


def test_a_high_cardinality_column_is_excluded_with_a_reason(
    experiment_client: TestClient,
) -> None:
    """Profiling findings reach the experiment record as context."""
    payload = run_experiment(
        experiment_client,
        high_cardinality_csv(120),
        filename="codes.csv",
        target_column="bucket",
        models=["random_forest_classifier"],
        folds=FOLDS,
        name="high cardinality",
    )

    # Either the column is excluded and the run succeeds, or nothing usable
    # remains and the API says so — both are correct, neither is a crash.
    if payload.status_code == 200:
        body = payload.json()
        assert "code" in body["preprocessing"]["excluded_columns"] or (
            "code" in body["preprocessing"]["identifier_columns"]
        )
    else:
        assert_envelope(payload, status_code=422, code="empty_feature_set")
