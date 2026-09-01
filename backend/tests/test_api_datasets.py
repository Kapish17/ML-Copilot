"""Contract tests for the dataset profiling endpoint."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.factories import build_csv, sample_csv, upload_payload

PROFILE_URL = "/api/v1/datasets/profile"


def _assert_strict_json(response: Any) -> dict[str, Any]:
    """Parse a response body, rejecting NaN and Infinity tokens.

    ``json.loads`` accepts those by default, but they are not valid JSON and
    would break a browser client, so the profiler must never emit them.
    """

    def reject(constant: str) -> None:
        raise AssertionError(f"response contains non-JSON constant {constant}")

    return json.loads(response.text, parse_constant=reject)


def test_profile_returns_expected_structure(client: TestClient) -> None:
    """A valid upload returns the documented response shape."""
    response = client.post(PROFILE_URL, files=upload_payload(sample_csv()))

    assert response.status_code == 200
    payload = _assert_strict_json(response)
    assert set(payload) == {
        "filename",
        "source_format",
        "generated_at",
        "dataset",
        "columns",
        "quality",
        "target",
    }
    assert payload["filename"] == "dataset.csv"
    assert payload["source_format"] == "csv"
    assert payload["target"] is None
    assert payload["dataset"]["row_count"] == 6
    assert payload["dataset"]["column_count"] == 6
    assert len(payload["columns"]) == 6


def test_profile_column_entries_are_typed(client: TestClient) -> None:
    """Each column carries counts, an inferred type and matching statistics."""
    response = client.post(PROFILE_URL, files=upload_payload(sample_csv()))
    columns = {column["name"]: column for column in response.json()["columns"]}

    age = columns["age"]
    assert age["inferred_type"] == "float"
    assert age["numeric_stats"]["mean"] is not None
    assert age["categorical_stats"] is None
    assert age["missing_count"] == 1

    city = columns["city"]
    assert city["inferred_type"] == "categorical"
    assert city["numeric_stats"] is None
    assert city["categorical_stats"]["top_values"][0]["count"] >= 1


def test_profile_reports_quality_issues(client: TestClient) -> None:
    """Duplicates, missing values and constant columns surface in the report."""
    response = client.post(PROFILE_URL, files=upload_payload(sample_csv()))
    quality = response.json()["quality"]
    codes = {issue["code"] for issue in quality["issues"]}

    assert quality["issue_count"] == len(quality["issues"])
    assert {"duplicate_rows", "missing_values", "constant_column"} <= codes


def test_profile_with_target_column(client: TestClient) -> None:
    """A target column is analysed and a task type suggested."""
    response = client.post(
        PROFILE_URL,
        files=upload_payload(sample_csv()),
        data={"target_column": "city"},
    )

    assert response.status_code == 200
    target = response.json()["target"]
    assert target["name"] == "city"
    assert target["task_suggestion"] == "classification"
    assert target["task_reason"]
    assert target["distribution"]


def test_profile_with_blank_target_column(client: TestClient) -> None:
    """An empty target field behaves the same as omitting it."""
    response = client.post(
        PROFILE_URL, files=upload_payload(sample_csv()), data={"target_column": ""}
    )

    assert response.status_code == 200
    assert response.json()["target"] is None


def test_missing_target_column_returns_422(client: TestClient) -> None:
    """An unknown target is rejected and the real columns are listed back."""
    response = client.post(
        PROFILE_URL,
        files=upload_payload(sample_csv()),
        data={"target_column": "not_a_column"},
    )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "target_column_not_found"
    assert "city" in error["details"]["available_columns"]


def test_unsupported_extension_returns_415(client: TestClient) -> None:
    """A non-CSV upload is refused."""
    response = client.post(
        PROFILE_URL, files=upload_payload(sample_csv(), filename="dataset.txt")
    )

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_file_type"


def test_empty_file_returns_400(client: TestClient) -> None:
    """A zero-byte upload is refused."""
    response = client.post(PROFILE_URL, files=upload_payload(b""))

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_file"


def test_malformed_csv_returns_422(client: TestClient) -> None:
    """A row with too many fields is reported as malformed CSV."""
    response = client.post(PROFILE_URL, files=upload_payload(b"a,b\n1,2\n3,4,5\n"))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "malformed_csv"


def test_header_only_csv_returns_422(client: TestClient) -> None:
    """A file with a header but no rows has nothing to profile."""
    response = client.post(PROFILE_URL, files=upload_payload(b"a,b\n"))

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "empty_dataset"


def test_duplicate_columns_return_422(client: TestClient) -> None:
    """Repeated header names are reported rather than silently renamed."""
    response = client.post(PROFILE_URL, files=upload_payload(b"a,a\n1,2\n"))

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "duplicate_columns"
    assert error["details"]["duplicate_columns"] == ["a"]


def test_oversized_upload_returns_413() -> None:
    """An upload beyond the configured limit is refused with 413."""
    application = create_app(Settings(max_upload_bytes=32))
    with TestClient(application) as sized_client:
        response = sized_client.post(PROFILE_URL, files=upload_payload(sample_csv()))

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_row_limit_returns_413() -> None:
    """A dataset with more rows than allowed is refused with 413."""
    application = create_app(Settings(max_dataset_rows=2))
    content = build_csv(["a"], [[1], [2], [3]])
    with TestClient(application) as limited_client:
        response = limited_client.post(PROFILE_URL, files=upload_payload(content))

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "dataset_too_large"


def test_missing_file_field_returns_invalid_request(client: TestClient) -> None:
    """A request without the file part fails validation, not the parser."""
    response = client.post(PROFILE_URL, data={"target_column": "city"})

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert error["details"]["errors"]


def test_get_is_not_allowed(client: TestClient) -> None:
    """Only POST is exposed, and the refusal uses the shared envelope."""
    response = client.get(PROFILE_URL)

    assert response.status_code == 405
    assert response.json()["error"]["code"] == "method_not_allowed"


@pytest.mark.parametrize(
    ("content", "filename"),
    [
        (b"", "dataset.csv"),
        (b"a,b\n1,2\n3,4,5\n", "dataset.csv"),
        (sample_csv(), "dataset.txt"),
    ],
)
def test_errors_share_one_envelope(
    client: TestClient, content: bytes, filename: str
) -> None:
    """Every failure has the same top-level shape for a frontend to consume."""
    response = client.post(PROFILE_URL, files=upload_payload(content, filename))
    payload = response.json()

    assert set(payload) == {"error"}
    assert set(payload["error"]) == {"code", "message", "details"}
    assert payload["error"]["message"]
    assert "Traceback" not in response.text


def test_unexpected_error_is_not_leaked() -> None:
    """An unhandled failure returns a generic message, never internals."""
    application = create_app()

    @application.get("/boom")
    def boom() -> None:
        raise RuntimeError("connection string with a secret")

    with TestClient(application, raise_server_exceptions=False) as failing_client:
        response = failing_client.get("/boom")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert "secret" not in response.text
    assert "Traceback" not in response.text
