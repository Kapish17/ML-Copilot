"""Regression tests for the production-hardening pass.

Every test in this file was written against a defect that existed. They are
grouped by the question they answer rather than by the module they touch,
because that is the shape the risks came in: a limit that was configured but
not enforced, a failure that could not be correlated, an error that carried a
value it should not have.

The tests deliberately reach for the *behaviour* rather than the
implementation. A future refactor is free to move any of this code; what it is
not free to do is put back the behaviour these tests describe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.middleware import MULTIPART_ENVELOPE_ALLOWANCE
from app.core.config import (
    MAX_CONFIGURABLE_UPLOAD_MB,
    Settings,
    get_settings,
)
from app.core.logging import REQUEST_ID_HEADER
from app.main import create_app
from tests.factories import learnable_classification_csv, upload_payload

# ---------------------------------------------------------------------------
# A failure a person can quote
# ---------------------------------------------------------------------------


@pytest.fixture
def failing_client() -> TestClient:
    """A client for an app with one route that always raises."""
    application = create_app()

    @application.get("/__test_failure")
    def _fail() -> None:
        """Raise the way an unexpected bug would."""
        raise RuntimeError("a failure nobody anticipated")

    return TestClient(application, raise_server_exceptions=False)


def test_an_unexpected_500_still_carries_its_request_id(
    failing_client: TestClient,
) -> None:
    """The one failure a user reports is the one they must be able to name.

    Starlette's unhandled-error handler runs above every middleware this
    application adds, so the id had already been unbound and the response had
    stopped passing through the code that stamps it. A 500 came back with no
    correlation id at all — the only status that could not be traced.
    """
    response = failing_client.get("/__test_failure")

    assert response.status_code == 500
    assert response.headers.get(REQUEST_ID_HEADER)
    assert response.json()["error"]["code"] == "internal_error"


def test_a_500_leaks_nothing_about_how_it_failed(
    failing_client: TestClient,
) -> None:
    """A generic message and empty details; the cause goes to the log."""
    payload = response_error(failing_client.get("/__test_failure"))

    assert payload["details"] == {}
    body = json.dumps(payload)
    for leaked in ("RuntimeError", "Traceback", "nobody anticipated", "/home"):
        assert leaked not in body


def test_a_caller_supplied_request_id_is_echoed_on_a_500(
    failing_client: TestClient,
) -> None:
    """Correlation works from the caller's side too, not only the server's."""
    response = failing_client.get(
        "/__test_failure", headers={REQUEST_ID_HEADER: "trace-abc-123"}
    )

    assert response.headers[REQUEST_ID_HEADER] == "trace-abc-123"


def test_a_405_says_which_methods_are_allowed(client: TestClient) -> None:
    """The envelope was right and the header was missing.

    `Allow` is required on a 405 by RFC 9110, and the handler was dropping the
    exception's headers on the floor while producing a response that looked
    entirely correct.
    """
    response = client.request("DELETE", "/health")

    assert response.status_code == 405
    assert response.headers.get("allow")
    assert response.json()["error"]["code"] == "method_not_allowed"


def test_the_413_status_table_uses_a_constant_every_supported_starlette_has() -> None:
    """The 413 entry must not depend on a Starlette-version-specific name.

    Starlette renamed its 413 constant from ``HTTP_413_REQUEST_ENTITY_TOO_LARGE``
    to ``HTTP_413_CONTENT_TOO_LARGE``; only the newest Starlette releases
    define the new name, and only the oldest ones lack it entirely. Importing
    ``app.api.error_handlers`` with an older Starlette installed (as pinned by
    some environments) used to raise ``AttributeError`` at import time, before
    the app could even start. Reimporting the module here — rather than only
    relying on the app already having started for every other test in this
    file — keeps that specific failure mode covered.
    """
    import importlib

    from app.api import error_handlers

    importlib.reload(error_handlers)

    assert error_handlers._HTTP_STATUS_CODES[413] == "file_too_large"


# ---------------------------------------------------------------------------
# Limits enforced where they are actually reachable
# ---------------------------------------------------------------------------


def test_an_oversized_body_is_refused_with_cors_headers(client: TestClient) -> None:
    """A browser has to be able to *read* the refusal.

    The body-limit middleware sat outside the CORS middleware, so its 413 went
    out without `Access-Control-Allow-Origin` — and a browser turns that into
    an opaque network failure. The dashboard showed "could not reach the
    server" for the one error that has a precise, actionable explanation.
    """
    settings = Settings()
    oversized = b'{"query":"' + b"x" * (settings.max_request_body_bytes + 1024) + b'"}'

    response = client.post(
        "/api/v1/search",
        content=oversized,
        headers={
            "content-type": "application/json",
            "Origin": "http://localhost:3000",
        },
    )

    assert response.status_code == 413
    assert response.headers.get("access-control-allow-origin") == (
        "http://localhost:3000"
    )
    assert response.json()["error"]["code"] == "request_body_too_large"


def test_a_multipart_flood_is_stopped_before_it_is_written_to_disk() -> None:
    """A streamed upload with no declared length used to be unbounded.

    Starlette parses the whole multipart body — writing each file part to a
    temporary file — before any route code runs, so the upload reader's own
    size check was reached only after the bytes had already landed. A client
    that never stops sending filled the disk.

    The request below is three times the upload limit and declares no
    `Content-Length`, which is the shape that bypassed every check.
    """
    settings = Settings()
    client = TestClient(create_app(settings))
    limit = settings.max_upload_bytes

    def flood():
        """Yield a multipart body that is far larger than the limit."""
        yield (
            b'--B\r\nContent-Disposition: form-data; name="file"; '
            b'filename="big.csv"\r\nContent-Type: text/csv\r\n\r\n'
        )
        sent = 0
        while sent < limit * 3:
            block = b"a" * (1024 * 1024)
            sent += len(block)
            yield block
        yield b"\r\n--B--\r\n"

    response = client.post(
        "/api/v1/datasets/profile",
        content=flood(),
        headers={"content-type": "multipart/form-data; boundary=B"},
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_body_too_large"


def test_a_legitimate_upload_is_not_caught_by_the_multipart_ceiling(
    client: TestClient,
) -> None:
    """The guard must not refuse the requests it exists to protect.

    The ceiling is the upload limit plus an envelope allowance, so a file at
    the limit still fits with its boundaries and text fields around it.
    """
    settings = Settings()

    assert MULTIPART_ENVELOPE_ALLOWANCE > 0
    response = client.post(
        "/api/v1/datasets/profile",
        files=upload_payload(learnable_classification_csv(rows=40), "small.csv"),
    )

    assert response.status_code == 200
    assert settings.max_upload_bytes + MULTIPART_ENVELOPE_ALLOWANCE > (
        settings.max_upload_bytes
    )


def test_an_encoded_feature_explosion_is_refused_before_training(
    experiment_client: TestClient,
) -> None:
    """The column limit counts the file; this counts what the model gets.

    One-hot encoding turns a categorical column into one feature per category,
    so a small file of wide categorical columns becomes a very large dense
    matrix — which cross-validation then copies per fold, per candidate. The
    column check could not see that, because it ran before encoding.
    """
    rows = 60
    header = ",".join(f"c{index}" for index in range(40)) + ",label\n"
    lines = [header]
    for row in range(rows):
        cells = ",".join(f"v{row % 40}_{column}" for column in range(40))
        lines.append(f"{cells},{'yes' if row % 2 else 'no'}\n")
    content = "".join(lines).encode("utf-8")

    settings = Settings(
        experiment_store_dir=Path("/tmp/unused-store"), max_encoded_features=50
    )
    client = TestClient(create_app(settings))

    response = client.post(
        "/api/v1/experiments/run",
        files=upload_payload(content, "wide.csv"),
        data={"target_column": "label"},
    )

    assert response.status_code == 413
    error = response.json()["error"]
    assert error["details"]["max_encoded_features"] == 50
    assert error["details"]["encoded_feature_count"] > 50


# ---------------------------------------------------------------------------
# Configuration that refuses to be wrong quietly
# ---------------------------------------------------------------------------


def test_a_limit_large_enough_to_be_no_limit_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`MAX_UPLOAD_MB=100000` does not configure a limit; it removes one."""
    monkeypatch.setenv("MAX_UPLOAD_MB", str(MAX_CONFIGURABLE_UPLOAD_MB + 1))
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="MAX_UPLOAD_MB must be <="):
        get_settings()

    get_settings.cache_clear()


def test_a_wildcard_cors_origin_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wildcard would let any page call this API from a visitor's browser.

    It is also useless here: the credential is a bearer token, which a browser
    does not attach on its own — so the wildcard gives away the protection an
    unauthenticated deployment has and buys nothing back.
    """
    monkeypatch.setenv("CORS_ALLOW_ORIGINS", "http://localhost:3000,*")
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="must list explicit origins"):
        get_settings()

    get_settings.cache_clear()


def test_a_relative_store_directory_is_resolved_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Otherwise the same configuration means a different directory per launch.

    A relative `EXPERIMENT_STORE_DIR` was resolved against the working
    directory, so a container restarted from elsewhere came up with an empty
    history and no error to explain it.
    """
    monkeypatch.setenv("EXPERIMENT_STORE_DIR", "data/experiments")
    get_settings.cache_clear()

    resolved = get_settings().experiment_store_dir

    assert resolved.is_absolute()
    get_settings.cache_clear()


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"min_cv_folds": 8, "max_cv_folds": 5}, "MIN_CV_FOLDS"),
        ({"default_cv_folds": 99}, "DEFAULT_CV_FOLDS"),
        (
            {"experiment_page_limit": 500, "max_experiment_page_limit": 200},
            "EXPERIMENT_PAGE_LIMIT",
        ),
    ],
)
def test_incoherent_limits_are_refused_at_construction(
    overrides: dict[str, Any], expected: str
) -> None:
    """Each limit was validated alone; none was validated against the others.

    A fold range with no acceptable value in it, or a default outside its own
    range, is a configuration that produces a confusing 4xx on every request
    instead of one clear failure at startup.
    """
    with pytest.raises(ValueError, match=expected):
        Settings(**overrides)


# ---------------------------------------------------------------------------
# What an error is allowed to say
# ---------------------------------------------------------------------------


def test_an_undeclared_form_field_is_refused_rather_than_ignored(
    experiment_client: TestClient,
) -> None:
    """`target_colum=churn` used to start a full training run on the wrong column.

    The field was silently dropped, the target fell back to "the last column by
    convention", and the caller received a complete, expensive, wrong
    experiment with a warning they had no reason to read. The agent endpoint
    already refused unknown fields; this is the endpoint where being wrong
    costs a training run.
    """
    response = experiment_client.post(
        "/api/v1/experiments/run",
        files=upload_payload(learnable_classification_csv(rows=60), "d.csv"),
        data={"target_colum": "renewed"},
    )

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "invalid_request"
    assert "target_colum" in json.dumps(error["details"])


def test_a_rejected_value_is_not_echoed_back_in_full() -> None:
    """A bounded request must not produce an unbounded answer.

    Pydantic quotes the offending value so a developer can see what was wrong
    with it — including for `string_too_long`, where the value is by definition
    longer than allowed, so the complaint carried the whole of it.
    """
    from app.api.error_handlers import (
        MAX_ECHOED_INPUT_CHARS,
        _sanitise_validation_errors,
    )

    over_long = "q" * 5000
    sanitised = _sanitise_validation_errors(
        [
            {
                "type": "string_too_long",
                "loc": ("body", "question"),
                "msg": "String should have at most 2000 characters",
                "input": over_long,
            }
        ]
    )
    echoed = sanitised[0]["input"]

    assert len(echoed) <= MAX_ECHOED_INPUT_CHARS + 1
    assert echoed.endswith("…"), "and it says that it was cut"
    assert echoed.startswith("qqq"), "while still showing what was sent"


def test_a_refused_field_is_still_not_echoed_at_all() -> None:
    """The stricter rule for `extra_forbidden` is unchanged.

    There is nothing to correct about a field that is not accepted, and its
    value is exactly what someone smuggles in — so it is dropped rather than
    shortened.
    """
    from app.api.error_handlers import _sanitise_validation_errors

    sanitised = _sanitise_validation_errors(
        [
            {
                "type": "extra_forbidden",
                "loc": ("body", "api_key"),
                "msg": "Extra inputs are not permitted",
                "input": "sk-a-smuggled-credential",
            }
        ]
    )

    assert "input" not in sanitised[0]
    assert "sk-a-smuggled-credential" not in json.dumps(sanitised)


def test_the_startup_line_describes_the_application_that_was_built(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The one log line read when something is already wrong must be true.

    It read the environment unconditionally, so an application built with
    explicit settings — every test, and any embedding deployment — announced
    the environment's configuration rather than its own. It would report
    `api_auth_enabled=False` for a service that had authentication on.
    """
    settings = Settings(
        api_auth_enabled=True,
        api_auth_key="a-key-long-enough-to-pass-validation-x",
        app_env="staging",
    )

    with caplog.at_level("INFO"):
        with TestClient(create_app(settings)):
            pass

    startup = next(
        record for record in caplog.records if "ML Copilot API started" in record.message
    )
    line = startup.getMessage()

    assert "api_auth_enabled=True" in line
    assert "environment=staging" in line
    assert settings.api_auth_key not in line


def response_error(response: Any) -> dict[str, Any]:
    """The error envelope of a response, asserting its shape first."""
    payload = response.json()
    assert set(payload["error"]) == {"code", "message", "details"}
    return payload["error"]


def test_the_application_starts_with_the_hardened_middleware_order() -> None:
    """CORS must wrap the body limit, and the request id must wrap both.

    Asserted on the built application rather than on the source, because the
    order is a property of what `create_app` produced and the failure it
    prevents is invisible in a unit test of any one middleware.
    """
    application: FastAPI = create_app()
    installed = [middleware.cls.__name__ for middleware in application.user_middleware]

    # `user_middleware` lists outermost first.
    assert installed.index("RequestContextMiddleware") < installed.index(
        "CORSMiddleware"
    )
    assert installed.index("CORSMiddleware") < installed.index(
        "RequestBodyLimitMiddleware"
    )
