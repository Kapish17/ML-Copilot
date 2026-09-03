"""Tests for request correlation, logging and the generated documentation.

Three things are asserted here, and they share a failure mode: each is
*invisible* when it breaks. A middleware that stops setting a header still
serves every request. A log line that quietly starts carrying an uploaded
filename still reads as a normal log. A tag description that never made it into
the schema leaves a documentation page that looks finished.

So the tests are about behaviour rather than wording:

**Correlation works, and cannot be turned into an injection vector.** A
response carries an id, a well-formed inbound id is honoured, and a malformed
one is replaced rather than written into a log line an operator will read as
though the server wrote it.

**The logs say enough and no more.** One line per request. A dataset's shape
but never its filename or its rows. A start-up line that reports whether a
credential is configured without going near its value.

**The schema documents itself.** Every tag a route uses has a description.
"""

from __future__ import annotations

import io
import logging

import pytest
from fastapi.testclient import TestClient

from app.api.middleware import VALID_REQUEST_ID, resolve_request_id
from app.core.config import Settings
from app.core.logging import (
    NO_REQUEST_ID,
    PROJECT_LOGGERS,
    REQUEST_ID_HEADER,
    RequestIdFilter,
    configure_logging,
    current_request_id,
)
from app.main import OPENAPI_TAGS, create_app
from starlette.datastructures import Headers

PROFILE_URL = "/api/v1/datasets/profile"

#: A credential-shaped value, so a test can prove it does not reach a log.
FAKE_KEY = "sk-test-do-not-log-000111222333"


def upload(content: bytes, name: str = "data.csv") -> dict[str, tuple]:
    """Build a multipart upload."""
    return {"file": (name, io.BytesIO(content), "text/csv")}


def tiny_csv() -> bytes:
    """Four rows that profile without trouble."""
    return b"a,b,label\n1,2,yes\n3,4,no\n5,6,yes\n7,8,no\n"


# ---------------------------------------------------------------------------
# The header
# ---------------------------------------------------------------------------


def test_every_response_carries_a_request_id(client: TestClient) -> None:
    """Including one nobody asked for."""
    response = client.get("/health")

    assert VALID_REQUEST_ID.match(response.headers[REQUEST_ID_HEADER])


def test_two_requests_get_two_different_ids(client: TestClient) -> None:
    """An id that repeats correlates nothing."""
    first = client.get("/health").headers[REQUEST_ID_HEADER]
    second = client.get("/health").headers[REQUEST_ID_HEADER]

    assert first != second


def test_an_error_response_carries_one_too(client: TestClient) -> None:
    """The case where correlation is actually wanted."""
    response = client.get("/api/v1/experiments/not-a-real-id")

    assert response.status_code >= 400
    assert VALID_REQUEST_ID.match(response.headers[REQUEST_ID_HEADER])


def test_a_well_formed_inbound_id_is_honoured(client: TestClient) -> None:
    """So a caller or a proxy can correlate across a hop it made first."""
    response = client.get("/health", headers={REQUEST_ID_HEADER: "caller-abc-123"})

    assert response.headers[REQUEST_ID_HEADER] == "caller-abc-123"


@pytest.mark.parametrize(
    "hostile",
    [
        "has spaces",
        "newline\ninjected",
        "escape\x1b[31m",
        "semi;colon",
        "x" * 65,
        "",
    ],
)
def test_a_malformed_inbound_id_is_replaced_not_echoed(
    client: TestClient, hostile: str
) -> None:
    """This value is written into log lines, so it is not the caller's to choose.

    A newline would let a caller forge a log entry and an escape sequence would
    let them colour someone's terminal. Both are refused by being ignored: the
    request is answered normally under a generated id.
    """
    response = client.get("/health", headers={REQUEST_ID_HEADER: hostile})

    returned = response.headers[REQUEST_ID_HEADER]
    assert returned != hostile
    assert VALID_REQUEST_ID.match(returned)


def test_the_resolver_is_the_only_thing_deciding(client: TestClient) -> None:
    """A unit-level check of the same rule, without an HTTP round trip."""
    assert resolve_request_id(Headers({REQUEST_ID_HEADER: "ok_123-ABC"})) == "ok_123-ABC"
    assert resolve_request_id(Headers({})) != ""
    assert resolve_request_id(Headers({REQUEST_ID_HEADER: "bad id"})) != "bad id"


def test_a_browser_may_read_the_header_it_was_sent() -> None:
    """Exposed through CORS, or a cross-origin dashboard cannot report it.

    Without `expose_headers` the header is present on the wire and unreadable
    from JavaScript, which is the sort of thing that looks fine in curl and is
    broken in the product.
    """
    application = create_app(Settings())
    cors = next(
        middleware
        for middleware in application.user_middleware
        if middleware.cls.__name__ == "CORSMiddleware"
    )

    assert REQUEST_ID_HEADER in cors.kwargs["expose_headers"]
    assert REQUEST_ID_HEADER in cors.kwargs["allow_headers"]


# ---------------------------------------------------------------------------
# The logs
# ---------------------------------------------------------------------------


def test_one_line_per_request_naming_method_path_and_status(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The line that turns "the API is slow" into "which call is slow"."""
    with caplog.at_level(logging.INFO, logger="app.api.middleware"):
        client.get("/health")

    lines = [record.getMessage() for record in caplog.records]
    assert len(lines) == 1
    assert "GET" in lines[0] and "/health" in lines[0] and "200" in lines[0]


def test_the_request_id_is_bound_while_the_request_runs(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Every record produced during a request carries that request's id.

    This is the whole point of the mechanism: the ingestion line from `app`
    and the request line from the middleware must be findable together.
    """
    with caplog.at_level(logging.INFO):
        response = client.post(
            PROFILE_URL, files=upload(tiny_csv()), headers={REQUEST_ID_HEADER: "corr-1"}
        )

    assert response.status_code == 200
    during = [
        record for record in caplog.records if record.name.startswith("app.")
    ]
    assert during, "no application records were produced"
    for record in during:
        assert RequestIdFilter().filter(record)
        assert record.request_id == "corr-1"


def test_nothing_is_bound_outside_a_request() -> None:
    """A background line gets a placeholder, not another request's id."""
    assert current_request_id() == NO_REQUEST_ID


def test_ingestion_logs_the_shape_and_never_the_filename(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The filename is chosen by the caller; the shape is the server's own fact.

    A filename in a log line is text a stranger wrote, appearing where an
    operator reads the server's own words — so the upload is logged by what it
    turned out to be, not by what it was called.
    """
    with caplog.at_level(logging.INFO, logger="app.services.datasets.service"):
        client.post(PROFILE_URL, files=upload(tiny_csv(), "../../etc/passwd.csv"))

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "4 rows x 3 columns" in text
    assert "passwd" not in text
    assert "etc" not in text


def test_no_dataset_value_reaches_the_log(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Shape and format, never a cell.

    An uploaded dataset is never written to disk; writing its contents to a log
    instead would defeat that entirely, and a log is the easiest place for it
    to happen by accident.
    """
    content = b"customer,secret_note\n1,extremely-distinctive-cell-value\n2,another\n"

    with caplog.at_level(logging.DEBUG):
        client.post(PROFILE_URL, files=upload(content))

    assert "extremely-distinctive-cell-value" not in caplog.text
    assert "secret_note" not in caplog.text


def test_a_rejected_upload_is_logged_by_its_stable_code(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """A failure worth counting in a log is a failure named the same way twice."""
    with caplog.at_level(logging.INFO, logger="app.services.datasets.service"):
        response = client.post(PROFILE_URL, files=upload(b"", "empty.csv"))

    code = response.json()["error"]["code"]
    assert code in "\n".join(record.getMessage() for record in caplog.records)


def test_the_start_up_line_reports_the_credential_without_reading_it(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Whether one is set is operational. What it is, is not."""
    monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)

    with caplog.at_level(logging.INFO, logger="app.main"):
        with TestClient(create_app(Settings())):
            pass

    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "llm_credential_configured=True" in text
    assert FAKE_KEY not in text
    assert "started" in text and "stopped" in text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_configuring_logging_twice_leaves_one_handler() -> None:
    """An application built per test must not multiply every line."""
    configure_logging("INFO")
    before = len(logging.getLogger().handlers)
    configure_logging("INFO")

    assert len(logging.getLogger().handlers) == before


def test_the_level_is_raised_only_on_this_project() -> None:
    """`LOG_LEVEL=DEBUG` must not turn on every installed package.

    The root logger's level is deliberately untouched: a record from `app` at
    INFO reaches the root handler regardless, because the level test happens on
    the logger the record was created on.
    """
    root_before = logging.getLogger().level

    configure_logging("DEBUG")

    for name in PROJECT_LOGGERS:
        assert logging.getLogger(name).level == logging.DEBUG
    assert logging.getLogger().level == root_before
    assert logging.getLogger("openpyxl").level == logging.NOTSET

    configure_logging("INFO")


def test_an_unusable_level_falls_back_rather_than_failing_to_start() -> None:
    """A typo in `LOG_LEVEL` is not a reason to refuse to serve traffic."""
    configure_logging("VERY-LOUD")

    assert logging.getLogger("app").level == logging.INFO

    configure_logging("INFO")


# ---------------------------------------------------------------------------
# The generated documentation
# ---------------------------------------------------------------------------


def test_every_tag_a_route_uses_is_described(client: TestClient) -> None:
    """Computed from the schema, so a new tag group fails until it is described.

    Five bare headings on `/docs` is what this prevents: a reader should learn
    what a group is for without opening an endpoint inside it.
    """
    schema = client.get("/openapi.json").json()

    used = {
        tag
        for operations in schema["paths"].values()
        for operation in operations.values()
        for tag in operation.get("tags", [])
    }
    described = {tag["name"]: tag["description"] for tag in schema.get("tags", [])}

    assert used, "no route is tagged at all"
    assert used <= set(described), f"undescribed tags: {sorted(used - set(described))}"
    for name, description in described.items():
        assert len(description) > 60, f"the {name} description says too little"


def test_the_documentation_describes_no_more_groups_than_exist(
    client: TestClient,
) -> None:
    """A description for a tag no route uses is documentation for nothing."""
    schema = client.get("/openapi.json").json()

    used = {
        tag
        for operations in schema["paths"].values()
        for operation in operations.values()
        for tag in operation.get("tags", [])
    }

    assert {tag["name"] for tag in OPENAPI_TAGS} == used


def test_the_documentation_makes_no_stale_claim(client: TestClient) -> None:
    """The description has been wrong before; these are the ways it was wrong."""
    schema = client.get("/openapi.json").json()
    described = schema["info"]["description"] + "".join(
        tag["description"] for tag in schema["tags"]
    )
    lowered = described.lower()

    assert "csv only" not in lowered
    assert "not implemented yet" not in lowered
    assert "coming soon" not in lowered
    assert "early development" not in lowered
    # The three formats and the agent are all present, and have been since
    # commits 15 and 12 respectively.
    assert ".xlsx" in lowered and "json" in lowered
    assert "agent" in lowered
