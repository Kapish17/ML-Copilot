"""Tests for the service-level endpoints exposed in this commit."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_root_returns_service_info(client: TestClient) -> None:
    """The root endpoint identifies the service and its version."""
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "ML Copilot API"
    assert payload["docs_url"] == "/docs"
    assert payload["version"]
    assert payload["environment"]


def test_health_reports_ok(client: TestClient) -> None:
    """The health endpoint reports a healthy service."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_response_shape(client: TestClient) -> None:
    """The health payload exposes exactly the documented fields."""
    payload = client.get("/health").json()

    assert set(payload) == {"status", "version", "environment"}


def test_unknown_route_returns_404(client: TestClient) -> None:
    """An unknown path answers 404 using the shared error envelope."""
    response = client.get("/not-a-route")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_openapi_documents_the_dataset_endpoint(client: TestClient) -> None:
    """The v1 dataset routes are mounted and documented."""
    paths = client.get("/openapi.json").json()["paths"]

    assert "/api/v1/datasets/profile" in paths
    assert "post" in paths["/api/v1/datasets/profile"]


# ---------------------------------------------------------------------------
# Cross-origin access for the dashboard
# ---------------------------------------------------------------------------


def test_a_browser_origin_on_the_allowlist_is_permitted() -> None:
    """The dashboard runs on another port, so its requests are cross-origin.

    Without this the browser refuses every call the frontend makes, which is
    why it is the one backend change Commit 16 required.
    """
    client = TestClient(
        create_app(Settings(cors_allow_origins=("http://localhost:3000",)))
    )

    response = client.get("/health", headers={"Origin": "http://localhost:3000"})

    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"] == "http://localhost:3000"
    )


def test_an_origin_off_the_allowlist_gets_no_permission() -> None:
    """An explicit list, never a wildcard."""
    client = TestClient(
        create_app(Settings(cors_allow_origins=("http://localhost:3000",)))
    )

    response = client.get("/health", headers={"Origin": "http://evil.example"})

    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_credentials_are_never_allowed_cross_origin() -> None:
    """Nothing here is authenticated, so no cookie may ride along."""
    client = TestClient(
        create_app(Settings(cors_allow_origins=("http://localhost:3000",)))
    )

    response = client.options(
        "/api/v1/datasets/profile",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert "access-control-allow-credentials" not in response.headers
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_an_empty_allowlist_installs_no_cross_origin_middleware() -> None:
    """A deployment serving the dashboard from this origin adds no surface.

    The request-context middleware is always installed and is named
    explicitly, so this stays a statement about CORS: an empty allowlist must
    add no cross-origin handling, not merely fewer middlewares than before.
    """
    application = create_app(Settings(cors_allow_origins=()))

    installed = [middleware.cls.__name__ for middleware in application.user_middleware]
    assert installed == ["RequestContextMiddleware"]
    assert "CORSMiddleware" not in installed
