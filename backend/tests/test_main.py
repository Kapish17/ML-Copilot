"""Tests for the service-level endpoints exposed in this commit."""

from __future__ import annotations

from fastapi.testclient import TestClient


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
