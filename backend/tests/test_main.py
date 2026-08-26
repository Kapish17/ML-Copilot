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
    """No other routes are registered in this commit."""
    assert client.get("/not-a-route").status_code == 404
