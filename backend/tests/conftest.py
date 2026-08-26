"""Shared pytest fixtures for the backend test suite."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app
from app.services.datasets import DatasetProfilingService


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """Provide a test client bound to the FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def settings() -> Settings:
    """Provide default settings, independent of the process environment."""
    return Settings()


@pytest.fixture(scope="session")
def service(settings: Settings) -> DatasetProfilingService:
    """Provide a dataset profiling service using default settings."""
    return DatasetProfilingService(settings)
