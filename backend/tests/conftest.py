"""Shared pytest fixtures for the backend test suite."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app, create_app
from app.services.datasets import DatasetProfilingService

# A finished experiment is now indexed into the retrieval index as it is run
# (see ``ExperimentRunner._index_for_knowledge``), through ``get_rag_config()``
# when a client is built with no ``rag_config`` override. That dependency is
# cached process-wide, so the first such request in the whole suite decides
# where every later one writes. Left unset, that would be the repository's own
# ``rag/index`` — exactly the "written into the repository's own store during
# a test run" problem ``experiment_store_dir`` below exists to prevent, here
# for the retrieval index instead of the experiment store. Set here, at import
# time and before any fixture runs, so every test file's default client stays
# isolated without each one needing its own override.
os.environ.setdefault("RAG_INDEX_DIR", tempfile.mkdtemp(prefix="ml-copilot-test-rag-"))


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


@pytest.fixture(scope="session")
def experiment_store_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An empty directory for experiment records, isolated per test session.

    Experiments are never written into the repository's own store during a
    test run, so a suite can be run repeatedly without accumulating history.
    """
    return tmp_path_factory.mktemp("experiment-store")


@pytest.fixture(scope="session")
def experiment_settings(experiment_store_dir: Path) -> Settings:
    """Settings whose experiment store points at the temporary directory."""
    return Settings(experiment_store_dir=experiment_store_dir)


@pytest.fixture(scope="session")
def experiment_client(experiment_settings: Settings) -> Iterator[TestClient]:
    """A client for an application whose experiment store is temporary."""
    with TestClient(create_app(experiment_settings)) as test_client:
        yield test_client
