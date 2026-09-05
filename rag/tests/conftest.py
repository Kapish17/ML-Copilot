"""Shared fixtures for the retrieval test suite.

Every fixture is function-scoped and writes into a temporary directory, so no
test touches the repository's own index and tests cannot leak state into each
other. Nothing here downloads a model or reaches the network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.config import RagConfig
from rag.indexing import RagIndexer
from rag.retrieval import RetrievalService
from rag.stores import LocalVectorStore
from rag.tests.factories import (
    FakeEmbeddingProvider,
    FakeExperimentRun,
    FakeExperimentStore,
    write_documentation,
)


@pytest.fixture
def index_dir(tmp_path: Path) -> Path:
    """An empty directory for one test's index."""
    return tmp_path / "index"


@pytest.fixture
def docs_root(tmp_path: Path) -> Path:
    """A project root holding the test documentation."""
    root = tmp_path / "project"
    root.mkdir()
    write_documentation(root)
    return root


@pytest.fixture
def config(index_dir: Path, docs_root: Path) -> RagConfig:
    """A configuration pointing at the temporary project and index.

    The chunk size is small so that the test documents produce several chunks
    each — a chunker is only interesting when it actually splits something.
    """
    return RagConfig(
        index_dir=index_dir,
        project_root=docs_root,
        documentation_files=("PREPROCESSING.md", "EVALUATION.md"),
        documentation_dir=None,
        chunk_size=400,
        chunk_overlap=60,
        min_chunk_size=80,
        embedding_dimension=32,
    )


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    """A deterministic embedding provider that needs no model."""
    return FakeEmbeddingProvider(dimension=32)


@pytest.fixture
def store(index_dir: Path) -> LocalVectorStore:
    """An empty local vector store."""
    return LocalVectorStore(index_dir)


@pytest.fixture
def indexer(
    config: RagConfig, store: LocalVectorStore, embeddings: FakeEmbeddingProvider
) -> RagIndexer:
    """An indexer wired to the temporary store and the fake provider."""
    return RagIndexer(config, store=store, embeddings=embeddings)


@pytest.fixture
def service(
    config: RagConfig, store: LocalVectorStore, embeddings: FakeEmbeddingProvider
) -> RetrievalService:
    """A retrieval service over the same store the indexer writes to."""
    return RetrievalService(config, store=store, embeddings=embeddings)


@pytest.fixture
def experiment_run() -> FakeExperimentRun:
    """One synthetic experiment record."""
    return FakeExperimentRun()


@pytest.fixture
def experiment_run_with_diagnostics() -> FakeExperimentRun:
    """A run that raised a signal, worded the way the ML layer words them."""
    return FakeExperimentRun(
        experiment_id="exp_ddd111222333_20260101T120000Z_0002",
        diagnostics=(
            {
                "code": "generalisation_gap",
                "severity": "warning",
                "message": (
                    "Potential overfitting signal: held-out performance is "
                    "materially below cross-validation performance."
                ),
                "details": {"relative_shortfall": 0.31},
            },
        ),
    )


@pytest.fixture
def experiment_store(experiment_run: FakeExperimentRun) -> FakeExperimentStore:
    """An experiment store holding one classification and one regression run."""
    return FakeExperimentStore(
        [
            experiment_run,
            FakeExperimentRun(
                experiment_id="exp_999888777666_20260102T090000Z_0002",
                name="price baseline",
                task_type="regression",
                selected_model="linear_regression",
                primary_metric="rmse",
                fingerprint="ffffeeee11112222",
                target_column="price",
                test_score=1957.67,
                selection_score=2291.11,
                tags=("housing",),
            ),
        ]
    )
