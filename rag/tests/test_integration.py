"""Tests against the real repository, and the optional neural provider.

Two things are checked here that a test with synthetic documents cannot show.

**What the shipped configuration actually indexes.** The allowlist, the
forbidden names and the directory rules are only worth anything if they hold
against this repository as it really is — with a ``.env.example`` in the root,
CSV fixtures in the tests, and a ``.git`` directory full of blobs.

**That the optional provider is genuinely optional.** The suite must run with
no PyTorch installed and no network. The one test that exercises the real
sentence-transformer model skips itself unless the package is installed *and*
the model is already cached, so it never downloads anything.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from rag.config import PROJECT_ROOT, RagConfig
from rag.embeddings import build_embedding_provider
from rag.indexing import RagIndexer
from rag.ingestion.documentation import discover_documentation_paths, is_forbidden_name
from rag.retrieval import RetrievalService
from rag.stores import LocalVectorStore
from rag.tests.factories import FakeEmbeddingProvider

#: Names that must never appear among the indexed files, whatever changes.
SENSITIVE_NAMES = (".env", ".env.example", "credentials.json", "secrets.json")


@pytest.fixture(scope="module")
def repository_config() -> RagConfig:
    """The shipped configuration, pointed at the real repository."""
    return RagConfig()


# --------------------------------------------------------------------------
# What the real configuration indexes
# --------------------------------------------------------------------------


def test_the_shipped_configuration_finds_the_project_readmes(
    repository_config: RagConfig,
) -> None:
    """The four READMEs are the documentation this project has."""
    references = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in discover_documentation_paths(repository_config)
    }

    assert {
        "README.md",
        "ml/README.md",
        "backend/README.md",
        "rag/README.md",
    } <= references


def test_no_environment_or_secret_file_is_indexed(
    repository_config: RagConfig,
) -> None:
    """The repository has a .env.example; it must never reach the index."""
    assert (PROJECT_ROOT / ".env.example").is_file(), "precondition for this test"

    names = {path.name for path in discover_documentation_paths(repository_config)}

    for sensitive in SENSITIVE_NAMES:
        assert sensitive not in names
        assert is_forbidden_name(sensitive, repository_config) is True


def test_no_source_code_dataset_or_git_object_is_indexed(
    repository_config: RagConfig,
) -> None:
    """Only Markdown documentation is a candidate — nothing else."""
    paths = discover_documentation_paths(repository_config)

    for path in paths:
        relative = path.relative_to(PROJECT_ROOT)
        assert path.suffix.lower() in repository_config.documentation_extensions
        assert ".git" not in relative.parts
        assert "data" not in relative.parts
        assert "__pycache__" not in relative.parts
        assert not path.name.endswith((".py", ".csv", ".json", ".npy", ".jsonl"))


def test_the_experiment_store_is_never_walked_as_documentation(
    repository_config: RagConfig,
) -> None:
    """Stored runs are ingested through their own adapter, not as files."""
    references = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in discover_documentation_paths(repository_config)
    }

    assert not any("experiments/runs" in reference for reference in references)
    assert "runs" in repository_config.forbidden_directory_names


# --------------------------------------------------------------------------
# The real documentation, end to end
# --------------------------------------------------------------------------


def test_the_real_documentation_indexes_and_retrieves(tmp_path: Path) -> None:
    """The whole path over this project's own READMEs, with the real provider.

    Uses the default hashing provider deliberately: it is what ships, it needs
    no download, and its retrieval quality is the number the evaluation
    reports.
    """
    config = RagConfig(index_dir=tmp_path / "index")
    store = LocalVectorStore(config.index_dir)
    indexer = RagIndexer(config, store=store)

    report = indexer.index_documentation()
    assert len(report.indexed) >= 3
    assert report.chunks_written > 20

    service = RetrievalService(config, store=LocalVectorStore(config.index_dir))
    response = service.search("how is data leakage prevented", top_k=5)

    assert not response.is_empty
    assert all(result.citation.startswith("docs:") for result in response)
    json.dumps(response.as_dict())


def test_the_real_documentation_answers_the_evaluation_set(tmp_path: Path) -> None:
    """Hit@5 and Recall@5 over the shipped questions.

    The thresholds are deliberately modest. This is a regression check — it
    catches a chunker or provider change that quietly stops retrieving the
    leakage section — not a claim that the default embeddings are excellent.
    """
    from rag.evaluation import evaluate_retrieval

    config = RagConfig(index_dir=tmp_path / "index")
    RagIndexer(config, store=LocalVectorStore(config.index_dir)).index_documentation()

    service = RetrievalService(config, store=LocalVectorStore(config.index_dir))
    report = evaluate_retrieval(service, k=5)

    assert report.query_count == 5
    assert report.hit_rate >= 0.8, report.as_text()
    assert report.recall >= 0.5, report.as_text()


def test_the_index_directory_holds_only_what_it_should(tmp_path: Path) -> None:
    """Three files, nothing else, and no document content in the manifest."""
    config = RagConfig(index_dir=tmp_path / "index")
    RagIndexer(
        config,
        store=LocalVectorStore(config.index_dir),
        embeddings=FakeEmbeddingProvider(dimension=32),
    ).index_documentation()

    names = sorted(path.name for path in config.index_dir.iterdir())
    manifest = json.loads((config.index_dir / "manifest.json").read_text("utf-8"))

    assert names == ["manifest.json", "records.jsonl", "vectors.npy"]
    assert "content" not in json.dumps(manifest)
    assert manifest["embedding_identifier"] == "fake-32"


# --------------------------------------------------------------------------
# The optional neural provider
# --------------------------------------------------------------------------


def _sentence_transformers_installed() -> bool:
    """Whether the optional dependency is importable."""
    return importlib.util.find_spec("sentence_transformers") is not None


def _model_is_cached() -> bool:
    """Whether the default model is already on disk.

    Checked so the test never triggers a download: an integration test that
    reaches the network is not one this suite is allowed to run.
    """
    if not _sentence_transformers_installed():
        return False
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache.is_dir():
        return False
    return any("all-MiniLM-L6-v2" in entry.name for entry in cache.iterdir())


@pytest.mark.skipif(
    not _model_is_cached(),
    reason="sentence-transformers is optional; the model is not installed and cached",
)
def test_the_neural_provider_embeds_when_it_is_available() -> None:
    """An optional integration check, skipped unless everything is local."""
    import numpy as np

    provider = build_embedding_provider(
        RagConfig(embedding_provider="sentence_transformer")
    )
    vectors = provider.embed_documents(
        ["leakage prevention during preprocessing", "docker compose configuration"]
    )
    query = provider.embed_query("how is target leakage avoided")

    assert vectors.shape[1] == provider.dimension
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert float(query @ vectors[0]) > float(query @ vectors[1])


def test_the_suite_does_not_require_the_optional_dependency() -> None:
    """The default path must work with nothing extra installed.

    Recorded as a test rather than a comment, because the guarantee is easy
    to break by adding one convenient top-level import.
    """
    provider = build_embedding_provider(RagConfig())

    assert provider.identifier.startswith("hashing-")
    assert provider.embed_query("anything").shape == (RagConfig().embedding_dimension,)
