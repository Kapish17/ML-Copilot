"""Tests for the indexer, the manifest and experiment synchronisation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.config import RagConfig
from rag.documents import SourceType, make_document_id
from rag.indexing import RagIndexer
from rag.manifest import MANIFEST_FILENAME, IndexManifest
from rag.stores import LocalVectorStore
from rag.tests.factories import (
    FakeEmbeddingProvider,
    FakeExperimentRun,
    FakeExperimentStore,
)

PREPROCESSING_ID = make_document_id(
    SourceType.PROJECT_DOCUMENTATION.value, "PREPROCESSING.md"
)


# --------------------------------------------------------------------------
# Documentation indexing
# --------------------------------------------------------------------------


def test_indexing_documentation_writes_chunks(indexer: RagIndexer) -> None:
    """The basic path: every allowed file becomes searchable chunks."""
    report = indexer.index_documentation()

    assert len(report.indexed) == 2
    assert report.chunks_written > 2
    assert indexer.store.count() == report.chunks_written
    assert PREPROCESSING_ID in report.indexed


def test_indexing_creates_a_manifest(indexer: RagIndexer, index_dir: Path) -> None:
    """The manifest is how the indexer knows what it has already done."""
    indexer.index_documentation()
    payload = json.loads((index_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))

    assert payload["document_count"] == 2
    assert payload["chunk_count"] > 2
    assert payload["embedding_identifier"] == "fake-32"
    assert payload["embedding_dimension"] == 32
    entry = next(
        item for item in payload["documents"] if item["document_id"] == PREPROCESSING_ID
    )
    assert entry["source_reference"] == "PREPROCESSING.md"
    assert entry["chunk_count"] == len(entry["chunk_ids"])
    assert entry["indexed_at"]


def test_the_manifest_holds_no_secret(indexer: RagIndexer, index_dir: Path) -> None:
    """Hashes, counts and identifiers only."""
    indexer.index_documentation()
    text = (index_dir / MANIFEST_FILENAME).read_text(encoding="utf-8").lower()

    for marker in ("api_key", "token", "password", "secret", "sk-"):
        assert marker not in text


def test_indexing_twice_creates_no_duplicates(indexer: RagIndexer) -> None:
    """Stable ids mean a repeat is an overwrite, not a second copy."""
    first = indexer.index_documentation()
    count = indexer.store.count()
    second = indexer.index_documentation()

    assert indexer.store.count() == count
    assert second.indexed == []
    assert len(second.skipped) == 2
    assert second.chunks_written == 0
    assert first.chunks_written > 0


def test_an_unchanged_document_is_not_re_embedded(
    indexer: RagIndexer, embeddings: FakeEmbeddingProvider
) -> None:
    """The manifest short-circuits the work before chunking, not after."""
    indexer.index_documentation()
    calls = embeddings.embed_document_calls
    indexer.index_documentation()

    assert embeddings.embed_document_calls == calls


def test_a_changed_document_is_reindexed(
    indexer: RagIndexer, docs_root: Path
) -> None:
    """New content must reach the index."""
    indexer.index_documentation()
    path = docs_root / "PREPROCESSING.md"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "\n## Scaling\n\nNumeric columns are standardised.\n",
        encoding="utf-8",
    )
    report = indexer.index_documentation()

    assert PREPROCESSING_ID in report.indexed
    assert report.chunks_written > 0
    assert any(
        "standardised" in (indexer.store.get(chunk_id).content)
        for chunk_id in indexer.store.chunk_ids_for_document(PREPROCESSING_ID)
    )


def test_a_shortened_document_leaves_no_orphan_chunks(
    indexer: RagIndexer, docs_root: Path
) -> None:
    """A deleted paragraph must disappear, not linger and still retrieve."""
    indexer.index_documentation()
    before = len(indexer.store.chunk_ids_for_document(PREPROCESSING_ID))

    (docs_root / "PREPROCESSING.md").write_text(
        "# Preprocessing Guide\n\nJust one short section now.\n", encoding="utf-8"
    )
    indexer.index_documentation()

    after = indexer.store.chunk_ids_for_document(PREPROCESSING_ID)
    assert len(after) < before
    assert all(
        "one-hot" not in indexer.store.get(chunk_id).content for chunk_id in after
    )


def test_a_removed_document_is_dropped_from_the_index(
    indexer: RagIndexer, docs_root: Path
) -> None:
    """Deleting a file removes it from retrieval on the next run."""
    indexer.index_documentation()
    (docs_root / "EVALUATION.md").unlink()
    report = indexer.index_documentation()

    evaluation_id = make_document_id(
        SourceType.PROJECT_DOCUMENTATION.value, "EVALUATION.md"
    )
    assert evaluation_id in report.removed
    assert evaluation_id not in indexer.store.document_ids()
    assert indexer.manifest.get(evaluation_id) is None


def test_forcing_reindexes_an_unchanged_document(indexer: RagIndexer) -> None:
    """The escape hatch when the chunker itself has changed."""
    indexer.index_documentation()
    report = indexer.index_documentation(force=True)

    assert len(report.indexed) == 2
    assert report.chunks_written > 0


def test_indexing_is_deterministic(config: RagConfig, index_dir: Path) -> None:
    """Two independent indexes of the same sources hold the same chunk ids."""

    def build(directory: Path) -> list[str]:
        """Index into a fresh directory and return the stored chunk ids."""
        store = LocalVectorStore(directory)
        RagIndexer(
            config.with_overrides(index_dir=directory),
            store=store,
            embeddings=FakeEmbeddingProvider(dimension=32),
        ).index_documentation()
        return sorted(
            chunk_id
            for document_id in store.document_ids()
            for chunk_id in store.chunk_ids_for_document(document_id)
        )

    assert build(index_dir / "a") == build(index_dir / "b")


# --------------------------------------------------------------------------
# Experiment indexing and synchronisation
# --------------------------------------------------------------------------


def test_indexing_experiments_writes_chunks(
    indexer: RagIndexer, experiment_store: FakeExperimentStore
) -> None:
    """Each experiment becomes several retrievable sections."""
    report = indexer.index_experiments(experiment_store)

    assert len(report.indexed) == 2
    assert report.chunks_written >= 4
    assert indexer.store.count() == report.chunks_written


def test_syncing_adds_a_new_experiment(
    indexer: RagIndexer, experiment_store: FakeExperimentStore
) -> None:
    """The operation to call after running experiments."""
    indexer.sync_experiments(experiment_store)
    before = indexer.store.count()

    experiment_store.add(
        FakeExperimentRun(experiment_id="exp_111222333444_20260103T080000Z_0003")
    )
    report = indexer.sync_experiments(experiment_store)

    assert len(report.indexed) == 1
    assert len(report.skipped) == 2
    assert indexer.store.count() > before


def test_syncing_updates_a_changed_experiment(
    indexer: RagIndexer, experiment_run: FakeExperimentRun
) -> None:
    """A record rewritten under the same id must not stay stale in the index."""
    store = FakeExperimentStore([experiment_run])
    indexer.sync_experiments(store)

    store.runs = [
        FakeExperimentRun(
            experiment_id=experiment_run.experiment_id, test_score=0.99
        )
    ]
    report = indexer.sync_experiments(store)

    document_id = make_document_id(
        SourceType.EXPERIMENT.value, experiment_run.experiment_id
    )
    contents = " ".join(
        indexer.store.get(chunk_id).content
        for chunk_id in indexer.store.chunk_ids_for_document(document_id)
    )

    assert document_id in report.indexed
    assert "0.9900" in contents
    assert "0.8500" not in contents


def test_syncing_prunes_a_deleted_experiment(
    indexer: RagIndexer, experiment_store: FakeExperimentStore
) -> None:
    """An experiment removed from the store leaves the index too."""
    indexer.sync_experiments(experiment_store)
    removed_id = experiment_store.runs[0].experiment_id
    experiment_store.remove(removed_id)

    report = indexer.sync_experiments(experiment_store)
    document_id = make_document_id(SourceType.EXPERIMENT.value, removed_id)

    assert document_id in report.removed
    assert document_id not in indexer.store.document_ids()


def test_syncing_can_keep_experiments_the_store_no_longer_has(
    indexer: RagIndexer, experiment_store: FakeExperimentStore
) -> None:
    """Pruning is the default, not the only option."""
    indexer.sync_experiments(experiment_store)
    experiment_store.remove(experiment_store.runs[0].experiment_id)
    report = indexer.sync_experiments(experiment_store, prune=False)

    assert report.removed == []
    assert len(indexer.store.document_ids()) == 2


def test_experiments_and_documentation_share_one_index(
    indexer: RagIndexer, experiment_store: FakeExperimentStore
) -> None:
    """One search should be able to reach both kinds of knowledge."""
    indexer.index_documentation()
    indexer.sync_experiments(experiment_store)
    stats = indexer.stats()

    assert stats["chunks_by_source_type"]["project_documentation"] > 0
    assert stats["chunks_by_source_type"]["experiment"] > 0
    assert stats["document_count"] == 4


# --------------------------------------------------------------------------
# Rebuilding and embedding changes
# --------------------------------------------------------------------------


def test_rebuilding_starts_from_nothing(
    indexer: RagIndexer, experiment_store: FakeExperimentStore
) -> None:
    """The way out of any inconsistency."""
    indexer.index_documentation()
    report = indexer.rebuild(experiment_store)

    assert report.rebuilt is True
    assert len(report.indexed) == 4
    assert indexer.store.count() > 0


def test_a_changed_embedding_provider_rebuilds_rather_than_mixes(
    config: RagConfig, index_dir: Path
) -> None:
    """Two embedding spaces in one index would produce confident nonsense."""
    store = LocalVectorStore(index_dir)
    RagIndexer(
        config, store=store, embeddings=FakeEmbeddingProvider(dimension=32)
    ).index_documentation()
    first_count = store.count()

    second = RagIndexer(
        config,
        store=LocalVectorStore(index_dir),
        embeddings=FakeEmbeddingProvider(dimension=16, name="other"),
    )
    report = second.index_documentation()

    assert report.rebuilt is True
    assert second.store.dimension == 16
    assert second.store.count() == first_count
    assert second.manifest.embedding_identifier == "other-16"


def test_removing_one_document_leaves_the_others(
    indexer: RagIndexer,
) -> None:
    """Targeted removal, for a source that should no longer be searchable."""
    indexer.index_documentation()
    deleted = indexer.remove_document(PREPROCESSING_ID)

    assert deleted > 0
    assert PREPROCESSING_ID not in indexer.store.document_ids()
    assert indexer.store.count() > 0


# --------------------------------------------------------------------------
# The manifest itself
# --------------------------------------------------------------------------


def test_an_absent_manifest_is_an_empty_one(tmp_path: Path) -> None:
    """An index that has never been built is not an error."""
    manifest = IndexManifest.load(tmp_path)

    assert manifest.entries == {}
    assert manifest.total_chunks == 0
    assert manifest.matches_embeddings("anything") is True


def test_a_manifest_round_trips(tmp_path: Path) -> None:
    """What is written is what comes back."""
    manifest = IndexManifest(embedding_identifier="fake-32", embedding_dimension=32)
    manifest.record(
        document_id="docs:one",
        source_type="project_documentation",
        source_reference="ONE.md",
        source_title="One",
        source_hash="abc123",
        chunk_ids=["docs:one#0000-aa", "docs:one#0001-bb"],
    )
    manifest.save(tmp_path)

    restored = IndexManifest.load(tmp_path)
    entry = restored.get("docs:one")

    assert restored.embedding_identifier == "fake-32"
    assert entry.chunk_count == 2
    assert entry.source_hash == "abc123"
    assert restored.total_chunks == 2


def test_the_manifest_decides_what_needs_work(tmp_path: Path) -> None:
    """Unchanged means skip; changed or unknown means index."""
    manifest = IndexManifest()
    manifest.record(
        document_id="docs:one",
        source_type="project_documentation",
        source_reference="ONE.md",
        source_title="One",
        source_hash="abc123",
        chunk_ids=["docs:one#0000-aa"],
    )

    assert manifest.needs_reindex("docs:one", "abc123") is False
    assert manifest.needs_reindex("docs:one", "def456") is True
    assert manifest.needs_reindex("docs:two", "abc123") is True


def test_a_corrupt_manifest_is_reported(tmp_path: Path) -> None:
    """Rebuilding is the fix, and the error says so."""
    (tmp_path / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")

    from rag.errors import CorruptIndexError

    with pytest.raises(CorruptIndexError, match="Rebuild"):
        IndexManifest.load(tmp_path)


def test_the_manifest_survives_a_new_indexer(
    config: RagConfig, index_dir: Path
) -> None:
    """A second process must know what the first already indexed."""
    RagIndexer(
        config,
        store=LocalVectorStore(index_dir),
        embeddings=FakeEmbeddingProvider(dimension=32),
    ).index_documentation()

    reopened = RagIndexer(
        config,
        store=LocalVectorStore(index_dir),
        embeddings=FakeEmbeddingProvider(dimension=32),
    )
    report = reopened.index_documentation()

    assert len(reopened.manifest.entries) == 2
    assert report.indexed == []
    assert len(report.skipped) == 2


def test_an_empty_document_is_removed_rather_than_indexed(
    indexer: RagIndexer, docs_root: Path
) -> None:
    """A file emptied of content should not keep its old chunks."""
    indexer.index_documentation()
    (docs_root / "EVALUATION.md").write_text("   \n\n", encoding="utf-8")
    report = indexer.index_documentation()

    evaluation_id = make_document_id(
        SourceType.PROJECT_DOCUMENTATION.value, "EVALUATION.md"
    )
    assert evaluation_id in report.removed
    assert evaluation_id not in indexer.store.document_ids()


def test_the_report_summarises_what_happened(indexer: RagIndexer) -> None:
    """Readable by a person and serialisable for a log."""
    payload = indexer.index_documentation().as_dict()

    json.dumps(payload)
    assert payload["indexed_count"] == 2
    assert payload["chunks_written"] > 0
    assert payload["skipped_count"] == 0
