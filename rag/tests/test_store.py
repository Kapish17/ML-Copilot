"""Tests for embedding providers and the persistent vector store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rag.config import RagConfig
from rag.documents import Chunk, SourceType
from rag.embeddings import (
    AVAILABLE_PROVIDERS,
    EmbeddingProvider,
    HashingEmbeddingProvider,
    build_embedding_provider,
)
from rag.embeddings.base import normalise
from rag.errors import (
    ConfigurationError,
    CorruptIndexError,
    EmbeddingDimensionError,
    EmbeddingProviderUnavailableError,
)
from rag.stores import LocalVectorStore, MetadataFilter, VectorRecord, VectorStore
from rag.tests.factories import FakeEmbeddingProvider


def make_chunk(
    chunk_id: str,
    content: str = "text",
    *,
    document_id: str = "docs:one",
    source_type: str = SourceType.PROJECT_DOCUMENTATION.value,
    **metadata,
) -> Chunk:
    """Build a chunk for a store test."""
    return Chunk(
        document_id=document_id,
        chunk_id=chunk_id,
        content=content,
        source_type=source_type,
        source_title="Test document",
        source_reference="TEST.md",
        citation=f"docs:test#{chunk_id}",
        metadata=metadata,
    )


def record(chunk: Chunk, vector: list[float]) -> VectorRecord:
    """Pair a chunk with a unit-length vector."""
    return VectorRecord(chunk=chunk, vector=normalise(np.array([vector]))[0])


# --------------------------------------------------------------------------
# Embedding providers
# --------------------------------------------------------------------------


def test_the_default_provider_satisfies_the_interface() -> None:
    """Retrieval depends on the protocol, so a provider must satisfy it."""
    provider = HashingEmbeddingProvider(dimension=64)

    assert isinstance(provider, EmbeddingProvider)
    assert provider.dimension == 64
    assert provider.identifier == "hashing-64"


def test_the_default_provider_loads_nothing_until_it_embeds() -> None:
    """Constructing a provider must not build or download anything."""
    provider = HashingEmbeddingProvider(dimension=64)

    assert provider._word_vectorizer is None, "no vectoriser before first use"
    provider.embed_query("anything")
    assert provider._word_vectorizer is not None


def test_every_vector_has_the_declared_dimension() -> None:
    """The store, the manifest and the search all rely on this."""
    provider = HashingEmbeddingProvider(dimension=48)
    documents = provider.embed_documents(["one text", "another text", ""])

    assert documents.shape == (3, 48)
    assert provider.embed_query("a question").shape == (48,)


def test_vectors_are_unit_length_so_a_dot_product_is_a_cosine() -> None:
    """The store never normalises again; it relies on this."""
    provider = HashingEmbeddingProvider(dimension=64)
    vectors = provider.embed_documents(["leakage prevention", "one-hot encoding"])

    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_embedding_is_deterministic_across_instances() -> None:
    """An index built yesterday must stay comparable to a query asked today."""
    first = HashingEmbeddingProvider(dimension=64).embed_query("data leakage")
    second = HashingEmbeddingProvider(dimension=64).embed_query("data leakage")

    assert np.array_equal(first, second)


def test_shared_words_score_higher_than_unrelated_text() -> None:
    """The default provider matches on term overlap; this is that property."""
    provider = HashingEmbeddingProvider(dimension=256)
    query = provider.embed_query("how is data leakage prevented")
    related, unrelated = provider.embed_documents(
        [
            "Leakage is prevented by fitting every transformer on training rows.",
            "Docker Compose describes the local container stack.",
        ]
    )

    assert float(query @ related) > float(query @ unrelated)


def test_empty_text_embeds_without_producing_nan() -> None:
    """A zero vector scores zero against everything; NaN would poison ranking."""
    vectors = HashingEmbeddingProvider(dimension=32).embed_documents(["", "   "])

    assert not np.isnan(vectors).any()


def test_normalising_leaves_a_zero_row_alone() -> None:
    """Dividing by a zero norm is the bug this prevents."""
    result = normalise(np.array([[0.0, 0.0], [3.0, 4.0]]))

    assert np.array_equal(result[0], np.array([0.0, 0.0], dtype=np.float32))
    assert np.allclose(result[1], np.array([0.6, 0.8]))


def test_the_provider_is_chosen_by_configuration() -> None:
    """Swapping providers is a setting, not a code change."""
    provider = build_embedding_provider(RagConfig(embedding_dimension=128))

    assert isinstance(provider, HashingEmbeddingProvider)
    assert provider.dimension == 128


def test_an_unknown_provider_name_is_refused() -> None:
    """With the available names, so the mistake is easy to fix."""
    with pytest.raises(ConfigurationError) as exc_info:
        build_embedding_provider(RagConfig(embedding_provider="magic"))

    assert exc_info.value.details["available"] == list(AVAILABLE_PROVIDERS)


def test_the_optional_provider_imports_without_its_dependency() -> None:
    """The module must load even where PyTorch is not installed."""
    from rag.embeddings.sentence_transformer import (
        DEFAULT_DIMENSION,
        DEFAULT_MODEL,
        SentenceTransformerEmbeddingProvider,
    )

    provider = SentenceTransformerEmbeddingProvider(dimension=DEFAULT_DIMENSION)

    assert provider.is_loaded is False, "constructing must not load the model"
    assert provider.dimension == DEFAULT_DIMENSION
    assert "MiniLM" in DEFAULT_MODEL


def test_the_optional_provider_explains_itself_when_unavailable() -> None:
    """A missing optional dependency is a message, not an import traceback."""
    from rag.embeddings.sentence_transformer import SentenceTransformerEmbeddingProvider

    provider = SentenceTransformerEmbeddingProvider(model_name="not/a-real-model")
    try:
        provider.embed_query("anything")
    except EmbeddingProviderUnavailableError as exc:
        assert "sentence-transformers" in exc.message or "Could not load" in exc.message
    else:  # pragma: no cover - only when the model is genuinely available
        pytest.skip("the model is installed and cached, so loading succeeded")


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------


def test_the_local_store_satisfies_the_interface() -> None:
    """Retrieval depends on the protocol, not on this implementation."""
    assert isinstance(LocalVectorStore(Path("unused")), VectorStore)


def test_constructing_a_store_touches_nothing(tmp_path: Path) -> None:
    """No directory is created until something is written."""
    directory = tmp_path / "index"
    LocalVectorStore(directory)

    assert not directory.exists()


def test_an_empty_store_answers_rather_than_failing(store: LocalVectorStore) -> None:
    """Searching before anything is indexed is a normal thing to do."""
    assert store.count() == 0
    assert store.dimension is None
    assert store.search(np.zeros(4), top_k=5) == []
    assert store.document_ids() == ()


def test_upsert_stores_chunks_and_their_vectors(store: LocalVectorStore) -> None:
    """The basic write path."""
    written = store.upsert(
        [
            record(make_chunk("a", "leakage prevention"), [1.0, 0.0]),
            record(make_chunk("b", "one-hot encoding"), [0.0, 1.0]),
        ]
    )

    assert written == 2
    assert store.count() == 2
    assert store.dimension == 2
    assert store.get("a").content == "leakage prevention"


def test_upsert_replaces_rather_than_duplicates(store: LocalVectorStore) -> None:
    """Writing the same chunk id twice leaves one row, with the newer content."""
    store.upsert([record(make_chunk("a", "first"), [1.0, 0.0])])
    store.upsert([record(make_chunk("a", "second"), [0.0, 1.0])])

    assert store.count() == 1
    assert store.get("a").content == "second"


def test_search_ranks_by_cosine_similarity(store: LocalVectorStore) -> None:
    """Closest first; the metric is documented and it is cosine."""
    store.upsert(
        [
            record(make_chunk("near", "near"), [1.0, 0.1]),
            record(make_chunk("far", "far"), [0.0, 1.0]),
        ]
    )
    hits = store.search(normalise(np.array([[1.0, 0.0]]))[0], top_k=2)

    assert [hit.chunk.chunk_id for hit in hits] == ["near", "far"]
    assert hits[0].score > hits[1].score
    assert store.similarity_metric == "cosine"


def test_search_honours_top_k(store: LocalVectorStore) -> None:
    """A caller gets what it asked for, not the whole index."""
    store.upsert(
        [record(make_chunk(str(index), "x"), [1.0, index / 10]) for index in range(10)]
    )

    assert len(store.search(np.array([1.0, 0.0]), top_k=3)) == 3
    assert len(store.search(np.array([1.0, 0.0]), top_k=100)) == 10


def test_search_applies_a_minimum_score(store: LocalVectorStore) -> None:
    """The threshold trades recall for precision."""
    store.upsert(
        [
            record(make_chunk("same", "same"), [1.0, 0.0]),
            record(make_chunk("orthogonal", "orthogonal"), [0.0, 1.0]),
        ]
    )
    hits = store.search(np.array([1.0, 0.0]), top_k=5, min_score=0.5)

    assert [hit.chunk.chunk_id for hit in hits] == ["same"]


def test_search_is_stable_for_ties(store: LocalVectorStore) -> None:
    """Identical query, identical index, identical ranking, every time."""
    store.upsert(
        [record(make_chunk(name, "identical text"), [1.0, 0.0]) for name in "abcde"]
    )
    query = np.array([1.0, 0.0])

    first = [hit.chunk.chunk_id for hit in store.search(query, top_k=5)]
    second = [hit.chunk.chunk_id for hit in store.search(query, top_k=5)]

    assert first == second == list("abcde")


def test_a_metadata_filter_restricts_the_candidates(store: LocalVectorStore) -> None:
    """Filtering happens before ranking, so top_k is filled from the subset."""
    store.upsert(
        [
            record(
                make_chunk("c1", "x", source_type="experiment", task_type="classification"),
                [1.0, 0.0],
            ),
            record(
                make_chunk("r1", "x", source_type="experiment", task_type="regression"),
                [1.0, 0.0],
            ),
            record(
                make_chunk("r2", "x", source_type="experiment", task_type="regression"),
                [0.9, 0.1],
            ),
        ]
    )
    hits = store.search(
        np.array([1.0, 0.0]),
        top_k=2,
        metadata_filter=MetadataFilter(equals={"task_type": "regression"}),
    )

    assert [hit.chunk.chunk_id for hit in hits] == ["r1", "r2"]


def test_a_filter_on_a_missing_key_matches_nothing(store: LocalVectorStore) -> None:
    """An absent value is not a wildcard."""
    store.upsert([record(make_chunk("a", "x"), [1.0, 0.0])])
    hits = store.search(
        np.array([1.0, 0.0]),
        top_k=5,
        metadata_filter=MetadataFilter(equals={"task_type": "classification"}),
    )

    assert hits == []


def test_a_filter_can_match_any_of_several_values(store: LocalVectorStore) -> None:
    """Used for 'these experiment ids'."""
    store.upsert(
        [
            record(make_chunk("a", "x", model="rf"), [1.0, 0.0]),
            record(make_chunk("b", "x", model="lr"), [1.0, 0.0]),
            record(make_chunk("c", "x", model="hgb"), [1.0, 0.0]),
        ]
    )
    hits = store.search(
        np.array([1.0, 0.0]),
        top_k=5,
        metadata_filter=MetadataFilter(any_of={"model": ("rf", "hgb")}),
    )

    assert sorted(hit.chunk.chunk_id for hit in hits) == ["a", "c"]


def test_a_filter_can_use_a_chunk_field_as_well_as_metadata(
    store: LocalVectorStore,
) -> None:
    """A caller should not have to know where the adapter put a value."""
    store.upsert(
        [
            record(make_chunk("a", "x", source_type="experiment"), [1.0, 0.0]),
            record(make_chunk("b", "x"), [1.0, 0.0]),
        ]
    )
    hits = store.search(
        np.array([1.0, 0.0]),
        top_k=5,
        metadata_filter=MetadataFilter(source_types=("experiment",)),
    )

    assert [hit.chunk.chunk_id for hit in hits] == ["a"]


def test_count_respects_a_filter(store: LocalVectorStore) -> None:
    """Used to tell 'nothing indexed' from 'nothing matched'."""
    store.upsert(
        [
            record(make_chunk("a", "x", source_type="experiment"), [1.0, 0.0]),
            record(make_chunk("b", "x"), [1.0, 0.0]),
        ]
    )

    assert store.count() == 2
    assert store.count(MetadataFilter(source_types=("experiment",))) == 1


def test_deleting_by_chunk_id_removes_only_those_rows(store: LocalVectorStore) -> None:
    """The matrix and the record list stay in step."""
    store.upsert(
        [record(make_chunk(name, name), [1.0, 0.0]) for name in ("a", "b", "c")]
    )

    assert store.delete(["b", "missing"]) == 1
    assert store.count() == 2
    assert store.get("b") is None
    assert len(store.search(np.array([1.0, 0.0]), top_k=5)) == 2


def test_deleting_a_document_removes_all_of_its_chunks(store: LocalVectorStore) -> None:
    """How a changed source is cleared before being rewritten."""
    store.upsert(
        [
            record(make_chunk("a", "x", document_id="docs:one"), [1.0, 0.0]),
            record(make_chunk("b", "x", document_id="docs:one"), [1.0, 0.0]),
            record(make_chunk("c", "x", document_id="docs:two"), [1.0, 0.0]),
        ]
    )

    assert store.delete_document("docs:one") == 2
    assert store.document_ids() == ("docs:two",)


def test_clear_empties_the_store_and_the_disk(store: LocalVectorStore) -> None:
    """Nothing left behind to be half-loaded next time."""
    store.upsert([record(make_chunk("a", "x"), [1.0, 0.0])])
    store.save()
    store.clear()

    assert store.count() == 0
    assert not store.vectors_path.exists()
    assert not store.records_path.exists()


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_the_index_survives_a_new_store_instance(index_dir: Path) -> None:
    """The whole point of a persistent store: reopening is enough."""
    writer = LocalVectorStore(index_dir)
    writer.upsert(
        [
            record(make_chunk("a", "leakage prevention", topic="leakage"), [1.0, 0.0]),
            record(make_chunk("b", "one-hot encoding", topic="encoding"), [0.0, 1.0]),
        ]
    )
    writer.save()

    reader = LocalVectorStore(index_dir)
    hits = reader.search(np.array([1.0, 0.0]), top_k=1)

    assert reader.count() == 2
    assert reader.dimension == 2
    assert hits[0].chunk.chunk_id == "a"
    assert hits[0].chunk.content == "leakage prevention"
    assert hits[0].chunk.metadata["topic"] == "leakage"
    assert hits[0].chunk.citation == "docs:test#a"


def test_the_stored_records_are_readable_json(index_dir: Path) -> None:
    """A record file that can be opened in an editor when something looks wrong."""
    store = LocalVectorStore(index_dir)
    store.upsert([record(make_chunk("a", "text"), [1.0, 0.0])])
    store.save()

    lines = store.records_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[0])

    assert len(lines) == 1
    assert payload["chunk_id"] == "a"
    assert "vector" not in payload, "vectors live in the matrix, not the records"


def test_saving_twice_does_not_grow_the_index(index_dir: Path) -> None:
    """Saving is a snapshot, not an append."""
    store = LocalVectorStore(index_dir)
    store.upsert([record(make_chunk("a", "text"), [1.0, 0.0])])
    store.save()
    first = store.records_path.stat().st_size
    store.save()

    assert store.records_path.stat().st_size == first


def test_a_deleted_chunk_stays_deleted_after_a_reload(index_dir: Path) -> None:
    """Deletion compacts both files, not just the one."""
    writer = LocalVectorStore(index_dir)
    writer.upsert(
        [record(make_chunk(name, name), [1.0, 0.0]) for name in ("a", "b", "c")]
    )
    writer.delete(["b"])
    writer.save()

    reader = LocalVectorStore(index_dir)

    assert reader.count() == 2
    assert reader.get("b") is None


# --------------------------------------------------------------------------
# Corruption and mismatch
# --------------------------------------------------------------------------


def test_an_unreadable_record_file_is_reported_not_guessed(index_dir: Path) -> None:
    """Returning the wrong text for the right score would be worse."""
    store = LocalVectorStore(index_dir)
    store.upsert([record(make_chunk("a", "text"), [1.0, 0.0])])
    store.save()
    store.records_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(CorruptIndexError, match="records"):
        LocalVectorStore(index_dir).count()


def test_a_mismatched_index_is_reported(index_dir: Path) -> None:
    """A matrix and a record file that disagree cannot both be right."""
    store = LocalVectorStore(index_dir)
    store.upsert(
        [record(make_chunk(name, name), [1.0, 0.0]) for name in ("a", "b", "c")]
    )
    store.save()
    lines = store.records_path.read_text(encoding="utf-8").splitlines()
    store.records_path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")

    with pytest.raises(CorruptIndexError) as exc_info:
        LocalVectorStore(index_dir).count()

    assert exc_info.value.details["vector_rows"] == 3
    assert exc_info.value.details["record_count"] == 2


def test_a_half_present_index_is_reported(index_dir: Path) -> None:
    """One file without the other is an incomplete index, not an empty one."""
    store = LocalVectorStore(index_dir)
    store.upsert([record(make_chunk("a", "text"), [1.0, 0.0])])
    store.save()
    store.vectors_path.unlink()

    with pytest.raises(CorruptIndexError, match="incomplete"):
        LocalVectorStore(index_dir).count()


def test_writing_a_different_dimension_is_refused(store: LocalVectorStore) -> None:
    """Mixing two embedding spaces would produce confident nonsense."""
    store.upsert([record(make_chunk("a", "text"), [1.0, 0.0])])

    with pytest.raises(EmbeddingDimensionError) as exc_info:
        store.upsert([record(make_chunk("b", "text"), [1.0, 0.0, 0.0])])

    assert exc_info.value.details["index_dimension"] == 2


def test_querying_with_a_different_dimension_is_refused(store: LocalVectorStore) -> None:
    """The same protection on the read path, with the same advice."""
    store.upsert([record(make_chunk("a", "text"), [1.0, 0.0])])

    with pytest.raises(EmbeddingDimensionError, match="rebuild"):
        store.search(np.array([1.0, 0.0, 0.0]), top_k=1)


def test_the_fake_provider_behaves_like_a_real_one() -> None:
    """The test double must not be looser than what it stands in for."""
    provider = FakeEmbeddingProvider(dimension=16)

    assert isinstance(provider, EmbeddingProvider)
    assert provider.is_loaded is False
    vectors = provider.embed_documents(["leakage prevention", "one-hot encoding"])
    assert vectors.shape == (2, 16)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert provider.is_loaded is True
