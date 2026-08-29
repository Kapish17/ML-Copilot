"""Building and maintaining the index.

The indexer is the only thing that writes to the vector store. It takes
documents from the ingestion adapters, chunks them, embeds the chunks and
upserts them — and it uses the manifest to avoid doing any of that when
nothing has changed.

::

    Document -> chunk -> embed -> upsert -> manifest entry

Three properties matter more than speed here.

**Indexing twice is not indexing twice.** Document and chunk ids are derived
from content and position, so re-indexing an unchanged source overwrites the
same rows with the same values. The manifest short-circuits it earlier still:
an unchanged source hash is skipped without being chunked.

**A changed source leaves nothing stale behind.** When a document's hash
changes, its old chunks are deleted before the new ones are written — not
merged with them. A paragraph deleted from a README disappears from the index
rather than lingering as an orphan that still retrieves.

**A changed embedding provider invalidates everything.** Vectors from two
providers are not comparable, so an index whose manifest records a different
provider is rebuilt rather than added to.

Experiment synchronisation is one-directional by design: this module reads the
experiment store, and ``ml/experiments`` has no idea the index exists. Nothing
about recording an experiment depends on RAG being present or working.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag.chunking import chunk_document
from rag.config import RagConfig
from rag.documents import Chunk, Document, SourceType
from rag.embeddings import EmbeddingProvider, build_embedding_provider
from rag.embeddings.base import batched
from rag.ingestion.documentation import load_documentation
from rag.ingestion.experiments import ExperimentStoreLike, load_experiments
from rag.manifest import IndexManifest
from rag.stores import LocalVectorStore, VectorRecord, VectorStore

logger = logging.getLogger(__name__)


@dataclass
class IndexReport:
    """What one indexing run did.

    Every number is a count of documents, not chunks, except where the name
    says otherwise — so a report reads as "four documents were unchanged, one
    was updated".
    """

    indexed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    chunks_written: int = 0
    chunks_deleted: int = 0
    rebuilt: bool = False

    @property
    def changed(self) -> bool:
        """True when the index was modified in any way."""
        return bool(self.indexed or self.removed or self.rebuilt)

    def extend(self, other: IndexReport) -> IndexReport:
        """Fold another report into this one, and return this one."""
        self.indexed.extend(other.indexed)
        self.skipped.extend(other.skipped)
        self.removed.extend(other.removed)
        self.failed.extend(other.failed)
        self.chunks_written += other.chunks_written
        self.chunks_deleted += other.chunks_deleted
        self.rebuilt = self.rebuilt or other.rebuilt
        return self

    def as_dict(self) -> dict[str, Any]:
        """Render the report as plain JSON-safe values."""
        return {
            "indexed": list(self.indexed),
            "skipped": list(self.skipped),
            "removed": list(self.removed),
            "failed": list(self.failed),
            "indexed_count": len(self.indexed),
            "skipped_count": len(self.skipped),
            "removed_count": len(self.removed),
            "chunks_written": self.chunks_written,
            "chunks_deleted": self.chunks_deleted,
            "rebuilt": self.rebuilt,
        }


class RagIndexer:
    """Chunks, embeds and stores documents, incrementally."""

    def __init__(
        self,
        config: RagConfig,
        *,
        store: VectorStore | None = None,
        embeddings: EmbeddingProvider | None = None,
    ) -> None:
        """Wire the indexer to its collaborators.

        Args:
            config: Every chunking, embedding and storage setting.
            store: Where vectors go. Defaults to a local store in the
                configured index directory.
            embeddings: How text becomes vectors. Defaults to the provider
                the configuration names; nothing is loaded until first use.
        """
        self._config = config
        self._store = store if store is not None else LocalVectorStore(config.index_dir)
        self._embeddings = (
            embeddings if embeddings is not None else build_embedding_provider(config)
        )
        self._manifest = IndexManifest.load(config.index_dir)

    @property
    def config(self) -> RagConfig:
        """The configuration this indexer runs with."""
        return self._config

    @property
    def store(self) -> VectorStore:
        """The vector store being written to."""
        return self._store

    @property
    def embeddings(self) -> EmbeddingProvider:
        """The embedding provider in use."""
        return self._embeddings

    @property
    def manifest(self) -> IndexManifest:
        """The manifest, as currently held in memory."""
        return self._manifest

    # -- Core ---------------------------------------------------------------

    def _embed_chunks(self, chunks: Sequence[Chunk]) -> list[VectorRecord]:
        """Embed chunks in configured batches, preserving order."""
        records: list[VectorRecord] = []
        for batch in batched(list(chunks), self._config.embedding_batch_size):
            vectors = self._embeddings.embed_documents([chunk.content for chunk in batch])
            records.extend(
                VectorRecord(chunk=chunk, vector=vectors[offset])
                for offset, chunk in enumerate(batch)
            )
        return records

    def _ensure_embedding_space(self) -> bool:
        """Clear the index if it was built with a different provider.

        Returns:
            bool: True when the index was cleared for this reason.
        """
        identifier = self._embeddings.identifier
        if self._manifest.matches_embeddings(identifier):
            return False
        logger.warning(
            "The index was built with embedding provider %r and is now being "
            "written with %r; rebuilding.",
            self._manifest.embedding_identifier,
            identifier,
        )
        self._store.clear()
        self._manifest.clear()
        return True

    def index_documents(
        self, documents: Iterable[Document], *, force: bool = False
    ) -> IndexReport:
        """Index a set of documents, skipping the unchanged ones.

        Args:
            documents: What to index.
            force: Re-chunk and re-embed even when the hash is unchanged.

        Returns:
            IndexReport: What was written, skipped, removed and what failed.
        """
        report = IndexReport(rebuilt=self._ensure_embedding_space())

        for document in documents:
            try:
                report.extend(self._index_one(document, force=force))
            except Exception as exc:  # noqa: BLE001 - one bad source, not all
                logger.warning(
                    "Failed to index %s: %s", document.document_id, type(exc).__name__
                )
                report.failed.append(document.document_id)

        self._finalise(report)
        return report

    def _index_one(self, document: Document, *, force: bool) -> IndexReport:
        """Index one document, replacing anything stored under it."""
        report = IndexReport()
        source_hash = document.source_hash

        if not force and not self._manifest.needs_reindex(
            document.document_id, source_hash
        ):
            report.skipped.append(document.document_id)
            return report

        chunks = chunk_document(document, self._config)
        if not chunks:
            # An empty source should not leave stale chunks behind either.
            removed = self._store.delete_document(document.document_id)
            if removed:
                report.chunks_deleted += removed
            self._manifest.forget(document.document_id)
            report.removed.append(document.document_id)
            return report

        # Delete first, then write: a shortened document must not keep the
        # chunks that used to sit past its new end.
        report.chunks_deleted += self._store.delete_document(document.document_id)
        records = self._embed_chunks(chunks)
        report.chunks_written += self._store.upsert(records)

        self._manifest.record(
            document_id=document.document_id,
            source_type=document.source_type,
            source_reference=document.source_reference,
            source_title=document.source_title,
            source_hash=source_hash,
            chunk_ids=[chunk.chunk_id for chunk in chunks],
        )
        report.indexed.append(document.document_id)
        return report

    def _finalise(self, report: IndexReport) -> None:
        """Persist the store and the manifest after a run that changed either."""
        self._manifest.embedding_identifier = self._embeddings.identifier
        self._manifest.embedding_dimension = self._store.dimension
        if hasattr(self._store, "save"):
            self._store.save()
        self._manifest.save(self._config.index_dir)

    # -- Sources -----------------------------------------------------------

    def index_documentation(self, *, force: bool = False) -> IndexReport:
        """Index the configured project documentation.

        Documents that no longer exist are dropped from the index, so
        deleting a file removes it from retrieval on the next run.
        """
        documents = list(load_documentation(self._config))
        report = self.index_documents(documents, force=force)

        present = {document.document_id for document in documents}
        for document_id in self._manifest.document_ids_for(
            SourceType.PROJECT_DOCUMENTATION.value
        ):
            if document_id not in present:
                report.chunks_deleted += self._store.delete_document(document_id)
                self._manifest.forget(document_id)
                report.removed.append(document_id)
        if report.removed:
            self._finalise(report)
        return report

    def index_experiments(
        self, store: ExperimentStoreLike, *, force: bool = False
    ) -> IndexReport:
        """Index every experiment the given store holds."""
        return self.index_documents(load_experiments(store, self._config), force=force)

    def sync_experiments(
        self, store: ExperimentStoreLike, *, prune: bool = True
    ) -> IndexReport:
        """Bring the index in step with the experiment store.

        New runs are added, changed runs are updated, and — with ``prune`` —
        runs no longer in the store are removed from the index. This is the
        operation to call after running experiments; it reads the store and
        writes the index, never the other way round.

        Args:
            store: The experiment store to read.
            prune: Whether to drop indexed experiments the store no longer
                has.

        Returns:
            IndexReport: What changed.
        """
        documents = list(load_experiments(store, self._config))
        report = self.index_documents(documents)

        if prune:
            present = {document.document_id for document in documents}
            for document_id in self._manifest.document_ids_for(
                SourceType.EXPERIMENT.value
            ):
                if document_id not in present:
                    report.chunks_deleted += self._store.delete_document(document_id)
                    self._manifest.forget(document_id)
                    report.removed.append(document_id)
            if report.removed:
                self._finalise(report)
        return report

    # -- Whole-index operations --------------------------------------------

    def rebuild(self, store: ExperimentStoreLike | None = None) -> IndexReport:
        """Discard the index and build it again from every source.

        The way out of any inconsistency: a changed embedding provider, a
        corrupt index that has been deleted, or simply wanting certainty.
        """
        self._store.clear()
        self._manifest.clear()

        report = IndexReport(rebuilt=True)
        report.extend(self.index_documentation(force=True))
        if store is not None:
            report.extend(self.index_experiments(store, force=True))
        report.rebuilt = True
        self._finalise(report)
        return report

    def remove_document(self, document_id: str) -> int:
        """Remove one document from the index and the manifest.

        Returns:
            int: How many chunks were deleted.
        """
        deleted = self._store.delete_document(document_id)
        self._manifest.forget(document_id)
        self._finalise(IndexReport())
        return deleted

    def stats(self) -> dict[str, Any]:
        """Describe the index: sizes, sources and the embedding identity."""
        payload: dict[str, Any] = {
            "embedding_identifier": self._embeddings.identifier,
            "embedding_dimension": self._store.dimension,
            "document_count": len(self._manifest.entries),
            "chunk_count": self._store.count(),
            "index_dir": str(self._config.index_dir),
        }
        if hasattr(self._store, "stats"):
            payload.update(self._store.stats())
        return payload


__all__ = ["IndexReport", "RagIndexer"]
