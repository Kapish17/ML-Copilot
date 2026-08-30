"""A persistent vector index on the local filesystem.

Three files in one directory, kept in step with each other:

::

    rag/index/
      vectors.npy    (rows, dimension) float32, one row per chunk
      records.jsonl  one JSON object per chunk, in the same row order
      manifest.json  what was indexed, from what, and with which embeddings

Row *i* of the matrix is the embedding of the chunk on line *i* of the record
file. That correspondence is the whole design, and it is checked on load: a
matrix and a record file of different lengths would return the wrong text for
the right score, which is worse than refusing to open.

**Similarity is cosine.** Every provider returns unit-length vectors, so the
cosine similarity of two vectors is their dot product, and searching is one
matrix-vector product over the candidate rows. Scores run from ``1.0``
(identical direction) through ``0.0`` (nothing in common) to ``-1.0``. In
practice the term-overlap default never produces a negative score for real
text, and ``0.0`` means "shares nothing at all".

**Metadata filtering happens first.** The filter selects candidate rows and
the dot product runs over those rows only, so asking for five classification
experiments searches classification experiments — it does not rank everything
and then throw most of it away.

**Writes are atomic.** Both files are written to temporary names, flushed,
``fsync``ed and moved into place with ``os.replace``, and the matrix is
written before the records. A process killed mid-save leaves the previous
complete index, never a half-written one.

This is a real index on disk, not a dictionary that forgets: it survives
process restarts, and reopening the directory is all that is needed to query
it again. It is deliberately simple — an exact, brute-force scan rather than
an approximate nearest-neighbour structure — which is the right trade for
thousands of chunks and the wrong one for millions. Replacing it means
implementing :class:`~rag.stores.base.VectorStore` elsewhere; nothing above
this module knows how any of it is stored. **Qdrant is not implemented.**
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from rag.documents import Chunk
from rag.embeddings.base import VECTOR_DTYPE
from rag.errors import CorruptIndexError, EmbeddingDimensionError
from rag.stores.base import MetadataFilter, SearchHit, VectorRecord

logger = logging.getLogger(__name__)

#: The matrix of embeddings.
VECTORS_FILENAME = "vectors.npy"
#: One JSON object per chunk, aligned with the matrix rows.
RECORDS_FILENAME = "records.jsonl"
#: Similarity metric, recorded so a reader of the directory knows what the
#: scores mean.
SIMILARITY_METRIC = "cosine"


class LocalVectorStore:
    """A :class:`~rag.stores.base.VectorStore` backed by files on disk."""

    def __init__(self, directory: Path | str) -> None:
        """Point the store at a directory.

        Nothing is read here. The index loads on first use and the directory
        is created on first write, so constructing a store has no side effect
        and costs nothing.

        Args:
            directory: Where the index files live.
        """
        self._directory = Path(directory)
        self._vectors: np.ndarray | None = None
        self._chunks: list[Chunk] = []
        self._row_by_chunk_id: dict[str, int] = {}
        self._loaded = False

    # -- Paths -------------------------------------------------------------

    @property
    def directory(self) -> Path:
        """The directory this store reads and writes."""
        return self._directory

    @property
    def vectors_path(self) -> Path:
        """Path of the embedding matrix."""
        return self._directory / VECTORS_FILENAME

    @property
    def records_path(self) -> Path:
        """Path of the chunk records."""
        return self._directory / RECORDS_FILENAME

    @property
    def similarity_metric(self) -> str:
        """The metric ``search`` scores with."""
        return SIMILARITY_METRIC

    @property
    def dimension(self) -> int | None:
        """Dimension of the stored vectors, or ``None`` when empty."""
        self._ensure_loaded()
        if self._vectors is None or self._vectors.shape[0] == 0:
            return None
        return int(self._vectors.shape[1])

    # -- Loading and saving ------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Read the index from disk once, on first use.

        Raises:
            CorruptIndexError: If the files cannot be read, or if the matrix
                and the records disagree about how many chunks there are.
        """
        if self._loaded:
            return
        self._loaded = True

        if not self.vectors_path.is_file() or not self.records_path.is_file():
            if self.vectors_path.is_file() != self.records_path.is_file():
                raise CorruptIndexError(
                    "The index is incomplete: one of the vector matrix and the "
                    "record file is missing. Rebuild the index.",
                    details={
                        "has_vectors": self.vectors_path.is_file(),
                        "has_records": self.records_path.is_file(),
                    },
                )
            self._vectors = None
            self._chunks = []
            self._row_by_chunk_id = {}
            return

        try:
            vectors = np.load(self.vectors_path, allow_pickle=False)
        except Exception as exc:  # noqa: BLE001 - any unreadable file
            raise CorruptIndexError(
                "The stored vector matrix could not be read. Rebuild the index.",
                details={"reason": type(exc).__name__},
            ) from exc

        chunks: list[Chunk] = []
        try:
            with self.records_path.open("r", encoding="utf-8") as handle:
                for number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    chunks.append(Chunk.from_dict(json.loads(line)))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise CorruptIndexError(
                "The stored chunk records could not be read. Rebuild the index.",
                details={"reason": type(exc).__name__, "line": number},
            ) from exc

        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise CorruptIndexError(
                "The vector matrix and the chunk records disagree about how "
                "many chunks are stored. Rebuild the index.",
                details={
                    "vector_rows": int(vectors.shape[0]) if vectors.ndim == 2 else None,
                    "record_count": len(chunks),
                },
            )

        self._vectors = vectors.astype(VECTOR_DTYPE, copy=False)
        self._chunks = chunks
        self._reindex()

    def _reindex(self) -> None:
        """Rebuild the chunk-id lookup after the rows change."""
        self._row_by_chunk_id = {
            chunk.chunk_id: row for row, chunk in enumerate(self._chunks)
        }

    def _atomic_write(self, path: Path, write) -> None:
        """Write a file through a temporary name and an atomic rename."""
        temporary = path.parent / f".{path.name}.{secrets.token_hex(4)}.tmp"
        try:
            with temporary.open("wb") as handle:
                write(handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self) -> None:
        """Persist the index to disk.

        The matrix is written first: if the process dies between the two
        writes, the record file still describes the previous, complete state
        and loading fails loudly on the row-count check rather than returning
        mismatched text.
        """
        self._ensure_loaded()
        self._directory.mkdir(parents=True, exist_ok=True)

        vectors = (
            self._vectors
            if self._vectors is not None
            else np.zeros((0, 0), dtype=VECTOR_DTYPE)
        )
        self._atomic_write(
            self.vectors_path, lambda handle: np.save(handle, vectors, allow_pickle=False)
        )

        def write_records(handle) -> None:
            """Write one JSON object per chunk, in row order."""
            for chunk in self._chunks:
                line = json.dumps(chunk.as_dict(), ensure_ascii=False, sort_keys=True)
                handle.write(line.encode("utf-8"))
                handle.write(b"\n")

        self._atomic_write(self.records_path, write_records)

    # -- Writing -----------------------------------------------------------

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        """Insert or replace records, keyed by chunk id.

        A chunk id already present is overwritten in place, which is what
        makes re-indexing an unchanged source a no-op rather than a duplicate.

        Args:
            records: The chunks and their embeddings.

        Returns:
            int: How many records were written.

        Raises:
            EmbeddingDimensionError: If a vector's length disagrees with what
                is already stored — almost always an index built with one
                embedding provider being written with another.
        """
        self._ensure_loaded()
        if not records:
            return 0

        incoming = np.asarray(
            [np.asarray(record.vector, dtype=VECTOR_DTYPE).ravel() for record in records],
            dtype=VECTOR_DTYPE,
        )
        width = int(incoming.shape[1])
        existing = self.dimension
        if existing is not None and width != existing:
            raise EmbeddingDimensionError(
                f"Cannot store {width}-dimensional vectors in an index built "
                f"with {existing} dimensions. Rebuild the index after changing "
                "the embedding provider.",
                details={"incoming_dimension": width, "index_dimension": existing},
            )

        if self._vectors is None or self._vectors.shape[0] == 0:
            self._vectors = np.zeros((0, width), dtype=VECTOR_DTYPE)

        replacements: list[tuple[int, int]] = []
        additions: list[int] = []
        for offset, record in enumerate(records):
            row = self._row_by_chunk_id.get(record.chunk_id)
            if row is None:
                additions.append(offset)
            else:
                replacements.append((row, offset))

        for row, offset in replacements:
            self._vectors[row] = incoming[offset]
            self._chunks[row] = records[offset].chunk

        if additions:
            self._vectors = np.vstack([self._vectors, incoming[additions]])
            self._chunks.extend(records[offset].chunk for offset in additions)

        self._reindex()
        return len(records)

    def delete(self, chunk_ids: Sequence[str]) -> int:
        """Remove records by chunk id.

        Returns:
            int: How many were present and removed.
        """
        self._ensure_loaded()
        rows = sorted(
            {
                self._row_by_chunk_id[chunk_id]
                for chunk_id in chunk_ids
                if chunk_id in self._row_by_chunk_id
            }
        )
        if not rows:
            return 0
        self._drop_rows(rows)
        return len(rows)

    def delete_document(self, document_id: str) -> int:
        """Remove every chunk belonging to one document."""
        self._ensure_loaded()
        rows = [
            row
            for row, chunk in enumerate(self._chunks)
            if chunk.document_id == document_id
        ]
        if not rows:
            return 0
        self._drop_rows(rows)
        return len(rows)

    def _drop_rows(self, rows: Sequence[int]) -> None:
        """Remove rows from both the matrix and the record list together."""
        removing = set(rows)
        keep = [row for row in range(len(self._chunks)) if row not in removing]
        self._chunks = [self._chunks[row] for row in keep]
        if self._vectors is not None:
            self._vectors = (
                self._vectors[keep]
                if keep
                else np.zeros((0, self._vectors.shape[1]), dtype=VECTOR_DTYPE)
            )
        self._reindex()

    def clear(self) -> None:
        """Remove everything, in memory and on disk."""
        self._ensure_loaded()
        self._vectors = None
        self._chunks = []
        self._row_by_chunk_id = {}
        self.vectors_path.unlink(missing_ok=True)
        self.records_path.unlink(missing_ok=True)

    # -- Reading -----------------------------------------------------------

    def _candidate_rows(self, metadata_filter: MetadataFilter | None) -> list[int]:
        """Return the rows a filter admits, in row order."""
        if metadata_filter is None or metadata_filter.is_empty:
            return list(range(len(self._chunks)))
        return [
            row
            for row, chunk in enumerate(self._chunks)
            if metadata_filter.matches(chunk)
        ]

    def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        min_score: float | None = None,
    ) -> list[SearchHit]:
        """Return the closest stored chunks, best first.

        Ties are broken by row order, which is insertion order, so an
        identical query against an identical index returns an identical
        ranking every time.

        Args:
            vector: The query embedding, unit length.
            top_k: Most results to return.
            metadata_filter: Restricts the candidates before ranking.
            min_score: Drops results scoring below this.

        Returns:
            list[SearchHit]: At most ``top_k`` hits, descending by score. An
            empty index returns an empty list rather than failing.

        Raises:
            EmbeddingDimensionError: If the query vector's length does not
                match the index.
        """
        self._ensure_loaded()
        if self._vectors is None or not self._chunks or top_k < 1:
            return []

        query = np.asarray(vector, dtype=VECTOR_DTYPE).ravel()
        if query.shape[0] != self._vectors.shape[1]:
            raise EmbeddingDimensionError(
                f"The query has {query.shape[0]} dimensions but the index was "
                f"built with {self._vectors.shape[1]}. The index was built with "
                "a different embedding provider; rebuild it.",
                details={
                    "query_dimension": int(query.shape[0]),
                    "index_dimension": int(self._vectors.shape[1]),
                },
            )

        rows = self._candidate_rows(metadata_filter)
        if not rows:
            return []

        # Both sides are unit length, so the dot product is the cosine.
        scores = self._vectors[rows] @ query

        wanted = min(top_k, len(rows))
        # argpartition finds the top `wanted` without sorting everything, then
        # only those are sorted. `-scores` because we want the largest.
        top = np.argpartition(-scores, wanted - 1)[:wanted] if wanted < len(rows) else np.arange(len(rows))
        # Sort by score descending, then by row ascending, so ties are stable.
        order = sorted(top, key=lambda index: (-float(scores[index]), rows[index]))

        hits: list[SearchHit] = []
        for index in order:
            score = float(scores[index])
            if min_score is not None and score < min_score:
                continue
            hits.append(SearchHit(chunk=self._chunks[rows[index]], score=score))
        return hits

    def count(self, metadata_filter: MetadataFilter | None = None) -> int:
        """Return how many chunks are stored, optionally matching a filter."""
        self._ensure_loaded()
        if metadata_filter is None or metadata_filter.is_empty:
            return len(self._chunks)
        return sum(1 for chunk in self._chunks if metadata_filter.matches(chunk))

    def document_ids(self) -> tuple[str, ...]:
        """Return the ids of every document with chunks in the store."""
        self._ensure_loaded()
        return tuple(dict.fromkeys(chunk.document_id for chunk in self._chunks))

    def chunk_ids_for_document(self, document_id: str) -> tuple[str, ...]:
        """Return the chunk ids stored for one document, in order."""
        self._ensure_loaded()
        return tuple(
            chunk.chunk_id
            for chunk in self._chunks
            if chunk.document_id == document_id
        )

    def get(self, chunk_id: str) -> Chunk | None:
        """Return one stored chunk, or ``None`` when it is not present."""
        self._ensure_loaded()
        row = self._row_by_chunk_id.get(chunk_id)
        return None if row is None else self._chunks[row]

    @property
    def is_built(self) -> bool:
        """Whether an index has been written to this directory.

        Answers "has anything been indexed here", not "is it readable" — a
        caller that needs the latter opens the store and lets
        :class:`~rag.errors.CorruptIndexError` say so. The distinction matters
        to anything that must tell "nothing was ever indexed" apart from
        "the index is broken", because the fix for each is different and
        answering "no relevant evidence" to either would be misleading.
        """
        return self.vectors_path.is_file() and self.records_path.is_file()

    def stats(self) -> dict[str, Any]:
        """Describe the index: size, shape and what is in it."""
        self._ensure_loaded()
        by_type: dict[str, int] = {}
        for chunk in self._chunks:
            by_type[chunk.source_type] = by_type.get(chunk.source_type, 0) + 1
        return {
            "chunk_count": len(self._chunks),
            "document_count": len(self.document_ids()),
            "dimension": self.dimension,
            "similarity_metric": SIMILARITY_METRIC,
            "chunks_by_source_type": by_type,
            "vectors_bytes": (
                self.vectors_path.stat().st_size if self.vectors_path.is_file() else 0
            ),
            "records_bytes": (
                self.records_path.stat().st_size if self.records_path.is_file() else 0
            ),
        }


__all__ = [
    "RECORDS_FILENAME",
    "SIMILARITY_METRIC",
    "VECTORS_FILENAME",
    "LocalVectorStore",
]
