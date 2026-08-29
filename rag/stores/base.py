"""The storage contract.

Retrieval depends on this interface, not on where vectors live. The only
implementation today writes files to a local directory; a Qdrant or pgvector
backend would implement the same five operations and nothing above it would
change — not the retrieval service, not the document model, not the embedding
provider, not an API caller. **Qdrant and PostgreSQL are not implemented.**

Metadata filters are part of the interface rather than something applied
afterwards, because *where* the filtering happens matters. Asking for the ten
best classification experiments should search among classification
experiments, not take the ten best chunks overall and discard the regressions
— which is how a filter applied after the fact silently returns three results
when it was asked for ten.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from rag.documents import Chunk


@dataclass(frozen=True)
class VectorRecord:
    """One chunk together with its embedding, ready to store."""

    chunk: Chunk
    vector: np.ndarray

    @property
    def chunk_id(self) -> str:
        """The chunk's identifier, which is the record's primary key."""
        return self.chunk.chunk_id

    @property
    def document_id(self) -> str:
        """The owning document's identifier."""
        return self.chunk.document_id


@dataclass(frozen=True)
class SearchHit:
    """One chunk that matched, and how well."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class MetadataFilter:
    """Which stored chunks a search may consider.

    Every condition is optional and they combine with "and". The values are
    matched against a chunk's own fields first and its metadata second, so a
    caller can filter on ``task_type`` without knowing whether the ingestion
    adapter put it in metadata or on the chunk.

    ``equals`` matches one value; ``any_of`` matches a set. A chunk missing
    the key never matches — an absent value is not a wildcard.
    """

    source_types: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    equals: Mapping[str, Any] = field(default_factory=dict)
    any_of: Mapping[str, Sequence[Any]] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """True when the filter admits everything."""
        return not (
            self.source_types or self.document_ids or self.equals or self.any_of
        )

    def _value(self, chunk: Chunk, key: str) -> Any:
        """Read a field from the chunk, falling back to its metadata."""
        if hasattr(chunk, key):
            return getattr(chunk, key)
        return chunk.metadata.get(key, _MISSING)

    def matches(self, chunk: Chunk) -> bool:
        """Return whether a chunk satisfies every condition."""
        if self.source_types and chunk.source_type not in self.source_types:
            return False
        if self.document_ids and chunk.document_id not in self.document_ids:
            return False
        for key, wanted in self.equals.items():
            if self._value(chunk, key) != wanted:
                return False
        for key, options in self.any_of.items():
            if self._value(chunk, key) not in tuple(options):
                return False
        return True

    def as_dict(self) -> dict[str, Any]:
        """Render the filter as plain values, for logging and responses."""
        return {
            "source_types": list(self.source_types),
            "document_ids": list(self.document_ids),
            "equals": dict(self.equals),
            "any_of": {key: list(value) for key, value in self.any_of.items()},
        }


class _Missing:
    """Sentinel for "this chunk has no such key", distinct from ``None``."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


_MISSING = _Missing()


@runtime_checkable
class VectorStore(Protocol):
    """Stores chunk embeddings and searches them by similarity."""

    @property
    def dimension(self) -> int | None:
        """Dimension of the stored vectors, or ``None`` when empty."""
        ...  # pragma: no cover - protocol

    def upsert(self, records: Sequence[VectorRecord]) -> int:
        """Insert or replace records, keyed by chunk id.

        Returns:
            int: How many records were written.
        """
        ...  # pragma: no cover - protocol

    def delete(self, chunk_ids: Sequence[str]) -> int:
        """Remove records by chunk id.

        Returns:
            int: How many were actually present and removed.
        """
        ...  # pragma: no cover - protocol

    def delete_document(self, document_id: str) -> int:
        """Remove every chunk belonging to one document.

        Returns:
            int: How many chunks were removed.
        """
        ...  # pragma: no cover - protocol

    def search(
        self,
        vector: np.ndarray,
        *,
        top_k: int,
        metadata_filter: MetadataFilter | None = None,
        min_score: float | None = None,
    ) -> list[SearchHit]:
        """Return the closest stored chunks, best first.

        Args:
            vector: The query embedding, unit length.
            top_k: Most results to return.
            metadata_filter: Restricts which chunks are candidates, applied
                **before** ranking.
            min_score: Drops results below this similarity.
        """
        ...  # pragma: no cover - protocol

    def count(self, metadata_filter: MetadataFilter | None = None) -> int:
        """Return how many chunks are stored, optionally matching a filter."""
        ...  # pragma: no cover - protocol

    def clear(self) -> None:
        """Remove everything."""
        ...  # pragma: no cover - protocol

    def document_ids(self) -> tuple[str, ...]:
        """Return the ids of every document with chunks in the store."""
        ...  # pragma: no cover - protocol


__all__ = [
    "MetadataFilter",
    "SearchHit",
    "VectorRecord",
    "VectorStore",
]
