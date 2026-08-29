"""What has been indexed, from what, and with which embeddings.

The manifest is the indexer's memory. Without it, "index the documentation"
would mean re-chunking and re-embedding every file every time; with it, the
indexer compares each source's current hash against the recorded one and does
work only where something changed.

One entry per document::

    {
      "document_id": "project_documentation:ml-readme",
      "source_type": "project_documentation",
      "source_reference": "ml/README.md",
      "source_hash": "3f1c9a2b7d4e5061",
      "chunk_count": 24,
      "chunk_ids": [...],
      "indexed_at": "2026-08-29T09:14:02+00:00"
    }

Plus the embedding provider's identifier for the index as a whole. That last
field is what catches the dangerous case: an index built with one embedding
provider and queried with another returns confident nonsense, because the
vectors are not in the same space. The indexer compares the identifiers and
rebuilds rather than mixing them.

**No secret is ever written here** — the manifest holds hashes, counts,
identifiers and timestamps, and no API key is involved in the first place.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.errors import CorruptIndexError

#: The manifest's own file.
MANIFEST_FILENAME = "manifest.json"
#: Bumped when the manifest's own layout changes.
MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ManifestEntry:
    """What is recorded about one indexed document."""

    document_id: str
    source_type: str
    source_reference: str
    source_hash: str
    chunk_count: int
    chunk_ids: tuple[str, ...] = ()
    indexed_at: str = ""
    source_title: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Render the entry as plain JSON-safe values."""
        return {
            "document_id": self.document_id,
            "source_type": self.source_type,
            "source_reference": self.source_reference,
            "source_title": self.source_title,
            "source_hash": self.source_hash,
            "chunk_count": self.chunk_count,
            "chunk_ids": list(self.chunk_ids),
            "indexed_at": self.indexed_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ManifestEntry:
        """Rebuild an entry from its stored form."""
        return cls(
            document_id=str(payload["document_id"]),
            source_type=str(payload.get("source_type", "")),
            source_reference=str(payload.get("source_reference", "")),
            source_title=str(payload.get("source_title", "")),
            source_hash=str(payload.get("source_hash", "")),
            chunk_count=int(payload.get("chunk_count", 0)),
            chunk_ids=tuple(payload.get("chunk_ids", ()) or ()),
            indexed_at=str(payload.get("indexed_at", "")),
        )


@dataclass
class IndexManifest:
    """The record of an index's contents, loaded from and saved to disk."""

    embedding_identifier: str = ""
    embedding_dimension: int | None = None
    entries: dict[str, ManifestEntry] = field(default_factory=dict)
    schema_version: str = MANIFEST_SCHEMA_VERSION
    updated_at: str = ""

    # -- Persistence -------------------------------------------------------

    @classmethod
    def load(cls, directory: Path) -> IndexManifest:
        """Read the manifest from a directory, or return an empty one.

        An absent manifest is not an error: it is what an index that has
        never been built looks like.

        Raises:
            CorruptIndexError: If the file exists but cannot be parsed.
        """
        path = Path(directory) / MANIFEST_FILENAME
        if not path.is_file():
            return cls()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CorruptIndexError(
                "The index manifest is not valid JSON. Rebuild the index.",
                details={"reason": type(exc).__name__},
            ) from exc
        if not isinstance(payload, dict):
            raise CorruptIndexError(
                "The index manifest must be a JSON object. Rebuild the index.",
                details={"found_type": type(payload).__name__},
            )

        try:
            entries = {
                str(entry["document_id"]): ManifestEntry.from_dict(entry)
                for entry in payload.get("documents", [])
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise CorruptIndexError(
                "The index manifest holds a malformed document entry. Rebuild "
                "the index.",
                details={"reason": type(exc).__name__},
            ) from exc

        return cls(
            embedding_identifier=str(payload.get("embedding_identifier", "")),
            embedding_dimension=payload.get("embedding_dimension"),
            entries=entries,
            schema_version=str(payload.get("schema_version", MANIFEST_SCHEMA_VERSION)),
            updated_at=str(payload.get("updated_at", "")),
        )

    def save(self, directory: Path) -> Path:
        """Write the manifest atomically, and return where it went."""
        path = Path(directory) / MANIFEST_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(timezone.utc).isoformat()

        payload = json.dumps(self.as_dict(), indent=2, sort_keys=True, ensure_ascii=False)
        temporary = path.parent / f".{MANIFEST_FILENAME}.{secrets.token_hex(4)}.tmp"
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path

    def as_dict(self) -> dict[str, Any]:
        """Render the whole manifest as plain JSON-safe values."""
        return {
            "schema_version": self.schema_version,
            "embedding_identifier": self.embedding_identifier,
            "embedding_dimension": self.embedding_dimension,
            "updated_at": self.updated_at,
            "document_count": len(self.entries),
            "chunk_count": self.total_chunks,
            "documents": [
                entry.as_dict()
                for entry in sorted(self.entries.values(), key=lambda e: e.document_id)
            ],
        }

    # -- Queries -----------------------------------------------------------

    @property
    def total_chunks(self) -> int:
        """How many chunks the manifest accounts for."""
        return sum(entry.chunk_count for entry in self.entries.values())

    def get(self, document_id: str) -> ManifestEntry | None:
        """Return the entry for a document, if it has been indexed."""
        return self.entries.get(document_id)

    def needs_reindex(self, document_id: str, source_hash: str) -> bool:
        """Return whether a document must be re-chunked and re-embedded.

        True when it has never been indexed, or when its content or metadata
        has changed since it was.
        """
        entry = self.entries.get(document_id)
        return entry is None or entry.source_hash != source_hash

    def matches_embeddings(self, identifier: str) -> bool:
        """Return whether the index was built with this embedding provider.

        An empty recorded identifier means an empty index, which matches
        anything.
        """
        return not self.embedding_identifier or self.embedding_identifier == identifier

    def document_ids_for(self, source_type: str) -> tuple[str, ...]:
        """Return the indexed document ids of one source type."""
        return tuple(
            sorted(
                entry.document_id
                for entry in self.entries.values()
                if entry.source_type == source_type
            )
        )

    # -- Updates -----------------------------------------------------------

    def record(
        self,
        *,
        document_id: str,
        source_type: str,
        source_reference: str,
        source_title: str,
        source_hash: str,
        chunk_ids: Iterable[str],
    ) -> ManifestEntry:
        """Record that a document has been indexed, replacing any earlier entry."""
        identifiers = tuple(chunk_ids)
        entry = ManifestEntry(
            document_id=document_id,
            source_type=source_type,
            source_reference=source_reference,
            source_title=source_title,
            source_hash=source_hash,
            chunk_count=len(identifiers),
            chunk_ids=identifiers,
            indexed_at=datetime.now(timezone.utc).isoformat(),
        )
        self.entries[document_id] = entry
        return entry

    def forget(self, document_id: str) -> ManifestEntry | None:
        """Remove a document's entry, returning it if there was one."""
        return self.entries.pop(document_id, None)

    def clear(self) -> None:
        """Forget everything, keeping the embedding identifier unset."""
        self.entries.clear()
        self.embedding_identifier = ""
        self.embedding_dimension = None


__all__ = [
    "MANIFEST_FILENAME",
    "MANIFEST_SCHEMA_VERSION",
    "IndexManifest",
    "ManifestEntry",
]
