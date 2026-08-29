"""What the retrieval layer stores and returns.

A :class:`Document` is one indexable source — a README, or one experiment
record. A :class:`Chunk` is the retrievable unit: a passage of that document
small enough to embed and specific enough to cite.

**Identity is derived, never generated.** A document's id comes from its source
type and reference; a chunk's id comes from its document id and its position;
a content hash comes from the text itself. No random UUID appears anywhere, so
indexing the same source twice produces the same ids, the second index updates
the first instead of duplicating it, and two machines building the same index
agree on every identifier.

Source types are an open vocabulary. :class:`SourceType` names the three that
exist today, and anything that follows the same slug rules is accepted — a
future ingestion adapter should not have to edit an enum in this module.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: Hash used for content hashes and derived identifiers.
HASH_ALGORITHM = "sha256"
#: Characters of digest kept in a document id. 12 hex characters is 48 bits —
#: far more than enough to separate the sources one project accumulates.
DOCUMENT_HASH_LENGTH = 12
#: Characters of digest kept in a content hash, used for change detection.
CONTENT_HASH_LENGTH = 16

#: Slugs are the vocabulary of both ids and citations, so they are restricted
#: to what is safe in a filename, a URL fragment and a JSON key.
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")

_SLUG_CLEANER = re.compile(r"[^a-z0-9._-]+")


class SourceType(str, Enum):
    """The kinds of knowledge the retrieval layer indexes.

    A ``str`` enum so a member serialises as its own value and comparisons
    against a plain string work. Ingestion adapters are free to use another
    slug; this enum names the ones that exist.
    """

    PROJECT_DOCUMENTATION = "project_documentation"
    EXPERIMENT = "experiment"
    ML_REFERENCE = "ml_reference"


def slugify(value: str) -> str:
    """Reduce arbitrary text to a slug usable in an id or a citation.

    Args:
        value: Any text — a heading, a path, a title.

    Returns:
        str: Lowercase, with runs of unsupported characters collapsed to a
        single hyphen, trimmed of leading and trailing separators. Empty input
        yields ``"untitled"`` rather than an empty slug.
    """
    lowered = value.strip().lower().replace("/", "-").replace("\\", "-")
    cleaned = _SLUG_CLEANER.sub("-", lowered).strip("-._")
    return cleaned[:128] or "untitled"


def content_hash(text: str) -> str:
    """Return the stable hash of a piece of text.

    Used to decide whether a source has changed since it was indexed. The
    text is encoded as UTF-8 and hashed as-is: whitespace is content, because
    it changes how the text chunks.
    """
    digest = hashlib.new(HASH_ALGORITHM, text.encode("utf-8")).hexdigest()
    return digest[:CONTENT_HASH_LENGTH]


def make_document_id(source_type: str, source_reference: str) -> str:
    """Derive a document's identifier from what it is and where it came from.

    Two documents of the same type and reference are the same document, on
    every machine and in every process. That is what makes re-indexing an
    update rather than a duplicate.

    Args:
        source_type: The source vocabulary term, e.g. ``"experiment"``.
        source_reference: What identifies the source within that type — a
            repository-relative path, or an experiment id.

    Returns:
        str: ``"<type-slug>:<reference-slug>"`` when the reference slugifies
        cleanly and briefly, otherwise the slug is replaced by a short digest
        of the reference so the id stays bounded and legal.
    """
    type_slug = slugify(str(source_type))
    reference = str(source_reference)
    reference_slug = slugify(reference)
    if not SLUG_PATTERN.match(reference_slug) or len(reference_slug) > 64:
        digest = hashlib.new(HASH_ALGORITHM, reference.encode("utf-8")).hexdigest()
        reference_slug = digest[:DOCUMENT_HASH_LENGTH]
    return f"{type_slug}:{reference_slug}"


def make_chunk_id(document_id: str, position: int, text: str) -> str:
    """Derive a chunk's identifier from its document, position and content.

    Position alone would make chunk 3 of a rewritten document collide with
    chunk 3 of the old one; content alone would make two identical passages
    collide. Both together give a chunk an identity that survives an unrelated
    edit elsewhere in the file and changes when the passage itself changes.

    Args:
        document_id: The owning document's id.
        position: Zero-based index of the chunk within the document.
        text: The chunk's content.

    Returns:
        str: ``"<document_id>#<position>-<digest>"``.
    """
    digest = hashlib.new(HASH_ALGORITHM, text.encode("utf-8")).hexdigest()
    return f"{document_id}#{position:04d}-{digest[:8]}"


def _jsonable(value: Any, *, depth: int = 0) -> Any:
    """Reduce a metadata value to something JSON can hold.

    Metadata is written to disk and returned over an API, so it may only
    contain plain values. Anything else is rendered as its string form rather
    than silently dropped — losing a filter key would be worse than storing a
    readable stand-in.
    """
    if depth > 6:
        return str(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # JSON has no NaN or infinity.
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, Enum):
        return _jsonable(value.value, depth=depth + 1)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item, depth=depth + 1) for item in value]
    return str(value)


def jsonable_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return metadata reduced to JSON-safe values, with string keys."""
    if not metadata:
        return {}
    return {str(key): _jsonable(value) for key, value in metadata.items()}


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage, with everything needed to cite it."""

    document_id: str
    chunk_id: str
    content: str
    source_type: str
    source_title: str
    source_reference: str
    #: Zero-based position within the document, so chunks can be re-ordered
    #: back into reading order after retrieval.
    position: int = 0
    #: The heading path this passage sits under, outermost first. Empty for
    #: sources that have no heading structure.
    heading_path: tuple[str, ...] = ()
    #: Stable citation reference; see :mod:`rag.citations`.
    citation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Normalise metadata so a chunk is always JSON-safe once built."""
        object.__setattr__(self, "metadata", jsonable_metadata(self.metadata))
        object.__setattr__(self, "heading_path", tuple(self.heading_path))

    @property
    def content_hash(self) -> str:
        """The hash of this chunk's text."""
        return content_hash(self.content)

    @property
    def heading(self) -> str | None:
        """The innermost heading this passage sits under, if any."""
        return self.heading_path[-1] if self.heading_path else None

    def as_dict(self) -> dict[str, Any]:
        """Render the chunk as plain JSON-safe values."""
        return {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source_type": self.source_type,
            "source_title": self.source_title,
            "source_reference": self.source_reference,
            "position": self.position,
            "heading_path": list(self.heading_path),
            "citation": self.citation,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Chunk:
        """Rebuild a chunk from its stored form."""
        return cls(
            document_id=str(payload["document_id"]),
            chunk_id=str(payload["chunk_id"]),
            content=str(payload["content"]),
            source_type=str(payload["source_type"]),
            source_title=str(payload.get("source_title", "")),
            source_reference=str(payload.get("source_reference", "")),
            position=int(payload.get("position", 0)),
            heading_path=tuple(payload.get("heading_path", ()) or ()),
            citation=str(payload.get("citation", "")),
            metadata=dict(payload.get("metadata", {}) or {}),
        )


@dataclass(frozen=True)
class Document:
    """One indexable source, before it is split into chunks."""

    source_type: str
    source_title: str
    source_reference: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Set explicitly only when an adapter has a better identity than the
    #: derived one; normally left alone.
    document_id: str = ""

    def __post_init__(self) -> None:
        """Derive the identifier and normalise metadata."""
        object.__setattr__(self, "metadata", jsonable_metadata(self.metadata))
        if not self.document_id:
            object.__setattr__(
                self,
                "document_id",
                make_document_id(self.source_type, self.source_reference),
            )

    @property
    def content_hash(self) -> str:
        """The hash of this document's text.

        The indexer compares this with the manifest to decide whether the
        document needs re-chunking and re-embedding.
        """
        return content_hash(self.content)

    @property
    def metadata_hash(self) -> str:
        """The hash of this document's metadata.

        Metadata can change while the text does not — an experiment's tags,
        say — and a stale filter value is as wrong as stale text, so this is
        part of what change detection compares.
        """
        return content_hash(json.dumps(self.metadata, sort_keys=True, default=str))

    @property
    def source_hash(self) -> str:
        """One hash covering both the content and the metadata."""
        return content_hash(f"{self.content_hash}:{self.metadata_hash}")

    def as_dict(self) -> dict[str, Any]:
        """Render the document's identity and metadata, without its text."""
        return {
            "document_id": self.document_id,
            "source_type": self.source_type,
            "source_title": self.source_title,
            "source_reference": self.source_reference,
            "content_hash": self.content_hash,
            "source_hash": self.source_hash,
            "metadata": dict(self.metadata),
        }


__all__ = [
    "CONTENT_HASH_LENGTH",
    "Chunk",
    "Document",
    "HASH_ALGORITHM",
    "SourceType",
    "content_hash",
    "jsonable_metadata",
    "make_chunk_id",
    "make_document_id",
    "slugify",
]
