"""Errors raised by the retrieval layer.

Plain Python exceptions with no HTTP meaning, matching ``ml/errors.py``. The
API layer translates them if and when RAG is exposed over HTTP; this package
never imports a web framework.

Each error carries an optional ``details`` mapping so a caller can act on the
specifics — which file, which provider, how many dimensions — without parsing
a message string.
"""

from __future__ import annotations

from typing import Any


class RagError(Exception):
    """Base class for every failure in the retrieval layer."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        """Store the message and any structured context.

        Args:
            message: Explanation written for a person reading a log or an API
                response. Never contains a stack trace.
            details: Machine-readable context.
        """
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}


class ConfigurationError(RagError):
    """A configuration value is missing, malformed or out of range."""


class EmbeddingError(RagError):
    """Base class for embedding failures."""


class EmbeddingProviderUnavailableError(EmbeddingError):
    """The requested embedding provider is not installed or cannot load."""


class EmbeddingDimensionError(EmbeddingError):
    """Vectors do not have the dimension the store or provider expects.

    Almost always means the index was built with one embedding provider and is
    being queried with another. Re-indexing is the fix; silently mixing spaces
    would produce confident nonsense.
    """


class IngestionError(RagError):
    """Base class for failures while turning a source into documents."""


class SourceNotFoundError(IngestionError):
    """A configured documentation source does not exist."""


class UnsafeSourceError(IngestionError):
    """A source was refused because indexing it would be unsafe.

    Raised for anything outside the configured roots, and for files that may
    hold credentials — ``.env`` and friends are never indexed, whatever the
    caller asks for.
    """


class VectorStoreError(RagError):
    """Base class for storage failures."""


class CorruptIndexError(VectorStoreError):
    """The stored index cannot be read, or its parts disagree.

    Raised rather than guessed at: a vector matrix and a record file that have
    drifted out of step would return the wrong text for the right score, which
    is worse than an error.
    """


class RetrievalError(RagError):
    """A query cannot be answered as asked."""


__all__ = [
    "ConfigurationError",
    "CorruptIndexError",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "EmbeddingProviderUnavailableError",
    "IngestionError",
    "RagError",
    "RetrievalError",
    "SourceNotFoundError",
    "UnsafeSourceError",
    "VectorStoreError",
]
