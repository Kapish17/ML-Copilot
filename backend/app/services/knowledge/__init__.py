"""Searching the knowledge base and answering from it, as an application service.

``filters``  request fields → the retrieval layer's own metadata filter
``errors``   the two refusals only an HTTP caller cares about
``service``  ``KnowledgeService`` — search and ask

Nothing in this package imports FastAPI, so the same service is drivable from
a script, a test or a future worker; the HTTP routes are one caller among
several. It computes nothing itself: retrieval belongs to ``rag/``, and
generation and grounding to ``llm/``.
"""

from app.services.knowledge.errors import (
    AnsweringFailedError,
    AnsweringUnavailableError,
    IndexNotBuiltError,
    KnowledgeError,
)
from app.services.knowledge.filters import (
    FILTERABLE_FIELDS,
    KNOWN_SOURCE_TYPES,
    build_filter,
)
from app.services.knowledge.service import KnowledgeService

__all__ = [
    "FILTERABLE_FIELDS",
    "KNOWN_SOURCE_TYPES",
    "AnsweringFailedError",
    "AnsweringUnavailableError",
    "IndexNotBuiltError",
    "KnowledgeError",
    "KnowledgeService",
    "build_filter",
]
