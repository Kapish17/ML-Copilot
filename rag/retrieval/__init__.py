"""Semantic retrieval over the indexed knowledge.

``results``  the structured evidence a search returns
``service``  the coordination: embed, filter, rank, attribute

Retrieval returns facts with citations. It does not read, summarise or answer
— those belong to a future model that receives this evidence. **No LLM
generation is implemented.**
"""

from rag.retrieval.results import RetrievalResponse, RetrievalResult
from rag.retrieval.service import RetrievalService, build_metadata_filter

__all__ = [
    "RetrievalResponse",
    "RetrievalResult",
    "RetrievalService",
    "build_metadata_filter",
]
