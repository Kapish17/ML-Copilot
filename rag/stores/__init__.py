"""Vector storage.

``base``   the :class:`VectorStore` contract, plus the record, hit and filter
           types every implementation speaks
``local``  the only implementation: a persistent index of files on disk

Storage is behind an interface so that a future Qdrant or pgvector backend
replaces one class and nothing else. **Neither is implemented.**
"""

from rag.stores.base import MetadataFilter, SearchHit, VectorRecord, VectorStore
from rag.stores.local import LocalVectorStore

__all__ = [
    "LocalVectorStore",
    "MetadataFilter",
    "SearchHit",
    "VectorRecord",
    "VectorStore",
]
