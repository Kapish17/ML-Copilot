"""ML Copilot retrieval layer.

Finds the evidence that bears on a question, from two kinds of knowledge the
project already produces:

- **project documentation** — what the system does and why it does it that way
- **experiment records** — what was actually run, and what came of it

::

    question -> embed -> filter by metadata -> rank by similarity
             -> ranked passages, each with a citation

**It returns evidence, never answers.** No text is generated, no conclusion is
drawn and no experiment result is interpreted. A later commit will hand this
evidence to a model and ask it to reason; this layer's job is to make sure
what it is handed is the right material, and that every sentence of an answer
can be traced back to a passage that exists. **LLM generation is not
implemented.**

The layer is independent: it imports neither FastAPI nor the backend, and
``ml/`` does not know it exists. Everything runs locally — the default
embedding provider needs no API key, no network and no model download, and no
document or experiment content leaves the machine. **Qdrant, PostgreSQL,
LangChain and LangGraph are not implemented.**

Typical use::

    from rag import RagConfig, RagIndexer, RetrievalService
    from ml.experiments import LocalExperimentStore

    config = RagConfig()
    indexer = RagIndexer(config)
    indexer.index_documentation()
    indexer.sync_experiments(LocalExperimentStore())

    service = RetrievalService(config)
    response = service.search("How is data leakage prevented?")
    for result in response:
        print(result.citation, round(result.score, 3))
"""

from rag.chunking import chunk_document, chunk_markdown
from rag.citations import build_citation, parse_citation
from rag.config import RagConfig, config_from_env
from rag.documents import Chunk, Document, SourceType, content_hash
from rag.embeddings import (
    EmbeddingProvider,
    HashingEmbeddingProvider,
    build_embedding_provider,
)
from rag.errors import (
    ConfigurationError,
    CorruptIndexError,
    EmbeddingDimensionError,
    RagError,
    RetrievalError,
    UnsafeSourceError,
)
from rag.evaluation import (
    DEFAULT_EVALUATION_QUERIES,
    EvaluationQuery,
    EvaluationReport,
    evaluate_retrieval,
)
from rag.indexing import IndexReport, RagIndexer
from rag.manifest import IndexManifest, ManifestEntry
from rag.retrieval import (
    RetrievalResponse,
    RetrievalResult,
    RetrievalService,
    build_metadata_filter,
)
from rag.stores import (
    LocalVectorStore,
    MetadataFilter,
    SearchHit,
    VectorRecord,
    VectorStore,
)

__all__ = [
    "DEFAULT_EVALUATION_QUERIES",
    "Chunk",
    "ConfigurationError",
    "CorruptIndexError",
    "Document",
    "EmbeddingDimensionError",
    "EmbeddingProvider",
    "EvaluationQuery",
    "EvaluationReport",
    "HashingEmbeddingProvider",
    "IndexManifest",
    "IndexReport",
    "LocalVectorStore",
    "ManifestEntry",
    "MetadataFilter",
    "RagConfig",
    "RagError",
    "RagIndexer",
    "RetrievalError",
    "RetrievalResponse",
    "RetrievalResult",
    "RetrievalService",
    "SearchHit",
    "SourceType",
    "UnsafeSourceError",
    "VectorRecord",
    "VectorStore",
    "build_citation",
    "build_embedding_provider",
    "build_metadata_filter",
    "chunk_document",
    "chunk_markdown",
    "config_from_env",
    "content_hash",
    "evaluate_retrieval",
    "parse_citation",
]
