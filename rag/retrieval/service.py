"""Finding the evidence for a question.

::

    question -> embed -> filter candidates -> rank by similarity
             -> apply threshold -> structured results with citations

The service coordinates; it does not compute. The embedding provider turns
text into a vector, the vector store filters and ranks, and this module turns
what comes back into results a caller can read and attribute.

**It never answers.** Given "which model performed best?", it returns the
passages that bear on the question, ranked, each with a citation. It does not
read them, compare them or say which model won — that requires reasoning over
the evidence, and reasoning is a later commit's job. **No LLM generation is
implemented.**

The convenience methods are the ones worth knowing:
:meth:`RetrievalService.search_documentation` and
:meth:`RetrievalService.search_experiments` are the same search with a source
filter, and :meth:`RetrievalService.search_experiments` additionally accepts
the experiment metadata that makes a question like "the best classification
run on this dataset" answerable as a filter rather than a full scan.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from rag.config import RagConfig
from rag.documents import SourceType
from rag.embeddings import EmbeddingProvider, build_embedding_provider
from rag.errors import RetrievalError
from rag.retrieval.results import RetrievalResponse, RetrievalResult
from rag.stores import LocalVectorStore, MetadataFilter, VectorStore

logger = logging.getLogger(__name__)


def build_metadata_filter(
    *,
    source_types: Sequence[str] = (),
    document_ids: Sequence[str] = (),
    equals: Mapping[str, Any] | None = None,
    any_of: Mapping[str, Sequence[Any]] | None = None,
) -> MetadataFilter:
    """Assemble a metadata filter, dropping conditions that are not set."""
    return MetadataFilter(
        source_types=tuple(source_types),
        document_ids=tuple(document_ids),
        equals={
            key: value for key, value in (equals or {}).items() if value is not None
        },
        any_of={
            key: tuple(value)
            for key, value in (any_of or {}).items()
            if value
        },
    )


class RetrievalService:
    """Returns ranked evidence for a question. Never an answer."""

    def __init__(
        self,
        config: RagConfig,
        *,
        store: VectorStore | None = None,
        embeddings: EmbeddingProvider | None = None,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            config: Supplies ``top_k``, the similarity threshold and the
                embedding provider to use.
            store: Where to search. Defaults to a local store in the
                configured index directory.
            embeddings: How the question becomes a vector. Must be the same
                provider the index was built with; the store raises if it is
                not, rather than returning nonsense.
        """
        self._config = config
        self._store = store if store is not None else LocalVectorStore(config.index_dir)
        self._embeddings = (
            embeddings if embeddings is not None else build_embedding_provider(config)
        )

    @property
    def config(self) -> RagConfig:
        """The configuration this service runs with."""
        return self._config

    @property
    def store(self) -> VectorStore:
        """The vector store being searched."""
        return self._store

    @property
    def embeddings(self) -> EmbeddingProvider:
        """The embedding provider in use."""
        return self._embeddings

    @property
    def similarity_metric(self) -> str:
        """The metric scores are expressed in."""
        return getattr(self._store, "similarity_metric", "cosine")

    def search(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        metadata_filter: MetadataFilter | None = None,
        source_types: Sequence[str] = (),
        equals: Mapping[str, Any] | None = None,
        any_of: Mapping[str, Sequence[Any]] | None = None,
    ) -> RetrievalResponse:
        """Find the passages that bear on a question.

        Args:
            question: What is being asked, in the caller's own words.
            top_k: Most results to return; the configured default when
                omitted.
            min_score: Minimum similarity a result must reach; the configured
                threshold when omitted.
            metadata_filter: A ready-made filter. When given, the
                ``source_types``, ``equals`` and ``any_of`` shortcuts are
                ignored.
            source_types: Restrict to these kinds of source.
            equals: Metadata keys that must match exactly.
            any_of: Metadata keys whose value must be one of several.

        Returns:
            RetrievalResponse: Ranked evidence, best first. Empty when the
            index holds nothing, when the filter admits nothing, or when
            nothing clears the threshold — the response distinguishes the
            three through ``candidate_count``.

        Raises:
            RetrievalError: If the question is blank.
            EmbeddingDimensionError: If the index was built with a different
                embedding provider.
        """
        text = (question or "").strip()
        if not text:
            raise RetrievalError(
                "A retrieval query cannot be empty.",
                details={"question": question},
            )

        resolved_k = self._config.resolve_top_k(top_k)
        threshold = self._config.resolve_threshold(min_score)
        active_filter = (
            metadata_filter
            if metadata_filter is not None
            else build_metadata_filter(
                source_types=source_types, equals=equals, any_of=any_of
            )
        )

        candidates = self._store.count(active_filter)
        if candidates == 0:
            return RetrievalResponse(
                question=text,
                results=(),
                top_k=resolved_k,
                similarity_threshold=threshold,
                similarity_metric=self.similarity_metric,
                filter_applied=active_filter.as_dict(),
                candidate_count=0,
            )

        vector = self._embeddings.embed_query(text)
        hits = self._store.search(
            vector,
            top_k=resolved_k,
            metadata_filter=active_filter,
            min_score=threshold,
        )

        results = tuple(
            RetrievalResult.from_chunk(hit.chunk, rank=rank, score=hit.score)
            for rank, hit in enumerate(hits, start=1)
        )
        return RetrievalResponse(
            question=text,
            results=results,
            top_k=resolved_k,
            similarity_threshold=threshold,
            similarity_metric=self.similarity_metric,
            filter_applied=active_filter.as_dict(),
            candidate_count=candidates,
        )

    def search_documentation(
        self, question: str, *, top_k: int | None = None, min_score: float | None = None
    ) -> RetrievalResponse:
        """Search only the project documentation."""
        return self.search(
            question,
            top_k=top_k,
            min_score=min_score,
            source_types=(SourceType.PROJECT_DOCUMENTATION.value,),
        )

    def search_experiments(
        self,
        question: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        task_type: str | None = None,
        dataset_fingerprint: str | None = None,
        target_column: str | None = None,
        selected_model: str | None = None,
        primary_metric: str | None = None,
        experiment_ids: Sequence[str] = (),
    ) -> RetrievalResponse:
        """Search only experiment records, narrowed by their metadata.

        This is the hybrid path: the metadata conditions restrict which
        experiments are candidates, and the semantic search ranks within
        them. "Which classification model did best on this dataset" is a
        filter on ``task_type`` and ``dataset_fingerprint`` plus a query
        about model performance — not a scan of the whole index.
        """
        return self.search(
            question,
            top_k=top_k,
            min_score=min_score,
            metadata_filter=build_metadata_filter(
                source_types=(SourceType.EXPERIMENT.value,),
                equals={
                    "task_type": task_type,
                    "dataset_fingerprint": dataset_fingerprint,
                    "target_column": target_column,
                    "selected_model": selected_model,
                    "primary_metric": primary_metric,
                },
                any_of={"experiment_id": tuple(experiment_ids)},
            ),
        )

    def stats(self) -> dict[str, Any]:
        """Describe what is searchable right now."""
        payload: dict[str, Any] = {
            "embedding_identifier": self._embeddings.identifier,
            "similarity_metric": self.similarity_metric,
            "chunk_count": self._store.count(),
            "top_k": self._config.top_k,
            "similarity_threshold": self._config.similarity_threshold,
        }
        if hasattr(self._store, "stats"):
            payload.update(self._store.stats())
        return payload


__all__ = ["RetrievalService", "build_metadata_filter"]
