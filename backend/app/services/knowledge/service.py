"""Searching the knowledge base, and answering from it.

The application service the two knowledge endpoints are adapters around. It
decides *when* to do things and *what to refuse*; it computes nothing.
Retrieval belongs to ``rag/``, generation and grounding to ``llm/``, and this
module holds neither an embedding nor a prompt.

::

    search:  query  → validate → check the index → RetrievalService → evidence
    ask:     question → validate → check the index and the provider
                                 → RAGAnswerService → grounded Answer

Two refusals happen here rather than deeper down, because they are only
distinguishable at the edge:

**An unbuilt index.** To a library caller an empty store honestly returns
nothing. To an HTTP client "no relevant evidence" would be a lie — the truth
is that nothing has been indexed. That is a 503 with instructions, not a 200
with an empty list.

**An unconfigured provider.** Checked before retrieval runs, so a request that
cannot possibly succeed does not spend an embedding pass first.

Everything else — no relevant evidence, a fabricated citation, a model that
declined — is a *result*, and comes back as one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from app.services.knowledge.errors import (
    AnsweringFailedError,
    AnsweringUnavailableError,
    IndexNotBuiltError,
)
from app.services.knowledge.filters import build_filter
from llm.answers import Answer
from llm.service import RAGAnswerService
from rag.config import RagConfig
from rag.retrieval import RetrievalResponse, RetrievalService

logger = logging.getLogger(__name__)

#: What to tell a caller whose index has never been built. Names the operation
#: rather than a path — the directory is configuration, not a client's
#: business.
INDEX_NOT_BUILT_MESSAGE = (
    "The retrieval index has not been built yet, so there is nothing to "
    "search. Build it by indexing the project documentation and synchronising "
    "the experiment store, then try again."
)


class KnowledgeService:
    """Answers "what do we know about this?" — as evidence, or as an answer."""

    def __init__(
        self,
        rag_config: RagConfig,
        *,
        retrieval: RetrievalService,
        answering: RAGAnswerService | None = None,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            rag_config: Supplies the query, ``top_k`` and threshold limits, so
                a library user and an HTTP client are held to the same rules.
            retrieval: The retrieval service to search with.
            answering: The grounded answer service. ``None`` when answering is
                not configured — search still works, and ``ask`` refuses with
                a 503 rather than the application failing to start.
        """
        self._config = rag_config
        self._retrieval = retrieval
        self._answering = answering

    @property
    def config(self) -> RagConfig:
        """The retrieval configuration in force."""
        return self._config

    @property
    def can_answer(self) -> bool:
        """Whether answer generation is configured and credentialed."""
        return self._answering is not None and self._answering.is_ready

    # -- Readiness ---------------------------------------------------------

    def _require_index(self) -> None:
        """Refuse if no index has been built.

        Raises:
            IndexNotBuiltError: If the store has never been written. A
                corrupt index is a different matter and is left to the
                retrieval layer, which raises
                :class:`~rag.errors.CorruptIndexError` when it tries to read
                it — the two are reported differently because the fix differs.
        """
        store = getattr(self._retrieval, "store", None)
        if store is not None and getattr(store, "is_built", True) is False:
            raise IndexNotBuiltError(
                INDEX_NOT_BUILT_MESSAGE,
                details={"index_built": False},
            )

    def _require_answering(self) -> RAGAnswerService:
        """Return the answer service, or refuse.

        Raises:
            AnsweringUnavailableError: If answering is not configured, or the
                provider has no credential. Checked before retrieval so a
                doomed request does no work.
        """
        if self._answering is None or not self._answering.is_ready:
            raise AnsweringUnavailableError(
                "Answer generation is not configured. Set an API key for the "
                "language-model provider to use this endpoint. Searching the "
                "knowledge base with POST /api/v1/search needs no credential.",
                details={"provider_ready": False},
            )
        return self._answering

    # -- Operations --------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        source_types: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> RetrievalResponse:
        """Find the evidence that bears on a query.

        Args:
            query: What is being asked, in the caller's own words.
            top_k: Most results to return; the configured default when
                omitted, and capped by ``max_top_k``.
            similarity_threshold: Minimum cosine similarity a result must
                reach.
            source_types: Restrict to these kinds of source.
            metadata: Named metadata filters, applied *before* ranking.

        Returns:
            RetrievalResponse: Ranked evidence, best first. Empty when nothing
            matched — which is a truthful answer, not a failure.

        Raises:
            ConfigurationError: If the query, limits or filters are unusable.
            IndexNotBuiltError: If nothing has been indexed yet.
            CorruptIndexError: If the index exists but cannot be read.
        """
        text = self._config.resolve_query(query)
        self._require_index()

        return self._retrieval.search(
            text,
            top_k=top_k,
            min_score=similarity_threshold,
            metadata_filter=build_filter(
                source_types=source_types, metadata=metadata
            ),
        )

    def ask(
        self,
        question: str,
        *,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        source_types: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> Answer:
        """Answer a question from retrieved evidence.

        The safety settings are the server's: there is no way for a caller to
        supply a system prompt, an endpoint, a credential, or to switch off
        grounding or citation validation. What a request may vary is how much
        evidence to look at and where to look for it.

        Args:
            question: What is being asked.
            top_k: How much evidence to retrieve.
            similarity_threshold: Minimum similarity for a passage to count.
            source_types: Restrict evidence to these kinds of source.
            metadata: Named metadata filters.

        Returns:
            Answer: A grounded answer with validated citations, or a status
            explaining why there is not one. ``insufficient_evidence`` and
            ``grounding_failed`` are returned as results, not raised.

        Raises:
            ConfigurationError: If the question, limits or filters are
                unusable.
            AnsweringUnavailableError: If answering is not configured.
            IndexNotBuiltError: If nothing has been indexed yet.
            CorruptIndexError: If the index exists but cannot be read.
        """
        text = self._config.resolve_query(question)
        answering = self._require_answering()
        self._require_index()

        # Validate the filter before generating: an unknown source type should
        # be a 400 about the request, not an answer built from evidence the
        # caller did not mean to search.
        build_filter(source_types=source_types, metadata=metadata)

        answer = answering.answer(
            text,
            top_k=top_k,
            source_types=tuple(source_types),
            equals=dict(metadata or {}) or None,
        )

        # The answer service returns provider and configuration failures as
        # statuses, which is right for a library caller. Over HTTP they are
        # errors: no answer was produced, and a client must not read one out
        # of the body. Results — grounded, insufficient, ungrounded — pass
        # straight through.
        if answer.status.is_failure:
            raise AnsweringFailedError.from_answer_code(
                answer.error_code,
                answer.answer,
                details={"status": answer.status.value},
            )
        return answer

    def describe(self) -> dict[str, Any]:
        """Describe what the knowledge endpoints can currently do.

        Safe to show a caller: it reports whether a credential is configured,
        never what it is, and names no filesystem location.
        """
        payload: dict[str, Any] = {
            "search_available": True,
            "answering_available": self.can_answer,
            "similarity_metric": getattr(
                self._retrieval, "similarity_metric", "cosine"
            ),
            "default_top_k": self._config.top_k,
            "max_top_k": self._config.max_top_k,
            "max_query_length": self._config.max_query_length,
        }
        store = getattr(self._retrieval, "store", None)
        payload["index_built"] = bool(getattr(store, "is_built", False))
        return payload


__all__ = ["INDEX_NOT_BUILT_MESSAGE", "KnowledgeService"]
