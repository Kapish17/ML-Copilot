"""Answering a question from retrieved evidence.

::

    question
       ↓  retrieve
    evidence
       ↓  is any of it good enough?      → no  → insufficient_evidence
       ↓  build prompt (evidence as data)
       ↓  generate                        → fails → provider/configuration error
    text
       ↓  validate citations              → fabricated → grounding_failed
       ↓                                  → none       → grounding_failed
    grounded answer + citations

The service coordinates and decides; it computes nothing. Retrieval belongs to
``rag/``, generation belongs to a provider, citation checking belongs to
:mod:`llm.grounding`. There is no SDK code here and no vendor name — the
service holds a retriever and a provider, both behind interfaces, which is
what lets the whole flow be tested with a fake of each.

**Two places it refuses to call the model at all.** When retrieval returns
nothing above the evidence threshold, there is nothing to ground an answer in,
so asking a model would only invite one to be invented — the service answers
``insufficient_evidence`` without spending a call. And when the provider has
no credential, it says so rather than failing mid-flight.

**The failure paths return an Answer, not an exception.** A caller asking a
question always gets the same object back, with a status saying what happened,
so "the provider timed out" and "the model fabricated a source" are handled
the same way as a good answer — by reading a field, not by catching something.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from llm.answers import (
    Answer,
    AnswerMetadata,
    AnswerStatus,
    build_failure,
)
from llm.config import LLMConfig
from llm.context import EvidenceContext, build_context
from llm.errors import (
    LLMConfigurationError,
    LLMError,
    LLMProviderError,
)
from llm.grounding import (
    build_citations,
    looks_like_injection_attempt,
    validate_citations,
)
from llm.messages import GenerationRequest, build_messages
from llm.prompts import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    INSUFFICIENT_EVIDENCE_MARKER,
    build_system_prompt,
    build_user_prompt,
)
from llm.providers.base import LLMProvider
from rag.retrieval import RetrievalResponse

logger = logging.getLogger(__name__)

#: Error codes reported on a failed answer, so a caller can branch without
#: matching on message text.
ERROR_CODES: dict[type[Exception], str] = {}


@runtime_checkable
class Retriever(Protocol):
    """The part of a retrieval service this module uses.

    Structural, so the answer service depends on the *shape* of a retriever
    rather than on a concrete one. ``rag.RetrievalService`` satisfies it, and
    so does a stand-in in a test.
    """

    def search(self, question: str, **kwargs: Any) -> RetrievalResponse:
        """Return ranked evidence for a question."""
        ...  # pragma: no cover - protocol


def _error_code(exc: LLMError) -> str:
    """Turn an exception class into a stable, snake_case code."""
    name = type(exc).__name__
    if name.startswith("LLM"):
        name = name[3:]
    if name.endswith("Error"):
        name = name[: -len("Error")]
    stem = "".join(
        f"_{char.lower()}" if char.isupper() else char for char in name
    ).lstrip("_")
    return f"llm_{stem}" if stem else "llm_error"


class RAGAnswerService:
    """Answers questions from retrieved evidence, and says how well."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        retriever: Retriever,
        provider: LLMProvider,
        propagate_retrieval_errors: bool = False,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            config: Every limit, threshold and generation setting.
            retriever: Where evidence comes from. Any object satisfying
                :class:`Retriever`.
            provider: How text is generated. Any object satisfying
                :class:`~llm.providers.base.LLMProvider`.
            propagate_retrieval_errors: What to do when retrieval itself
                fails. ``False`` (the default) degrades to
                ``insufficient_evidence``, which is right for a script or a
                notebook: the user asked a question and the honest answer is
                that it cannot be answered. ``True`` lets the failure through,
                which is right for a caller that must tell *"there is no
                relevant evidence"* apart from *"the retrieval system is
                broken"* — an HTTP API, for instance, where the first is a
                200 and the second is a 503 someone needs to act on.
        """
        self._config = config
        self._retriever = retriever
        self._provider = provider
        self._propagate_retrieval_errors = propagate_retrieval_errors

    @property
    def config(self) -> LLMConfig:
        """The configuration this service runs with."""
        return self._config

    @property
    def provider(self) -> LLMProvider:
        """The provider generating the answers."""
        return self._provider

    @property
    def is_ready(self) -> bool:
        """Whether a generation call could be attempted right now."""
        return self._provider.is_ready

    # -- The flow ----------------------------------------------------------

    def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        source_types: Sequence[str] = (),
        equals: dict[str, Any] | None = None,
    ) -> Answer:
        """Answer a question from retrieved evidence.

        Args:
            question: What is being asked, in the user's own words.
            top_k: Chunks to retrieve; the configured maximum when omitted.
            source_types: Restrict evidence to these kinds of source.
            equals: Metadata the evidence must match — the hybrid path, for a
                question about one task type or one dataset.

        Returns:
            Answer: Grounded text with citations, or a status explaining why
            there is none. This method does not raise for an expected failure;
            it reports it.
        """
        asked = (question or "").strip()
        if not asked:
            return build_failure(
                question=question or "",
                status=AnswerStatus.CONFIGURATION_ERROR,
                message="A question is required.",
                error_code="empty_question",
            )

        started = time.perf_counter()
        retrieval = self._retrieve(asked, top_k=top_k, source_types=source_types, equals=equals)
        context = build_context(retrieval.results, self._config)
        warnings: list[str] = []

        if context.truncated:
            warnings.append(
                f"Only {context.context_count} of {context.retrieved_count} "
                "retrieved passages fitted the context limit; the rest were "
                "left out. Lower-ranked evidence was dropped first."
            )
        if context.below_threshold_count:
            warnings.append(
                f"{context.below_threshold_count} retrieved passage(s) scored "
                f"below the {self._config.min_evidence_score} evidence "
                "threshold and were not used."
            )
        if looks_like_injection_attempt(context):
            warnings.append(
                "A retrieved passage contains text that reads like an "
                "instruction. Retrieved content is supplied to the model as "
                "untrusted data and is never followed as an instruction."
            )

        if context.is_empty:
            return self._insufficient(asked, context, warnings)

        try:
            return self._generate(asked, context, warnings, started)
        except LLMConfigurationError as exc:
            return self._failure(asked, context, exc, AnswerStatus.CONFIGURATION_ERROR, warnings)
        except LLMProviderError as exc:
            return self._failure(asked, context, exc, AnswerStatus.PROVIDER_ERROR, warnings)
        except LLMError as exc:  # pragma: no cover - defensive
            return self._failure(asked, context, exc, AnswerStatus.PROVIDER_ERROR, warnings)

    # -- Steps -------------------------------------------------------------

    def _retrieve(
        self,
        question: str,
        *,
        top_k: int | None,
        source_types: Sequence[str],
        equals: dict[str, Any] | None,
    ) -> RetrievalResponse:
        """Fetch evidence, handling a retrieval failure as configured.

        By default a broken index produces "I cannot answer that" rather than
        a stack trace: the caller asked a question, and the honest response to
        a question we cannot gather evidence for is that we cannot answer it.

        With ``propagate_retrieval_errors`` the failure is raised instead, so
        a caller that distinguishes "nothing relevant was found" from "the
        retrieval system is broken" can report them differently. Answering
        "no evidence" to a corrupt index tells the user their question is
        unanswerable when the truth is that something needs fixing.

        Raises:
            Exception: Whatever retrieval raised, when the service was built
                with ``propagate_retrieval_errors=True``.
        """
        wanted = top_k if top_k is not None else self._config.max_retrieved_chunks
        try:
            return self._retriever.search(
                question,
                top_k=wanted,
                source_types=tuple(source_types),
                equals=equals,
            )
        except Exception as exc:  # noqa: BLE001 - any retrieval failure
            logger.warning("Retrieval failed: %s", type(exc).__name__)
            if self._propagate_retrieval_errors:
                raise
            return RetrievalResponse(question=question)

    def _generate(
        self,
        question: str,
        context: EvidenceContext,
        warnings: list[str],
        started: float,
    ) -> Answer:
        """Build the prompt, call the provider and validate what comes back."""
        request = GenerationRequest(
            messages=build_messages(
                build_system_prompt(), build_user_prompt(question, context)
            ),
            model=self._config.model,
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_output_tokens,
            timeout_seconds=self._config.timeout_seconds,
        )
        result = self._provider.generate(request)

        metadata = self._metadata(
            context,
            provider=result.provider,
            model=result.model,
            latency=round(time.perf_counter() - started, 3),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
        )

        if result.is_truncated:
            warnings.append(
                "The model stopped because it reached the output token limit, "
                "so the answer may be incomplete."
            )

        text = result.text.strip()

        # The model's declared way of saying the evidence does not answer the
        # question. Honoured as an honest refusal rather than being mistaken
        # for an answer that failed to cite anything.
        if INSUFFICIENT_EVIDENCE_MARKER in text:
            explanation = text.replace(INSUFFICIENT_EVIDENCE_MARKER, "").strip()
            return Answer(
                question=question,
                answer=explanation or INSUFFICIENT_EVIDENCE_ANSWER,
                status=AnswerStatus.INSUFFICIENT_EVIDENCE,
                metadata=metadata,
                warnings=tuple(warnings),
                allowed_citations=context.allowed_citations,
            )

        report = validate_citations(text, context)

        if not report.is_grounded:
            return Answer(
                question=question,
                answer=text,
                status=AnswerStatus.GROUNDING_FAILED,
                citations=build_citations(report.valid, context),
                rejected_citations=report.fabricated,
                allowed_citations=context.allowed_citations,
                metadata=metadata,
                warnings=tuple([*warnings, *report.reasons]),
                error_code="grounding_failed",
            )

        return Answer(
            question=question,
            answer=text,
            status=AnswerStatus.GROUNDED,
            citations=build_citations(report.valid, context),
            allowed_citations=context.allowed_citations,
            metadata=metadata,
            warnings=tuple(warnings),
        )

    def _insufficient(
        self, question: str, context: EvidenceContext, warnings: list[str]
    ) -> Answer:
        """Decline to answer, without calling the model.

        There is nothing to ground an answer in, so a generation call could
        only produce an ungrounded one. Not making it is both cheaper and
        safer.
        """
        return Answer(
            question=question,
            answer=INSUFFICIENT_EVIDENCE_ANSWER,
            status=AnswerStatus.INSUFFICIENT_EVIDENCE,
            metadata=self._metadata(context, provider=self._provider.name),
            warnings=tuple(warnings),
            allowed_citations=(),
        )

    def _failure(
        self,
        question: str,
        context: EvidenceContext,
        exc: LLMError,
        status: AnswerStatus,
        warnings: list[str],
    ) -> Answer:
        """Report a provider or configuration failure as an answer.

        The exception's message is used because every message in this layer is
        written for a reader and carries no credential, path or SDK internal —
        see ``llm/errors.py``. Nothing from a vendor's exception reaches here.
        """
        logger.warning("%s during generation: %s", type(exc).__name__, exc.message)
        return build_failure(
            question=question,
            status=status,
            message=exc.message,
            error_code=_error_code(exc),
            metadata=self._metadata(context, provider=self._provider.name),
            warnings=warnings,
        )

    def _metadata(
        self,
        context: EvidenceContext,
        *,
        provider: str = "",
        model: str = "",
        latency: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        finish_reason: str | None = None,
    ) -> AnswerMetadata:
        """Assemble the metadata reported alongside an answer."""
        return AnswerMetadata(
            provider=provider or self._provider.name,
            model=model or self._config.model,
            retrieved_count=context.retrieved_count,
            context_count=context.context_count,
            context_truncated=context.truncated,
            context_characters=context.character_count,
            approximate_context_tokens=context.approximate_tokens,
            below_threshold_count=context.below_threshold_count,
            latency_seconds=latency,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    def describe(self) -> dict[str, Any]:
        """Describe how this service is configured, without any credential."""
        return {
            **self._config.describe(),
            "provider_ready": self.is_ready,
        }


__all__ = ["RAGAnswerService", "Retriever"]
