"""What a grounded question-answering call returns.

An :class:`Answer` is the text, the sources that back it, and an honest
statement of how much to trust it. The status is not decoration: it is the
difference between "here is an answer supported by these passages" and "the
model wrote something and none of it is backed", and a caller that ignores it
is using an ungrounded system with extra steps.

Everything here is JSON-safe and free of internals. No embedding vector, no
provider object, no raw response, no prompt, and no credential — the prompt
in particular is deliberately absent, because it contains the retrieved
evidence and returning it on every answer would put a large, quotable copy of
the corpus into every log that captures a response.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AnswerStatus(str, Enum):
    """How much an answer can be relied on.

    A closed vocabulary rather than free-form strings, so a caller can branch
    on it and a test can assert it.
    """

    #: Answered from retrieved evidence, with at least one valid citation and
    #: no fabricated ones. The only status that means "usable as an answer".
    GROUNDED = "grounded"
    #: Retrieval found nothing worth grounding in, or the model said the
    #: evidence does not cover the question. No claim is being made.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    #: The model produced text, but it cannot be trusted: it cited a source
    #: that was never retrieved, or it cited nothing while evidence was
    #: available. The text is returned so a human can see what happened; it is
    #: not presented as an answer.
    GROUNDING_FAILED = "grounding_failed"
    #: The provider failed — timeout, rate limit, outage, unusable response.
    #: Nothing was answered.
    PROVIDER_ERROR = "provider_error"
    #: The layer is not configured to generate: no API key, no SDK, an unknown
    #: provider. Nothing was attempted.
    CONFIGURATION_ERROR = "configuration_error"

    @property
    def is_usable(self) -> bool:
        """True only for an answer a caller may present as an answer."""
        return self is AnswerStatus.GROUNDED

    @property
    def is_failure(self) -> bool:
        """True when nothing was answered, as opposed to answered badly."""
        return self in (
            AnswerStatus.PROVIDER_ERROR,
            AnswerStatus.CONFIGURATION_ERROR,
        )


@dataclass(frozen=True)
class Citation:
    """One source that backs part of an answer.

    Built from retrieved evidence, never from the generated text: the model
    supplies an identifier, and everything else here is looked up from what
    was actually retrieved. That is what makes a citation's title and score
    trustworthy even when the model's prose is not.
    """

    citation_id: str
    source_type: str
    source_title: str
    source_reference: str
    relevance_score: float
    #: A short opening extract of the cited passage, so a reader can see what
    #: was cited without a second lookup. Never the whole passage.
    excerpt: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Render the citation as plain JSON-safe values."""
        return {
            "citation_id": self.citation_id,
            "source_type": self.source_type,
            "source_title": self.source_title,
            "source_reference": self.source_reference,
            "relevance_score": self.relevance_score,
            "excerpt": self.excerpt,
        }


@dataclass(frozen=True)
class AnswerMetadata:
    """How an answer was produced.

    Enough to reproduce, audit or debug a call, and nothing more. No prompt,
    no raw response, no request identifiers, no credential.
    """

    provider: str = ""
    model: str = ""
    retrieved_count: int = 0
    context_count: int = 0
    context_truncated: bool = False
    context_characters: int = 0
    approximate_context_tokens: int = 0
    below_threshold_count: int = 0
    latency_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Render the metadata as plain JSON-safe values."""
        return {
            "provider": self.provider,
            "model": self.model,
            "retrieved_count": self.retrieved_count,
            "context_count": self.context_count,
            "context_truncated": self.context_truncated,
            "context_characters": self.context_characters,
            "approximate_context_tokens": self.approximate_context_tokens,
            "below_threshold_count": self.below_threshold_count,
            "latency_seconds": self.latency_seconds,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "finish_reason": self.finish_reason,
        }


@dataclass(frozen=True)
class Answer:
    """A grounded answer, or an honest account of why there is not one."""

    question: str
    answer: str
    status: AnswerStatus
    citations: tuple[Citation, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: AnswerMetadata = field(default_factory=AnswerMetadata)
    #: Citation identifiers the model produced that were **not** in the
    #: retrieved evidence. Reported rather than quietly removed: a fabricated
    #: source is the most important thing a reader can know about an answer.
    rejected_citations: tuple[str, ...] = ()
    #: Exactly what the model was permitted to cite, for auditing.
    allowed_citations: tuple[str, ...] = ()
    #: The error code when the status is a failure, e.g. ``"llm_timeout"``.
    error_code: str | None = None

    @property
    def is_grounded(self) -> bool:
        """True only when the answer is backed by retrieved evidence."""
        return self.status is AnswerStatus.GROUNDED

    @property
    def citation_ids(self) -> tuple[str, ...]:
        """The identifiers of the sources backing this answer."""
        return tuple(citation.citation_id for citation in self.citations)

    def as_dict(self) -> dict[str, Any]:
        """Render the whole answer as plain JSON-safe values."""
        return {
            "question": self.question,
            "answer": self.answer,
            "status": self.status.value,
            "is_grounded": self.is_grounded,
            "citations": [citation.as_dict() for citation in self.citations],
            "citation_ids": list(self.citation_ids),
            "rejected_citations": list(self.rejected_citations),
            "allowed_citations": list(self.allowed_citations),
            "warnings": list(self.warnings),
            "error_code": self.error_code,
            "metadata": self.metadata.as_dict(),
        }


def build_failure(
    *,
    question: str,
    status: AnswerStatus,
    message: str,
    error_code: str | None = None,
    metadata: AnswerMetadata | None = None,
    warnings: Sequence[str] = (),
) -> Answer:
    """Build an answer that reports a failure rather than answering.

    Used for every path that does not produce grounded text, so a caller
    always receives the same object and never has to distinguish "an answer"
    from "an exception that was caught somewhere".
    """
    return Answer(
        question=question,
        answer=message,
        status=status,
        metadata=metadata or AnswerMetadata(),
        warnings=tuple(warnings),
        error_code=error_code,
    )


__all__ = [
    "Answer",
    "AnswerMetadata",
    "AnswerStatus",
    "Citation",
    "build_failure",
]
