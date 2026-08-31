"""What an agent run returns.

One object, JSON-safe throughout, whose ``status`` is the field that matters.
An agent that ran three tools and produced a beautifully worded answer with a
fabricated citation has not answered the question, and the result says so
rather than burying it.

What is deliberately **not** in here:

- No prompt. Not the system prompt, not the rendered observations.
- No chain-of-thought, no reasoning trace, no field named anything like it.
  The planner's optional one-line note about *what* it chose is recorded with
  the tool call as metadata; how it decided is not returned, not stored and
  not logged.
- No raw model output beyond the answer itself.
- No credential, no filesystem path, no fitted model, no DataFrame.

A caller gets what was done and what came back. That is enough to audit a run
— which tools ran, with what, in what order, and which evidence the answer
rests on — without turning the model's internal monologue into an API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from agent.observations import ensure_json_safe


class AgentStatus(str, Enum):
    """How one agent run turned out."""

    #: An answer supported by the observations, with every citation valid.
    COMPLETED = "completed"
    #: Real work was done and is reported, but something is missing — a tool
    #: was unavailable, a budget ran out mid-plan. The answer covers what is
    #: there and says what is not.
    PARTIAL = "partial"
    #: Nothing the tools returned supports an answer. Not a failure: the
    #: honest response to a question this system cannot answer.
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    #: An answer was produced but cannot be trusted: it cited a source that
    #: was never retrieved, or cited nothing while evidence was available.
    #: The text is returned so a person can see what happened; it is not an
    #: answer.
    GROUNDING_FAILED = "grounding_failed"
    #: The run could not be completed: the planner was unusable, or it never
    #: produced a decision.
    FAILED = "failed"

    @property
    def is_answer(self) -> bool:
        """True only when the text may be presented to a user as an answer."""
        return self is AgentStatus.COMPLETED

    @property
    def is_failure(self) -> bool:
        """True when no answer was produced at all."""
        return self is AgentStatus.FAILED


@dataclass(frozen=True)
class AgentCitation:
    """One source backing part of an answer.

    Only the identifier came from the model. Everything beside it was read
    from the observation the passage actually came from, so a citation's title
    and score are trustworthy even when the prose is not.
    """

    citation_id: str
    source_type: str = ""
    source_title: str = ""
    source_reference: str = ""
    score: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Render the citation as plain JSON-safe values."""
        return {
            "citation_id": self.citation_id,
            "source_type": self.source_type,
            "source_title": self.source_title,
            "source_reference": self.source_reference,
            "score": self.score,
        }


@dataclass(frozen=True)
class AgentResult:
    """One complete agent run, as the caller receives it."""

    question: str
    status: AgentStatus
    final_answer: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    citations: tuple[AgentCitation, ...] = ()
    #: Identifiers the model produced that were never retrieved. Reported
    #: rather than removed — a fabricated source is the most important thing
    #: to know about an answer.
    rejected_citations: tuple[str, ...] = ()
    #: Exactly what the model was permitted to cite, for audit.
    allowed_citations: tuple[str, ...] = ()
    experiment_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    iterations: int = 0
    tool_call_count: int = 0
    #: Stable code when the run failed or was stopped.
    error_code: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float | None = None

    @property
    def is_answer(self) -> bool:
        """True only when the answer may be presented as one."""
        return self.status.is_answer

    @property
    def citation_ids(self) -> tuple[str, ...]:
        """The identifiers actually used, in order."""
        return tuple(citation.citation_id for citation in self.citations)

    def as_dict(self) -> dict[str, Any]:
        """Render the whole result as plain JSON-safe values."""
        return {
            "question": self.question,
            "status": self.status.value,
            "final_answer": self.final_answer,
            "is_answer": self.is_answer,
            "tool_calls": ensure_json_safe(self.tool_calls),
            "observations": ensure_json_safe(self.observations),
            "citations": [citation.as_dict() for citation in self.citations],
            "citation_ids": list(self.citation_ids),
            "rejected_citations": list(self.rejected_citations),
            "allowed_citations": list(self.allowed_citations),
            "experiment_ids": list(self.experiment_ids),
            "warnings": list(self.warnings),
            "iterations": self.iterations,
            "tool_call_count": self.tool_call_count,
            "error_code": self.error_code,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "duration_ms": self.duration_ms,
        }


__all__ = ["AgentCitation", "AgentResult", "AgentStatus"]
