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
class WorkflowStepReport:
    """One planned step, and what became of it."""

    step: str
    tool: str
    #: The short label from the plan — what this step was *for*. A label, not
    #: reasoning: it says "Explain the winning model", never why the planner
    #: thought that was next.
    purpose: str
    #: ``ok``, ``unavailable``, ``rejected``, ``failed`` — or ``skipped``, when
    #: no call was made at all because the step it needed produced nothing, or
    #: a limit ended the run first.
    status: str
    depends_on: tuple[str, ...] = ()
    #: Why it did not run or did not work. An authored sentence.
    reason: str | None = None

    @property
    def succeeded(self) -> bool:
        """Whether this step produced a usable result."""
        return self.status == "ok"

    def as_dict(self) -> dict[str, Any]:
        """Render the step as plain JSON-safe values."""
        return {
            "step": self.step,
            "tool": self.tool,
            "purpose": self.purpose,
            "status": self.status,
            "depends_on": list(self.depends_on),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class WorkflowReport:
    """The plan a run executed, beside what actually happened to it.

    This is the *whole* of what a caller learns about planning. It says what
    was going to be done, in what order, and how each part turned out. It does
    not say how the planner arrived at any of it, because that is not returned,
    stored or logged — see the note at the top of this module.
    """

    goal: str
    objective: str
    steps: tuple[WorkflowStepReport, ...] = ()

    @property
    def completed_step_count(self) -> int:
        """How many planned steps produced a result."""
        return sum(1 for step in self.steps if step.succeeded)

    @property
    def is_complete(self) -> bool:
        """Whether every planned step produced a result."""
        return bool(self.steps) and self.completed_step_count == len(self.steps)

    def summary_lines(self) -> list[str]:
        """The plan as a person reads it, one numbered line per step."""
        return [
            f"{index}. {step.purpose}"
            for index, step in enumerate(self.steps, start=1)
        ]

    def executed_lines(self) -> list[str]:
        """What was actually carried out, for the answer prompt.

        Not the same list as :meth:`summary_lines`: a step that was skipped is
        described as skipped. The distinction is the point — an answer written
        from the plan rather than from the execution is how "and then I
        explained the model" gets written about a step that never ran.
        """
        lines: list[str] = []
        for index, step in enumerate(self.steps, start=1):
            suffix = "" if step.succeeded else f" — NOT DONE ({step.status})"
            lines.append(f"{index}. {step.purpose}{suffix}")
        return lines

    def as_dict(self) -> dict[str, Any]:
        """Render the report as plain JSON-safe values."""
        return {
            "goal": self.goal,
            "objective": self.objective,
            "steps": [step.as_dict() for step in self.steps],
            "summary": self.summary_lines(),
            "planned_step_count": len(self.steps),
            "completed_step_count": self.completed_step_count,
            "is_complete": self.is_complete,
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
    #: The plan this run executed, when it had one. ``None`` for a run that
    #: took the one-decision-at-a-time path, which is why every field of it is
    #: optional in the API schema: a client written before plans existed sees
    #: the same response it always did.
    workflow: WorkflowReport | None = None

    @property
    def tools_used(self) -> tuple[str, ...]:
        """Every tool this run called, in order of first use."""
        seen: list[str] = []
        for call in self.tool_calls:
            name = str(call.get("tool_name", ""))
            if name and name not in seen:
                seen.append(name)
        return tuple(seen)

    def execution_summary(self) -> dict[str, Any]:
        """A short, honest account of the run's shape.

        Enough for a client to say "3 of 4 steps completed" without reading the
        observations, and deliberately not enough to reconstruct anything the
        planner thought. ``partial`` is the field worth reading: it is true
        whenever the answer covers less than the question asked for, whatever
        the reason.
        """
        return {
            "planned": self.workflow is not None,
            "steps_planned": len(self.workflow.steps) if self.workflow else 0,
            "steps_completed": (
                self.workflow.completed_step_count if self.workflow else 0
            ),
            "workflow_complete": bool(self.workflow and self.workflow.is_complete),
            "tools_used": list(self.tools_used),
            "tool_call_count": self.tool_call_count,
            "partial": self.status is AgentStatus.PARTIAL,
            "stopped_by": self.error_code,
        }

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
            "workflow": self.workflow.as_dict() if self.workflow else None,
            "execution_summary": self.execution_summary(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "duration_ms": self.duration_ms,
        }


__all__ = [
    "AgentCitation",
    "AgentResult",
    "AgentStatus",
    "WorkflowReport",
    "WorkflowStepReport",
]
