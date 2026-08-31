"""What a run knows so far, and what it has left to spend.

The state is deliberately the *only* memory an agent run has. There is no
module-level accumulator, no cache keyed on the question, nothing carried
between runs. Two runs of the same question start identically, which is what
makes the loop testable at all.

It is also deliberately dull data. Everything in it is JSON-legal by the time
it arrives — observations are made safe as they are recorded, arguments are
summarised, citations are strings. A DataFrame, a fitted pipeline, a SHAP
explainer or a provider client cannot be in here, and a test asserts it by
serialising the whole thing.

**Budgets live here rather than in the loop.** The loop asks the state whether
it may spend, and the state answers. Keeping the arithmetic in one place is
why there is no path that forgets to decrement, and why a limit cannot be
reached "almost" — :meth:`ExecutionState.remaining_tool_calls` is the single
truth about what is left.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent.config import AgentConfig
from agent.observations import Observation, ObservationStatus, ensure_json_safe

#: How a call identifier is built: the run's own ordinal, not anything the
#: planner chose. Deterministic, collision-free within a run, and incapable of
#: carrying a path, a credential or anything else a model wrote.
CALL_ID_TEMPLATE = "call-{ordinal:02d}"


@dataclass
class ExecutionState:
    """Everything one agent run has done, and everything it may still do."""

    question: str
    config: AgentConfig
    #: Every recorded call, in order.
    observations: list[Observation] = field(default_factory=list)
    #: Planning turns taken, whether or not they led to a tool call.
    iterations: int = 0
    #: Non-fatal notes for the caller: a truncated observation, a tool that
    #: reported itself unavailable, a budget that ran out.
    warnings: list[str] = field(default_factory=list)
    started_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # -- Counters ----------------------------------------------------------

    @property
    def tool_call_count(self) -> int:
        """How many tool calls have been recorded, including failures.

        A rejected or failed call still costs budget. Otherwise a planner
        could retry a broken call indefinitely without ever spending
        anything, which is the loop the budget exists to prevent.
        """
        return len(self.observations)

    @property
    def remaining_tool_calls(self) -> int:
        """How many more tool calls this run may make."""
        return max(0, self.config.max_tool_calls - self.tool_call_count)

    @property
    def remaining_iterations(self) -> int:
        """How many more planning turns this run may take."""
        return max(0, self.config.max_iterations - self.iterations)

    @property
    def context_chars(self) -> int:
        """Characters of observation text accumulated so far."""
        return sum(
            len(str(observation.output)) + len(observation.error or "")
            for observation in self.observations
        )

    @property
    def remaining_context_chars(self) -> int:
        """Characters of observation text this run may still accumulate."""
        return max(0, self.config.max_context_chars - self.context_chars)

    def can_call_tool(self) -> bool:
        """Whether every budget still allows one more tool call."""
        return (
            self.remaining_tool_calls > 0
            and self.remaining_iterations > 0
            and self.remaining_context_chars > 0
        )

    def exhausted_budget(self) -> str | None:
        """Name the budget that has run out, or ``None`` while all remain.

        Reported so a partial result can say *which* limit stopped it — "it
        stopped" is much less useful than "it used all six tool calls".
        """
        if self.remaining_tool_calls <= 0:
            return "max_tool_calls"
        if self.remaining_iterations <= 0:
            return "max_iterations"
        if self.remaining_context_chars <= 0:
            return "max_context_chars"
        return None

    # -- Recording ---------------------------------------------------------

    def next_call_id(self) -> str:
        """The identifier for the next call.

        Derived from the run's own ordinal. The planner never supplies one:
        an identifier a model chooses is an identifier that can be made to
        look like a path, collide with another call, or carry text into a log.
        """
        return CALL_ID_TEMPLATE.format(ordinal=self.tool_call_count + 1)

    def record(self, observation: Observation) -> Observation:
        """Store one observation and return it."""
        self.observations.append(observation)
        if observation.status is ObservationStatus.UNAVAILABLE:
            self.warn(
                f"'{observation.tool_name}' could not provide a result: "
                f"{observation.error or observation.output.get('reason', 'unavailable')}."
            )
        return observation

    def begin_iteration(self) -> int:
        """Count one planning turn and return the new total."""
        self.iterations += 1
        return self.iterations

    def warn(self, message: str) -> None:
        """Record a note for the caller, without repeating it."""
        if message not in self.warnings:
            self.warnings.append(message)

    # -- Views -------------------------------------------------------------

    @property
    def successful_observations(self) -> tuple[Observation, ...]:
        """Only the calls that produced a usable result."""
        return tuple(item for item in self.observations if item.succeeded)

    def citations(self) -> tuple[str, ...]:
        """Every citation identifier the run actually retrieved.

        This is the complete set the final answer may cite. An identifier that
        is not in here was not retrieved, and citing it is a fabrication no
        matter how plausible it looks.
        """
        found: list[str] = []
        for observation in self.observations:
            for citation in observation.citations:
                if citation not in found:
                    found.append(citation)
        return tuple(found)

    def experiment_ids(self) -> tuple[str, ...]:
        """Every experiment this run created or read, in order of appearance."""
        found: list[str] = []
        for observation in self.observations:
            identifier = observation.output.get("experiment_id")
            if isinstance(identifier, str) and identifier and identifier not in found:
                found.append(identifier)
        return tuple(found)

    def tool_calls(self) -> list[dict[str, Any]]:
        """A compact record of what was called, with what, and how it went."""
        return [
            {
                "call_id": observation.call_id,
                "tool_name": observation.tool_name,
                "status": observation.status.value,
                "arguments": ensure_json_safe(observation.input_summary),
                "duration_ms": observation.duration_ms,
            }
            for observation in self.observations
        ]

    def as_dict(self) -> dict[str, Any]:
        """Render the whole state as plain JSON-safe values.

        What is **not** here is as deliberate as what is: no prompt, no
        planner reasoning, no raw model output, no fitted model, no data. A
        caller sees what was decided and what came back, not how the decision
        was reached.
        """
        return {
            "question": self.question,
            "iterations": self.iterations,
            "tool_call_count": self.tool_call_count,
            "remaining_tool_calls": self.remaining_tool_calls,
            "tool_calls": self.tool_calls(),
            "observations": [item.as_dict() for item in self.observations],
            "citations": list(self.citations()),
            "experiment_ids": list(self.experiment_ids()),
            "warnings": list(self.warnings),
            "budgets": self.config.as_dict(),
        }


__all__ = ["CALL_ID_TEMPLATE", "ExecutionState"]
