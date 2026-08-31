"""Failures of the agent layer.

Every one of these is a *refusal or a breakdown of the orchestration itself* —
never a failure of the work a tool was asked to do. A dataset that cannot be
profiled, a model that will not fit, a query that matches nothing: those are
answered by the underlying layers in their own vocabulary and arrive here as
observations, not exceptions.

What lives here is the smaller and more important set: the planner asked for
something that does not exist, asked for it with arguments that do not
validate, produced text that is not a decision at all, or asked for more work
than the budget allows. These are the moments where an unbounded agent would
improvise. This one stops and says what happened.

Each error carries a stable ``code`` so a caller — a test, a log, and in a
later commit an HTTP layer — can tell the cases apart without matching on
prose.
"""

from __future__ import annotations

from typing import Any


class AgentError(Exception):
    """Base class for every failure of the agent layer."""

    #: Stable, machine-readable identifier for this kind of failure.
    code = "agent_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        """Record the message and any structured, already-safe details.

        Args:
            message: Explanation written for a person reading a client. It
                must not contain a credential, a filesystem path or a
                provider's own exception text.
            details: Extra machine-readable context. Whatever goes in here is
                assumed to have been sanitised by the caller.
        """
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details or {})

    def as_dict(self) -> dict[str, Any]:
        """Render the failure as plain JSON-safe values."""
        return {"code": self.code, "message": self.message, "details": self.details}


# ---------------------------------------------------------------------------
# Registry and tools
# ---------------------------------------------------------------------------


class ToolError(AgentError):
    """Base class for anything that went wrong with a tool."""

    code = "tool_error"


class UnknownToolError(ToolError):
    """The planner named a tool that is not registered.

    The single most important refusal in the package. There is no fallback
    that tries to run it anyway, no fuzzy match to the nearest name, and no
    dynamic import: a name that is not in the registry cannot be executed by
    any path.
    """

    code = "unknown_tool"


class DuplicateToolError(ToolError):
    """Two tools were registered under one name.

    A programming error, caught at registration rather than at the moment one
    silently shadows the other.
    """

    code = "duplicate_tool"


class ToolValidationError(ToolError):
    """The arguments for a tool did not validate.

    Raised *before* the tool runs. A tool never sees an argument it has not
    approved, so validation cannot be skipped by a planner that words its
    request cleverly.
    """

    code = "invalid_tool_arguments"


class ToolExecutionError(ToolError):
    """A tool raised while doing its work.

    The underlying exception is logged, never returned: a stack trace, a
    filesystem path or a provider's message must not reach a client through
    an observation.
    """

    code = "tool_execution_failed"


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class PlannerError(AgentError):
    """Base class for a planner that could not be used."""

    code = "planner_error"


class MalformedPlanError(PlannerError):
    """The planner produced something that is not a decision.

    Prose, a code block, a half-written JSON object, an instruction to run a
    shell command — all the same thing from here: text that does not parse as
    one of the two moves the protocol allows. It is reported as malformed
    rather than interpreted, because interpreting it is exactly how an agent
    ends up doing something nobody asked for.
    """

    code = "malformed_plan"


class PlannerUnavailableError(PlannerError):
    """The planner's provider could not be used at all.

    A missing credential or an uninstalled SDK. Nothing was attempted, and
    retrying without changing the configuration will not help.
    """

    code = "planner_unavailable"


class PlannerProviderError(PlannerError):
    """The planner's provider failed while generating.

    Someone else's service: a timeout, a rate limit, an outage. The vendor's
    exception is caught and replaced — its message can carry a request URL,
    headers or an echoed payload.
    """

    code = "planner_provider_error"


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


class BudgetExhaustedError(AgentError):
    """The run reached one of its hard limits.

    Not an error in the sense that something broke — the limits exist to be
    reached. It is raised so the orchestrator stops at a single, obvious
    place, and is turned into a structured partial result rather than being
    propagated to a caller.
    """

    code = "budget_exhausted"


class AgentConfigurationError(AgentError):
    """The agent was configured with values it cannot run under."""

    code = "agent_configuration_error"


__all__ = [
    "AgentConfigurationError",
    "AgentError",
    "BudgetExhaustedError",
    "DuplicateToolError",
    "MalformedPlanError",
    "PlannerError",
    "PlannerProviderError",
    "PlannerUnavailableError",
    "ToolError",
    "ToolExecutionError",
    "ToolValidationError",
    "UnknownToolError",
]
