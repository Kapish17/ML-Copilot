"""Failures that belong to the API's view of the agent.

Two conditions the agent layer has no opinion about, because they only matter
to something serving requests.

**A budget a client asked for that the server will not grant.** ``agent/``
takes an :class:`~agent.config.AgentConfig` and runs under it; it has no
concept of "the limit somebody else configured", so it cannot refuse a larger
one. The refusal has to happen here, before a config is built.

**A planner with no credential.** The orchestrator already reports this as a
failed run, and that is right for a library caller. This error exists so the
API can refuse *before* any work is done, and answer with the status the
specification asks for rather than after a planning call has been attempted.
"""

from __future__ import annotations

from app.core.errors import MLCopilotError


class AgentServiceError(MLCopilotError):
    """Base class for API-level failures of the agent endpoint."""

    code = "agent_service_error"
    status_code = 500


class AgentBudgetError(AgentServiceError):
    """A requested budget exceeds the server's configured limit.

    Rejected rather than silently capped. Capping would let a client believe
    it had been granted six tool calls when it had three, and the difference
    only shows up as a partial result it cannot explain. Saying no, with the
    limit named, is the shorter conversation.
    """

    code = "invalid_agent_budget"
    status_code = 422


class AgentUnavailableError(AgentServiceError):
    """The agent cannot run because its planner is not configured.

    Raised before the orchestrator is entered, so a request that cannot
    possibly succeed does not spend a planning call first.
    """

    code = "llm_not_configured"
    status_code = 503


__all__ = ["AgentBudgetError", "AgentServiceError", "AgentUnavailableError"]
