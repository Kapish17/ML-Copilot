"""Asking a model what to do next, through the provider abstraction.

The planner is a thin thing on purpose. It builds a prompt from the registry
and the observations, asks the provider for one short response, and hands the
text to :func:`agent.plans.parse_plan`. It does not decide whether the chosen
tool exists, whether the arguments are valid, or whether the budget allows the
call — those are the orchestrator's and the registry's, and keeping them out
of here means a planner cannot be the thing that authorises a call.

It reaches the model only through :class:`~llm.providers.base.LLMProvider`.
No SDK is imported, no endpoint is named, no credential is read here — the
same contract the answer service has kept since Commit 10, for the same
reason: swapping the model is a change to which provider is constructed, not a
change to this file.

Provider failures are translated into agent errors. A vendor's exception can
carry a request URL, a header or an echoed payload, so it is caught and
replaced with an authored message under a stable code.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from agent.config import AgentConfig
from agent.errors import PlannerProviderError, PlannerUnavailableError
from agent.plans import PlanStep, parse_plan
from agent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    build_answer_prompt,
    build_planner_prompt,
    render_observations,
    render_tool_catalogue,
)
from llm.errors import (
    LLMConfigurationError,
    LLMDependencyError,
    LLMError,
)
from llm.messages import GenerationRequest, build_messages

logger = logging.getLogger(__name__)

#: How each provider failure is described to a caller. Authored sentences, so
#: nothing a vendor wrote is ever repeated.
PROVIDER_FAILURES: dict[str, str] = {
    "LLMTimeoutError": "it did not respond in time",
    "LLMRateLimitError": "it is currently rate limiting requests",
    "LLMAuthenticationError": "it rejected the configured credential",
    "LLMUnavailableError": "it could not be reached",
    "LLMResponseError": "it returned nothing usable",
    "LLMContextTooLargeError": "the request exceeded its context window",
}

#: The stable code reported beside each of those.
PROVIDER_CODES: dict[str, str] = {
    "LLMTimeoutError": "timeout",
    "LLMRateLimitError": "rate_limited",
    "LLMAuthenticationError": "authentication_failed",
    "LLMUnavailableError": "unavailable",
    "LLMResponseError": "invalid_response",
    "LLMContextTooLargeError": "context_too_large",
}


@runtime_checkable
class Planner(Protocol):
    """Chooses the next action, and writes the final answer."""

    def decide(
        self,
        question: str,
        *,
        tool_definitions: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        remaining_tool_calls: int,
        context: dict[str, Any] | None = None,
    ) -> PlanStep:
        """Choose the next action.

        Args:
            question: What is being asked.
            tool_definitions: Every registered tool, as the planner sees them.
            observations: What has been observed so far.
            remaining_tool_calls: How much budget is left.
            context: Named facts about this run — flags, names and counts
                supplied by the application. Never content: a dataset's rows
                reach a planner as a tool observation or not at all.

        Raises:
            MalformedPlanError: If the response is not a decision.
            PlannerUnavailableError: If the planner is not configured.
            PlannerProviderError: If the provider failed.
        """
        ...  # pragma: no cover - protocol

    def write_answer(
        self,
        question: str,
        *,
        observations: list[dict[str, Any]],
        allowed_citations: list[str],
    ) -> str:
        """Write the final answer from the observations."""
        ...  # pragma: no cover - protocol

    @property
    def is_ready(self) -> bool:
        """Whether the planner could be asked right now."""
        ...  # pragma: no cover - protocol


class LLMPlanner:
    """A planner backed by any :class:`~llm.providers.base.LLMProvider`."""

    def __init__(
        self,
        provider: Any,
        *,
        config: AgentConfig | None = None,
        model: str = "",
        max_answer_tokens: int = 900,
    ) -> None:
        """Wire the planner to a provider and the run's limits.

        Args:
            provider: Anything satisfying the provider protocol. The fake
                provider from Commit 10 satisfies it, which is how the whole
                agent suite runs offline.
            config: Timeouts, temperature and output limits.
            model: Model identifier to request. Empty lets the provider use
                its configured default.
            max_answer_tokens: Output limit for the final answer, which is
                longer than a decision.
        """
        self._provider = provider
        self._config = config or AgentConfig()
        self._model = model
        self._max_answer_tokens = max_answer_tokens

    @property
    def is_ready(self) -> bool:
        """Whether the provider has what it needs to be called."""
        return bool(getattr(self._provider, "is_ready", False))

    @property
    def provider_name(self) -> str:
        """Which provider is behind this planner."""
        return str(getattr(self._provider, "name", "unknown"))

    def _generate(self, system_prompt: str, user_prompt: str, *, timeout: float, max_tokens: int) -> str:
        """Ask the provider once, translating its failures."""
        request = GenerationRequest(
            messages=build_messages(system_prompt, user_prompt),
            model=self._model,
            temperature=float(self._config.planner_temperature),
            max_output_tokens=max_tokens,
            timeout_seconds=timeout,
        )
        try:
            result = self._provider.generate(request)
        except (LLMConfigurationError, LLMDependencyError) as exc:
            raise PlannerUnavailableError(
                "The agent's language-model provider is not configured. Set an "
                "API key for the provider to use the agent.",
                details={"provider": self.provider_name},
            ) from exc
        except LLMError as exc:
            # The vendor's own message is deliberately not passed through: it
            # can carry a request URL, a header or an echoed payload. What
            # reaches a caller is an authored sentence and a stable code.
            logger.warning(
                "Planner provider failed (%s): %s", type(exc).__name__, exc
            )
            raise PlannerProviderError(
                "The agent could not reach its language-model provider: "
                f"{PROVIDER_FAILURES.get(type(exc).__name__, 'the request failed')}.",
                details={
                    "provider": self.provider_name,
                    "failure": PROVIDER_CODES.get(type(exc).__name__, "provider_error"),
                },
            ) from exc

        return getattr(result, "text", "") or ""

    def decide(
        self,
        question: str,
        *,
        tool_definitions: list[dict[str, Any]],
        observations: list[dict[str, Any]],
        remaining_tool_calls: int,
        context: dict[str, Any] | None = None,
    ) -> PlanStep:
        """Choose the next action."""
        prompt = build_planner_prompt(
            question,
            tool_catalogue=render_tool_catalogue(tool_definitions),
            observations=render_observations(
                observations, limit=self._config.max_context_chars
            ),
            remaining_tool_calls=remaining_tool_calls,
            context=context,
        )
        text = self._generate(
            PLANNER_SYSTEM_PROMPT,
            prompt,
            timeout=self._config.planner_timeout_seconds,
            max_tokens=self._config.planner_max_output_tokens,
        )
        return parse_plan(text)

    def write_answer(
        self,
        question: str,
        *,
        observations: list[dict[str, Any]],
        allowed_citations: list[str],
    ) -> str:
        """Write the final answer from the observations."""
        prompt = build_answer_prompt(
            question,
            observations=render_observations(
                observations, limit=self._config.max_context_chars
            ),
            allowed_citations=allowed_citations,
        )
        return self._generate(
            ANSWER_SYSTEM_PROMPT,
            prompt,
            timeout=self._config.answer_timeout_seconds,
            max_tokens=self._max_answer_tokens,
        )


__all__ = ["LLMPlanner", "Planner"]
