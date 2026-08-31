"""What the agent is allowed to spend, and how it behaves while spending it.

Every limit in this module exists for the same reason: an agent that decides
its own next move can decide to keep going. The bounds are not tuning
parameters, they are the difference between a system that stops and one that
does not, so they are explicit, validated, and configurable in one place
rather than defaulted at each call site.

Four of them are hard stops:

``max_tool_calls``      how much work one question may cause
``max_iterations``      how many planning turns, tool call or not
``max_context_chars``   how much accumulated observation text may be carried
``max_answer_length``   how long the final answer may be

Reaching any of them ends the run with a structured partial result. None of
them can be raised by a request, by a planner, or by anything a tool observed
— a limit a model can talk its way past is not a limit.

**No credential is held here.** Like :class:`~llm.config.LLMConfig`, this
object is safe to log, compare and put in a test failure message. The
language-model settings the planner needs are the LLM layer's own
configuration and stay there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

from agent.errors import AgentConfigurationError

# -- Budgets ----------------------------------------------------------------
#: Tool calls one run may make. Six is enough for the longest sensible chain —
#: profile, run, explain, search, and room to recover from one failure — and
#: small enough that a planner stuck in a loop is stopped in seconds.
DEFAULT_MAX_TOOL_CALLS = 6
#: Planning turns one run may take, whether or not a turn leads to a tool call.
#:
#: The two budgets are independent, and either may be the one that binds. The
#: default leaves it to ``max_tool_calls``: eight turns is more than six calls
#: can consume, so a run that spends its whole tool budget still has a turn
#: left to write its answer. Setting this at or below ``max_tool_calls`` is
#: allowed and makes *this* the binding limit — which is a legitimate way to
#: cap a run by planning effort rather than by work done.
DEFAULT_MAX_ITERATIONS = 8
#: Characters of observation text one run may accumulate. Bounds both the
#: prompt the planner sees and the state a caller receives.
DEFAULT_MAX_CONTEXT_CHARS = 24_000
#: Characters the final answer may run to.
DEFAULT_MAX_ANSWER_LENGTH = 4_000

# -- Per-observation limits -------------------------------------------------
#: Characters of any single observation shown to the planner. A tool that
#: returns something enormous is truncated rather than allowed to consume the
#: whole context budget by itself.
DEFAULT_MAX_OBSERVATION_CHARS = 6_000

# -- Planner ----------------------------------------------------------------
#: Sampling temperature for planning. Zero by default: a planner that picks a
#: different tool each time it is asked the same question is not auditable.
DEFAULT_PLANNER_TEMPERATURE = 0.0
#: Seconds one planning call may take.
DEFAULT_PLANNER_TIMEOUT_SECONDS = 30.0
#: Seconds the final-answer call may take. Separate because it is a longer
#: piece of writing over more evidence.
DEFAULT_ANSWER_TIMEOUT_SECONDS = 45.0
#: Output tokens one planning decision may use. A decision is a short JSON
#: object, so this is deliberately small — it also means a planner that starts
#: writing an essay is cut off rather than paid for.
DEFAULT_PLANNER_MAX_OUTPUT_TOKENS = 400

#: Environment variables this module reads. Named here so the documentation,
#: the tests and ``.env.example`` cannot drift from the implementation.
ENVIRONMENT_VARIABLES: tuple[str, ...] = (
    "AGENT_MAX_TOOL_CALLS",
    "AGENT_MAX_ITERATIONS",
    "AGENT_MAX_CONTEXT_CHARS",
    "AGENT_MAX_ANSWER_LENGTH",
    "AGENT_MAX_OBSERVATION_CHARS",
    "AGENT_PLANNER_TEMPERATURE",
    "AGENT_PLANNER_TIMEOUT_SECONDS",
    "AGENT_ANSWER_TIMEOUT_SECONDS",
    "AGENT_PLANNER_MAX_OUTPUT_TOKENS",
)


def _positive_int(name: str, value: int) -> None:
    """Raise unless ``value`` is a positive integer."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise AgentConfigurationError(
            f"{name} must be a positive integer.", details={name: value}
        )


def _positive_number(name: str, value: float) -> None:
    """Raise unless ``value`` is a positive number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise AgentConfigurationError(
            f"{name} must be a positive number.", details={name: value}
        )


@dataclass(frozen=True)
class AgentConfig:
    """Every limit and setting one agent run operates under."""

    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_iterations: int = DEFAULT_MAX_ITERATIONS
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS
    max_answer_length: int = DEFAULT_MAX_ANSWER_LENGTH
    max_observation_chars: int = DEFAULT_MAX_OBSERVATION_CHARS
    planner_temperature: float = DEFAULT_PLANNER_TEMPERATURE
    planner_timeout_seconds: float = DEFAULT_PLANNER_TIMEOUT_SECONDS
    answer_timeout_seconds: float = DEFAULT_ANSWER_TIMEOUT_SECONDS
    planner_max_output_tokens: int = DEFAULT_PLANNER_MAX_OUTPUT_TOKENS

    def __post_init__(self) -> None:
        """Reject a configuration the agent could not run safely under."""
        _positive_int("max_tool_calls", self.max_tool_calls)
        _positive_int("max_iterations", self.max_iterations)
        _positive_int("max_context_chars", self.max_context_chars)
        _positive_int("max_answer_length", self.max_answer_length)
        _positive_int("max_observation_chars", self.max_observation_chars)
        _positive_int("planner_max_output_tokens", self.planner_max_output_tokens)
        _positive_number("planner_timeout_seconds", self.planner_timeout_seconds)
        _positive_number("answer_timeout_seconds", self.answer_timeout_seconds)

        if isinstance(self.planner_temperature, bool) or not isinstance(
            self.planner_temperature, (int, float)
        ):
            raise AgentConfigurationError(
                "planner_temperature must be a number.",
                details={"planner_temperature": self.planner_temperature},
            )
        if not 0.0 <= float(self.planner_temperature) <= 2.0:
            raise AgentConfigurationError(
                "planner_temperature must be between 0.0 and 2.0.",
                details={"planner_temperature": self.planner_temperature},
            )

        if self.max_observation_chars > self.max_context_chars:
            raise AgentConfigurationError(
                "max_observation_chars must not exceed max_context_chars.",
                details={
                    "max_observation_chars": self.max_observation_chars,
                    "max_context_chars": self.max_context_chars,
                },
            )

    def with_overrides(self, **changes: Any) -> AgentConfig:
        """Return a validated copy with some fields replaced."""
        return replace(self, **changes)

    def as_dict(self) -> dict[str, Any]:
        """Render the configuration as plain JSON-safe values.

        Safe to log and to return: there is no credential in this object to
        leave out.
        """
        return {
            "max_tool_calls": self.max_tool_calls,
            "max_iterations": self.max_iterations,
            "max_context_chars": self.max_context_chars,
            "max_answer_length": self.max_answer_length,
            "max_observation_chars": self.max_observation_chars,
            "planner_temperature": self.planner_temperature,
            "planner_timeout_seconds": self.planner_timeout_seconds,
            "answer_timeout_seconds": self.answer_timeout_seconds,
            "planner_max_output_tokens": self.planner_max_output_tokens,
        }


def _read_int(name: str, values: dict[str, Any]) -> None:
    """Read one integer environment variable into ``values``."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return
    try:
        values[name.removeprefix("AGENT_").lower()] = int(raw)
    except ValueError as exc:
        raise AgentConfigurationError(
            f"{name} must be an integer.", details={"variable": name}
        ) from exc


def _read_float(name: str, values: dict[str, Any]) -> None:
    """Read one floating-point environment variable into ``values``."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return
    try:
        values[name.removeprefix("AGENT_").lower()] = float(raw)
    except ValueError as exc:
        raise AgentConfigurationError(
            f"{name} must be a number.", details={"variable": name}
        ) from exc


def config_from_env(**overrides: Any) -> AgentConfig:
    """Build the configuration from ``AGENT_*`` variables.

    Explicit ``overrides`` win over the environment, which wins over the
    defaults. An unset or blank variable is simply not read, so a partially
    configured environment is normal rather than an error.

    Raises:
        AgentConfigurationError: If a variable is present but unreadable, or
            if the resulting combination is not one the agent can run under.
    """
    values: dict[str, Any] = {}
    for name in (
        "AGENT_MAX_TOOL_CALLS",
        "AGENT_MAX_ITERATIONS",
        "AGENT_MAX_CONTEXT_CHARS",
        "AGENT_MAX_ANSWER_LENGTH",
        "AGENT_MAX_OBSERVATION_CHARS",
        "AGENT_PLANNER_MAX_OUTPUT_TOKENS",
    ):
        _read_int(name, values)
    for name in (
        "AGENT_PLANNER_TEMPERATURE",
        "AGENT_PLANNER_TIMEOUT_SECONDS",
        "AGENT_ANSWER_TIMEOUT_SECONDS",
    ):
        _read_float(name, values)

    values.update(overrides)
    return AgentConfig(**values)


__all__ = [
    "DEFAULT_ANSWER_TIMEOUT_SECONDS",
    "DEFAULT_MAX_ANSWER_LENGTH",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_OBSERVATION_CHARS",
    "DEFAULT_MAX_TOOL_CALLS",
    "DEFAULT_PLANNER_MAX_OUTPUT_TOKENS",
    "DEFAULT_PLANNER_TEMPERATURE",
    "DEFAULT_PLANNER_TIMEOUT_SECONDS",
    "ENVIRONMENT_VARIABLES",
    "AgentConfig",
    "config_from_env",
]
