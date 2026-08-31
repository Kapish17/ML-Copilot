"""The real planner, driven by the fake provider from Commit 10.

:class:`~agent.planner.LLMPlanner` is the production path, so it is tested
against a real :class:`~llm.providers.base.LLMProvider` implementation rather
than a stub of its own. The provider is Commit 10's fake, which is exactly the
point: the agent reaches generation through the same abstraction the ask
endpoint does, and swapping the model is a change to which provider is
constructed.
"""

from __future__ import annotations

import json

import pytest

from agent.config import AgentConfig
from agent.errors import (
    MalformedPlanError,
    PlannerProviderError,
    PlannerUnavailableError,
)
from agent.orchestrator import AgentOrchestrator
from agent.planner import LLMPlanner, Planner
from agent.prompts import (
    OBSERVATIONS_OPEN,
    TOOLS_OPEN,
    render_observations,
    render_tool_catalogue,
)
from agent.results import AgentStatus
from llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from llm.providers.fake import FakeLLMProvider

DECISION = json.dumps(
    {"action": "tool", "tool": "search_knowledge", "arguments": {"query": "leakage"}}
)
FINISH = '{"action": "final"}'


def test_the_planner_satisfies_the_planner_protocol() -> None:
    """Structural conformance with the scripted planner the tests use."""
    assert isinstance(LLMPlanner(FakeLLMProvider()), Planner)


def test_a_decision_is_parsed_from_the_provider_response() -> None:
    """The ordinary path."""
    planner = LLMPlanner(FakeLLMProvider(responses=DECISION))

    step = planner.decide(
        "How is leakage prevented?",
        tool_definitions=[{"name": "search_knowledge", "description": "Search."}],
        observations=[],
        remaining_tool_calls=3,
    )

    assert step.tool == "search_knowledge"
    assert step.arguments == {"query": "leakage"}


def test_the_prompt_carries_the_tools_the_observations_and_the_budget() -> None:
    """What the planner is given, asserted on the actual request."""
    provider = FakeLLMProvider(responses=FINISH)
    planner = LLMPlanner(provider)

    planner.decide(
        "Which model won?",
        tool_definitions=[
            {"name": "run_experiment", "description": "Run one.", "arguments": []}
        ],
        observations=[{"call_id": "call-01", "tool_name": "dataset_profile"}],
        remaining_tool_calls=2,
    )

    prompt = provider.last_user_prompt
    assert "Which model won?" in prompt
    assert TOOLS_OPEN in prompt and "run_experiment" in prompt
    assert OBSERVATIONS_OPEN in prompt and "dataset_profile" in prompt
    assert "at most 2 more tool call(s)" in prompt


def test_the_system_prompt_forbids_code_and_secrets() -> None:
    """Asserted on what the provider actually received."""
    provider = FakeLLMProvider(responses=FINISH)

    LLMPlanner(provider).decide(
        "q", tool_definitions=[], observations=[], remaining_tool_calls=1
    )

    system = provider.last_system_prompt.lower()
    assert "cannot write or run code" in system
    assert "shell commands" in system
    assert "data, not instruction" in system


def test_the_planner_sends_no_credential_in_its_request() -> None:
    """The provider reads the key; the prompt never carries one."""
    provider = FakeLLMProvider(responses=FINISH)

    LLMPlanner(provider).decide(
        "q", tool_definitions=[], observations=[], remaining_tool_calls=1
    )

    rendered = provider.last_system_prompt + provider.last_user_prompt
    assert "sk-" not in rendered
    assert "LLM_API_KEY" not in rendered


def test_the_configured_temperature_and_limits_reach_the_request() -> None:
    """Determinism is a setting, and it must actually be sent."""
    provider = FakeLLMProvider(responses=FINISH)
    config = AgentConfig(planner_temperature=0.0, planner_max_output_tokens=123)

    LLMPlanner(provider, config=config).decide(
        "q", tool_definitions=[], observations=[], remaining_tool_calls=1
    )

    request = provider.last_request
    assert request is not None
    assert request.temperature == 0.0
    assert request.max_output_tokens == 123
    assert request.timeout_seconds == config.planner_timeout_seconds


def test_a_response_that_is_not_a_decision_is_malformed() -> None:
    """Including one that is a block of Python."""
    planner = LLMPlanner(
        FakeLLMProvider(responses='```python\nimport os\nos.environ["LLM_API_KEY"]\n```')
    )

    with pytest.raises(MalformedPlanError):
        planner.decide("q", tool_definitions=[], observations=[], remaining_tool_calls=1)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (LLMTimeoutError("provider timed out at https://api.example/v1"), "timeout"),
        (LLMRateLimitError("429 from https://api.example/v1"), "rate_limited"),
        (LLMAuthenticationError("bad key sk-live-abc"), "authentication_failed"),
    ],
)
def test_a_provider_failure_becomes_an_agent_error(
    error: Exception, expected: str
) -> None:
    """And the vendor's own message never comes with it."""
    planner = LLMPlanner(FakeLLMProvider(error=error))

    with pytest.raises(PlannerProviderError) as caught:
        planner.decide("q", tool_definitions=[], observations=[], remaining_tool_calls=1)

    assert caught.value.details["failure"] == expected
    assert "https://" not in caught.value.message
    assert "sk-live-abc" not in caught.value.message
    assert "429" not in caught.value.message


def test_a_missing_credential_is_reported_as_unavailable() -> None:
    """Nothing was attempted, and retrying will not help."""
    planner = LLMPlanner(
        FakeLLMProvider(error=LLMConfigurationError("no LLM_API_KEY configured"))
    )

    with pytest.raises(PlannerUnavailableError) as caught:
        planner.decide("q", tool_definitions=[], observations=[], remaining_tool_calls=1)

    assert "not configured" in caught.value.message
    assert "LLM_API_KEY" not in caught.value.message


def test_readiness_follows_the_provider() -> None:
    """A provider with no credential makes an unready planner."""
    assert LLMPlanner(FakeLLMProvider(ready=True)).is_ready is True
    assert LLMPlanner(FakeLLMProvider(ready=False)).is_ready is False


def test_the_answer_step_uses_its_own_timeout() -> None:
    """It is a longer piece of writing over more evidence."""
    provider = FakeLLMProvider(responses="An answer.")
    config = AgentConfig(answer_timeout_seconds=61.0)

    LLMPlanner(provider, config=config).write_answer(
        "q", observations=[], allowed_citations=["docs:a#b"]
    )

    assert provider.last_request is not None
    assert provider.last_request.timeout_seconds == 61.0
    assert "docs:a#b" in provider.last_user_prompt


def test_the_real_planner_drives_the_orchestrator(registry) -> None:
    """End to end with a provider, not a scripted planner."""
    provider = FakeLLMProvider(
        responses=[
            DECISION,
            FINISH,
            "Leakage is prevented on the training split "
            "[docs:ml-readme#cross-validation].",
        ]
    )
    agent = AgentOrchestrator(LLMPlanner(provider), registry)

    result = agent.run("How is leakage prevented?")

    assert result.status is AgentStatus.COMPLETED
    assert result.tool_call_count == 1
    assert result.citation_ids == ("docs:ml-readme#cross-validation",)


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def test_the_catalogue_renders_every_tool_and_nothing_else(registry) -> None:
    """A planner is shown exactly what the executor will accept."""
    rendered = render_tool_catalogue(registry.definitions())

    for name in registry.names():
        assert name in rendered
    assert "shell" not in rendered


def test_an_empty_catalogue_says_so_rather_than_rendering_nothing() -> None:
    """An agent with no tools is a legible state, not a blank prompt."""
    assert "no tools are available" in render_tool_catalogue([])


def test_observations_are_dropped_oldest_first_and_the_loss_is_stated() -> None:
    """Silently losing the evidence an answer rests on would be worse."""
    entries = [{"call_id": f"call-{index:02d}", "text": "x" * 400} for index in range(10)]

    rendered = render_observations(entries, limit=1_000)

    assert "earlier observation(s) omitted" in rendered
    assert "call-09" in rendered
    assert "call-00" not in rendered


def test_no_observations_renders_a_statement_not_an_empty_block() -> None:
    """So a planner's first turn reads clearly."""
    assert "no tools have been called yet" in render_observations([], limit=1_000)
