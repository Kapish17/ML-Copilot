"""Execution state, JSON safety and the budgets.

The budget tests are the ones that matter most: they are the reason this
agent stops. Each asserts not just that the run ends but that it ends with a
structured result naming the limit that ended it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from agent.config import AgentConfig, config_from_env
from agent.errors import AgentConfigurationError
from agent.observations import (
    Observation,
    ObservationStatus,
    ensure_json_safe,
    summarise_arguments,
)
from agent.state import ExecutionState


def observation(
    call_id: str = "call-01",
    *,
    tool: str = "search_knowledge",
    status: ObservationStatus = ObservationStatus.OK,
    output: dict[str, Any] | None = None,
    citations: tuple[str, ...] = (),
) -> Observation:
    """Build an observation, for brevity."""
    return Observation(
        call_id=call_id,
        tool_name=tool,
        status=status,
        output=output or {"status": "ok"},
        citations=citations,
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def test_a_fresh_state_has_spent_nothing(config: AgentConfig) -> None:
    """Two runs of the same question start identically."""
    state = ExecutionState(question="What happened?", config=config)

    assert state.tool_call_count == 0
    assert state.iterations == 0
    assert state.remaining_tool_calls == config.max_tool_calls
    assert state.can_call_tool() is True
    assert state.exhausted_budget() is None


def test_call_ids_come_from_the_run_not_the_planner(config: AgentConfig) -> None:
    """An identifier a model chooses could carry a path or collide."""
    state = ExecutionState(question="q", config=config)

    assert state.next_call_id() == "call-01"
    state.record(observation("call-01"))
    assert state.next_call_id() == "call-02"


def test_citations_are_collected_in_order_without_duplicates(
    config: AgentConfig,
) -> None:
    """This set is exactly what an answer may cite."""
    state = ExecutionState(question="q", config=config)
    state.record(observation("call-01", citations=("docs:a", "docs:b")))
    state.record(observation("call-02", citations=("docs:b", "docs:c")))

    assert state.citations() == ("docs:a", "docs:b", "docs:c")


def test_experiment_ids_are_collected_from_observations(
    config: AgentConfig,
) -> None:
    """So a caller can find the runs an answer talks about."""
    state = ExecutionState(question="q", config=config)
    state.record(observation("call-01", output={"experiment_id": "exp_1"}))
    state.record(observation("call-02", output={"experiment_id": "exp_1"}))

    assert state.experiment_ids() == ("exp_1",)


def test_an_unavailable_observation_records_a_warning(config: AgentConfig) -> None:
    """A caller should not have to read the observations to notice."""
    state = ExecutionState(question="q", config=config)
    state.record(
        Observation(
            call_id="call-01",
            tool_name="explain_experiment",
            status=ObservationStatus.UNAVAILABLE,
            output={"reason": "fitted_model_not_persisted"},
            error="The fitted model is not available.",
        )
    )

    assert any("explain_experiment" in warning for warning in state.warnings)


def test_the_state_serialises_as_json(config: AgentConfig) -> None:
    """The whole thing, exactly as a caller would receive it."""
    state = ExecutionState(question="q", config=config)
    state.record(observation(citations=("docs:a",)))

    json.dumps(state.as_dict())


def test_the_state_holds_no_live_objects(config: AgentConfig) -> None:
    """A DataFrame smuggled into an output becomes a type name, not a repr."""

    class PretendFrame:
        """Stands in for anything with a revealing repr."""

        def __repr__(self) -> str:
            return "<PretendFrame at /home/someone/secret.csv>"

    state = ExecutionState(question="q", config=config)
    state.record(observation(output={"frame": PretendFrame()}))

    rendered = json.dumps(state.as_dict())
    assert "<PretendFrame>" in rendered
    assert "secret.csv" not in rendered


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------


def test_non_finite_floats_become_null() -> None:
    """``NaN`` and ``Infinity`` are not JSON, and pandas produces both."""
    assert ensure_json_safe({"a": float("nan"), "b": float("inf")}) == {
        "a": None,
        "b": None,
    }


def test_nested_structures_are_walked() -> None:
    """Sets and tuples become lists; keys become strings."""
    assert ensure_json_safe({1: ("a", {"b"}), "c": [{"d": 2}]}) == {
        "1": ["a", ["b"]],
        "c": [{"d": 2}],
    }


def test_deeply_nested_structures_are_cut_rather_than_recursing() -> None:
    """A cycle or a very deep structure must not blow the stack."""
    payload: dict[str, Any] = {}
    node = payload
    for _ in range(40):
        node["next"] = {}
        node = node["next"]

    json.dumps(ensure_json_safe(payload))


def test_long_argument_values_are_summarised() -> None:
    """A long argument cannot fill the state."""
    summary = summarise_arguments({"query": "x" * 5_000})

    assert len(summary["query"]) < 300
    assert summary["query"].endswith("[truncated]")


# ---------------------------------------------------------------------------
# Configuration and budgets
# ---------------------------------------------------------------------------


def test_the_default_configuration_is_usable() -> None:
    """And leaves a turn to answer after spending its whole tool budget."""
    config = AgentConfig()

    assert config.max_iterations > config.max_tool_calls
    json.dumps(config.as_dict())


def test_either_budget_may_be_the_binding_one() -> None:
    """A run can be capped by planning effort rather than by work done."""
    config = AgentConfig(max_tool_calls=10, max_iterations=3)
    state = ExecutionState(question="q", config=config)
    for _ in range(3):
        state.begin_iteration()

    assert state.exhausted_budget() == "max_iterations"


@pytest.mark.parametrize(
    "changes",
    [
        {"max_tool_calls": 0},
        {"max_iterations": 0},
        {"max_context_chars": -1},
        {"max_answer_length": 0},
        {"planner_temperature": 5.0},
        {"planner_temperature": -1.0},
        {"planner_timeout_seconds": 0},
        {"max_observation_chars": 100_000},
    ],
)
def test_an_unusable_configuration_is_refused(changes: dict[str, Any]) -> None:
    """A limit that cannot be honoured is caught at construction."""
    with pytest.raises(AgentConfigurationError):
        AgentConfig(**changes)


def test_configuration_reads_agent_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AGENT_*``, with explicit overrides winning."""
    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "3")
    monkeypatch.setenv("AGENT_MAX_ITERATIONS", "5")
    monkeypatch.setenv("AGENT_PLANNER_TEMPERATURE", "0.2")

    config = config_from_env()
    assert (config.max_tool_calls, config.max_iterations) == (3, 5)
    assert config.planner_temperature == 0.2

    assert config_from_env(max_tool_calls=2).max_tool_calls == 2


def test_a_blank_environment_variable_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partially configured environment is normal, not an error."""
    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "   ")

    assert config_from_env().max_tool_calls == AgentConfig().max_tool_calls


def test_an_unreadable_environment_variable_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Present but nonsense is a configuration error, not a silent default."""
    monkeypatch.setenv("AGENT_MAX_TOOL_CALLS", "lots")

    with pytest.raises(AgentConfigurationError):
        config_from_env()


def test_the_configuration_holds_no_credential() -> None:
    """It is safe to log, compare and put in a failure message.

    ``planner_max_output_tokens`` is a budget, not a credential — so the
    check is on the credential-shaped names, and on every value being a
    number.
    """
    payload = AgentConfig().as_dict()
    rendered = json.dumps(payload).lower()

    for forbidden in ("api_key", "secret", "password", "credential", "url", "sk-"):
        assert forbidden not in rendered
    assert all(isinstance(value, (int, float)) for value in payload.values())


def test_the_tool_budget_is_spent_by_failures_too(config: AgentConfig) -> None:
    """Otherwise a planner could retry a broken call for ever."""
    state = ExecutionState(question="q", config=AgentConfig(max_tool_calls=2))
    state.record(observation("call-01", status=ObservationStatus.REJECTED))
    state.record(observation("call-02", status=ObservationStatus.FAILED))

    assert state.remaining_tool_calls == 0
    assert state.exhausted_budget() == "max_tool_calls"
    assert state.can_call_tool() is False


def test_the_iteration_budget_is_reported_separately() -> None:
    """A planner that never calls a tool still stops."""
    state = ExecutionState(
        question="q", config=AgentConfig(max_tool_calls=6, max_iterations=7)
    )
    for _ in range(7):
        state.begin_iteration()

    assert state.exhausted_budget() == "max_iterations"


def test_the_context_budget_is_reported_when_observations_grow() -> None:
    """Bounding what the planner sees and what the caller receives."""
    state = ExecutionState(
        question="q",
        config=AgentConfig(max_context_chars=200, max_observation_chars=200),
    )
    state.record(observation(output={"text": "x" * 500}))

    assert state.exhausted_budget() == "max_context_chars"
