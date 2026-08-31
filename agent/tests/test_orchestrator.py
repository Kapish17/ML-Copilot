"""The bounded loop: workflows that work, and every way one is stopped.

The scripted planner is what makes these possible. A real model produces a
malformed decision or a fabricated citation only occasionally and never on
demand; here each of those is one line of script, so the behaviour that
matters can be asserted rather than hoped for.
"""

from __future__ import annotations

import json

from agent.config import AgentConfig
from agent.errors import PlannerProviderError, PlannerUnavailableError
from agent.planners.fake import PLANS
from agent.plans import PlanStep
from agent.results import AgentStatus
from agent.tests.factories import FakeArtifacts, FakeExecutor

CITATION = "docs:ml-readme#cross-validation"
SECOND_CITATION = "docs:ml-readme#leakage"


def tool_step(name: str, **arguments: object) -> PlanStep:
    """Build a scripted tool call."""
    return PlanStep(action="tool", tool=name, arguments=dict(arguments))


FINAL = PlanStep(action="final")


# ---------------------------------------------------------------------------
# Successful workflows
# ---------------------------------------------------------------------------


def test_a_single_tool_workflow_completes(build_agent) -> None:
    """Search, then answer — the shortest useful run."""
    agent, planner = build_agent(
        [tool_step("search_knowledge", query="How is cross-validation used?"), FINAL],
        answer=f"Selection happens on the training rows only [{CITATION}].",
    )

    result = agent.run("How does model selection work?")

    assert result.status is AgentStatus.COMPLETED
    assert result.is_answer is True
    assert result.tool_call_count == 1
    assert result.citation_ids == (CITATION,)
    assert planner.decide_count == 2


def test_a_question_needing_no_tool_is_answered_directly(build_agent) -> None:
    """The planner may finish on its first turn."""
    agent, _ = build_agent([FINAL], answer="Cross-validation splits the training data.")

    result = agent.run("What is cross-validation?")

    assert result.tool_call_count == 0
    # Nothing was observed, so there is nothing to ground an answer in.
    assert result.status is AgentStatus.INSUFFICIENT_EVIDENCE


def test_a_dataset_then_experiment_workflow_completes(build_agent) -> None:
    """The chain the product idea is built around."""
    agent, _ = build_agent(
        [
            tool_step("dataset_profile", dataset="sales", target_column="churned"),
            tool_step("run_experiment", dataset="sales", target_column="churned"),
            FINAL,
        ],
        answer=(
            "Experiment exp_20260101T000000Z_abc123 selected "
            "random_forest_classifier, scoring 0.86 F1 on the held-out test set."
        ),
    )

    result = agent.run("Which model performs best on the sales data?")

    assert result.status is AgentStatus.COMPLETED
    assert result.tool_call_count == 2
    assert result.experiment_ids == ("exp_20260101T000000Z_abc123",)
    assert [call["tool_name"] for call in result.tool_calls] == [
        "dataset_profile",
        "run_experiment",
    ]


def test_an_experiment_then_explanation_workflow_completes(
    build_agent, dataset_source, artifacts, store
) -> None:
    """The full four-step chain, ending in a live explanation."""
    from agent.tools import build_default_registry
    from agent.tests.factories import FakeProfiler

    def explain_global(model, X, y, **kwargs):
        return _Explanation(
            {
                "status": "available",
                "method": "shap",
                "model_name": "random_forest_classifier",
                "task_type": "classification",
                "feature_importances": [
                    {"feature": "income", "importance": 0.42, "rank": 1}
                ],
            }
        )

    registry = build_default_registry(
        source=dataset_source,
        profiler=FakeProfiler(),
        executor=FakeExecutor(artifacts=FakeArtifacts()),
        lookup=store,
        artifacts=artifacts,
        explain_global=explain_global,
        available_models=("logistic_regression",),
    )

    agent, _ = build_agent(
        [
            tool_step("dataset_profile", dataset="sales"),
            tool_step("run_experiment", dataset="sales"),
            tool_step("explain_experiment", experiment_id="exp_20260101T000000Z_abc123"),
            FINAL,
        ],
        answer="Income drives exp_20260101T000000Z_abc123 most strongly.",
        tools=registry,
    )

    result = agent.run("Why does the winning model predict what it does?")

    assert result.status is AgentStatus.COMPLETED
    assert result.tool_call_count == 3
    explanation = result.observations[-1]
    assert explanation["status"] == "ok"
    assert explanation["output"]["source"] == "recomputed"


def test_a_mixed_search_and_experiment_workflow_completes(build_agent) -> None:
    """Evidence and results in the same run, with citations from the evidence."""
    agent, _ = build_agent(
        [
            tool_step("search_knowledge", query="Which models are supported?"),
            tool_step("run_experiment", dataset="sales", target_column="churned"),
            FINAL,
        ],
        answer=(
            f"Selection is cross-validated [{CITATION}]. On this data "
            "exp_20260101T000000Z_abc123 chose random_forest_classifier."
        ),
    )

    result = agent.run("Which model wins here, and how was it chosen?")

    assert result.status is AgentStatus.COMPLETED
    assert result.citation_ids == (CITATION,)
    assert result.experiment_ids == ("exp_20260101T000000Z_abc123",)


def test_the_planner_sees_the_previous_observations(build_agent) -> None:
    """Which is what lets a second decision depend on the first result."""
    agent, planner = build_agent(
        [tool_step("search_knowledge", query="x"), FINAL],
        answer=f"Yes [{CITATION}].",
    )

    agent.run("q")

    assert planner.decide_calls[0]["observations"] == []
    assert planner.decide_calls[1]["observations"][0]["tool_name"] == "search_knowledge"


def test_the_planner_is_shown_exactly_the_registered_tools(build_agent) -> None:
    """No hidden tool, and none withheld."""
    agent, planner = build_agent([FINAL])

    agent.run("q")

    shown = {definition["name"] for definition in planner.decide_calls[0]["tool_definitions"]}
    assert shown == {
        "dataset_profile",
        "run_experiment",
        "search_knowledge",
        "explain_experiment",
    }


# ---------------------------------------------------------------------------
# Rejections
# ---------------------------------------------------------------------------


def test_an_unknown_tool_is_rejected_and_the_run_continues(build_agent) -> None:
    """The planner can correct itself; what it cannot do is succeed."""
    agent, _ = build_agent(
        [
            tool_step("run_shell", command="ls -la"),
            tool_step("search_knowledge", query="How is leakage prevented?"),
            FINAL,
        ],
        answer=f"Preprocessing is fitted on the training split [{SECOND_CITATION}].",
    )

    result = agent.run("How is leakage prevented?")

    rejected = result.observations[0]
    assert rejected["status"] == "rejected"
    assert rejected["error_code"] == "unknown_tool"
    assert result.status is AgentStatus.PARTIAL
    assert result.tool_call_count == 2


def test_invalid_arguments_are_rejected_before_the_tool_runs(
    build_agent, retrieval
) -> None:
    """The retrieval service is never reached."""
    agent, _ = build_agent(
        [tool_step("search_knowledge", query="x", top_k=999_999), FINAL]
    )

    result = agent.run("q")

    assert result.observations[0]["status"] == "rejected"
    assert result.observations[0]["error_code"] == "invalid_tool_arguments"
    assert retrieval.calls == []


def test_a_failing_tool_becomes_an_observation_not_a_crash(
    build_agent, dataset_source, artifacts, store
) -> None:
    """And its real cause never reaches the result."""
    from agent.tools import build_default_registry
    from agent.tests.factories import FakeProfiler

    registry = build_default_registry(
        source=dataset_source,
        profiler=FakeProfiler(),
        executor=FakeExecutor(
            error=RuntimeError("connection to /var/run/db.sock failed: key=sk-secret")
        ),
        lookup=store,
        artifacts=artifacts,
        available_models=("logistic_regression",),
    )

    agent, _ = build_agent(
        [tool_step("run_experiment", dataset="sales"), FINAL], tools=registry
    )
    result = agent.run("Which model wins?")

    failed = result.observations[0]
    assert failed["status"] == "failed"
    assert failed["error_code"] == "tool_execution_failed"

    rendered = json.dumps(result.as_dict())
    for leaked in ("/var/run/db.sock", "sk-secret", "RuntimeError", "Traceback"):
        assert leaked not in rendered


def test_an_unavailable_tool_result_makes_the_run_partial(build_agent) -> None:
    """The answer is kept; the gap is stated rather than filled."""
    agent, _ = build_agent(
        [
            tool_step("explain_experiment", experiment_id="exp_gone", scope="prediction"),
            tool_step("search_knowledge", query="How is leakage prevented?"),
            FINAL,
        ],
        answer=f"Preprocessing is fitted on the training split only [{CITATION}].",
    )

    result = agent.run("Why did the model predict that?")

    assert result.status is AgentStatus.PARTIAL
    assert result.observations[0]["status"] == "unavailable"
    assert any("explain_experiment" in warning for warning in result.warnings)


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


def test_the_tool_budget_stops_a_planner_that_keeps_calling(build_agent) -> None:
    """The loop always terminates, and says which limit stopped it."""
    agent, _ = build_agent(
        PLANS["too_many_calls"],
        config=AgentConfig(max_tool_calls=3, max_iterations=5),
        answer=f"Partial findings [{CITATION}].",
    )

    result = agent.run("Tell me everything.")

    assert result.tool_call_count == 3
    assert result.error_code == "max_tool_calls"
    assert result.status is AgentStatus.PARTIAL
    assert any("limit of tool calls" in warning for warning in result.warnings)


def test_the_iteration_budget_stops_a_planner_that_never_finishes(
    build_agent,
) -> None:
    """A planner that only ever asks for rejected tools still stops."""
    agent, _ = build_agent(
        [tool_step("not_a_tool") for _ in range(20)],
        config=AgentConfig(max_tool_calls=20, max_iterations=4),
    )

    result = agent.run("q")

    assert result.iterations == 4
    assert result.tool_call_count == 4
    assert result.error_code == "max_iterations"
    assert any("planning steps" in warning for warning in result.warnings)


def test_the_context_budget_stops_a_run_that_accumulates_too_much(
    build_agent, dataset_source, artifacts, store
) -> None:
    """A tool returning a great deal cannot be called indefinitely."""
    from agent.tools import build_default_registry
    from agent.tests.factories import FakeProfiler, FakeRetrieval, FakeRetrievalResult

    registry = build_default_registry(
        source=dataset_source,
        profiler=FakeProfiler(),
        retrieval=FakeRetrieval(
            [FakeRetrievalResult("docs:x#y", content="a" * 4_000)]
        ),
        lookup=store,
        artifacts=artifacts,
    )

    agent, _ = build_agent(
        [tool_step("search_knowledge", query=f"q{index}") for index in range(10)],
        config=AgentConfig(
            max_tool_calls=10, max_iterations=12, max_context_chars=3_000,
            max_observation_chars=3_000,
        ),
        tools=registry,
        answer="Partial findings so far [docs:x#y].",
    )

    result = agent.run("q")

    assert result.error_code == "max_context_chars"
    assert result.tool_call_count < 10
    assert any("observed material" in warning for warning in result.warnings)


def test_a_run_that_uses_its_whole_tool_budget_can_still_answer(
    build_agent,
) -> None:
    """Which is why max_iterations must exceed max_tool_calls."""
    agent, planner = build_agent(
        [tool_step("search_knowledge", query="How is leakage prevented?")] * 4,
        config=AgentConfig(max_tool_calls=2, max_iterations=4),
        answer=f"Findings [{CITATION}].",
    )

    result = agent.run("q")

    assert result.tool_call_count == 2
    assert result.final_answer
    assert len(planner.answer_calls) == 1


# ---------------------------------------------------------------------------
# Planner failures
# ---------------------------------------------------------------------------


def test_a_malformed_planner_response_ends_the_run_safely(build_agent) -> None:
    """Not a decision, so nothing was called and nothing was executed."""
    agent, _ = build_agent(PLANS["malformed"])

    result = agent.run("q")

    assert result.status is AgentStatus.FAILED
    assert result.error_code == "malformed_plan"
    assert result.tool_call_count == 0


def test_a_planner_provider_failure_is_reported_structurally(build_agent) -> None:
    """A vendor's own message never reaches the result."""
    agent, _ = build_agent(
        [],
        error=PlannerProviderError(
            "The agent could not reach its language-model provider: it did not "
            "respond in time.",
            details={"provider": "fake"},
        ),
    )

    result = agent.run("q")

    assert result.status is AgentStatus.FAILED
    assert result.error_code == "planner_provider_error"
    assert "did not respond in time" in result.final_answer


def test_an_unconfigured_planner_is_reported_structurally(build_agent) -> None:
    """No credential means a clear refusal, not a crash."""
    agent, _ = build_agent(
        [],
        error=PlannerUnavailableError(
            "The agent's language-model provider is not configured."
        ),
    )

    result = agent.run("q")

    assert result.status is AgentStatus.FAILED
    assert result.error_code == "planner_unavailable"


def test_a_provider_failure_while_answering_is_reported(build_agent) -> None:
    """The work already done is kept in the result."""
    agent, _ = build_agent(
        [tool_step("search_knowledge", query="x"), FINAL],
        answer_error=PlannerProviderError("The provider could not be reached."),
    )

    result = agent.run("q")

    assert result.status is AgentStatus.FAILED
    assert result.tool_call_count == 1
    assert result.observations[0]["status"] == "ok"


# ---------------------------------------------------------------------------
# The result object
# ---------------------------------------------------------------------------


def test_the_result_serialises_as_json(build_agent) -> None:
    """Everything a caller receives is plain values."""
    agent, _ = build_agent(
        [tool_step("search_knowledge", query="x"), FINAL],
        answer=f"Yes [{CITATION}].",
    )

    json.dumps(agent.run("q").as_dict())


def test_the_result_exposes_no_reasoning_trace(build_agent) -> None:
    """No prompt, no chain-of-thought, no raw model output beyond the answer."""
    agent, _ = build_agent(
        [
            PlanStep(
                action="tool",
                tool="search_knowledge",
                arguments={"query": "x"},
                reason="I should look this up first",
            ),
            FINAL,
        ],
        answer=f"Yes [{CITATION}].",
    )

    payload = agent.run("q").as_dict()
    rendered = json.dumps(payload).lower()

    for forbidden in (
        "chain_of_thought",
        "chain of thought",
        "reasoning",
        "scratchpad",
        "system_prompt",
        "you are the planning step",
    ):
        assert forbidden not in rendered


def test_the_run_records_its_timing(build_agent) -> None:
    """Enough to see how long a question took, without a profiler."""
    agent, _ = build_agent([FINAL])

    result = agent.run("q")

    assert result.started_at is not None
    assert result.completed_at is not None
    assert result.duration_ms is not None and result.duration_ms >= 0


def test_the_artifact_cache_is_cleared_when_a_run_ends(
    build_agent, artifacts
) -> None:
    """Nothing a run put in memory outlives it."""
    agent, _ = build_agent([tool_step("run_experiment", dataset="sales"), FINAL])

    agent.run("q")

    assert len(artifacts) == 0


class _Explanation:
    """An explanation object shaped like the explainability layer's own."""

    def __init__(self, payload: dict) -> None:
        """Hold the payload."""
        self._payload = payload

    def as_dict(self) -> dict:
        """Render the payload."""
        return self._payload
