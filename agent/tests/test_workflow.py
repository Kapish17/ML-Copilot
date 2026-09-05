"""Tests for planning a whole workflow, and for running one.

The agent could already call several tools. What it could not do was *plan*,
and the difference shows up in three places this module is about.

**A plan is checked before anything runs.** A step naming a tool nobody
registered makes the whole plan invalid — not a rejected call at step four with
three calls already spent. The same goes for a plan that is too long, that uses
one tool over and over, or that asks a step to wait on a later one.

**A later step is given values by the executor, not by the model.** The chain
this commit exists for is ``run_experiment`` → ``explain_experiment``, and the
experiment id travels between them as a reference this code resolves from the
observation. Nothing asks a language model to read an id out of one tool's
output and type it into another's arguments.

**A plan that half-works is reported as half-working.** The useful part is
kept, the part that did not happen says so, and the answer is told which is
which — because an agent that describes work it did not do is worse than one
that admits it stopped.

Everything here is offline: the planner is scripted, the services are doubles,
and the plans go through the real parser on the way in.
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest

from agent.config import AgentConfig
from agent.errors import PlannerProviderError
from agent.orchestrator import AgentOrchestrator
from agent.registry import ToolRegistry
from agent.schemas import ArgumentSchema
from agent.tools.base import BaseTool, ToolResult
from agent.planners.fake import WORKFLOWS, FakePlanner, NoScriptedWorkflow
from agent.results import AgentStatus
from agent.workflow import (
    REFERENCEABLE_FIELDS,
    MalformedWorkflowError,
    WorkflowStep,
    as_reference,
    parse_workflow,
    resolve_arguments,
)

#: The tools the fixtures register. A plan may name these and nothing else.
KNOWN = ("dataset_profile", "run_experiment", "explain_experiment", "search_knowledge")


class SlowTool(BaseTool):
    """A registered tool that takes longer than a very small time budget.

    Real enough to be executed by the real path — the registry approves it and
    its (empty) schema validates — and slow enough that the deadline is reached
    while it is running rather than by luck.
    """

    tool_name = "slow_tool"
    tool_description = "Takes a while, for testing the run deadline."

    @property
    def schema(self) -> ArgumentSchema:
        """No arguments."""
        return ArgumentSchema()

    def run(self, arguments: Any) -> ToolResult:
        """Sleep past the deadline, then return something ordinary."""
        time.sleep(0.05)
        return ToolResult(output={"status": "ok", "slept": True})


def plan(**payload: Any) -> str:
    """Render a plan the way a model would return it."""
    return json.dumps(payload)


def parse(payload: Any, *, max_steps: int = 5, max_tool_repeats: int = 2):
    """Parse a plan against the fixture registry's tool names."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return parse_workflow(
        text,
        known_tools=KNOWN,
        max_steps=max_steps,
        max_tool_repeats=max_tool_repeats,
    )


# ---------------------------------------------------------------------------
# Parsing a plan
# ---------------------------------------------------------------------------


def test_a_plan_becomes_ordered_steps_with_ids_this_code_assigned() -> None:
    """Positions, not names the planner chose.

    An identifier a model picks is one that can be made to collide, to look
    like a path, or to carry text into a log line. These come from the step's
    place in the list and nowhere else.
    """
    workflow = parse(WORKFLOWS["profile_experiment_explain"])

    assert [step.step_id for step in workflow.steps] == ["step-1", "step-2", "step-3"]
    assert workflow.tools == ("dataset_profile", "run_experiment", "explain_experiment")
    assert workflow.goal
    assert workflow.objective


def test_the_plan_summary_is_what_a_person_reads() -> None:
    """Numbered labels — what will be done, never how it was decided."""
    workflow = parse(WORKFLOWS["profile_experiment_explain"])

    assert workflow.summary_lines() == [
        "1. Profile the dataset",
        "2. Compare models",
        "3. Explain the winning model",
    ]


def test_a_step_carries_no_arguments_into_the_rendered_plan() -> None:
    """The plan is shown to a person; the values are not part of it.

    Arguments are the one place a planner could put text of its own choosing
    into that display. What a call actually received is reported by its
    observation, already summarised.
    """
    workflow = parse(WORKFLOWS["profile_experiment_explain"])
    rendered = json.dumps(workflow.as_dict())

    assert "arguments" not in rendered
    assert "churned" not in rendered


def test_one_step_is_a_complete_plan() -> None:
    """Because one step is enough for a great many questions."""
    workflow = parse(WORKFLOWS["search_only"])

    assert len(workflow) == 1
    assert workflow.steps[0].tool == "search_knowledge"


@pytest.mark.parametrize(
    "scenario,reason",
    [
        ("unknown_tool", "unknown_tool"),
        ("too_many_steps", "too_many_steps"),
        ("repeated_tool", "tool_repeated"),
        ("forward_dependency", "forward_dependency"),
        ("forbidden_reference", "unknown_reference_field"),
    ],
)
def test_an_invalid_plan_is_refused_as_a_plan(scenario: str, reason: str) -> None:
    """Before a step runs, not after several have.

    Each of these is the shape of a different failure, and each is refused with
    a code naming which — an unknown tool, a plan too long to be one, one tool
    used over and over, a step waiting on a later step, and a reference to a
    field a later step may not read.
    """
    with pytest.raises(MalformedWorkflowError) as caught:
        parse(WORKFLOWS[scenario])

    assert caught.value.details["reason"] == reason


def test_a_step_cannot_depend_on_itself() -> None:
    """Which is the smallest cycle there is."""
    with pytest.raises(MalformedWorkflowError):
        parse(plan(steps=[{"tool": "run_experiment", "depends_on": ["step-1"]}]))


def test_a_plan_cannot_express_a_cycle_at_all() -> None:
    """Not detected — unrepresentable.

    Dependencies point backwards by construction, so there is no plan whose
    steps wait on each other. That is why execution needs no scheduler and no
    cycle check: it is one pass down a list.
    """
    with pytest.raises(MalformedWorkflowError) as caught:
        parse(
            plan(
                steps=[
                    {"tool": "run_experiment", "depends_on": ["step-2"]},
                    {"tool": "explain_experiment", "depends_on": ["step-1"]},
                ]
            )
        )

    assert caught.value.details["reason"] == "forward_dependency"


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I would start by loading the data.",
        "```python\nimport os\nos.environ['LLM_API_KEY']\n```",
        "[1, 2, 3]",
        '{"goal": "no steps here"}',
        '{"steps": []}',
        '{"steps": "search everything"}',
    ],
)
def test_a_response_that_is_not_a_plan_is_refused(text: str) -> None:
    """Including one that is entirely code.

    There is no branch of this parser that reaches an interpreter. A Python
    block is text that does not parse as a plan, which is the same outcome as
    an empty reply.
    """
    with pytest.raises(MalformedWorkflowError):
        parse(text)


def test_a_plan_may_not_ask_for_a_field_outside_the_allowlist() -> None:
    """The allowlist is the whole reference vocabulary."""
    for field_name in ("__class__", "output", "candidates", "top_features"):
        assert field_name not in REFERENCEABLE_FIELDS
        with pytest.raises(MalformedWorkflowError):
            parse(
                plan(
                    steps=[
                        {"tool": "run_experiment", "arguments": {"dataset": "sales"}},
                        {
                            "tool": "explain_experiment",
                            "arguments": {
                                "experiment_id": {
                                    "from_step": "step-1",
                                    "field": field_name,
                                }
                            },
                        },
                    ]
                )
            )


def test_free_text_in_a_plan_is_bounded() -> None:
    """A goal is a line, and a purpose is a label."""
    workflow = parse(
        plan(
            goal="g" * 5_000,
            steps=[
                {
                    "tool": "search_knowledge",
                    "purpose": "p" * 5_000,
                    "arguments": {"query": "x"},
                }
            ],
        )
    )

    assert len(workflow.goal) <= 300
    assert len(workflow.steps[0].purpose) <= 120


# ---------------------------------------------------------------------------
# References between steps
# ---------------------------------------------------------------------------


def test_only_the_one_reference_shape_is_a_reference() -> None:
    """A string that looks like a placeholder is a string."""
    assert as_reference({"from_step": "step-1", "field": "experiment_id"}) == (
        "step-1",
        "experiment_id",
    )
    for value in (
        "{experiment_id}",
        "$step-1.experiment_id",
        {"from_step": "step-1"},
        {"from_step": "step-1", "field": "experiment_id", "extra": 1},
        {"from_step": 1, "field": "experiment_id"},
        ["step-1", "experiment_id"],
        None,
    ):
        assert as_reference(value) is None


def test_a_reference_is_filled_from_what_the_earlier_step_produced() -> None:
    """The chain this commit exists for, in one assertion."""
    step = WorkflowStep(
        step_id="step-2",
        tool="explain_experiment",
        purpose="Explain",
        arguments={"experiment_id": {"from_step": "step-1", "field": "experiment_id"}},
    )

    resolution = resolve_arguments(step, {"step-1": {"experiment_id": "exp_abc"}})

    assert resolution.ok
    assert resolution.arguments == {"experiment_id": "exp_abc"}


@pytest.mark.parametrize(
    "outputs,code",
    [
        ({}, "dependency_unavailable"),
        ({"step-1": {}}, "reference_not_found"),
        ({"step-1": {"experiment_id": None}}, "reference_not_found"),
        ({"step-1": {"experiment_id": ["a", "b"]}}, "reference_not_found"),
        ({"step-1": {"experiment_id": {"nested": 1}}}, "reference_not_found"),
        ({"step-1": {"experiment_id": "  "}}, "reference_not_found"),
        ({"step-1": {"experiment_id": "x" * 500}}, "reference_not_found"),
    ],
)
def test_an_unresolvable_reference_blocks_its_step(outputs: dict, code: str) -> None:
    """Never a default, never an empty string, never the reference's own text.

    A step that cannot be given what it needs does not run. Filling the gap
    would mean calling a tool with a value nobody supplied.
    """
    step = WorkflowStep(
        step_id="step-2",
        tool="explain_experiment",
        purpose="Explain",
        arguments={"experiment_id": {"from_step": "step-1", "field": "experiment_id"}},
    )

    resolution = resolve_arguments(step, outputs)

    assert not resolution.ok
    assert resolution.blocked_code == code
    assert resolution.blocked_reason


def test_a_declared_dependency_is_required_even_without_a_reference() -> None:
    """`depends_on` means what it says, whether or not a value is read."""
    step = WorkflowStep(
        step_id="step-2",
        tool="search_knowledge",
        purpose="Search",
        arguments={"query": "anything"},
        depends_on=("step-1",),
    )

    assert not resolve_arguments(step, {}).ok
    assert resolve_arguments(step, {"step-1": {"status": "ok"}}).ok


def test_a_reference_implies_a_dependency_the_planner_forgot_to_declare() -> None:
    """A model that writes one and not the other still gets correct ordering."""
    workflow = parse(
        plan(
            steps=[
                {"tool": "run_experiment", "arguments": {"dataset": "sales"}},
                {
                    "tool": "explain_experiment",
                    "arguments": {
                        "experiment_id": {"from_step": "step-1", "field": "experiment_id"}
                    },
                },
            ]
        )
    )

    assert workflow.steps[1].depends_on == ()
    assert workflow.steps[1].requires == ("step-1",)


# ---------------------------------------------------------------------------
# Running a plan
# ---------------------------------------------------------------------------


def test_a_planned_run_executes_every_step_in_order(build_agent) -> None:
    """Three tools, once each, in the order they were planned."""
    agent, planner = build_agent(
        workflow=WORKFLOWS["profile_experiment_explain"],
        answer="The random forest won [exp_1a2b3c4d5e6f_20260101T000000Z_abcd].",
    )

    result = agent.run("Analyse this dataset and tell me which model won.")

    assert [call["tool_name"] for call in result.tool_calls] == [
        "dataset_profile",
        "run_experiment",
        "explain_experiment",
    ]
    assert result.workflow is not None
    assert result.workflow.is_complete
    assert result.workflow.completed_step_count == 3
    # One planning call for the whole run, not one per step.
    assert len(planner.plan_calls) == 1
    assert planner.decide_count == 0


def test_the_experiment_id_travels_between_steps_without_the_model(
    build_agent, executor
) -> None:
    """The point of the whole mechanism.

    `explain_experiment` is called with the id `run_experiment` actually
    produced. Nothing asked a language model to read it out of one output and
    type it into another's arguments.
    """
    agent, _ = build_agent(workflow=WORKFLOWS["profile_experiment_explain"])

    result = agent.run("Explain the winner")

    explain = next(
        call for call in result.tool_calls if call["tool_name"] == "explain_experiment"
    )
    experiment = next(
        obs for obs in result.observations if obs["tool_name"] == "run_experiment"
    )
    assert explain["status"] == "ok"
    assert explain["arguments"]["experiment_id"] == experiment["output"]["experiment_id"]


def test_the_execution_summary_says_what_shape_the_run_had(build_agent) -> None:
    """Enough to say "3 of 3 steps" without reading the observations."""
    agent, _ = build_agent(workflow=WORKFLOWS["profile_experiment_explain"])

    summary = agent.run("Analyse this").execution_summary()

    assert summary["planned"] is True
    assert summary["steps_planned"] == 3
    assert summary["steps_completed"] == 3
    assert summary["workflow_complete"] is True
    assert summary["tools_used"] == [
        "dataset_profile",
        "run_experiment",
        "explain_experiment",
    ]


def test_a_run_without_a_plan_reports_no_workflow(build_agent) -> None:
    """The fallback path is visible rather than disguised as a one-step plan."""
    from agent.planners.fake import PLANS

    agent, _ = build_agent(PLANS["search_then_final"])

    result = agent.run("How is cross-validation used?")

    assert result.workflow is None
    assert result.execution_summary()["planned"] is False
    assert result.tool_call_count == 1


def test_a_planner_that_cannot_plan_falls_back_without_cost(build_agent) -> None:
    """A wasted planning attempt buys no iteration and changes no accounting."""
    from agent.planners.fake import PLANS

    agent, _ = build_agent(
        PLANS["search_then_final"],
        workflow_error=NoScriptedWorkflow("nothing scripted"),
    )

    result = agent.run("How is cross-validation used?")

    assert result.workflow is None
    assert result.tool_call_count == 1
    assert result.iterations == 2


def test_a_provider_failure_while_planning_does_not_retry_in_the_loop(
    build_agent,
) -> None:
    """Falling back would spend a second call to fail the same way."""
    agent, planner = build_agent(
        workflow_error=PlannerProviderError("The provider is unreachable.")
    )

    result = agent.run("anything")

    assert result.status is AgentStatus.FAILED
    assert planner.decide_count == 0


# ---------------------------------------------------------------------------
# Partial results
# ---------------------------------------------------------------------------


def test_a_failed_step_keeps_the_work_that_did_succeed(build_agent) -> None:
    """profile → experiment → explain, where the experiment cannot run.

    The profile worked, so it is reported. The experiment names a dataset that
    does not exist, so the call is refused before it runs. The explanation
    needed the experiment's id, so it is skipped with a reason. The run still
    returns, reports all three, and never claims the whole workflow succeeded.
    """
    agent, _ = build_agent(workflow=WORKFLOWS["partial_chain"])

    result = agent.run("Profile this and explain the best model")

    assert result.status is AgentStatus.PARTIAL
    assert result.workflow is not None
    assert not result.workflow.is_complete
    assert result.workflow.completed_step_count == 1
    statuses = {step.step: step.status for step in result.workflow.steps}
    # "rejected", not "unavailable": the dataset argument's allowed values are
    # read from the session's own dataset names, so a name nobody registered
    # never reaches the tool at all.
    assert statuses == {"step-1": "ok", "step-2": "rejected", "step-3": "skipped"}
    assert "did not produce" in (result.workflow.steps[2].reason or "")


def test_a_run_where_nothing_succeeded_says_so_rather_than_reporting_a_partial(
    build_agent,
) -> None:
    """"Partial" would imply there is something to report. There is not.

    Every step of this plan failed, so the honest status is that nothing the
    tools returned supports an answer — the same answer the agent has always
    given when it has nothing.
    """
    agent, _ = build_agent(workflow=WORKFLOWS["failing_dependency"])

    result = agent.run("Explain the winning model")

    assert result.status is AgentStatus.INSUFFICIENT_EVIDENCE
    assert result.workflow.completed_step_count == 0


def test_a_skipped_step_never_ran_and_is_not_a_tool_call(build_agent) -> None:
    """"Skipped" is its own word because no call was made.

    A caller can tell "the registry refused this" from "this was never
    attempted", and only the first spends a tool call.
    """
    agent, _ = build_agent(workflow=WORKFLOWS["partial_chain"])

    result = agent.run("Profile this and explain the best model")

    assert result.tool_call_count == 2
    assert [call["tool_name"] for call in result.tool_calls] == [
        "dataset_profile",
        "run_experiment",
    ]


def test_an_independent_step_runs_even_when_another_failed(build_agent) -> None:
    """Only the steps that needed the failure are skipped.

    The profile here names a dataset that does not exist and is refused; the
    search does not depend on it and must still happen, because half an answer
    is better than none and the search half is real.
    """
    agent, _ = build_agent(
        workflow={
            "goal": "Two unrelated things",
            "steps": [
                {
                    "tool": "dataset_profile",
                    "purpose": "Profile the dataset",
                    "arguments": {"dataset": "no_such_dataset"},
                },
                {
                    "tool": "search_knowledge",
                    "purpose": "Search the documentation",
                    "arguments": {"query": "leakage prevention"},
                },
            ],
        },
        answer="Leakage is prevented on the training split "
        "[docs:ml-readme#cross-validation].",
    )

    result = agent.run("Two things please")

    statuses = {step.step: step.status for step in result.workflow.steps}
    assert statuses["step-1"] == "rejected"
    assert statuses["step-2"] == "ok"
    assert result.status is AgentStatus.PARTIAL


def test_the_answer_is_told_which_steps_did_not_happen(build_agent) -> None:
    """So it cannot describe work that did not happen.

    The prompt carries what was *carried out*, not what was planned — a
    skipped step is marked NOT DONE, which is the difference between an
    honest partial answer and a confident wrong one.
    """
    agent, planner = build_agent(workflow=WORKFLOWS["partial_chain"])

    agent.run("Profile this and explain the best model")

    executed = planner.answer_calls[-1]["plan_summary"]
    assert executed == [
        "1. Profile the dataset",
        "2. Compare models — NOT DONE (rejected)",
        "3. Explain the winning model — NOT DONE (skipped)",
    ]


def test_the_answer_is_told_what_the_run_was_for(build_agent) -> None:
    """A list of observations does not carry the point of the exercise."""
    agent, planner = build_agent(workflow=WORKFLOWS["profile_experiment_explain"])

    agent.run("Which model is best?")

    assert (
        planner.answer_calls[-1]["objective"]
        == "Name the winning model and say why it was selected"
    )


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


def test_a_plan_longer_than_the_limit_is_refused_and_the_run_falls_back(
    build_agent,
) -> None:
    """An over-long plan costs nothing: no step of it runs."""
    from agent.planners.fake import PLANS

    agent, _ = build_agent(
        PLANS["search_then_final"], workflow=WORKFLOWS["too_many_steps"]
    )

    result = agent.run("Search everything")

    assert result.workflow is None
    assert result.tool_call_count == 1


def test_the_planner_is_told_the_step_limit_it_must_plan_within(
    build_agent,
) -> None:
    """The plan's own limit, which is not the call budget.

    The two bound different things — how much may be planned, and how much may
    be done — and keeping them separate is what lets a run report "you asked
    for four things and I could afford one" instead of silently refusing the
    plan.
    """
    agent, planner = build_agent(
        workflow=WORKFLOWS["search_only"],
        config=AgentConfig(max_tool_calls=6, max_workflow_steps=3, max_tool_repeats=2),
    )

    agent.run("anything")

    assert planner.plan_calls[0]["max_steps"] == 3
    assert planner.plan_calls[0]["max_tool_repeats"] == 2


def test_a_budget_reached_mid_plan_stops_and_accounts_for_every_step(
    build_agent,
) -> None:
    """The remaining steps are recorded as not run, not left unexplained."""
    agent, _ = build_agent(
        workflow=WORKFLOWS["profile_experiment_explain"],
        config=AgentConfig(max_tool_calls=1),
    )

    result = agent.run("Analyse this")

    assert result.tool_call_count == 1
    assert result.error_code == "max_tool_calls"
    statuses = [step.status for step in result.workflow.steps]
    assert statuses == ["ok", "skipped", "skipped"]
    assert all(step.reason for step in result.workflow.steps[1:])


def test_the_clock_stops_a_run_between_steps(registry, artifacts) -> None:
    """A deadline the run can actually honour.

    The first step takes longer than the whole budget, so the second is never
    started. Note what that means and what it does not: the clock is checked
    *between* steps, so the slow step itself ran to completion — nothing here
    executes a tool in something it could abandon. Stated rather than dressed
    up as a cancellation this architecture cannot perform.
    """
    slow = ToolRegistry()
    slow.register(SlowTool())
    slow.register(next(registry.get(name) for name in registry.names()))

    agent = AgentOrchestrator(
        FakePlanner(
            workflow={
                "goal": "Two steps, one of them slow",
                "steps": [
                    {"tool": "slow_tool", "purpose": "Take a while", "arguments": {}},
                    {
                        "tool": slow.names()[1],
                        "purpose": "The step that never starts",
                        "arguments": {},
                    },
                ],
            }
        ),
        slow,
        config=AgentConfig(max_run_seconds=0.02),
        artifacts=artifacts,
    )

    result = agent.run("Analyse this")

    assert result.error_code == "max_run_seconds"
    assert result.tool_call_count == 1
    assert any("time limit" in warning for warning in result.warnings)
    assert result.workflow.steps[1].status == "skipped"


def test_every_limit_appears_in_the_reported_budgets() -> None:
    """A limit that is not reported is one nobody can check."""
    budgets = AgentConfig().as_dict()

    for name in (
        "max_tool_calls",
        "max_iterations",
        "max_workflow_steps",
        "max_tool_repeats",
        "max_run_seconds",
        "max_context_chars",
    ):
        assert name in budgets


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("run_shell", {"command": "rm -rf /"}),
        ("python", {"code": "import os; os.environ"}),
        ("read_file", {"path": "../../.env"}),
        ("http_get", {"url": "http://169.254.169.254/"}),
        ("sql", {"query": "SELECT * FROM users"}),
        ("load_model", {"path": "/tmp/evil.joblib"}),
    ],
)
def test_a_plan_cannot_introduce_a_capability_the_registry_lacks(
    build_agent, tool: str, arguments: dict
) -> None:
    """Shell, Python, files, network, SQL, an arbitrary model path.

    Every one of them is refused the same way and for the same reason: the
    name is not in the registry, so the plan is not a plan this agent can run.
    Nothing is executed, and the run continues on the fallback path.
    """
    agent, _ = build_agent(
        workflow={"goal": "escape", "steps": [{"tool": tool, "arguments": arguments}]}
    )

    result = agent.run("do something dangerous")

    assert result.workflow is None
    assert result.tool_call_count == 0
    assert tool not in json.dumps(result.as_dict())


def test_a_planned_call_still_passes_through_argument_validation(build_agent) -> None:
    """Planning is not an authorisation.

    A plan may name a registered tool and still be wrong about its arguments.
    The schema decides, exactly as it does on the adaptive path, and the call
    is rejected before the tool runs.
    """
    agent, _ = build_agent(
        workflow={
            "goal": "search too hard",
            "steps": [
                {
                    "tool": "search_knowledge",
                    "purpose": "Search",
                    "arguments": {"query": "anything", "top_k": 999_999},
                }
            ],
        }
    )

    result = agent.run("search")

    assert result.workflow.steps[0].status == "rejected"
    assert result.tool_calls[0]["status"] == "rejected"


def test_a_planned_call_cannot_smuggle_an_undeclared_argument(build_agent) -> None:
    """An undeclared field is a rejected call, not an ignored one."""
    agent, _ = build_agent(
        workflow={
            "goal": "smuggle",
            "steps": [
                {
                    "tool": "search_knowledge",
                    "purpose": "Search",
                    "arguments": {"query": "x", "model_path": "/etc/passwd"},
                }
            ],
        }
    )

    result = agent.run("search")

    assert result.tool_calls[0]["status"] == "rejected"
    assert "/etc/passwd" not in json.dumps(result.as_dict())


def test_the_planning_prompt_contains_no_observed_text(build_agent) -> None:
    """Planning happens before anything has run.

    Which means the plan cannot be steered by a retrieved document: there is
    no observation to steer it with. It is the one prompt in the system with
    no untrusted text in it at all.
    """
    agent, planner = build_agent(workflow=WORKFLOWS["search_only"])

    agent.run("anything")

    assert "observations" not in planner.plan_calls[0]


def test_a_plan_reports_no_reasoning_anywhere(build_agent) -> None:
    """What was done and what came back — never how it was decided."""
    agent, _ = build_agent(workflow=WORKFLOWS["profile_experiment_explain"])

    rendered = json.dumps(agent.run("Analyse this").as_dict()).lower()

    for forbidden in ("chain_of_thought", "reasoning", "thought", "scratchpad"):
        assert forbidden not in rendered


def test_a_planned_run_is_json_safe_throughout(build_agent) -> None:
    """No DataFrame, no fitted model, no provider object can reach a caller."""
    agent, _ = build_agent(workflow=WORKFLOWS["profile_experiment_explain"])

    json.dumps(agent.run("Analyse this").as_dict())
