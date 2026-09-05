"""A plan made once, up front, and everything that keeps it a plan.

The agent has always been able to run several tools: the loop asks the planner
for one decision, executes it, and asks again. What it could not do is *plan* —
each turn saw only what had happened, so "profile this dataset, run an
experiment on it, then explain the winner" was three unrelated guesses that
happened to come out in a sensible order, and the third one had to be told the
experiment id by a model reading it back out of the second one's output.

This module adds the missing shape. A planner may now answer the question
"what will this take?" once, with a whole workflow:

.. code-block:: json

    {"goal": "Find and explain the best model for renewals",
     "objective": "Name the winning model and say why it was selected",
     "steps": [
       {"tool": "dataset_profile", "purpose": "Profile the uploaded dataset",
        "arguments": {"dataset": "uploaded_dataset"}},
       {"tool": "run_experiment", "purpose": "Compare models",
        "arguments": {"dataset": "uploaded_dataset", "target_column": "renewed"}},
       {"tool": "explain_experiment", "purpose": "Explain the winner",
        "depends_on": ["step-2"],
        "arguments": {"experiment_id": {"from_step": "step-2",
                                        "field": "experiment_id"}}}]}

---------------------------------------------------------------------------
Why this is still bounded
---------------------------------------------------------------------------
A plan is more capable than a single decision, so it is worth being precise
about what has *not* changed.

**Nothing here executes.** This module parses and validates. The orchestrator
runs steps through exactly the path it always has — the registry decides
whether a tool exists, the tool's schema decides whether the arguments are
acceptable — and neither can be reached any other way.

**A plan cannot name a tool that does not exist.** The registered names are
passed in and checked at parse time, so an invented tool is a rejected *plan*
rather than a rejected call at step four. This is stricter than the loop, which
could only reject a name once it was asked for.

**A plan cannot loop.** ``depends_on`` may only name an *earlier* step. That is
not a cycle check, it is the absence of a way to express a cycle: the steps are
a list, dependencies point backwards, and execution is one pass from first to
last. There is no scheduler, no re-planning, and no path that revisits a step.

**A plan is finite before it starts.** The step count is capped, and so is how
many times one tool may appear. Both are checked here, before a single step
runs — a plan of forty searches is refused as a plan rather than discovered
halfway through.

---------------------------------------------------------------------------
References, and why they are this narrow
---------------------------------------------------------------------------
A later step needs values from an earlier one, and the obvious way to get them
is to ask the model to copy them across. That is the way that goes wrong: it
puts tool output through a language model and back out again, where it can be
mistyped, hallucinated, or — for anything larger than an id — spend the whole
context budget.

So a reference is a small closed form, ``{"from_step": ..., "field": ...}``,
resolved *by this code* from the structured observation the earlier step
produced. Three rules keep it from becoming a query language:

1. **Only an allowlisted field name.** :data:`REFERENCEABLE_FIELDS` is the
   complete set. There is no dotted path, no index, no wildcard and no
   expression, so a reference cannot reach into a nested structure or walk to
   somewhere it was not meant to go.
2. **Only a scalar.** A resolved value must be a string, number or boolean. A
   list or an object cannot be carried into a tool argument this way.
3. **Only a step that succeeded.** An unresolved reference makes its step
   unrunnable, and it is skipped with a stated reason — never guessed at,
   never filled with a default, never passed through as the literal text of
   the reference.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent.errors import MalformedPlanError

#: Longest planner response this module will attempt to parse. A workflow is a
#: short object; anything longer is a model writing prose.
MAX_WORKFLOW_CHARS = 20_000

#: Caps on the free text a plan may carry. These strings are written by a model
#: and shown to a person, so they are bounded like every other such value.
MAX_GOAL_CHARS = 300
MAX_PURPOSE_CHARS = 120
MAX_OBJECTIVE_CHARS = 300

#: How a step is identified. Assigned **here**, from the step's position, and
#: never read from the planner's response — an identifier a model chooses is
#: one that can be made to collide, to look like a path, or to carry text into
#: a log line.
STEP_ID_TEMPLATE = "step-{ordinal}"
_STEP_ID = re.compile(r"\Astep-(?P<ordinal>\d{1,3})\Z")

#: The keys a reference object uses. Both are required; anything else in the
#: object is a malformed plan rather than an ignored extra.
REFERENCE_STEP_KEY = "from_step"
REFERENCE_FIELD_KEY = "field"

#: Every field a later step may read from an earlier one. The complete list:
#: identifiers and short labels that one tool produces and another needs.
#:
#: What is deliberately absent is everything else — scores, row counts, column
#: lists, retrieved passages, feature importances. Those belong in the answer,
#: not in a tool argument, and a reference that could reach them would be a way
#: to move observed content into a call without anything checking it.
REFERENCEABLE_FIELDS: frozenset[str] = frozenset(
    {
        "experiment_id",
        "dataset",
        "target_column",
        "task_type",
        "selected_model",
        "primary_metric",
    }
)


class MalformedWorkflowError(MalformedPlanError):
    """A planner's workflow was not a workflow.

    A subclass of :class:`~agent.errors.MalformedPlanError` on purpose: to
    every caller above, "the planner did not produce something usable" is one
    situation with one handling, whether what it failed to produce was a single
    decision or a whole plan.
    """

    code = "malformed_workflow"


@dataclass(frozen=True)
class WorkflowStep:
    """One planned tool call, and what it needs before it can run."""

    #: Assigned from position, never by the planner.
    step_id: str
    #: A registered tool name, checked against the registry at parse time.
    tool: str
    #: A short label saying what this step is *for*, shown to a person as part
    #: of the plan. It is a label, not reasoning — see ``agent/README.md``.
    purpose: str
    #: Arguments as planned. Values may be reference objects, which are
    #: resolved against earlier observations immediately before the call.
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Earlier steps this one needs. Always earlier, so a plan is a sequence.
    depends_on: tuple[str, ...] = ()

    @property
    def references(self) -> tuple[str, ...]:
        """Every step id this step's arguments refer to, deduplicated."""
        found: list[str] = []
        for value in self.arguments.values():
            reference = as_reference(value)
            if reference is not None and reference[0] not in found:
                found.append(reference[0])
        return tuple(found)

    @property
    def requires(self) -> tuple[str, ...]:
        """Every earlier step this one needs, declared or implied.

        A reference in an argument *is* a dependency whether or not the planner
        also listed it in ``depends_on``, so the two are unioned rather than
        trusted separately. A model that writes one and forgets the other still
        gets a correctly ordered run.
        """
        combined = list(self.depends_on)
        for step_id in self.references:
            if step_id not in combined:
                combined.append(step_id)
        return tuple(combined)

    def as_dict(self) -> dict[str, Any]:
        """Render the step as plain JSON-safe values.

        **The arguments are not included.** A plan is shown to a person as
        "what will be done", and the values are the one place a planner could
        put text of its own choosing into that display. What a call actually
        received is reported by its observation, already summarised.
        """
        return {
            "step": self.step_id,
            "tool": self.tool,
            "purpose": self.purpose,
            "depends_on": list(self.depends_on),
        }


@dataclass(frozen=True)
class Workflow:
    """A whole plan: what it is for, what it will do, and what to say at the end."""

    goal: str
    steps: tuple[WorkflowStep, ...]
    #: What the final answer should accomplish. Carried into the answer prompt
    #: so the writing step knows what the run was *for*, which a list of
    #: observations does not say.
    objective: str = ""

    def __len__(self) -> int:
        """How many steps the plan has."""
        return len(self.steps)

    @property
    def tools(self) -> tuple[str, ...]:
        """Every tool the plan uses, in order of first appearance."""
        seen: list[str] = []
        for step in self.steps:
            if step.tool not in seen:
                seen.append(step.tool)
        return tuple(seen)

    def step(self, step_id: str) -> WorkflowStep | None:
        """Return one step by id, or ``None``."""
        for candidate in self.steps:
            if candidate.step_id == step_id:
                return candidate
        return None

    def summary_lines(self) -> list[str]:
        """The plan as a person reads it: one numbered line per step.

        This is the *whole* of what a caller is shown about planning. It says
        what will be done and in what order; it does not say how the planner
        decided, because that is not returned, stored or logged anywhere.
        """
        return [
            f"{index}. {step.purpose}" for index, step in enumerate(self.steps, start=1)
        ]

    def as_dict(self) -> dict[str, Any]:
        """Render the plan as plain JSON-safe values."""
        return {
            "goal": self.goal,
            "objective": self.objective,
            "steps": [step.as_dict() for step in self.steps],
            "summary": self.summary_lines(),
        }


def as_reference(value: Any) -> tuple[str, str] | None:
    """Read a value as a reference to an earlier step, or return ``None``.

    Recognises exactly one shape — a two-key object naming a step and a field —
    and nothing else. A string that merely looks like a placeholder is not a
    reference: it is a string, and it reaches the tool's schema as one.

    Returns:
        tuple[str, str] | None: The step id and field name, or ``None`` when
        the value is an ordinary argument.
    """
    if not isinstance(value, Mapping):
        return None
    if set(value) != {REFERENCE_STEP_KEY, REFERENCE_FIELD_KEY}:
        return None
    step_id = value.get(REFERENCE_STEP_KEY)
    field_name = value.get(REFERENCE_FIELD_KEY)
    if not isinstance(step_id, str) or not isinstance(field_name, str):
        return None
    return step_id.strip(), field_name.strip()


def _clean(value: Any, limit: int) -> str:
    """Return a bounded single-line string, or an empty one."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


def _step_ordinal(step_id: str) -> int | None:
    """Return the position a step id names, or ``None`` if it is not one."""
    match = _STEP_ID.match(step_id.strip())
    return int(match.group("ordinal")) if match else None


def _parse_step(
    payload: Any,
    *,
    ordinal: int,
    known_tools: frozenset[str],
) -> WorkflowStep:
    """Turn one entry of the plan into a step, or refuse the plan.

    Raises:
        MalformedWorkflowError: If the entry is not an object, names no tool,
            names a tool that is not registered, carries arguments that are not
            an object, or declares a dependency that is not an earlier step.
    """
    if not isinstance(payload, Mapping):
        raise MalformedWorkflowError(
            f"Step {ordinal} of the plan is not an object.",
            details={"reason": "step_not_an_object", "step": ordinal},
        )

    tool = payload.get("tool") or payload.get("tool_name")
    if not isinstance(tool, str) or not tool.strip():
        raise MalformedWorkflowError(
            f"Step {ordinal} of the plan does not name a tool.",
            details={"reason": "missing_tool", "step": ordinal},
        )
    tool = tool.strip()

    # The refusal that matters most, and it happens here rather than at
    # execution: a plan that names a tool nobody registered is not a plan this
    # agent can run, and finding that out before step one is better than
    # finding it out at step four with three calls already spent.
    if tool not in known_tools:
        raise MalformedWorkflowError(
            f"Step {ordinal} of the plan names '{tool[:80]}', which is not an "
            "available tool. No tool was called and nothing was executed.",
            details={"reason": "unknown_tool", "step": ordinal, "tool": tool[:80]},
        )

    arguments = payload.get("arguments", payload.get("input", {}))
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise MalformedWorkflowError(
            f"Step {ordinal}'s arguments must be an object of named values.",
            details={"reason": "invalid_arguments", "step": ordinal},
        )

    raw_dependencies = payload.get("depends_on") or payload.get("after") or []
    if isinstance(raw_dependencies, str):
        raw_dependencies = [raw_dependencies]
    if not isinstance(raw_dependencies, Sequence):
        raise MalformedWorkflowError(
            f"Step {ordinal}'s dependencies must be a list of step ids.",
            details={"reason": "invalid_depends_on", "step": ordinal},
        )

    dependencies: list[str] = []
    for entry in raw_dependencies:
        if not isinstance(entry, str):
            raise MalformedWorkflowError(
                f"Step {ordinal} declares a dependency that is not a step id.",
                details={"reason": "invalid_depends_on", "step": ordinal},
            )
        position = _step_ordinal(entry)
        # Backwards only. This is what makes a cycle unrepresentable rather
        # than merely detected: there is no way to write a plan whose steps
        # wait on each other, so execution is one pass and always terminates.
        if position is None or not 1 <= position < ordinal:
            raise MalformedWorkflowError(
                f"Step {ordinal} depends on '{str(entry)[:40]}', which is not "
                "an earlier step. A step may only depend on one before it.",
                details={
                    "reason": "forward_dependency",
                    "step": ordinal,
                    "depends_on": str(entry)[:40],
                },
            )
        identifier = STEP_ID_TEMPLATE.format(ordinal=position)
        if identifier not in dependencies:
            dependencies.append(identifier)

    for name, value in arguments.items():
        reference = as_reference(value)
        if reference is None:
            continue
        step_id, field_name = reference
        position = _step_ordinal(step_id)
        if position is None or not 1 <= position < ordinal:
            raise MalformedWorkflowError(
                f"Step {ordinal}'s '{str(name)[:40]}' refers to "
                f"'{step_id[:40]}', which is not an earlier step.",
                details={
                    "reason": "forward_reference",
                    "step": ordinal,
                    "argument": str(name)[:40],
                },
            )
        if field_name not in REFERENCEABLE_FIELDS:
            raise MalformedWorkflowError(
                f"Step {ordinal}'s '{str(name)[:40]}' asks for "
                f"'{field_name[:40]}', which is not a field a later step may "
                "read from an earlier one.",
                details={
                    "reason": "unknown_reference_field",
                    "step": ordinal,
                    "field": field_name[:40],
                    "allowed": sorted(REFERENCEABLE_FIELDS),
                },
            )

    purpose = _clean(payload.get("purpose") or payload.get("reason"), MAX_PURPOSE_CHARS)
    return WorkflowStep(
        step_id=STEP_ID_TEMPLATE.format(ordinal=ordinal),
        tool=tool,
        purpose=purpose or f"Run {tool}",
        arguments=dict(arguments),
        depends_on=tuple(dependencies),
    )


def parse_workflow(
    text: str,
    *,
    known_tools: Sequence[str],
    max_steps: int,
    max_tool_repeats: int,
) -> Workflow:
    """Turn a planner's response into a validated plan, or refuse it.

    Every check happens here, before anything runs. A plan that survives this
    function names only registered tools, is no longer than the limit, uses no
    tool more often than the limit, and has dependencies that point only
    backwards — so the executor's job is to run a list, not to police one.

    Args:
        text: Exactly what the planner produced.
        known_tools: The registered tool names. A step naming anything else
            makes the whole plan malformed.
        max_steps: Most steps a plan may have.
        max_tool_repeats: Most times one tool may appear in a plan.

    Returns:
        Workflow: The validated plan.

    Raises:
        MalformedWorkflowError: For every one of the failures above, and for a
            response that is empty, over-long, not JSON, not an object, or
            carries no steps. Malformed is where it stops: this function never
            reads the text as an instruction, never extracts a code block and
            never guesses at intent.
    """
    if not isinstance(text, str) or not text.strip():
        raise MalformedWorkflowError(
            "The planner returned no plan.", details={"reason": "empty"}
        )
    if len(text) > MAX_WORKFLOW_CHARS:
        raise MalformedWorkflowError(
            "The planner's plan was too long to be a plan.",
            details={"reason": "too_long", "length": len(text)},
        )

    payload = _load_object(text)

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, Sequence) or isinstance(raw_steps, (str, bytes)):
        raise MalformedWorkflowError(
            "A plan must carry a list of steps.",
            details={"reason": "missing_steps"},
        )
    if not raw_steps:
        raise MalformedWorkflowError(
            "A plan must have at least one step.",
            details={"reason": "no_steps"},
        )
    if len(raw_steps) > max_steps:
        raise MalformedWorkflowError(
            f"A plan may have at most {max_steps} steps; this one has "
            f"{len(raw_steps)}.",
            details={
                "reason": "too_many_steps",
                "steps": len(raw_steps),
                "maximum": max_steps,
            },
        )

    allowed = frozenset(known_tools)
    steps = tuple(
        _parse_step(entry, ordinal=index, known_tools=allowed)
        for index, entry in enumerate(raw_steps, start=1)
    )

    # One tool used over and over is the shape a runaway plan takes: six
    # searches, or the same experiment run again because the first answer was
    # not liked. Capped here so it is refused as a plan.
    counts: dict[str, int] = {}
    for step in steps:
        counts[step.tool] = counts.get(step.tool, 0) + 1
        if counts[step.tool] > max_tool_repeats:
            raise MalformedWorkflowError(
                f"A plan may use '{step.tool}' at most {max_tool_repeats} "
                "time(s).",
                details={
                    "reason": "tool_repeated",
                    "tool": step.tool,
                    "maximum": max_tool_repeats,
                },
            )

    return Workflow(
        goal=_clean(payload.get("goal"), MAX_GOAL_CHARS),
        steps=steps,
        objective=_clean(
            payload.get("objective") or payload.get("final_answer_objective"),
            MAX_OBJECTIVE_CHARS,
        ),
    )


#: Strips a ```json fence, the way :mod:`agent.plans` does. Formatting only:
#: what is inside still has to be a plan.
_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*(?P<body>.*?)\s*```\s*$", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _load_object(text: str) -> Mapping[str, Any]:
    """Read the response as a JSON object, or refuse it.

    ``json.loads`` and nothing else — never a literal evaluator, never a
    template. A response that is a block of Python is text that does not parse
    as an object, which is the same outcome as an empty reply.
    """
    candidate = text.strip()
    fenced = _FENCE.match(candidate)
    if fenced:
        candidate = fenced.group("body")

    try:
        payload = json.loads(candidate)
    except (ValueError, TypeError):
        match = _OBJECT.search(candidate)
        if match is None:
            raise MalformedWorkflowError(
                "The planner's plan was not a plan. Expected a JSON object "
                "with a list of steps; no tool was called and nothing was "
                "executed.",
                details={"reason": "not_json"},
            ) from None
        try:
            payload = json.loads(match.group(0))
        except (ValueError, TypeError):
            raise MalformedWorkflowError(
                "The planner's plan was not a plan. Expected a JSON object "
                "with a list of steps; no tool was called and nothing was "
                "executed.",
                details={"reason": "not_json"},
            ) from None

    if not isinstance(payload, Mapping):
        raise MalformedWorkflowError(
            "A plan must be a JSON object.", details={"reason": "not_an_object"}
        )
    return payload


#: Longest a resolved reference value may be. An id is short; a value longer
#: than this is not the kind of thing a reference is for, and passing it into a
#: tool argument would be moving observed content somewhere it was not checked.
MAX_RESOLVED_VALUE_CHARS = 200


@dataclass(frozen=True)
class Resolution:
    """The outcome of preparing one step's arguments.

    Either the arguments are ready, or the step cannot run and there is a
    stated reason. There is no third case: a reference that could not be
    resolved is never replaced with a default, an empty string, or the literal
    text of the reference itself.
    """

    arguments: dict[str, Any] = field(default_factory=dict)
    #: A sentence for the caller when the step cannot run. ``None`` when it can.
    blocked_reason: str | None = None
    #: A stable code beside it, for a test and for a log.
    blocked_code: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the step may proceed."""
        return self.blocked_reason is None


def resolve_arguments(
    step: WorkflowStep, outputs: Mapping[str, Mapping[str, Any]]
) -> Resolution:
    """Fill a step's references from what earlier steps actually produced.

    This is the whole of the dependency mechanism, and it is deliberately
    unexciting: for each argument that is a reference, look up the named step's
    observed output, read one allowlisted field, check it is a short scalar,
    and use it. Anything else blocks the step.

    Args:
        step: The planned step.
        outputs: Structured output per completed step id. A step that failed,
            was skipped or has not run is simply absent, which is what makes
            "the step it needed did not produce anything" the default outcome
            rather than a special case.

    Returns:
        Resolution: The arguments to call with, or a reason the step cannot run.
    """
    for required in step.requires:
        if required not in outputs:
            return Resolution(
                blocked_reason=(
                    f"This step needed the result of {required}, which did not "
                    "produce one."
                ),
                blocked_code="dependency_unavailable",
            )

    resolved: dict[str, Any] = {}
    for name, value in step.arguments.items():
        reference = as_reference(value)
        if reference is None:
            resolved[name] = value
            continue

        source_id, field_name = reference
        source = outputs.get(source_id)
        if source is None:
            return Resolution(
                blocked_reason=(
                    f"This step needed '{field_name}' from {source_id}, which "
                    "did not produce a result."
                ),
                blocked_code="dependency_unavailable",
            )

        # Only from the allowlist, and only ever one level deep. Checked again
        # here rather than trusted from parse time: this function is reachable
        # on its own, and a check that only happens somewhere else is a check
        # that stops happening when a caller changes.
        if field_name not in REFERENCEABLE_FIELDS:
            return Resolution(
                blocked_reason=(
                    f"This step asked for '{field_name}', which is not a field "
                    "a later step may read."
                ),
                blocked_code="unknown_reference_field",
            )

        # A scalar, or nothing. A list or an object cannot be carried into a
        # tool argument this way — that would be a route for observed content
        # to enter a call in a shape the schema was not written to expect.
        found = source.get(field_name)
        if not isinstance(found, (str, int, float)):
            return Resolution(
                blocked_reason=(
                    f"{source_id} did not produce a usable '{field_name}'."
                ),
                blocked_code="reference_not_found",
            )
        if isinstance(found, str) and (
            not found.strip() or len(found) > MAX_RESOLVED_VALUE_CHARS
        ):
            return Resolution(
                blocked_reason=(
                    f"{source_id}'s '{field_name}' was not a usable value."
                ),
                blocked_code="reference_not_found",
            )
        resolved[name] = found

    return Resolution(arguments=resolved)


__all__ = [
    "MAX_GOAL_CHARS",
    "MAX_RESOLVED_VALUE_CHARS",
    "Resolution",
    "resolve_arguments",
    "MAX_OBJECTIVE_CHARS",
    "MAX_PURPOSE_CHARS",
    "MAX_WORKFLOW_CHARS",
    "REFERENCEABLE_FIELDS",
    "REFERENCE_FIELD_KEY",
    "REFERENCE_STEP_KEY",
    "STEP_ID_TEMPLATE",
    "MalformedWorkflowError",
    "Workflow",
    "WorkflowStep",
    "as_reference",
    "parse_workflow",
]
