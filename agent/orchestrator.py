"""The loop. Bounded, explicit, and the only thing that executes anything.

::

    while the budget holds:
        ask the planner for a decision
        if it asked for a tool:
            is the tool registered?          no  -> rejected observation
            do the arguments validate?       no  -> rejected observation
            run it, record what came back
            continue
        if it asked to finish:
            write the answer, check its grounding, return

    budget spent -> a partial result saying which limit stopped it

Everything about this loop is meant to be readable in one sitting, because
every safety property of the agent is a property of these forty lines.

**It always terminates.** Every path through the body either records an
observation, which costs tool budget, or returns. There is no branch that
retries without spending, and no branch that continues after a budget check
fails. A planner stuck asking for the same broken tool runs out and stops.

**Rejection is cheap and visible.** An unknown tool and an invalid argument
set do not raise out of the run — they become observations with
``status: rejected``, which the planner sees on its next turn. That is
deliberate: a planner that mistypes a tool name should be able to correct
itself, and a caller should be able to see that it tried. What it must never
do is succeed, and it cannot: the rejection happens before any tool code runs.

**A failing tool does not fail the run.** An exception is caught, logged with
its real cause, and recorded as a failed observation carrying an authored
message. A stack trace, a filesystem path or a provider's own words never
reach the state.

**The planner is never the thing that authorises a call.** It names a tool;
the registry decides whether that name exists, and the tool's schema decides
whether the arguments are acceptable. There is no path from planner output to
execution that does not pass through both.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from agent.config import AgentConfig
from agent.errors import (
    AgentError,
    MalformedPlanError,
    PlannerProviderError,
    PlannerUnavailableError,
    ToolValidationError,
    UnknownToolError,
)
from agent.grounding import (
    build_agent_citations,
    check_citations,
    coerce_length,
    evidence_from_observations,
    has_reportable_results,
    is_abstention,
    render_observations_for_answer,
    strip_abstention_marker,
    unsupported_experiment_ids,
)
from agent.observations import (
    Observation,
    ObservationStatus,
    ensure_json_safe,
    summarise_arguments,
)
from agent.registry import ToolRegistry
from agent.results import AgentResult, AgentStatus
from agent.state import ExecutionState

logger = logging.getLogger(__name__)

#: Said when a run stops because a limit was reached rather than because the
#: work finished.
BUDGET_MESSAGES: dict[str, str] = {
    "max_tool_calls": (
        "The agent reached its limit of tool calls for one question and "
        "stopped. The result below covers what it had already found."
    ),
    "max_iterations": (
        "The agent reached its limit of planning steps for one question and "
        "stopped. The result below covers what it had already found."
    ),
    "max_context_chars": (
        "The agent reached its limit on how much observed material one "
        "question may accumulate and stopped. The result below covers what "
        "it had already found."
    ),
}


class AgentOrchestrator:
    """Runs one question to an answer, within hard limits."""

    def __init__(
        self,
        planner: Any,
        registry: ToolRegistry,
        *,
        config: AgentConfig | None = None,
        artifacts: Any = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """Wire the orchestrator to its planner, its tools and its limits.

        Args:
            planner: Anything satisfying :class:`~agent.planner.Planner`.
            registry: The complete allowlist. The planner is shown exactly
                this and the executor accepts exactly this.
            config: Every budget and timeout.
            artifacts: Optional in-memory cache of fitted models, cleared when
                a run finishes so nothing outlives the question that made it.
            context: Named facts about this run, shown to the planner each
                turn — a flag saying a dataset was supplied, the name to
                address it by, its shape. **Facts, never content.** Only
                scalars survive rendering, and this package neither knows nor
                cares what the facts are about; a caller that put a cell value
                in here would be handing whoever wrote that cell a line in the
                prompt, which is why the tools' structured observations are
                the only route content takes.
        """
        self._planner = planner
        self._registry = registry
        self._config = config or AgentConfig()
        self._artifacts = artifacts
        self._context = {
            key: value
            for key, value in (context or {}).items()
            if isinstance(value, (str, bool, int, float))
        }

    @property
    def config(self) -> AgentConfig:
        """The limits in force."""
        return self._config

    @property
    def context(self) -> dict[str, Any]:
        """The run's facts, as the planner is shown them."""
        return dict(self._context)

    # -- One tool call -----------------------------------------------------

    def _reject(
        self, state: ExecutionState, tool_name: str, error: AgentError, arguments: Any
    ) -> Observation:
        """Record a call that was refused before anything ran.

        Only the argument *names* are recorded, never their values. A rejected
        call's arguments never passed validation, so they are unvalidated text
        of unknown length and content — a path the planner invented, a
        credential someone tried to smuggle in, a whole document. Naming the
        fields is enough for the planner to correct itself, and the error
        message already says which field was wrong and what was allowed.
        """
        names = sorted(str(key) for key in arguments) if isinstance(arguments, dict) else []
        return state.record(
            Observation(
                call_id=state.next_call_id(),
                tool_name=tool_name,
                status=ObservationStatus.REJECTED,
                input_summary={"argument_names": names},
                error=error.message,
                error_code=error.code,
            )
        )

    def _execute(self, state: ExecutionState, step: Any) -> Observation:
        """Validate and run one requested tool call.

        The order is the whole point and never varies: look the name up, then
        validate the arguments, then run. Neither of the first two steps can
        be skipped, and the third is unreachable without them.
        """
        tool_name = step.tool or ""

        try:
            tool = self._registry.get(tool_name)
        except UnknownToolError as exc:
            logger.info("Planner requested unregistered tool %r", tool_name)
            return self._reject(state, tool_name, exc, step.arguments)

        try:
            validated = tool.schema.validate(step.arguments)
        except ToolValidationError as exc:
            logger.info("Rejected arguments for tool %r: %s", tool_name, exc.message)
            return self._reject(state, tool_name, exc, step.arguments)

        call_id = state.next_call_id()
        started = time.perf_counter()
        try:
            result = tool.run(validated)
        except Exception:  # noqa: BLE001 - a tool must not break the run
            # The real cause is logged; what reaches the state is authored.
            logger.exception("Tool %r failed", tool_name)
            return state.record(
                Observation(
                    call_id=call_id,
                    tool_name=tool_name,
                    status=ObservationStatus.FAILED,
                    input_summary=summarise_arguments(validated),
                    error=(
                        f"The '{tool_name}' tool failed while running and "
                        "returned no result."
                    ),
                    error_code="tool_execution_failed",
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            )

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        output = ensure_json_safe(result.output)
        if not isinstance(output, dict):  # pragma: no cover - defensive
            output = {"value": output}

        return state.record(
            Observation(
                call_id=call_id,
                tool_name=tool_name,
                status=(
                    ObservationStatus.OK if result.available else ObservationStatus.UNAVAILABLE
                ),
                input_summary=summarise_arguments(validated),
                output=output,
                error=None if result.available else str(output.get("message") or result.reason),
                error_code=None if result.available else result.reason,
                duration_ms=duration_ms,
                citations=tuple(result.citations),
            )
        )

    # -- The run -----------------------------------------------------------

    def run(self, question: str) -> AgentResult:
        """Answer one question, or explain why it could not be answered."""
        state = ExecutionState(question=question, config=self._config)
        started = time.perf_counter()
        definitions = self._registry.definitions()

        try:
            return self._loop(state, definitions, started)
        finally:
            # Nothing a run put in memory outlives it. A fitted model held to
            # explain an experiment is dropped here, so the cache cannot grow
            # across questions or leak between them.
            if self._artifacts is not None:
                self._artifacts.clear()

    def _loop(
        self, state: ExecutionState, definitions: list[dict[str, Any]], started: float
    ) -> AgentResult:
        """The bounded loop."""
        while True:
            exhausted = state.exhausted_budget()
            if exhausted is not None:
                state.warn(BUDGET_MESSAGES[exhausted])
                return self._finalise(
                    state, started, stopped_by=exhausted, allow_answer=True
                )

            state.begin_iteration()

            try:
                step = self._planner.decide(
                    state.question,
                    tool_definitions=definitions,
                    observations=render_observations_for_answer(state.observations),
                    remaining_tool_calls=state.remaining_tool_calls,
                    context=dict(self._context),
                )
            except MalformedPlanError as exc:
                # Not a decision. Nothing was called and nothing was executed —
                # including when the "response" was a block of Python.
                logger.info("Planner produced no usable decision: %s", exc.message)
                state.warn(
                    "The planner did not produce a usable decision, so the run "
                    "stopped."
                )
                return self._failed(state, started, exc)
            except (PlannerUnavailableError, PlannerProviderError) as exc:
                return self._failed(state, started, exc)

            if step.is_final:
                return self._finalise(state, started, stopped_by=None, allow_answer=True)

            self._execute(state, step)

    # -- Ending ------------------------------------------------------------

    def _failed(
        self, state: ExecutionState, started: float, error: AgentError
    ) -> AgentResult:
        """Return a structured failure, keeping whatever work was done."""
        return self._build(
            state,
            started,
            status=AgentStatus.FAILED,
            answer=error.message,
            error_code=error.code,
        )

    def _finalise(
        self,
        state: ExecutionState,
        started: float,
        *,
        stopped_by: str | None,
        allow_answer: bool,
    ) -> AgentResult:
        """Write the answer and check it against what was observed."""
        context = evidence_from_observations(state.observations)
        allowed = tuple(context.allowed_citations)

        if not allow_answer:  # pragma: no cover - reserved for future callers
            return self._build(
                state, started, status=AgentStatus.PARTIAL, answer="", error_code=stopped_by
            )

        try:
            text = self._planner.write_answer(
                state.question,
                observations=render_observations_for_answer(state.observations),
                allowed_citations=list(allowed),
            )
        except (PlannerUnavailableError, PlannerProviderError) as exc:
            return self._failed(state, started, exc)

        text, was_cut = coerce_length(text or "", self._config.max_answer_length)
        if was_cut:
            state.warn(
                "The answer was longer than the configured limit and was cut."
            )

        # Nothing to answer from, or the model said as much itself.
        if is_abstention(text) or not has_reportable_results(state.observations):
            return self._build(
                state,
                started,
                status=AgentStatus.INSUFFICIENT_EVIDENCE,
                answer=strip_abstention_marker(text)
                or (
                    "The available tools did not return enough to answer that "
                    "question."
                ),
                allowed=allowed,
                error_code=stopped_by,
            )

        report = check_citations(text, context)
        invented = unsupported_experiment_ids(text, state.observations)

        if report.has_fabrications or invented:
            # Reported, never repaired. Guessing which real source was meant
            # would turn an obvious failure into a subtle one.
            for reason in report.reasons:
                state.warn(reason)
            if invented:
                state.warn(
                    "The answer referred to "
                    + ", ".join(f"'{item}'" for item in invented)
                    + ", which no tool produced in this run."
                )
            return self._build(
                state,
                started,
                status=AgentStatus.GROUNDING_FAILED,
                answer=text,
                allowed=allowed,
                rejected=tuple(report.fabricated) + invented,
                error_code="grounding_failed",
            )

        # Evidence was retrieved but the answer cites none of it.
        if allowed and not report.valid:
            for reason in report.reasons:
                state.warn(reason)
            return self._build(
                state,
                started,
                status=AgentStatus.GROUNDING_FAILED,
                answer=text,
                allowed=allowed,
                error_code="grounding_failed",
            )

        unavailable = any(
            observation.status is ObservationStatus.UNAVAILABLE
            for observation in state.observations
        )
        rejected_or_failed = any(
            observation.status
            in {ObservationStatus.REJECTED, ObservationStatus.FAILED}
            for observation in state.observations
        )
        partial = bool(stopped_by) or unavailable or rejected_or_failed

        return self._build(
            state,
            started,
            status=AgentStatus.PARTIAL if partial else AgentStatus.COMPLETED,
            answer=text,
            allowed=allowed,
            citation_ids=report.valid,
            context=context,
            error_code=stopped_by,
        )

    def _build(
        self,
        state: ExecutionState,
        started: float,
        *,
        status: AgentStatus,
        answer: str,
        allowed: tuple[str, ...] = (),
        rejected: tuple[str, ...] = (),
        citation_ids: tuple[str, ...] = (),
        context: Any = None,
        error_code: str | None = None,
    ) -> AgentResult:
        """Assemble the result."""
        completed_at = datetime.now(timezone.utc)
        citations = (
            build_agent_citations(citation_ids, context)
            if context is not None and citation_ids
            else ()
        )
        return AgentResult(
            question=state.question,
            status=status,
            final_answer=answer,
            tool_calls=state.tool_calls(),
            observations=[item.as_dict() for item in state.observations],
            citations=citations,
            rejected_citations=rejected,
            allowed_citations=allowed,
            experiment_ids=state.experiment_ids(),
            warnings=tuple(state.warnings),
            iterations=state.iterations,
            tool_call_count=state.tool_call_count,
            error_code=error_code,
            started_at=state.started_at,
            completed_at=completed_at,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )


__all__ = ["BUDGET_MESSAGES", "AgentOrchestrator"]
