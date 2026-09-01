"""Running the agent for one HTTP request.

The application service the agent endpoint is an adapter around. Like its
siblings it decides *when* to do things and *what to refuse*, and computes
nothing: the whole run — planning, tool selection, execution, grounding — is
:class:`~agent.orchestrator.AgentOrchestrator`'s, built in Commit 12 and
unchanged here.

::

    ask:  question → check the planner → resolve the budget
                   → build the registry for this run's dataset
                   → AgentOrchestrator.run → AgentResult

**The dataset is a loan, not a possession.** A request may bring one; if it
does, the registry built for that run includes the two tools that need data
and the planner is told a dataset is available. The reference lives in the
:class:`~app.services.agent.datasets.RequestDataset` the caller holds, and
falls out of scope when the request ends. Nothing here stores it, and the
service holds no dataset of its own between calls.

Three things happen here rather than deeper down, because they are only
distinguishable at the edge:

**An unconfigured planner.** The orchestrator reports this as a failed run,
which is right for a library caller. Over HTTP it should be refused before any
work is attempted, and answered as a 503 that says what to set.

**A requested budget.** ``agent/`` runs under whatever configuration it is
given; the notion of "smaller than what the server allows" belongs to a server.
See :mod:`app.services.agent.budgets`.

**A failed run.** ``completed``, ``partial``, ``insufficient_evidence`` and
``grounding_failed`` are results and pass straight through as 200. ``failed``
is not: no answer was produced, and a client must not read one out of the body.
It becomes an HTTP error carrying the code the failure mapped to.

**One orchestrator per question.** The service holds the collaborators — the
planner, a way to build the registry, the artifact cache — and assembles an
orchestrator for each run. That is what makes a request-level budget and a
request-level dataset safe: two questions asked at once get separate
registries over separate sources, so neither can see the other's data or
limits, because neither mutates anything shared.

Nothing in this package imports FastAPI, so the same service is drivable from
a script, a test or a future worker.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from agent.config import AgentConfig
from agent.orchestrator import AgentOrchestrator
from agent.registry import ToolRegistry
from agent.results import AgentResult, AgentStatus
from app.core.agent_errors import translate_run_failure
from app.core.errors import MLCopilotError
from app.services.agent.budgets import resolve_config
from app.services.agent.datasets import RequestDataset
from app.services.agent.errors import AgentUnavailableError

logger = logging.getLogger(__name__)

#: What to tell a caller whose planner has no credential. Names what to set
#: rather than a path, and says what still works without one.
NOT_CONFIGURED_MESSAGE = (
    "The agent is not configured. Set an API key for the language-model "
    "provider to use this endpoint. Searching the knowledge base with "
    "POST /api/v1/search needs no credential."
)


class AgentRunFailedError(MLCopilotError):
    """A run finished without producing an answer.

    Carries the code and status its cause mapped to, so a planner timeout and
    a missing credential reach the client as different things without needing
    a class each.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Build the error with the status its cause maps to."""
        super().__init__(message, details=details)
        # Instance attributes shadow the class ones the handler reads.
        self.code = code
        self.status_code = status_code

    @classmethod
    def from_result(cls, result: AgentResult) -> AgentRunFailedError:
        """Build the error for a failed run."""
        code, status_code = translate_run_failure(result.error_code)
        return cls(
            result.final_answer or "The agent could not complete the request.",
            code=code,
            status_code=status_code,
            details={"status": result.status.value},
        )


class AgentService:
    """Answers a question by orchestrating the system's own capabilities."""

    def __init__(
        self,
        *,
        planner: Any | None,
        registry_factory: Callable[[Any], ToolRegistry] | None,
        config: AgentConfig,
        artifacts: Any = None,
    ) -> None:
        """Wire the service to the agent's collaborators and the server's limits.

        Args:
            planner: The planner to decide with. ``None`` when the agent
                cannot be built at all — ``ask`` then refuses with a 503
                rather than the application failing to start.
            registry_factory: Builds the tool allowlist for one run, over the
                dataset that run was given. A factory rather than a registry
                because the dataset is not known until the request is parsed.
            config: The server's limits. A request may lower these, never
                raise them.
            artifacts: The in-memory cache of fitted models, so an experiment
                run during a question can be explained in the same question.
                Cleared by the orchestrator when a run ends.
        """
        self._planner = planner
        self._registry_factory = registry_factory
        self._config = config
        self._artifacts = artifacts

    @property
    def config(self) -> AgentConfig:
        """The server's limits."""
        return self._config

    @property
    def can_run(self) -> bool:
        """Whether a question could be answered right now."""
        return (
            self._planner is not None
            and self._registry_factory is not None
            and bool(getattr(self._planner, "is_ready", False))
        )

    def registry_for(self, dataset: RequestDataset | None = None) -> ToolRegistry:
        """Build the allowlist for a run with, or without, a dataset."""
        if self._registry_factory is None:
            raise AgentUnavailableError(
                NOT_CONFIGURED_MESSAGE, details={"provider_ready": False}
            )
        return self._registry_factory(dataset.source() if dataset else None)

    def tool_names(self, dataset: RequestDataset | None = None) -> tuple[str, ...]:
        """The tools a run with this dataset may use.

        Reported on every answer. A tool is registered only when the service
        it wraps is available, so this is also how a client sees that a
        question asked without a dataset had no ``run_experiment`` to reach
        for.
        """
        if self._registry_factory is None:
            return ()
        return self.registry_for(dataset).names()

    def _build(
        self, config: AgentConfig, dataset: RequestDataset | None
    ) -> AgentOrchestrator:
        """Assemble one orchestrator for one run.

        Raises:
            AgentUnavailableError: If there is no planner, no registry, or the
                planner has no credential. Checked first so a doomed request
                does no work.
        """
        if not self.can_run:
            raise AgentUnavailableError(
                NOT_CONFIGURED_MESSAGE, details={"provider_ready": False}
            )
        return AgentOrchestrator(
            self._planner,
            self.registry_for(dataset),
            config=config,
            artifacts=self._artifacts,
            context=dataset.planner_context() if dataset else {"dataset_available": False},
        )

    def ask(
        self,
        question: str,
        *,
        budgets: dict[str, Any] | None = None,
        dataset: RequestDataset | None = None,
    ) -> AgentResult:
        """Answer one question, within the limits the server allows.

        Args:
            question: What is being asked, in the caller's own words.
            budgets: Requested limits. May only lower the server's.
            dataset: A dataset lent to this run. When given, the two
                dataset-dependent tools are registered and the planner is told
                one is available — a flag, a name and a shape, never a row.
                The reference is not kept: this method holds it for the
                duration of the call and nothing else does.

        Returns:
            AgentResult: The run, with a status saying how it turned out.
            ``completed``, ``partial``, ``insufficient_evidence`` and
            ``grounding_failed`` are all returned as results.

        Raises:
            AgentUnavailableError: If the agent is not configured.
            AgentBudgetError: If a requested budget exceeds the server's.
            AgentRunFailedError: If the run produced no answer at all.
        """
        config = resolve_config(self._config, budgets)
        result = self._build(config, dataset).run(question)

        if result.status is AgentStatus.FAILED:
            logger.info(
                "Agent run failed for a request: %s", result.error_code or "unknown"
            )
            raise AgentRunFailedError.from_result(result)

        return result

    def describe(self) -> dict[str, Any]:
        """Describe what the agent endpoints can currently do.

        ``tools`` is what a question asked *without* a dataset can reach.
        Uploading one adds the two dataset-dependent tools for that request
        only, which the response's own ``tools_available`` reports.

        Safe to show a caller: it reports whether a credential is configured,
        never what it is, and names no filesystem location.
        """
        return {
            "agent_available": self.can_run,
            "tools": list(self.tool_names()),
            "dataset_upload_supported": True,
            "max_tool_calls": self._config.max_tool_calls,
            "max_iterations": self._config.max_iterations,
            "max_context_chars": self._config.max_context_chars,
            "max_answer_length": self._config.max_answer_length,
        }


__all__ = ["NOT_CONFIGURED_MESSAGE", "AgentRunFailedError", "AgentService"]
