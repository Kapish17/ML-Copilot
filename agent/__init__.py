"""ML Copilot — the bounded agent layer.

Lets a language model decide which of the system's existing capabilities a
question needs, and then run them — within limits it cannot raise, through
tools it cannot add to, over evidence it cannot invent.

``config``        the budgets and timeouts one run operates under
``errors``        the refusals and breakdowns of orchestration itself
``schemas``       typed argument declarations, and the validation of a call
``plans``         the two decisions a planner may make, and parsing one
``prompts``       what the planner and the answerer are told to distrust
``planner``       asking a model what to do, through the provider abstraction
``planners``      a deterministic scripted planner, for tests
``registry``      the allowlist
``tools``         the four capabilities, over the services that already exist
``observations``  what came back, JSON-safe and treated as data
``state``         what a run knows so far and what it may still spend
``grounding``     checking the answer against what was actually observed
``results``       what a run returns
``orchestrator``  the bounded loop

**The agent can only execute explicitly registered tools.** The agent never
executes arbitrary Python, shell commands, HTTP requests, or filesystem
operations.

Those two sentences are the design. There is no generic ``execute`` tool, no
code evaluation, no subprocess, no HTTP client and no filesystem handle
anywhere in this package; a planner's response is parsed as one of two
declared decisions or rejected as malformed; a tool name that is not in the
registry cannot be run by any path; and arguments are validated against a
declared schema before a tool sees them.

The agent orchestrates. It computes nothing: profiling belongs to the dataset
service, training and selection to the experiment runner, ranking to the
retrieval layer, SHAP to the explainability layer, and generation to the
provider abstraction. Every one of those is reached through a structural
protocol, so this package imports no web framework, no SDK, and neither
pandas, numpy, scikit-learn nor SHAP.

**No HTTP endpoint is implemented.** This is the library layer; exposing it
over FastAPI is a later commit.
"""

from agent.config import AgentConfig, config_from_env
from agent.errors import (
    AgentConfigurationError,
    AgentError,
    BudgetExhaustedError,
    DuplicateToolError,
    MalformedPlanError,
    PlannerError,
    PlannerProviderError,
    PlannerUnavailableError,
    ToolError,
    ToolExecutionError,
    ToolValidationError,
    UnknownToolError,
)
from agent.observations import Observation, ObservationStatus, ensure_json_safe
from agent.orchestrator import AgentOrchestrator
from agent.planner import LLMPlanner, Planner
from agent.planners.fake import FakePlanner
from agent.plans import PlanStep, parse_plan
from agent.registry import ToolRegistry
from agent.results import AgentCitation, AgentResult, AgentStatus
from agent.schemas import ArgumentField, ArgumentSchema
from agent.state import ExecutionState
from agent.tools import (
    DatasetProfileTool,
    ExperimentArtifactCache,
    ExplainExperimentTool,
    InMemoryDatasetSource,
    RunExperimentTool,
    SearchKnowledgeTool,
    Tool,
    ToolResult,
    build_default_registry,
)

__all__ = [
    "AgentCitation",
    "AgentConfig",
    "AgentConfigurationError",
    "AgentError",
    "AgentOrchestrator",
    "AgentResult",
    "AgentStatus",
    "ArgumentField",
    "ArgumentSchema",
    "BudgetExhaustedError",
    "DatasetProfileTool",
    "DuplicateToolError",
    "ExecutionState",
    "ExperimentArtifactCache",
    "ExplainExperimentTool",
    "FakePlanner",
    "InMemoryDatasetSource",
    "LLMPlanner",
    "MalformedPlanError",
    "Observation",
    "ObservationStatus",
    "PlanStep",
    "Planner",
    "PlannerError",
    "PlannerProviderError",
    "PlannerUnavailableError",
    "RunExperimentTool",
    "SearchKnowledgeTool",
    "Tool",
    "ToolError",
    "ToolExecutionError",
    "ToolRegistry",
    "ToolResult",
    "ToolValidationError",
    "UnknownToolError",
    "build_default_registry",
    "config_from_env",
    "ensure_json_safe",
    "parse_plan",
]
