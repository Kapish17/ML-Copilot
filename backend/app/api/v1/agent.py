"""The agent endpoint: ask a question, let the system choose how to answer it.

Two handlers, each doing the same three things and nothing else: take the
request, hand it to the agent service, validate the structured result against a
response schema. No planning, no tool selection, no budget arithmetic and no
grounding check happens here — those belong to ``agent/`` and to
:mod:`app.services.agent.service`, and a test asserts that these handlers stay
under three statements.

Failures propagate. Agent errors, the two API-level refusals and a run that
produced no answer are all turned into the one documented envelope by the
application's exception handlers, so no handler builds an error response by
hand.

One thing decides the status codes, and it is the same rule the knowledge
endpoints follow: **a question that was processed but could not be answered
well is a result, not a failure.** A partial run, a run with no relevant
evidence and a run whose answer failed its grounding check are all 200 with a
status saying so. Only a run that produced nothing at all — an unusable
planner — is an HTTP error.
"""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.dependencies import AgentServiceDep
from app.schemas.agent import AgentAskRequest, AgentAskResponse, AgentStatusResponse
from app.schemas.errors import ErrorResponse

router = APIRouter(tags=["agent"])

#: The example bodies shown in the generated documentation. Four outcomes, all
#: of them 200, because telling them apart is the thing a client most needs to
#: get right — and reading them side by side is the quickest way to see that
#: `status`, not `final_answer`, is the field that matters.
_ASK_EXAMPLES: dict[str, dict[str, object]] = {
    "completed": {
        "summary": "Answered, and every citation checks out",
        "description": (
            "The only status whose `final_answer` may be shown to a user as "
            "an answer."
        ),
        "value": {
            "question": "How does this project select a model?",
            "status": "completed",
            "final_answer": (
                "Models are compared by cross-validation on the training rows "
                "only, and the winner is measured once on the untouched test "
                "set [docs:ml-readme#cross-validation]."
            ),
            "is_answer": True,
            "tool_calls": [
                {
                    "call_id": "call-01",
                    "tool_name": "search_knowledge",
                    "status": "ok",
                    "arguments": {"query": "how is a model selected"},
                    "duration_ms": 12.4,
                }
            ],
            "citation_ids": ["docs:ml-readme#cross-validation"],
            "rejected_citations": [],
            "experiment_ids": [],
            "warnings": [],
            "iterations": 2,
            "tool_call_count": 1,
            "tools_available": ["search_knowledge", "explain_experiment"],
        },
    },
    "partial": {
        "summary": "Real work done, something missing",
        "description": (
            "The experiment ran and is reported in full; the explanation was "
            "unavailable and the gap is stated rather than filled."
        ),
        "value": {
            "question": "Which model performed best, and why?",
            "status": "partial",
            "final_answer": (
                "Experiment exp_86494cff7a45_20260101T120000Z_a1b2 selected "
                "random_forest_classifier, scoring 0.86 F1 on the held-out "
                "test set. No feature explanation was available for it."
            ),
            "is_answer": False,
            "citation_ids": [],
            "rejected_citations": [],
            "experiment_ids": ["exp_86494cff7a45_20260101T120000Z_a1b2"],
            "warnings": [
                "'explain_experiment' could not provide a result: a per-row "
                "explanation requires the fitted model, which is not persisted."
            ],
            "iterations": 4,
            "tool_call_count": 3,
            "tools_available": [
                "dataset_profile",
                "run_experiment",
                "search_knowledge",
                "explain_experiment",
            ],
        },
    },
    "insufficient_evidence": {
        "summary": "Nothing observed supports an answer",
        "description": (
            "An honest refusal, not a failure. The question may simply have "
            "no answer in this project's documentation or history."
        ),
        "value": {
            "question": "What were last quarter's marketing results?",
            "status": "insufficient_evidence",
            "final_answer": (
                "Nothing in the indexed documentation or experiment history "
                "covers this question."
            ),
            "is_answer": False,
            "citation_ids": [],
            "rejected_citations": [],
            "experiment_ids": [],
            "warnings": [],
            "iterations": 2,
            "tool_call_count": 1,
            "tools_available": ["search_knowledge", "explain_experiment"],
        },
    },
    "grounding_failed": {
        "summary": "The answer cited something that was never retrieved",
        "description": (
            "The text is returned so a person can see what happened; it is "
            "not an answer. The fabricated identifier is reported, never "
            "repaired."
        ),
        "value": {
            "question": "How does this project prevent leakage?",
            "status": "grounding_failed",
            "final_answer": "Leakage is prevented by design [docs:internal-notes].",
            "is_answer": False,
            "citation_ids": [],
            "rejected_citations": ["docs:internal-notes"],
            "allowed_citations": ["docs:ml-readme#leakage-prevention"],
            "experiment_ids": [],
            "warnings": [
                "The answer cited 'docs:internal-notes', which was not in the "
                "retrieved evidence."
            ],
            "iterations": 2,
            "tool_call_count": 1,
            "tools_available": ["search_knowledge", "explain_experiment"],
        },
    },
}

_ASK_ERRORS: dict[int | str, dict[str, object]] = {
    status.HTTP_200_OK: {
        "description": (
            "The run finished. Read `status`: `completed`, `partial`, "
            "`insufficient_evidence` and `grounding_failed` are all returned "
            "here, because each is a real outcome of a valid request."
        ),
        "content": {"application/json": {"examples": _ASK_EXAMPLES}},
    },
    status.HTTP_400_BAD_REQUEST: {
        "model": ErrorResponse,
        "description": "The agent is configured with limits it cannot run under.",
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": (
            "The request body does not match the schema — a blank or "
            "over-long question, a field the endpoint does not define, or a "
            "budget larger than the server allows."
        ),
    },
    status.HTTP_502_BAD_GATEWAY: {
        "model": ErrorResponse,
        "description": (
            "The planner's language-model provider failed — timeout, rate "
            "limit, outage — or produced something that was not a decision. "
            "No tool was run and nothing was executed."
        ),
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": (
            "The agent is not configured: no language-model credential. "
            "`POST /api/v1/search` continues to work without one."
        ),
    },
}


@router.post(
    "/agent/ask",
    response_model=AgentAskResponse,
    responses=_ASK_ERRORS,
    summary="Answer a question by orchestrating the system's own capabilities",
)
def agent_ask(agent: AgentServiceDep, request: AgentAskRequest) -> AgentAskResponse:
    """Let the system decide which of its capabilities a question needs.

    Where `POST /api/v1/ask` answers from retrieved documents in a single
    step, this endpoint may profile a dataset, run a cross-validated
    experiment, explain the winning model and search the project's history —
    choosing the sequence itself, then writing an answer from what those steps
    actually returned.

    **The agent can only execute explicitly registered tools.** **The agent
    never executes arbitrary Python, shell commands, HTTP requests, or
    filesystem operations.** There is no generic "execute" tool, no code
    evaluation, no subprocess and no filesystem handle anywhere in the agent;
    a planner's response is parsed as one of two declared decisions or
    rejected; a tool name that is not registered cannot be run by any path;
    and arguments are validated against a declared schema before a tool sees
    them. `tools_available` on every response says exactly what could be
    chosen from.

    Read `status` before using `final_answer`:

    * `completed` — supported by the observations, every citation real. The
      only status a caller may present to a user as an answer.
    * `partial` — real work was done and is reported, but something is
      missing: a tool was unavailable, or a budget ran out. The gap is stated
      in `warnings` rather than filled in.
    * `insufficient_evidence` — nothing observed supports an answer.
    * `grounding_failed` — the answer cited a source that was never retrieved,
      or cited nothing while evidence existed. `rejected_citations` names what
      was invented; it is never repaired.

    All four are **HTTP 200**: the request was valid and the work was done.

    **Execution is bounded.** A run may make at most a configured number of
    tool calls and planning steps, and accumulate a limited amount of observed
    material; reaching any limit ends the run with a `partial` result naming
    it. A request may make those limits *smaller* — never larger.

    **No chain-of-thought is returned.** What comes back is which tool was
    chosen, the validated arguments, what the tool returned, and the answer.
    How the planner decided is not returned, stored or logged, and there is no
    field for it.

    Safety settings belong to the server: a request cannot supply a system
    prompt, a provider endpoint, a credential, a model, an estimator, a tool,
    a filesystem path, or a switch that turns off grounding or citation
    validation. Requires a language-model credential; without one this returns
    503 while the other endpoints continue to work.
    """
    result = agent.ask(request.question, budgets=request.budgets())
    return AgentAskResponse.model_validate(
        {**result.as_dict(), "tools_available": list(agent.tool_names())}
    )


@router.get(
    "/agent/status",
    response_model=AgentStatusResponse,
    summary="Report what the agent can currently do",
)
def agent_status(agent: AgentServiceDep) -> AgentStatusResponse:
    """Describe whether the agent is available, its tools and its limits.

    Exists so a client can tell "the agent is not configured" from "the
    question had no answer" before asking, and so it need not hard-code limits
    the server already knows. Reports whether a credential is configured,
    never what it is, and names no filesystem location.
    """
    return AgentStatusResponse.model_validate(agent.describe())
