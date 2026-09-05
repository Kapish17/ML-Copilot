"""FastAPI application entrypoint for the ML Copilot backend.

The module exposes a ``create_app`` factory plus a module-level ``app`` that
``uvicorn app.main:app`` loads. Feature routes live under ``app/api``; this
module only assembles the application.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent.config import AgentConfig
from agent.tools.datasets import InMemoryDatasetSource
from app.api.dependencies import (
    SettingsDep,
    get_agent_config,
    get_dataset_source,
    get_llm_config,
    get_llm_provider,
    get_rag_config,
)
from app.api.error_handlers import register_exception_handlers
from app.api.middleware import RequestBodyLimitMiddleware, RequestContextMiddleware
from app.api.v1 import api_router
from app.core.config import Settings, get_settings
from app.core.logging import REQUEST_ID_HEADER, configure_logging
from app.schemas.system import HealthStatus, ServiceInfo
from llm.config import LLMConfig, config_from_env as llm_config_from_env
from llm.providers import LLMProvider
from rag.config import RagConfig, config_from_env as rag_config_from_env

logger = logging.getLogger(__name__)

DOCS_URL = "/docs"

#: Descriptions for the tag groups the generated documentation renders. Without
#: these, `/docs` shows five bare headings and a reader has to open an endpoint
#: to learn what a group is for.
OPENAPI_TAGS: list[dict[str, str]] = [
    {
        "name": "system",
        "description": (
            "Service identity and liveness. `/health` is what both container "
            "healthchecks call."
        ),
    },
    {
        "name": "datasets",
        "description": (
            "Upload a CSV, Excel (`.xlsx`) or JSON file and get its structure, "
            "per-column statistics, data-quality findings and an optional "
            "target analysis. Nothing is trained and nothing is stored: the "
            "file is parsed in memory for the request and released."
        ),
    },
    {
        "name": "experiments",
        "description": (
            "Run a complete experiment and read the history back. One run "
            "profiles the data, prepares it with a leakage-safe split, "
            "cross-validates every candidate on the training rows, retrains "
            "the winner and measures it **once** on the untouched test set, "
            "explains it with SHAP and stores the result under the dataset's "
            "content fingerprint. Cross-validated and held-out scores are "
            "always reported as separate fields."
        ),
    },
    {
        "name": "knowledge",
        "description": (
            "Search the project's own documentation and its experiment "
            "history, and answer questions from what was retrieved. Search "
            "needs no credential. An answer that cannot be grounded in "
            "retrieved evidence is returned as a status, not as an error."
        ),
    },
    {
        "name": "agent",
        "description": (
            "Let the system choose which of its own capabilities a question "
            "needs, then answer from what those steps returned. Bounded by "
            "construction: four registered tools, typed arguments validated "
            "before anything runs, and budgets a request may lower and never "
            "raise. No chain-of-thought is returned."
        ),
    },
]


def _allow_browser_origins(application: FastAPI, settings: Settings) -> None:
    """Let the dashboard, served from another origin, call this API.

    The frontend is a separate service on a separate port, so every request it
    makes is cross-origin and a browser will refuse it without this. It is the
    one change the frontend required in the backend, and it is deliberately
    narrow: an **explicit list of origins** from configuration, no wildcard, no
    credentials, and only the methods and headers this API actually uses.

    ``allow_credentials`` stays off because nothing here is authenticated —
    there is no cookie and no session to protect, and turning it on would
    forbid the wildcard-free list from ever being widened safely later. When
    the allowlist is empty the middleware is not installed at all, so a
    deployment that serves the frontend from the same origin gains no
    cross-origin surface it did not ask for.

    Args:
        application: The application being assembled.
        settings: Active settings, holding the configured origins.
    """
    origins = list(settings.cors_allow_origins)
    if not origins:
        return

    application.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        # `X-Request-ID` is accepted so a caller may supply its own, and
        # exposed so a browser can read the one the server chose. It is an
        # opaque per-request label — not a session, not a user identifier, and
        # not stored anywhere.
        #
        # **`Authorization` is deliberately absent**, so a browser on an
        # allowed origin cannot send the API key cross-origin at all. That is
        # not an oversight and it breaks nothing this project ships: the
        # dashboard holds no key, because a browser bundle is readable by every
        # visitor and a "secret" in one is not secret. Server-to-server callers
        # are unaffected — CORS is a browser rule and applies to none of them —
        # and the supported way to put a browser in front of a protected
        # deployment is a server-side proxy that adds the header, whose own
        # requests are not cross-origin browser requests either.
        #
        # So the only thing this refusal blocks is the anti-pattern, and it
        # blocks it loudly, in the console, at the first attempt.
        allow_headers=["Content-Type", REQUEST_ID_HEADER],
        # `allow_credentials` stays off: bearer authentication travels in a
        # header the caller sets explicitly, never in a cookie a browser
        # attaches on its own, so nothing here needs ambient credentials.
        expose_headers=[REQUEST_ID_HEADER],
        max_age=600,
    )


def _describe_startup(settings: Settings) -> dict[str, object]:
    """Return the facts worth logging once, at start-up.

    Booleans and counts. Whether a credential is configured is operationally
    important — it is the difference between "the agent is broken" and "the
    agent was never switched on" — and it is answerable without going anywhere
    near the value itself. Neither the key, nor the variable's contents, nor
    any absolute path appears here.

    Args:
        settings: The active settings.

    Returns:
        dict: Plain values, safe to write to a log.
    """
    facts: dict[str, object] = {
        "version": settings.app_version,
        "environment": settings.app_env,
        # Whether the API is protected is the first thing to want from a log
        # when a deployment behaves unexpectedly. The key itself is not here
        # and is not derivable from anything that is.
        "api_auth_enabled": settings.api_auth_enabled,
        "cors_origins": len(settings.cors_allow_origins),
        "formats": ",".join(settings.supported_dataset_extensions),
    }
    try:
        facts["llm_credential_configured"] = llm_config_from_env().has_api_key
    except Exception:  # pragma: no cover - configuration is best-effort here
        facts["llm_credential_configured"] = False
    try:
        facts["retrieval_index_present"] = rag_config_from_env().index_dir.is_dir()
    except Exception:  # pragma: no cover - configuration is best-effort here
        facts["retrieval_index_present"] = False
    return facts


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Configure logging, announce what is running, and say when it stops.

    The startup line is the one log entry that answers "what is this process,
    and what can it do?" without a request having to arrive first — which is
    exactly the question asked of a container that is up but behaving
    unexpectedly.
    """
    settings = get_settings()
    configure_logging(settings.log_level)

    facts = _describe_startup(settings)
    logger.info(
        "ML Copilot API started (%s)",
        " ".join(f"{key}={value}" for key, value in facts.items()),
    )
    try:
        yield
    finally:
        logger.info("ML Copilot API stopped")


def create_app(
    settings: Settings | None = None,
    *,
    rag_config: RagConfig | None = None,
    llm_config: LLMConfig | None = None,
    llm_provider: LLMProvider | None = None,
    agent_config: AgentConfig | None = None,
    dataset_source: InMemoryDatasetSource | None = None,
) -> FastAPI:
    """Build and configure the FastAPI application.

    Nothing expensive happens here. No index is read, no embedding model is
    loaded and no language-model client is built, so the application starts
    with no retrieval index present and no API key configured — the endpoints
    that need either say so when they are called.

    Args:
        settings: Optional settings override. When given, it replaces the
            ``get_settings`` dependency for this application instance, which
            lets tests run against custom limits without touching the
            environment.
        rag_config: Optional retrieval configuration override, for pointing
            the index at a temporary directory in a test.
        llm_config: Optional language-model configuration override.
        llm_provider: Optional provider override. A test passes the
            deterministic fake here so the whole HTTP → retrieval → model →
            grounding path runs offline.
        agent_config: Optional agent budget override, for running an endpoint
            against smaller limits than the environment configures.
        dataset_source: Optional datasets the agent may name. Empty by
            default — nothing populates it in the default wiring, so the two
            dataset-dependent tools are simply not registered. A test, and a
            later commit that gives the endpoint data to work on, supplies
            one here.

    Returns:
        FastAPI: A configured application instance.
    """
    config = settings or get_settings()
    application = FastAPI(
        title=config.app_name,
        version=config.app_version,
        summary="Backend service for ML Copilot, an AI data scientist assistant.",
        description=(
            "Profile a tabular dataset, run a complete machine-learning "
            "experiment on it, and read back the history of what has been "
            "run.\n\n"
            "**CSV, Excel (.xlsx) and JSON are supported; the ML pipeline is "
            "intentionally format-agnostic** — everything downstream of "
            "ingestion works on a standardised DataFrame.\n\n"
            "An experiment profiles the data, prepares it with a leakage-safe "
            "train/test split, cross-validates every candidate model on the "
            "training rows only, retrains the winner and measures it **once** "
            "on the untouched test set, explains it with SHAP, and stores the "
            "whole run as a record identified by the dataset's content "
            "fingerprint.\n\n"
            "The knowledge endpoints search that documentation and history, "
            "and answer questions from it. **POST /api/v1/ask returns "
            "evidence-grounded answers; the LLM is not the source of "
            "truth.** "
            "Every project-specific claim must come from a retrieved passage, "
            "every citation is checked against the passages actually "
            "supplied, and an answer citing a source that was not retrieved "
            "is rejected rather than quietly cleaned up.\n\n"
            "**POST /api/v1/agent/ask** goes one step further: it lets the "
            "system choose which of its own capabilities a question needs — "
            "profiling a dataset, running an experiment, explaining the "
            "winner, searching the history — and then answers from what those "
            "steps actually returned. **The agent can only execute explicitly "
            "registered tools.** **The agent never executes arbitrary Python, "
            "shell commands, HTTP requests, or filesystem operations.** "
            "Execution is bounded by hard limits a request may lower and "
            "never raise, and no chain-of-thought is returned.\n\n"
            "Runs are synchronous. Records and the retrieval index are local "
            "files: MLflow, any database, Qdrant, model serving, background "
            "workers, authentication, streaming, conversation memory and any "
            "agent framework (LangChain, LangGraph, AutoGen, CrewAI) are not "
            "implemented.\n\n"
            "**Authentication is optional and off by default**, which is what "
            "lets the local and demo stack run with no secret at all. A "
            "deployment that sets `API_AUTH_ENABLED=true` requires "
            "`Authorization: Bearer <key>` on every endpoint marked with a "
            "padlock; health, service info and the three capability endpoints "
            "stay open. This is a single shared key, not identity: there are "
            "no users, no roles and no expiry, and it must be carried over "
            "TLS in any remote deployment.\n\n"
            "Every failure returns the same envelope: "
            "`{\"error\": {\"code\", \"message\", \"details\"}}`. An answer "
            "that could not be grounded is **not** a failure — it returns 200 "
            "with a status saying so."
        ),
        docs_url=DOCS_URL,
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
    )

    if settings is not None:
        application.dependency_overrides[get_settings] = lambda: settings
    if rag_config is not None:
        application.dependency_overrides[get_rag_config] = lambda: rag_config
    if llm_config is not None:
        application.dependency_overrides[get_llm_config] = lambda: llm_config
    if llm_provider is not None:
        application.dependency_overrides[get_llm_provider] = lambda: llm_provider
    if agent_config is not None:
        application.dependency_overrides[get_agent_config] = lambda: agent_config
    if dataset_source is not None:
        application.dependency_overrides[get_dataset_source] = lambda: dataset_source

    _allow_browser_origins(application, config)
    # Below the request-id middleware, so an oversized body is still logged
    # with an id and still answers with one — a caller who hits the limit gets
    # a failure they can quote, like every other failure here.
    application.add_middleware(
        RequestBodyLimitMiddleware, max_bytes=config.max_request_body_bytes
    )
    # Added last, so it sits outside the CORS middleware and every response —
    # including a preflight that CORS answers on its own — carries a request id
    # and produces one log line.
    application.add_middleware(RequestContextMiddleware)
    register_exception_handlers(application)
    application.include_router(api_router)

    @application.get(
        "/",
        response_model=ServiceInfo,
        tags=["system"],
        summary="Describe the running service",
    )
    def read_root(active_settings: SettingsDep) -> ServiceInfo:
        """Return basic information about the running service."""
        return ServiceInfo(
            name=active_settings.app_name,
            version=active_settings.app_version,
            environment=active_settings.app_env,
            docs_url=DOCS_URL,
            authentication_required=active_settings.api_auth_enabled,
        )

    @application.get(
        "/health",
        response_model=HealthStatus,
        tags=["system"],
        summary="Report service liveness",
    )
    def health_check(active_settings: SettingsDep) -> HealthStatus:
        """Report whether the service is up and able to serve requests."""
        return HealthStatus(
            status="ok",
            version=active_settings.app_version,
            environment=active_settings.app_env,
        )

    return application


app = create_app()
