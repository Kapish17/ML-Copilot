"""FastAPI application entrypoint for the ML Copilot backend.

The module exposes a ``create_app`` factory plus a module-level ``app`` that
``uvicorn app.main:app`` loads. Feature routes live under ``app/api``; this
module only assembles the application.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.dependencies import (
    SettingsDep,
    get_llm_config,
    get_llm_provider,
    get_rag_config,
)
from app.api.error_handlers import register_exception_handlers
from app.api.v1 import api_router
from app.core.config import Settings, get_settings
from app.schemas.system import HealthStatus, ServiceInfo
from llm.config import LLMConfig
from llm.providers import LLMProvider
from rag.config import RagConfig

DOCS_URL = "/docs"


def create_app(
    settings: Settings | None = None,
    *,
    rag_config: RagConfig | None = None,
    llm_config: LLMConfig | None = None,
    llm_provider: LLMProvider | None = None,
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
            "**CSV is currently supported; the ML pipeline is intentionally "
            "format-agnostic** — everything downstream of ingestion works on "
            "a standardised DataFrame.\n\n"
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
            "Runs are synchronous. Records and the retrieval index are local "
            "files: MLflow, any database, Qdrant, model serving, background "
            "workers, authentication, streaming, conversation memory and "
            "agent features are not implemented.\n\n"
            "Every failure returns the same envelope: "
            "`{\"error\": {\"code\", \"message\", \"details\"}}`. An answer "
            "that could not be grounded is **not** a failure — it returns 200 "
            "with a status saying so."
        ),
        docs_url=DOCS_URL,
    )

    if settings is not None:
        application.dependency_overrides[get_settings] = lambda: settings
    if rag_config is not None:
        application.dependency_overrides[get_rag_config] = lambda: rag_config
    if llm_config is not None:
        application.dependency_overrides[get_llm_config] = lambda: llm_config
    if llm_provider is not None:
        application.dependency_overrides[get_llm_provider] = lambda: llm_provider

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
