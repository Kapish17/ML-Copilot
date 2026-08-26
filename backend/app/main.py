"""FastAPI application entrypoint for the ML Copilot backend.

The module exposes a ``create_app`` factory plus a module-level ``app`` that
``uvicorn app.main:app`` loads. Feature routes live under ``app/api``; this
module only assembles the application.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.api.dependencies import SettingsDep
from app.api.error_handlers import register_exception_handlers
from app.api.v1 import api_router
from app.core.config import Settings, get_settings
from app.schemas.system import HealthStatus, ServiceInfo

DOCS_URL = "/docs"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    Args:
        settings: Optional settings override. When given, it replaces the
            ``get_settings`` dependency for this application instance, which
            lets tests run against custom limits without touching the
            environment.

    Returns:
        FastAPI: A configured application instance.
    """
    config = settings or get_settings()
    application = FastAPI(
        title=config.app_name,
        version=config.app_version,
        summary="Backend service for ML Copilot, an AI data scientist assistant.",
        docs_url=DOCS_URL,
    )

    if settings is not None:
        application.dependency_overrides[get_settings] = lambda: settings

    register_exception_handlers(application)
    application.include_router(api_router)

    @application.get("/", response_model=ServiceInfo, tags=["system"])
    def read_root(active_settings: SettingsDep) -> ServiceInfo:
        """Return basic information about the running service."""
        return ServiceInfo(
            name=active_settings.app_name,
            version=active_settings.app_version,
            environment=active_settings.app_env,
            docs_url=DOCS_URL,
        )

    @application.get("/health", response_model=HealthStatus, tags=["system"])
    def health_check(active_settings: SettingsDep) -> HealthStatus:
        """Report whether the service is up and able to serve requests."""
        return HealthStatus(
            status="ok",
            version=active_settings.app_version,
            environment=active_settings.app_env,
        )

    return application


app = create_app()
