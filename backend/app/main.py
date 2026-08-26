"""FastAPI application entrypoint for the ML Copilot backend.

This commit intentionally exposes only two service-level endpoints. Feature
routers are introduced in later commits.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.core.config import Settings, get_settings
from app.schemas.system import HealthStatus, ServiceInfo

DOCS_URL = "/docs"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    Args:
        settings: Optional settings override, mainly useful for tests.

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

    @application.get("/", response_model=ServiceInfo, tags=["system"])
    def read_root() -> ServiceInfo:
        """Return basic information about the running service."""
        return ServiceInfo(
            name=config.app_name,
            version=config.app_version,
            environment=config.app_env,
            docs_url=DOCS_URL,
        )

    @application.get("/health", response_model=HealthStatus, tags=["system"])
    def health_check() -> HealthStatus:
        """Report whether the service is up and able to serve requests."""
        return HealthStatus(
            status="ok",
            version=config.app_version,
            environment=config.app_env,
        )

    return application


app = create_app()
