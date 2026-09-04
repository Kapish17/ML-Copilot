"""Schemas for the service-level (non-domain) endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ServiceInfo(BaseModel):
    """Basic identity information returned by the root endpoint."""

    name: str = Field(..., description="Human readable service name.")
    version: str = Field(..., description="Current API version.")
    environment: str = Field(..., description="Environment the service runs in.")
    docs_url: str = Field(..., description="Path to the interactive API docs.")
    authentication_required: bool = Field(
        ...,
        description=(
            "Whether the protected endpoints need an `Authorization: Bearer "
            "<key>` header on this deployment. A boolean about the "
            "*configuration*, never about the key: it says an API key is "
            "required, not what it is, how long it is, or whether one was "
            "supplied. It is also not a disclosure — anyone can learn the same "
            "fact by making one request and reading the 401 — and publishing "
            "it here lets a client say so up front instead of failing on the "
            "user's first action."
        ),
    )


class HealthStatus(BaseModel):
    """Liveness information returned by the health endpoint."""

    status: str = Field(..., description="Health indicator, 'ok' when healthy.")
    version: str = Field(..., description="Current API version.")
    environment: str = Field(..., description="Environment the service runs in.")
