"""Schemas for the service-level (non-domain) endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ServiceInfo(BaseModel):
    """Basic identity information returned by the root endpoint."""

    name: str = Field(..., description="Human readable service name.")
    version: str = Field(..., description="Current API version.")
    environment: str = Field(..., description="Environment the service runs in.")
    docs_url: str = Field(..., description="Path to the interactive API docs.")


class HealthStatus(BaseModel):
    """Liveness information returned by the health endpoint."""

    status: str = Field(..., description="Health indicator, 'ok' when healthy.")
    version: str = Field(..., description="Current API version.")
    environment: str = Field(..., description="Environment the service runs in.")
