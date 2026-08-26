"""Schemas for the API error envelope.

All failures — domain errors, request-validation errors and unexpected
failures — are returned in the same shape so a frontend can handle them with
one code path.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Machine-readable description of a single failure."""

    code: str = Field(..., description="Stable error code, e.g. 'malformed_csv'.")
    message: str = Field(..., description="Human readable explanation.")
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured context such as limits or column names.",
    )


class ErrorResponse(BaseModel):
    """Envelope returned for every non-2xx response."""

    error: ErrorDetail
