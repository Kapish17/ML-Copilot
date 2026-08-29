"""Translation of exceptions into the API error envelope.

Every failure leaves the API in the same shape::

    {"error": {"code": "...", "message": "...", "details": {...}}}

That holds for every layer: dataset validation, preprocessing, model selection,
explainability and experiment storage all answer in the same shape, even though
only the backend's own errors know anything about HTTP. Errors from ``ml`` are
translated by :mod:`app.core.ml_errors`.

Unexpected exceptions are logged server-side and answered with a generic
message, so stack traces and internal details never reach an API consumer.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.errors import MLCopilotError
from app.core.ml_errors import MLError, is_client_error, translate_ml_error
from app.schemas.errors import ErrorDetail, ErrorResponse

logger = logging.getLogger(__name__)

INTERNAL_ERROR_MESSAGE = "An unexpected error occurred while handling the request."

#: Stable codes for the HTTP statuses Starlette raises on its own.
_HTTP_STATUS_CODES = {
    status.HTTP_400_BAD_REQUEST: "bad_request",
    status.HTTP_404_NOT_FOUND: "not_found",
    status.HTTP_405_METHOD_NOT_ALLOWED: "method_not_allowed",
    status.HTTP_413_CONTENT_TOO_LARGE: "file_too_large",
    status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: "unsupported_file_type",
}


def build_error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    """Build a JSON response carrying the standard error envelope.

    Args:
        status_code: HTTP status to answer with.
        code: Stable, machine-readable error code.
        message: Explanation safe to show to an API consumer.
        details: Optional structured context.

    Returns:
        JSONResponse: The serialised error envelope.
    """
    payload = ErrorResponse(
        error=ErrorDetail(code=code, message=message, details=details or {})
    )
    return JSONResponse(status_code=status_code, content=jsonable_encoder(payload))


async def handle_application_error(_: Request, exc: MLCopilotError) -> JSONResponse:
    """Return the envelope for an expected, typed application error."""
    return build_error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
    )


async def handle_ml_error(request: Request, exc: MLError) -> JSONResponse:
    """Return the envelope for an error raised by the ML or experiment layer.

    The ML packages know nothing about HTTP, so the mapping to a status lives
    in :mod:`app.core.ml_errors`. A failure that maps to 5xx is logged with its
    real cause and answered with a generic message; a 4xx is the caller's to
    fix, so its message is passed through with any filesystem path removed.
    """
    code, status_code, message, details = translate_ml_error(exc)
    if not is_client_error(exc):
        logger.exception(
            "ML layer %s while processing %s %s",
            type(exc).__name__,
            request.method,
            request.url.path,
        )
    return build_error_response(
        status_code=status_code, code=code, message=message, details=details
    )


async def handle_validation_error(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return the envelope for a malformed request (missing field, bad form)."""
    return build_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        code="invalid_request",
        message="The request could not be processed. Check the submitted fields.",
        details={"errors": jsonable_encoder(exc.errors())},
    )


async def handle_http_exception(
    _: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Return the envelope for Starlette's own HTTP errors, e.g. 404 and 405."""
    return build_error_response(
        status_code=exc.status_code,
        code=_HTTP_STATUS_CODES.get(exc.status_code, "http_error"),
        message=str(exc.detail),
    )


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Log an unexpected failure and answer with a generic message."""
    logger.exception(
        "Unhandled %s while processing %s %s",
        type(exc).__name__,
        request.method,
        request.url.path,
    )
    return build_error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="internal_error",
        message=INTERNAL_ERROR_MESSAGE,
    )


def register_exception_handlers(application: FastAPI) -> None:
    """Attach every exception handler to the application."""
    application.add_exception_handler(MLCopilotError, handle_application_error)
    application.add_exception_handler(MLError, handle_ml_error)
    application.add_exception_handler(RequestValidationError, handle_validation_error)
    application.add_exception_handler(StarletteHTTPException, handle_http_exception)
    application.add_exception_handler(Exception, handle_unexpected_error)
