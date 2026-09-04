"""Translation of exceptions into the API error envelope.

Every failure leaves the API in the same shape::

    {"error": {"code": "...", "message": "...", "details": {...}}}

That holds for every layer: dataset validation, preprocessing, model selection,
explainability, experiment storage, retrieval and answer generation all answer
in the same shape, even though only the backend's own errors know anything
about HTTP. Errors from ``ml`` are translated by :mod:`app.core.ml_errors`;
errors from ``rag`` and ``llm`` by :mod:`app.core.knowledge_errors`; errors
from ``agent`` by :mod:`app.core.agent_errors`.

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

from app.core.agent_errors import (
    AgentError,
    is_client_error as is_client_agent_error,
    translate_agent_error,
)
from app.core.errors import MLCopilotError
from app.core.knowledge_errors import (
    LLMError,
    RagError,
    is_client_error as is_client_knowledge_error,
    translate_knowledge_error,
)
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
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Build a JSON response carrying the standard error envelope.

    Args:
        status_code: HTTP status to answer with.
        code: Stable, machine-readable error code.
        message: Explanation safe to show to an API consumer.
        details: Optional structured context.
        headers: Optional response headers the status requires — in practice
            only ``WWW-Authenticate`` on a 401.

    Returns:
        JSONResponse: The serialised error envelope.
    """
    payload = ErrorResponse(
        error=ErrorDetail(code=code, message=message, details=details or {})
    )
    return JSONResponse(
        status_code=status_code, content=jsonable_encoder(payload), headers=headers
    )


async def handle_application_error(_: Request, exc: MLCopilotError) -> JSONResponse:
    """Return the envelope for an expected, typed application error.

    Authentication failures arrive here like any other typed error, which is
    the point: a 401 is the same envelope as a 413 or a 422, with a stable
    code, a safe message and no traceback. It is not a second error format and
    it is never a 500.
    """
    return build_error_response(
        status_code=exc.status_code,
        code=exc.code,
        message=exc.message,
        details=exc.details,
        headers=exc.headers,
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


async def handle_knowledge_error(
    request: Request, exc: RagError | LLMError
) -> JSONResponse:
    """Return the envelope for a retrieval or language-model failure.

    Neither package knows about HTTP, so the mapping lives in
    :mod:`app.core.knowledge_errors`. A provider failure is logged with its
    real cause and answered as a 502; a missing credential or an unbuilt index
    is a 503 whose message says what to set, because that is a problem only a
    human can fix and guessing wastes their time.

    Note what does *not* arrive here: ``insufficient_evidence`` and
    ``grounding_failed`` are results, not exceptions. They travel as 200 with
    a status field.
    """
    code, status_code, message, details = translate_knowledge_error(exc)
    if not is_client_knowledge_error(exc):
        logger.warning(
            "%s while processing %s %s: %s",
            type(exc).__name__,
            request.method,
            request.url.path,
            getattr(exc, "message", ""),
        )
    return build_error_response(
        status_code=status_code, code=code, message=message, details=details
    )


async def handle_agent_error(request: Request, exc: AgentError) -> JSONResponse:
    """Return the envelope for a failure of the agent layer.

    ``agent/`` knows nothing about HTTP, so the mapping lives in
    :mod:`app.core.agent_errors`. A planner whose provider failed, or which
    produced something that was not a decision, is a 502; a planner with no
    credential is a 503 whose message says what to set.

    Note what does *not* arrive here. ``partial``, ``insufficient_evidence``
    and ``grounding_failed`` are results, not exceptions — they travel as 200
    with a status field. And an unknown tool or an invalid argument set does
    not usually reach here either: inside a run those become *rejected
    observations*, so the planner can correct itself and the client can see
    that it tried.
    """
    code, status_code, message, details = translate_agent_error(exc)
    if not is_client_agent_error(exc):
        logger.warning(
            "%s while processing %s %s: %s",
            type(exc).__name__,
            request.method,
            request.url.path,
            getattr(exc, "message", ""),
        )
    return build_error_response(
        status_code=status_code, code=code, message=message, details=details
    )


def _sanitise_validation_errors(errors: list[Any]) -> list[Any]:
    """Strip the echoed value from "this field is not allowed" errors.

    Pydantic reports the offending value alongside the complaint, which helps
    a developer fix a wrong type. It does not help for ``extra_forbidden``:
    the field is refused outright, so there is nothing to correct, and the
    value is exactly the kind of thing someone smuggles in — a credential, an
    endpoint, a prompt. Naming the field is the whole of the useful message;
    quoting what was in it only puts it back on the wire and into the logs.
    """
    sanitised: list[Any] = []
    for error in errors:
        if isinstance(error, dict) and error.get("type") == "extra_forbidden":
            error = {key: value for key, value in error.items() if key != "input"}
        sanitised.append(error)
    return sanitised


async def handle_validation_error(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return the envelope for a malformed request (missing field, bad form)."""
    return build_error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        code="invalid_request",
        message="The request could not be processed. Check the submitted fields.",
        details={"errors": _sanitise_validation_errors(jsonable_encoder(exc.errors()))},
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
    application.add_exception_handler(RagError, handle_knowledge_error)
    application.add_exception_handler(LLMError, handle_knowledge_error)
    application.add_exception_handler(AgentError, handle_agent_error)
    application.add_exception_handler(RequestValidationError, handle_validation_error)
    application.add_exception_handler(StarletteHTTPException, handle_http_exception)
    application.add_exception_handler(Exception, handle_unexpected_error)
