"""Request-scoped context: one id per request, and one log line per request.

A plain ASGI middleware rather than Starlette's ``BaseHTTPMiddleware``. The
base class wraps every request in an extra task and rebuilds the request and
response objects to do it, and none of that is needed here: this middleware
reads one header, sets one context variable, adds one response header and logs
one line. Thirty lines of the raw protocol are cheaper and easier to reason
about than the machinery that would hide them.

What it produces, per request::

    2026-09-03T11:20:14 INFO  app.api.middleware [3f9a1c7e2b8d4a15]
        POST /api/v1/experiments/run -> 200 in 4820.1ms

and the same id in the response's ``X-Request-ID`` header, so the line above
can be found from the client side of a problem.

**What is never logged here.** No request body, no query string, no headers, no
uploaded filename. The path template is a route the application defines; a
query string and a filename are attacker-chosen text, and text a caller chooses
does not belong in a log line that an operator will read as though the server
wrote it.
"""

from __future__ import annotations

import logging
import re
import time

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import (
    REQUEST_ID_HEADER,
    bind_request_id,
    new_request_id,
    reset_request_id,
)

logger = logging.getLogger(__name__)

#: What an inbound request id may look like. Deliberately narrow: this value
#: is written into log lines, and a caller who could put a newline or an ANSI
#: escape in it could forge log entries or garble a terminal. Anything that
#: does not match is replaced with a generated id rather than rejected — the
#: request is not at fault, only the label it suggested.
VALID_REQUEST_ID = re.compile(r"\A[A-Za-z0-9_-]{1,64}\Z")


def resolve_request_id(headers: Headers) -> str:
    """Return the id to use for this request.

    An inbound ``X-Request-ID`` is honoured when it is well-formed, so a proxy
    or a client can correlate across a hop it made first. Otherwise one is
    generated. Either way the request has an id before anything else runs.

    Args:
        headers: The request headers.

    Returns:
        str: A safe request id.
    """
    candidate = headers.get(REQUEST_ID_HEADER, "")
    if VALID_REQUEST_ID.match(candidate):
        return candidate
    return new_request_id()


class RequestContextMiddleware:
    """Bind a request id, return it in a header, and log the outcome."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the application below this middleware."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI event stream."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = resolve_request_id(Headers(scope=scope))
        token = bind_request_id(request_id)
        started = time.perf_counter()
        status_code: int | None = None

        async def send_with_request_id(message: Message) -> None:
            """Stamp the response, and remember what status it carried."""
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                # `setdefault`, so a handler that set its own id keeps it.
                MutableHeaders(scope=message).setdefault(REQUEST_ID_HEADER, request_id)
            await send(message)

        method = scope.get("method", "?")
        path = scope.get("path", "?")

        try:
            await self.app(scope, receive, send_with_request_id)
        except Exception:
            # The traceback is logged by the application's own unhandled-error
            # handler, which runs *outside* this middleware and therefore after
            # the id is unbound. This line is what carries the id, so the
            # traceback can be tied to a request; logging the exception again
            # here would only duplicate it.
            logger.warning(
                "%s %s -> unhandled error after %.1fms",
                method,
                path,
                (time.perf_counter() - started) * 1000,
            )
            raise
        else:
            elapsed = (time.perf_counter() - started) * 1000
            # A failed response is the one worth noticing in a quiet log.
            level = logging.WARNING if (status_code or 500) >= 500 else logging.INFO
            logger.log(
                level, "%s %s -> %s in %.1fms", method, path, status_code, elapsed
            )
        finally:
            reset_request_id(token)


__all__ = ["RequestContextMiddleware", "VALID_REQUEST_ID", "resolve_request_id"]
