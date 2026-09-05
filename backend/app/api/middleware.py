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

import json
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


class RequestBodyLimitMiddleware:
    """Refuse a JSON body larger than the configured ceiling.

    **Why this cannot be a dependency.** FastAPI reads and parses the body
    before it solves a route's dependencies, so by the time any code the route
    declares could object, the megabytes are already dictionaries in memory.
    The only place to say no before that happens is here, above the
    application.

    **Why it exists at all.** Every expensive path in this project is bounded,
    and `MAX_PREDICTION_RECORDS` bounds a prediction request's *rows*. It says
    nothing about their *size*: five hundred records each carrying a very long
    string is a legal request under the row limit and an unbounded one under
    any other measure. This is the same bound in the other dimension, and it is
    ``MAX_UPLOAD_MB`` applied to the body shape that does not go through the
    upload reader.

    **What it does not touch.** Multipart requests — the dataset uploads — are
    passed straight through. They have their own, larger limit and their own
    streaming reader in ``app.services.datasets.validation``, and a second
    ceiling here would only be a confusing way to change that limit.

    Two checks, because one is not enough:

    1. ``Content-Length``, when the client sends one. Refused immediately,
       before a single body byte is read.
    2. A running total while the body streams, for a chunked request that
       declares no length. The count stops at the ceiling, so a client that
       lies about its length does not get further than one that is honest.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        """Wrap the application, refusing bodies above ``max_bytes``."""
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one ASGI event stream."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        content_type = headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            await self.app(scope, receive, send)
            return

        declared = _declared_length(headers)
        if declared is not None and declared > self.max_bytes:
            await self._refuse(scope, send)
            return

        received = 0
        exceeded = False

        async def counted_receive() -> Message:
            """Pass the body through, counting it as it goes."""
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    # An empty final chunk, so the application below sees a
                    # short body and fails its own validation rather than
                    # hanging waiting for the rest. The response it produces is
                    # discarded: `guarded_send` answers instead.
                    return {"type": "http.request", "body": b"", "more_body": False}
            return message

        async def guarded_send(message: Message) -> None:
            """Send the application's response, unless the body was too large."""
            if exceeded:
                return
            await send(message)

        await self.app(scope, counted_receive, guarded_send)
        if exceeded:
            await self._refuse(scope, send)

    async def _refuse(self, scope: Scope, send: Send) -> None:
        """Answer with the project's one error envelope, and nothing else.

        Written here rather than raised, because an exception thrown from
        middleware never reaches the application's handlers — it is above them.
        The shape is the documented envelope so a client parses this failure
        exactly like every other one.
        """
        body = json.dumps(
            {
                "error": {
                    "code": "request_body_too_large",
                    "message": (
                        "The request body is larger than this service accepts. "
                        f"The limit is {self.max_bytes // (1024 * 1024)} MB."
                    ),
                    "details": {"max_bytes": self.max_bytes},
                }
            }
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def _declared_length(headers: Headers) -> int | None:
    """Return a usable ``Content-Length``, or ``None``.

    A header that is absent, blank or not a number is treated as absent: it is
    the streaming counter's job to bound those, and refusing the request for a
    malformed header would answer a different question than the one asked.
    """
    raw = headers.get("content-length")
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


__all__ = [
    "VALID_REQUEST_ID",
    "RequestBodyLimitMiddleware",
    "RequestContextMiddleware",
    "resolve_request_id",
]
