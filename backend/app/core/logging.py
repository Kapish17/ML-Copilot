"""Logging setup, and the request id that ties a run of log lines together.

Two things live here, and they exist for one reason: when something goes wrong
in this application, the interesting evidence is spread across five packages.
A single upload can produce an ingestion line from ``app``, a run line from
``ml``, a retrieval line from ``rag`` and a generation line from ``llm``. Read
in a log they interleave with every other request in flight and say nothing
about which of them belong together.

**A request id fixes that.** One value per request, generated at the edge,
carried in a :class:`contextvars.ContextVar`, attached to every record by a
filter, and returned to the caller in ``X-Request-ID``. A user who reports a
problem can quote that header, and it selects exactly their request's lines out
of the whole log.

**A formatter makes it visible.** A record can carry the id and still be
useless if the format string never prints it, so this module installs one that
does. It is deliberately conservative: one handler, never a second, and it
raises the level only on this project's own loggers — a third-party package
that logs enthusiastically at INFO is not this application's log.

Nothing here reads a credential, and no function in this module takes one.
"""

from __future__ import annotations

import logging
import secrets
import sys
from contextvars import ContextVar, Token

#: The header a request id travels in, both directions. The `X-` prefix is
#: conventional rather than standard, which is exactly why it is spelled the
#: same everywhere: the middleware, the CORS allowance and the documentation.
REQUEST_ID_HEADER = "X-Request-ID"

#: What a log line shows when there is no request in scope — application
#: startup, a background import, a test. Better than an empty column.
NO_REQUEST_ID = "-"

#: The loggers this application owns. Only these have their level set, so
#: configuring INFO here never turns on a chatty dependency.
PROJECT_LOGGERS = ("app", "ml", "rag", "llm", "agent")

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s [%(request_id)s] %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

#: Marks the handler this module installed, so a second call finds it instead
#: of adding another and doubling every line.
_HANDLER_MARKER = "_ml_copilot_handler"

_request_id: ContextVar[str] = ContextVar("request_id", default=NO_REQUEST_ID)


def new_request_id() -> str:
    """Return a fresh request id.

    Sixteen hex characters from :mod:`secrets`. Unpredictable, so it cannot be
    guessed or counted to infer traffic volume, and short enough to quote in a
    support message.
    """
    return secrets.token_hex(8)


def current_request_id() -> str:
    """Return the id of the request being handled, or ``"-"`` outside one."""
    return _request_id.get()


def bind_request_id(request_id: str) -> Token[str]:
    """Attach a request id to the current context.

    Args:
        request_id: The value to bind.

    Returns:
        A token to pass to :func:`reset_request_id` when the request ends.
    """
    return _request_id.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    """Detach the request id bound by :func:`bind_request_id`."""
    _request_id.reset(token)


class RequestIdFilter(logging.Filter):
    """Give every record a ``request_id`` attribute.

    A filter rather than a `LoggerAdapter` or an `extra=` at each call site,
    because records reaching this handler come from five packages and from
    libraries none of them control. Anything without an id gets ``"-"``, so the
    format string can print it unconditionally and no call site has to know
    this mechanism exists.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Populate ``record.request_id`` and keep the record."""
        if not hasattr(record, "request_id"):
            record.request_id = current_request_id()
        return True


def _existing_handler(root: logging.Logger) -> logging.Handler | None:
    """Return the handler this module installed, if it is already there."""
    for handler in root.handlers:
        if getattr(handler, _HANDLER_MARKER, False):
            return handler
    return None


def configure_logging(level: str = "INFO") -> None:
    """Install the application's log handler and set its loggers' level.

    Idempotent: calling it twice leaves one handler, so an application built
    per test does not multiply the output.

    The root logger's own level is left alone. A record from ``app`` at INFO
    reaches the root handler because the level test happens at the logger it
    was created on, not at its ancestors — which is what keeps a third-party
    package's INFO chatter out while letting this project's through.

    Args:
        level: A level name such as ``"INFO"`` or ``"WARNING"``. An
            unrecognised value falls back to INFO rather than raising, since
            failing to start over a typo in ``LOG_LEVEL`` helps nobody.
    """
    resolved = logging.getLevelNamesMapping().get(str(level).upper(), logging.INFO)

    root = logging.getLogger()
    handler = _existing_handler(root)
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        handler.addFilter(RequestIdFilter())
        setattr(handler, _HANDLER_MARKER, True)
        root.addHandler(handler)

    for name in PROJECT_LOGGERS:
        logging.getLogger(name).setLevel(resolved)


__all__ = [
    "NO_REQUEST_ID",
    "PROJECT_LOGGERS",
    "REQUEST_ID_HEADER",
    "RequestIdFilter",
    "bind_request_id",
    "configure_logging",
    "current_request_id",
    "new_request_id",
    "reset_request_id",
]
