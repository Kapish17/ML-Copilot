"""A lightweight, in-process guard against too many agent runs at once.

Not a distributed rate limiter, and deliberately not built as one. This is a
single small deployment, and the problem worth solving locally is: two clicks
of "Ask" before the first request's spinner appears, two browser tabs asking
at once, or a client retrying an already-in-flight request too eagerly. Each
of those turns one user action into several concurrent calls against a
free-tier language-model quota that is already measured in single-digit
requests per minute — see ``llm/providers/gemini_provider.py``.

A semaphore, sized small, acquired without blocking. A request that cannot
get a slot right away is refused immediately — before a registry is built,
before a planner is asked for anything, and before a single token is spent.
That is the cheapest possible refusal, and the opposite of what an aggressive
retry policy does: it stops one request from becoming several, rather than
turning one request into several by retrying it.

**What this is not.** It does not de-duplicate by question text, by dataset,
or by client identity — that would need a key, a TTL and a cache, which is
more machinery than a portfolio deployment's traffic justifies. It bounds
*how many* agent runs this one process will attempt at once, full stop. A
second, genuinely different question from a second, genuine user is throttled
exactly the same as an accidental duplicate would be; on a quota this small,
serialising the two is the more honest behaviour anyway — both would likely
fail from quota exhaustion if run concurrently, and one succeeding while the
other waits briefly is strictly better than both failing.

**Why in-process and not shared across instances.** Render's free tier runs
one instance; nothing here assumes otherwise, and nothing here would work
correctly if it were not true — nothing here is shared across processes.
Multiplying instances would need a shared store (Redis, a database row) to
mean the same thing, which is exactly the paid, external infrastructure the
optimisation for this deployment does not call for.
"""

from __future__ import annotations

import os
import threading

#: How many agent runs this process will attempt at once, when no
#: environment override is set. Small on purpose: it is not a capacity limit
#: for a well-provisioned server, it is a guard against one user's request
#: accidentally becoming several against a five-requests-per-minute quota.
DEFAULT_MAX_CONCURRENT_REQUESTS = 2

#: The environment variable this module reads.
ENVIRONMENT_VARIABLE = "AGENT_MAX_CONCURRENT_REQUESTS"


class AgentConcurrencyLimiter:
    """Bounds how many agent runs may be in flight across the whole process.

    Wraps a :class:`threading.Semaphore` rather than an ``asyncio`` primitive
    because the agent's own run is synchronous, blocking code executed either
    directly (``POST /agent/ask``) or inside a thread pool (``POST
    /agent/ask-with-dataset`` awaits dataset loading, then calls the same
    synchronous ``AgentService.ask``) — a thread-safe primitive is the one
    that is correct under both.
    """

    def __init__(self, max_concurrent: int) -> None:
        """Build a limiter that allows at most ``max_concurrent`` runs at once.

        Raises:
            ValueError: If ``max_concurrent`` is not a positive integer.
        """
        if not isinstance(max_concurrent, int) or isinstance(max_concurrent, bool) or max_concurrent < 1:
            raise ValueError("max_concurrent must be a positive integer.")
        self._max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)

    @property
    def max_concurrent(self) -> int:
        """The configured ceiling, for reporting in an error's details."""
        return self._max_concurrent

    def try_acquire(self) -> bool:
        """Reserve one slot if one is free right now, without waiting.

        Returns:
            bool: Whether a slot was reserved. A caller that gets ``False``
            must not call :meth:`release` — nothing was acquired.
        """
        return self._semaphore.acquire(blocking=False)

    def release(self) -> None:
        """Give back a slot reserved by a successful :meth:`try_acquire`."""
        self._semaphore.release()


def limiter_from_env() -> AgentConcurrencyLimiter:
    """Build the limiter from ``AGENT_MAX_CONCURRENT_REQUESTS``.

    Raises:
        ValueError: If the variable is set to something other than a positive
            integer.
    """
    raw = os.getenv(ENVIRONMENT_VARIABLE, "").strip()
    if not raw:
        return AgentConcurrencyLimiter(DEFAULT_MAX_CONCURRENT_REQUESTS)
    try:
        max_concurrent = int(raw)
    except ValueError as exc:
        raise ValueError(f"{ENVIRONMENT_VARIABLE} must be an integer.") from exc
    return AgentConcurrencyLimiter(max_concurrent)


__all__ = [
    "DEFAULT_MAX_CONCURRENT_REQUESTS",
    "ENVIRONMENT_VARIABLE",
    "AgentConcurrencyLimiter",
    "limiter_from_env",
]
