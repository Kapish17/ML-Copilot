"""The in-process safeguard against too many concurrent agent runs.

Two layers, tested separately: :class:`AgentConcurrencyLimiter` itself (a
plain semaphore wrapper with no framework dependency), and
:class:`AgentService.ask` honouring one when it is given one. Neither test
touches HTTP or the real orchestrator — the limiter's job is to refuse
*before* any of that is built, so a unit test that never builds it is the
faithful one.
"""

from __future__ import annotations

import pytest

from agent.config import AgentConfig
from agent.results import AgentResult, AgentStatus
from app.services.agent.errors import AgentTooManyRequestsError
from app.services.agent.service import AgentService
from app.services.agent.throttle import (
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    ENVIRONMENT_VARIABLE,
    AgentConcurrencyLimiter,
    limiter_from_env,
)


# ---------------------------------------------------------------------------
# AgentConcurrencyLimiter
# ---------------------------------------------------------------------------


def test_a_limiter_allows_up_to_its_configured_number_at_once() -> None:
    """Exactly ``max_concurrent`` slots, no more."""
    limiter = AgentConcurrencyLimiter(2)

    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False, "a third slot should not be free"


def test_a_released_slot_becomes_available_again() -> None:
    """Releasing is what makes the guard temporary rather than permanent."""
    limiter = AgentConcurrencyLimiter(1)

    assert limiter.try_acquire() is True
    assert limiter.try_acquire() is False

    limiter.release()

    assert limiter.try_acquire() is True


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_a_limiter_refuses_a_non_positive_integer_size(bad: object) -> None:
    """A limiter of zero or fewer slots would refuse every request outright."""
    with pytest.raises(ValueError):
        AgentConcurrencyLimiter(bad)  # type: ignore[arg-type]


def test_the_default_size_is_read_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """No environment variable set means the small, documented default."""
    monkeypatch.delenv(ENVIRONMENT_VARIABLE, raising=False)

    limiter = limiter_from_env()

    assert limiter.max_concurrent == DEFAULT_MAX_CONCURRENT_REQUESTS


def test_the_size_is_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment may size the guard to its own quota."""
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, "5")

    assert limiter_from_env().max_concurrent == 5


def test_a_non_integer_environment_value_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A silently-ignored typo would leave the safeguard mis-sized."""
    monkeypatch.setenv(ENVIRONMENT_VARIABLE, "not-a-number")

    with pytest.raises(ValueError):
        limiter_from_env()


# ---------------------------------------------------------------------------
# AgentService.ask, with a throttle
# ---------------------------------------------------------------------------


class _ReadyPlanner:
    """The minimum a planner must offer for ``AgentService.can_run`` to hold."""

    is_ready = True


class _StubOrchestrator:
    """Stands in for :class:`~agent.orchestrator.AgentOrchestrator`.

    Records nothing interesting about the run itself — this suite is about
    whether the throttle was consulted at all, not about planning or tools,
    which are exercised end to end elsewhere.
    """

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def run(self, question: str) -> AgentResult:
        return AgentResult(question=question, status=AgentStatus.COMPLETED, final_answer="ok")


@pytest.fixture
def service(monkeypatch: pytest.MonkeyPatch) -> AgentService:
    """A service whose orchestrator never really runs, over a real throttle."""
    monkeypatch.setattr(
        "app.services.agent.service.AgentOrchestrator", _StubOrchestrator
    )
    return AgentService(
        planner=_ReadyPlanner(),
        registry_factory=lambda dataset: object(),
        config=AgentConfig(),
        throttle=AgentConcurrencyLimiter(1),
    )


def test_a_run_that_fits_the_throttle_proceeds(service: AgentService) -> None:
    """The ordinary case: nothing else in flight, so the run happens."""
    result = service.ask("How does this project select a model?")

    assert result.status is AgentStatus.COMPLETED


def test_a_second_concurrent_run_is_refused_without_running(
    service: AgentService,
) -> None:
    """A slot already held means the second caller never reaches the planner."""
    assert service._throttle is not None
    held = service._throttle.try_acquire()
    assert held, "the fixture's throttle should start with a free slot"

    try:
        with pytest.raises(AgentTooManyRequestsError) as caught:
            service.ask("A second, concurrent question.")
        assert caught.value.status_code == 429
        assert caught.value.code == "agent_too_many_requests"
        assert caught.value.details["max_concurrent_requests"] == 1
    finally:
        service._throttle.release()


def test_the_slot_is_released_after_a_successful_run(service: AgentService) -> None:
    """A run must not leak its slot, or the guard ratchets down to zero."""
    service.ask("First question.")
    service.ask("Second question, after the first released its slot.")


def test_the_slot_is_released_even_when_the_run_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing run still gives back what it borrowed."""

    class _ExplodingOrchestrator(_StubOrchestrator):
        def run(self, question: str) -> AgentResult:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "app.services.agent.service.AgentOrchestrator", _ExplodingOrchestrator
    )
    throttle = AgentConcurrencyLimiter(1)
    service = AgentService(
        planner=_ReadyPlanner(),
        registry_factory=lambda dataset: object(),
        config=AgentConfig(),
        throttle=throttle,
    )

    with pytest.raises(RuntimeError):
        service.ask("A question that will fail.")

    # The slot came back, so a following request is not wrongly throttled.
    assert throttle.try_acquire() is True


def test_no_throttle_means_no_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A service built with ``throttle=None`` behaves as it always did."""
    monkeypatch.setattr(
        "app.services.agent.service.AgentOrchestrator", _StubOrchestrator
    )
    service = AgentService(
        planner=_ReadyPlanner(),
        registry_factory=lambda dataset: object(),
        config=AgentConfig(),
        throttle=None,
    )

    for _ in range(5):
        assert service.ask("q").status is AgentStatus.COMPLETED
