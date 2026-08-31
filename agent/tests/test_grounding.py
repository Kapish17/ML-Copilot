"""Grounding the final answer in what was actually observed.

The rule is Commit 10's, applied to a wider set of evidence: a citation is
valid exactly when the run retrieved it. These tests cover the three outcomes
that matter — a valid citation, a fabricated one, and a missing one — plus the
agent-specific case Commit 10 never had, an invented experiment id.
"""

from __future__ import annotations

import pytest

from agent.grounding import (
    check_citations,
    coerce_length,
    evidence_from_observations,
    is_abstention,
    unsupported_experiment_ids,
)
from agent.observations import Observation, ObservationStatus
from agent.plans import PlanStep
from agent.results import AgentStatus

REAL = "docs:ml-readme#cross-validation"
SECOND = "docs:ml-readme#leakage"
FABRICATED = "docs:secret-internal"

FINAL = PlanStep(action="final")


def tool_step(name: str, **arguments: object) -> PlanStep:
    """Build a scripted tool call."""
    return PlanStep(action="tool", tool=name, arguments=dict(arguments))


def search_observation(*citations: str) -> Observation:
    """An observation carrying retrieved passages."""
    return Observation(
        call_id="call-01",
        tool_name="search_knowledge",
        status=ObservationStatus.OK,
        output={
            "status": "ok",
            "results": [
                {
                    "citation_id": citation,
                    "score": 0.8,
                    "content": "Retrieved text.",
                    "source_type": "project_documentation",
                    "source_title": "ML Copilot",
                    "source_reference": "ml/README.md",
                }
                for citation in citations
            ],
        },
        citations=citations,
    )


# ---------------------------------------------------------------------------
# The evidence set
# ---------------------------------------------------------------------------


def test_the_allowed_set_is_exactly_what_was_retrieved() -> None:
    """No more, and no less."""
    context = evidence_from_observations([search_observation(REAL, SECOND)])

    assert context.allowed_citations == (REAL, SECOND)


def test_a_failed_observation_contributes_no_evidence() -> None:
    """A tool that did not produce a result cannot back a claim."""
    failed = Observation(
        call_id="call-01",
        tool_name="search_knowledge",
        status=ObservationStatus.FAILED,
        output={"results": [{"citation_id": REAL}]},
    )

    assert evidence_from_observations([failed]).allowed_citations == ()


# ---------------------------------------------------------------------------
# Citation validation
# ---------------------------------------------------------------------------


def test_a_retrieved_citation_is_valid() -> None:
    """The case the specification spells out."""
    context = evidence_from_observations([search_observation(REAL)])

    report = check_citations(f"Selection is cross-validated [{REAL}].", context)

    assert report.is_grounded is True
    assert report.valid == (REAL,)
    assert report.fabricated == ()


def test_an_unretrieved_citation_is_a_grounding_failure() -> None:
    """The other case the specification spells out."""
    context = evidence_from_observations([search_observation(REAL)])

    report = check_citations(f"Trust me [{FABRICATED}].", context)

    assert report.is_grounded is False
    assert report.fabricated == (FABRICATED,)
    assert report.valid == ()


def test_a_fabricated_citation_is_never_repaired() -> None:
    """It is reported as invented, not swapped for the nearest real one."""
    context = evidence_from_observations([search_observation(REAL)])

    report = check_citations(
        f"Selection is cross-validated [{FABRICATED}].", context
    )

    assert FABRICATED in report.fabricated
    assert REAL not in report.valid


def test_citing_nothing_while_evidence_exists_is_a_failure() -> None:
    """An uncited claim about this project is not grounded in anything."""
    context = evidence_from_observations([search_observation(REAL)])

    report = check_citations("Cross-validation is used.", context)

    assert report.is_grounded is False
    assert report.valid == ()


def test_an_invented_experiment_id_is_detected() -> None:
    """A fabricated result looks like a record a person can go and read."""
    observations = [
        Observation(
            call_id="call-01",
            tool_name="run_experiment",
            status=ObservationStatus.OK,
            output={"experiment_id": "exp_real_001"},
        )
    ]

    invented = unsupported_experiment_ids(
        "Runs exp_real_001 and exp_invented_999 both scored well.", observations
    )

    assert invented == ("exp_invented_999",)


def test_an_observed_experiment_id_is_not_flagged() -> None:
    """The check must not fire on ids the run actually produced."""
    observations = [
        Observation(
            call_id="call-01",
            tool_name="run_experiment",
            status=ObservationStatus.OK,
            output={"experiment_id": "exp_real_001"},
        )
    ]

    assert unsupported_experiment_ids("Run exp_real_001 won.", observations) == ()


# ---------------------------------------------------------------------------
# Abstention and length
# ---------------------------------------------------------------------------


def test_the_abstention_marker_is_a_declared_protocol() -> None:
    """Read as a line, not guessed at from phrasing."""
    assert is_abstention("INSUFFICIENT_EVIDENCE") is True
    assert is_abstention("Some text\nINSUFFICIENT_EVIDENCE\nmore") is True
    assert is_abstention("There is insufficient evidence here.") is False


def test_an_over_long_answer_is_cut() -> None:
    """The limit bounds what a caller receives, whatever the model does."""
    text, was_cut = coerce_length("x" * 5_000, 100)

    assert was_cut is True
    assert len(text) <= 101


# ---------------------------------------------------------------------------
# End to end, through the orchestrator
# ---------------------------------------------------------------------------


def test_a_valid_citation_completes_the_run(build_agent) -> None:
    """And the citation object is built from the retrieved passage."""
    agent, _ = build_agent(
        [tool_step("search_knowledge", query="x"), FINAL],
        answer=f"Selection happens on the training rows only [{REAL}].",
    )

    result = agent.run("How does selection work?")

    assert result.status is AgentStatus.COMPLETED
    assert result.citation_ids == (REAL,)
    citation = result.citations[0]
    assert citation.source_reference == "ml/README.md"
    assert citation.source_title == "ML Copilot — ML Layer"


def test_a_fabricated_citation_fails_the_run(build_agent) -> None:
    """The text is returned so a person can see it; it is not an answer."""
    agent, _ = build_agent(
        [tool_step("search_knowledge", query="x"), FINAL],
        answer=f"Trust me, this is right [{FABRICATED}].",
    )

    result = agent.run("q")

    assert result.status is AgentStatus.GROUNDING_FAILED
    assert result.is_answer is False
    assert result.rejected_citations == (FABRICATED,)
    assert result.error_code == "grounding_failed"
    assert result.final_answer  # kept, so a human can see what happened


def test_a_missing_citation_fails_the_run(build_agent) -> None:
    """Evidence was retrieved and the answer used none of it."""
    agent, _ = build_agent(
        [tool_step("search_knowledge", query="x"), FINAL],
        answer="Cross-validation is used throughout.",
    )

    result = agent.run("q")

    assert result.status is AgentStatus.GROUNDING_FAILED
    assert result.citation_ids == ()
    assert result.allowed_citations == (REAL, SECOND)


def test_an_invented_experiment_id_fails_the_run(build_agent) -> None:
    """A fabricated result is treated as seriously as a fabricated source."""
    agent, _ = build_agent(
        [tool_step("run_experiment", dataset="sales"), FINAL],
        answer="Runs exp_20260101T000000Z_abc123 and exp_made_up_42 both won.",
    )

    result = agent.run("q")

    assert result.status is AgentStatus.GROUNDING_FAILED
    assert "exp_made_up_42" in result.rejected_citations


def test_an_abstaining_answer_is_reported_as_insufficient(build_agent) -> None:
    """An honest refusal, not a failure."""
    agent, _ = build_agent(
        [tool_step("search_knowledge", query="x"), FINAL],
        answer="INSUFFICIENT_EVIDENCE",
    )

    result = agent.run("What is the airspeed velocity of a swallow?")

    assert result.status is AgentStatus.INSUFFICIENT_EVIDENCE
    assert result.is_answer is False
    assert "INSUFFICIENT_EVIDENCE" not in result.final_answer


def test_a_run_with_no_results_at_all_is_insufficient(build_agent) -> None:
    """Nothing observed means nothing to ground an answer in."""
    agent, _ = build_agent(
        [tool_step("not_a_tool"), FINAL], answer="Here is a confident answer."
    )

    result = agent.run("q")

    assert result.status is AgentStatus.INSUFFICIENT_EVIDENCE


def test_an_answer_with_no_evidence_but_real_results_is_allowed(
    build_agent,
) -> None:
    """An experiment result is a fact from a tool, not a citable passage."""
    agent, _ = build_agent(
        [tool_step("run_experiment", dataset="sales"), FINAL],
        answer=(
            "Experiment exp_20260101T000000Z_abc123 selected "
            "random_forest_classifier with an F1 of 0.86."
        ),
    )

    result = agent.run("Which model won?")

    assert result.status is AgentStatus.COMPLETED
    assert result.allowed_citations == ()


def test_the_answer_step_is_shown_only_the_allowed_citations(
    build_agent,
) -> None:
    """It is told exactly what it may cite, and nothing else."""
    agent, planner = build_agent(
        [tool_step("search_knowledge", query="x"), FINAL],
        answer=f"Yes [{REAL}].",
    )

    agent.run("q")

    assert planner.answer_calls[0]["allowed_citations"] == [REAL, SECOND]


@pytest.mark.parametrize("marker", ["api_key", "sk-", "system prompt"])
def test_the_answer_step_receives_no_secret(build_agent, marker: str) -> None:
    """What it is given is the question, the observations and the citations."""
    agent, planner = build_agent(
        [tool_step("search_knowledge", query="x"), FINAL], answer=f"Yes [{REAL}]."
    )

    agent.run("q")

    payload = str(planner.answer_calls[0])
    assert marker not in payload
