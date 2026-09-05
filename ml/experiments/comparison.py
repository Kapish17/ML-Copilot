"""Putting past runs side by side.

The value of a stored history is the question "is this better than what we did
last time?", and that question only means something between runs judged the
same way. Comparing an F1 of 0.82 against an RMSE of 5276 is not a close call
to be handled carefully — it is meaningless, so it is refused.

Where a set of runs *is* comparable, the ranking takes its direction from the
same metric definitions the rest of the project uses, so a smaller RMSE wins
and a larger F1 wins without this module deciding that for itself.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ml.errors import IncomparableExperimentsError
from ml.evaluation.metrics import (
    MetricDirection,
    format_metric_spread,
    format_metric_value,
    metric_label,
)
from ml.experiments.run import CROSS_VALIDATION_STRATEGY, ExperimentRun
from ml.experiments.store import shared_metric_direction


@dataclass(frozen=True)
class ComparisonRow:
    """One run's line in a comparison table.

    Carries the context a comparison is unreadable without: what chose the
    model, what measured it, and how much data was behind each number. Two F1
    scores mean different things when one came from 4,000 training rows and
    the other from 90, and a row that hides that invites the wrong conclusion.

    Every field is read from the stored record. Nothing here is recomputed or
    inferred, and a value the run did not store is ``None`` rather than a
    plausible substitute.
    """

    experiment_id: str
    created_at: str
    name: str
    model_name: str
    strategy: str
    selection_score: float | None
    selection_score_std: float | None
    test_score: float | None
    baseline_score: float | None
    improvement: float | None
    #: Added after the first comparison shipped; defaulted so a caller that
    #: builds a row positionally still works.
    task_type: str = ""
    primary_metric: str = ""
    train_row_count: int | None = None
    test_row_count: int | None = None
    feature_count: int | None = None
    #: How many diagnostics on this run are worth more than a glance, so a
    #: table can mark the runs to read carefully without carrying every
    #: sentence. See :mod:`ml.evaluation.diagnostics`.
    warning_count: int = 0
    #: True when the test score played no part in choosing the model.
    is_unbiased: bool = False
    #: One sentence on why this run's model won, composed from its own
    #: recorded numbers — never written by a language model.
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Render the row as plain, JSON-friendly values."""
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "name": self.name,
            "model_name": self.model_name,
            "strategy": self.strategy,
            "task_type": self.task_type,
            "primary_metric": self.primary_metric,
            "selection_score": self.selection_score,
            "selection_score_std": self.selection_score_std,
            "test_score": self.test_score,
            "baseline_score": self.baseline_score,
            "improvement": self.improvement,
            "train_row_count": self.train_row_count,
            "test_row_count": self.test_row_count,
            "feature_count": self.feature_count,
            "warning_count": self.warning_count,
            "is_unbiased": self.is_unbiased,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class ExperimentComparison:
    """Several runs of one kind, ranked by their final test score."""

    task_type: str
    primary_metric: str
    direction: MetricDirection
    rows: tuple[ComparisonRow, ...]

    @property
    def higher_is_better(self) -> bool:
        """True when a larger score is a better result."""
        return self.direction is MetricDirection.HIGHER_IS_BETTER

    @property
    def all_cross_validated(self) -> bool:
        """True when every run here chose its model without the test set."""
        return all(row.strategy == CROSS_VALIDATION_STRATEGY for row in self.rows)

    def best(self) -> ComparisonRow | None:
        """The best-scoring run, or ``None`` when none has a score."""
        scored = [row for row in self.rows if row.test_score is not None]
        return scored[0] if scored else None

    def as_table(self) -> list[dict[str, Any]]:
        """Render the comparison as a list of serialisable rows."""
        return [row.as_dict() for row in self.rows]

    def summary(self) -> dict[str, Any]:
        """Return a serialisable description of the comparison."""
        best = self.best()
        return {
            "task_type": self.task_type,
            "primary_metric": self.primary_metric,
            #: Sent so a reader labels its columns the same way this table
            #: does, without keeping its own copy of the metric names.
            "primary_metric_label": metric_label(self.primary_metric),
            "direction": self.direction.value,
            "run_count": len(self.rows),
            "best_experiment_id": best.experiment_id if best else None,
            "runs": self.as_table(),
        }

    def as_text(self) -> str:
        """Render the comparison as a readable text table.

        The score columns are labelled with the metric and with where each
        number came from, so a selection score is never read as a test score,
        and the cross-validated column carries its own spread — a mean across
        folds without it invites more confidence than the run earned.

        A run with diagnostics worth reading is marked, not scored down: the
        marker says "read this one's notes", never "this one is wrong".
        """
        label = metric_label(self.primary_metric)
        # "CV" only when it is true of every row. A comparison may mix
        # strategies, and a holdout run's selecting score is the held-out score
        # rather than a fold mean.
        selection_label = (
            f"CV {label}" if self.all_cross_validated else f"Selection {label}"
        )
        headers = (
            "Experiment",
            "Model",
            selection_label,
            f"Held-out {label}",
            "Baseline",
            "Improvement",
            "Train rows",
            "Features",
            "Notes",
        )

        def cell(value: float | None) -> str:
            """Format one number, or mark it absent."""
            return format_metric_value(value)

        def count(value: int | None) -> str:
            """Format a row or feature count, or mark it absent."""
            return f"{value:,}" if value is not None else "-"

        body = [
            (
                row.experiment_id,
                row.model_name,
                format_metric_spread(row.selection_score, row.selection_score_std),
                cell(row.test_score),
                cell(row.baseline_score),
                f"{row.improvement:+.4f}" if row.improvement is not None else "-",
                count(row.train_row_count),
                count(row.feature_count),
                f"{row.warning_count} to review" if row.warning_count else "-",
            )
            for row in self.rows
        ]

        widths = [
            max(len(headers[index]), *(len(line[index]) for line in body))
            if body
            else len(headers[index])
            for index in range(len(headers))
        ]
        header_line = "  ".join(
            text.ljust(widths[index]) if index < 2 else text.rjust(widths[index])
            for index, text in enumerate(headers)
        )
        lines = [header_line, "-" * len(header_line)]
        lines.extend(
            "  ".join(
                text.ljust(widths[index]) if index < 2 else text.rjust(widths[index])
                for index, text in enumerate(line)
            )
            for line in body
        )
        direction = "higher is better" if self.higher_is_better else "lower is better"
        lines.append("")
        lines.append(
            f"Ranked by held-out {label} ({direction}) on {self.task_type} runs."
        )
        lines.append(
            "The selection column is the score that chose each model: for a "
            "cross-validated run, the mean across folds ± the spread between "
            "them — how much the folds disagreed, not a confidence interval. "
            "The held-out column is a separate measurement, taken once after "
            "the choice."
        )
        if not self.all_cross_validated:
            lines.append(
                "Some of these runs chose their model on the held-out rows, so "
                "for those the two columns are one measurement used twice. "
                "Their notes say so."
            )
        return "\n".join(lines)


def compare_experiments(runs: Sequence[ExperimentRun]) -> ExperimentComparison:
    """Rank historical runs against each other.

    Args:
        runs: The runs to compare. They must share a task and a primary
            metric; anything else cannot be ranked and is refused.

    Returns:
        ExperimentComparison: The runs ordered best first.

    Raises:
        IncomparableExperimentsError: If the runs are empty, or judged by
            different metrics, or solve different tasks.
    """
    if not runs:
        raise IncomparableExperimentsError(
            "There are no runs to compare.", details={"run_count": 0}
        )

    direction = shared_metric_direction(runs)
    higher_is_better = direction is MetricDirection.HIGHER_IS_BETTER

    def rank(run: ExperimentRun) -> tuple[int, float]:
        """Order by final test score, keeping unscored runs at the end."""
        value = run.evaluation.primary_metric_value
        if value is None:
            return (1, 0.0)
        return (0, -value if higher_is_better else value)

    ordered = sorted(runs, key=rank)
    rows = tuple(
        ComparisonRow(
            experiment_id=run.experiment_id,
            created_at=run.created_at.isoformat(),
            name=run.name,
            model_name=run.selected_model,
            strategy=run.selection.strategy,
            selection_score=run.selection.selection_score,
            selection_score_std=run.selection.selection_score_std,
            test_score=run.evaluation.primary_metric_value,
            baseline_score=run.evaluation.baseline_metrics.get(run.primary_metric),
            improvement=run.evaluation.baseline_comparison.get(
                "absolute_improvement"
            ),
            task_type=run.task_type,
            primary_metric=run.primary_metric,
            train_row_count=run.preprocessing.train_row_count,
            test_row_count=run.evaluation.test_row_count,
            feature_count=run.feature_count,
            warning_count=run.evaluation.warning_count,
            is_unbiased=run.evaluation.is_unbiased,
            rationale=run.selection.selection_rationale,
        )
        for run in ordered
    )

    return ExperimentComparison(
        task_type=ordered[0].task_type,
        primary_metric=ordered[0].primary_metric,
        direction=direction,
        rows=rows,
    )
