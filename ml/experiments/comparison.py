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
from ml.evaluation.metrics import MetricDirection
from ml.experiments.run import ExperimentRun
from ml.experiments.store import shared_metric_direction


@dataclass(frozen=True)
class ComparisonRow:
    """One run's line in a comparison table."""

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

    def as_dict(self) -> dict[str, Any]:
        """Render the row as plain, JSON-friendly values."""
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "name": self.name,
            "model_name": self.model_name,
            "strategy": self.strategy,
            "selection_score": self.selection_score,
            "selection_score_std": self.selection_score_std,
            "test_score": self.test_score,
            "baseline_score": self.baseline_score,
            "improvement": self.improvement,
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
            "direction": self.direction.value,
            "run_count": len(self.rows),
            "best_experiment_id": best.experiment_id if best else None,
            "runs": self.as_table(),
        }

    def as_text(self) -> str:
        """Render the comparison as a readable text table.

        The score columns are labelled with the metric and with where each
        number came from, so a selection score is never read as a test score.
        """
        label = self.primary_metric.upper()
        headers = (
            "Experiment",
            "Model",
            f"CV {label}",
            f"Test {label}",
            "Baseline",
            "Improvement",
        )

        def cell(value: float | None) -> str:
            """Format one number, or mark it absent."""
            return f"{value:.4f}" if value is not None else "-"

        body = [
            (
                row.experiment_id,
                row.model_name,
                cell(row.selection_score),
                cell(row.test_score),
                cell(row.baseline_score),
                f"{row.improvement:+.4f}" if row.improvement is not None else "-",
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
        lines.append(f"Ranked by test {label} ({direction}) on {self.task_type} runs.")
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
        )
        for run in ordered
    )

    return ExperimentComparison(
        task_type=ordered[0].task_type,
        primary_metric=ordered[0].primary_metric,
        direction=direction,
        rows=rows,
    )
