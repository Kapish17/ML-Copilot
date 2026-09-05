import Link from "next/link";
import { Badge } from "@/components/common/Badge";
import { DataTable, Td, Th } from "@/components/common/DataTable";
import { experimentHref } from "@/lib/citations";
import type { ExperimentComparison as Comparison } from "@/lib/api/types";
import {
  directionLabel,
  formatCount,
  formatMetric,
  formatSigned,
  formatTimestamp,
  metricDirection,
  metricLabel,
} from "@/lib/format";

/**
 * Two or more runs ranked against each other.
 *
 * The backend decides the ranking and reports which direction the metric
 * runs in, so this component never sorts or judges — it renders the order it
 * was given and marks the run the backend called best. Re-deriving "best"
 * here would be a second, quietly divergent implementation of a decision the
 * backend already made.
 */
export function ExperimentComparisonView({
  comparison,
}: {
  comparison: Comparison;
}) {
  const direction = metricDirection(
    comparison.primary_metric,
    comparison.direction,
  );
  const metric = metricLabel(comparison.primary_metric);
  // "CV" only when it is true of every row. A history can hold runs chosen by
  // cross-validation and runs chosen on the held-out rows, and one heading has
  // to be true of everything under it.
  const allCrossValidated = comparison.runs.every(
    (run) => run.strategy === "cross_validation",
  );
  const selectionHeading = allCrossValidated
    ? `CV ${metric}`
    : `Selection ${metric}`;

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
        <Badge tone="accent">{metricLabel(comparison.primary_metric)}</Badge>
        <Badge tone="neutral" glyph={comparison.higher_is_better ? "↑" : "↓"}>
          {directionLabel(direction)}
        </Badge>
        <Badge tone="neutral">{comparison.task_type}</Badge>
        <span className="text-ink-500">{comparison.run_count} runs</span>
      </div>

      <DataTable
        caption={`Runs compared on ${metricLabel(comparison.primary_metric)}, best first`}
        head={
          <tr>
            <Th>Run</Th>
            <Th>Model</Th>
            <Th>Created</Th>
            <Th numeric>{selectionHeading}</Th>
            <Th numeric>Held-out {metric}</Th>
            <Th numeric>Baseline</Th>
            <Th numeric>Improvement</Th>
            <Th numeric>Train rows</Th>
            <Th numeric>Features</Th>
          </tr>
        }
      >
        {comparison.runs.map((run) => {
          const best = run.experiment_id === comparison.best_experiment_id;
          return (
            <tr key={run.experiment_id} className={best ? "bg-accent-50/60" : undefined}>
              <Th scope="row" className="normal-case tracking-normal">
                <Link
                  href={experimentHref(run.experiment_id)}
                  className="font-medium text-accent-700 underline decoration-accent-300 underline-offset-2 hover:text-accent-900"
                >
                  {run.name}
                </Link>
                {best && (
                  <span className="ml-2 font-sans">
                    <Badge tone="accent" glyph="★">
                      best
                    </Badge>
                  </span>
                )}
                {/*
                  Beside the run's name rather than in a column of its own: a
                  marker at the far right of nine columns is a marker nobody
                  reads, and the whole point of it is to be seen before the
                  score is.
                */}
                {run.warning_count ? (
                  <span className="ml-2 font-sans">
                    <Badge tone="warn" glyph="!">
                      {run.warning_count} to review
                    </Badge>
                  </span>
                ) : null}
                <span className="mt-0.5 block break-id font-mono text-xs font-normal text-ink-500">
                  {run.experiment_id}
                </span>
              </Th>
              <Td className="font-mono text-xs">{run.model_name}</Td>
              <Td className="whitespace-nowrap text-xs text-ink-600">
                {formatTimestamp(run.created_at)}
              </Td>
              <Td numeric className="text-ink-600">
                {formatMetric(run.selection_score)}
                {run.selection_score_std !== null && (
                  <span className="block text-xs">
                    ± {formatMetric(run.selection_score_std)}
                  </span>
                )}
              </Td>
              <Td numeric className="font-semibold">
                {formatMetric(run.test_score)}
              </Td>
              <Td numeric className="text-ink-600">
                {formatMetric(run.baseline_score)}
              </Td>
              <Td numeric>{formatSigned(run.improvement)}</Td>
              <Td numeric className="text-ink-600">
                {formatCount(run.train_row_count)}
              </Td>
              <Td numeric className="text-ink-600">
                {formatCount(run.feature_count)}
              </Td>
            </tr>
          );
        })}
      </DataTable>

      {/*
        The two score columns are easy to conflate, and conflating them is the
        single most misleading thing this table could do. Said once, plainly,
        under the numbers they describe.
      */}
      <p className="mt-2 text-xs leading-relaxed text-ink-500">
        {selectionHeading} is the score that chose each model: for a
        cross-validated run, the mean across the folds of the training rows,
        with ± the spread between those folds — how much they disagreed, not a
        confidence interval. The held-out column is a separate measurement,
        taken once after the model was chosen.
        {!allCrossValidated &&
          " Some of these runs chose their model on the held-out rows, so for those the two columns are one measurement used twice."}{" "}
        Runs marked <span className="font-medium">to review</span> raised
        diagnostics worth reading on the run&rsquo;s own page.
      </p>
    </div>
  );
}
