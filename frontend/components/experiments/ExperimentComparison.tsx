import Link from "next/link";
import { Badge } from "@/components/common/Badge";
import { DataTable, Td, Th } from "@/components/common/DataTable";
import { experimentHref } from "@/lib/citations";
import type { ExperimentComparison as Comparison } from "@/lib/api/types";
import {
  directionLabel,
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
            <Th numeric>CV mean</Th>
            <Th numeric>Test score</Th>
            <Th numeric>Baseline</Th>
            <Th numeric>Improvement</Th>
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
            </tr>
          );
        })}
      </DataTable>
    </div>
  );
}
