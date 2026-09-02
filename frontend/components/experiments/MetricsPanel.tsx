import { Badge } from "@/components/common/Badge";
import { DataTable, Td, Th } from "@/components/common/DataTable";
import { ConfusionMatrix } from "./ConfusionMatrix";
import type { ExperimentEvaluation } from "@/lib/api/types";
import {
  directionLabel,
  formatCount,
  formatMetric,
  formatSigned,
  metricDirection,
  metricLabel,
} from "@/lib/format";

/**
 * Everything measured on the held-out test set.
 *
 * Whatever the backend measured is rendered — accuracy, precision, recall,
 * F1 and ROC-AUC for a classifier; MAE, MSE, RMSE and R² for a regressor —
 * because the metric set follows the task and the frontend does not need to
 * know which task it is looking at to show it. Each row states its own
 * direction, so a table mixing R² with RMSE stays readable.
 *
 * Metrics the backend could not compute are listed separately with its
 * reason, rather than shown as a blank that reads like a zero.
 */
export function MetricsPanel({
  evaluation,
}: {
  evaluation: ExperimentEvaluation;
}) {
  const entries = Object.entries(evaluation.metrics);
  const baselineMetrics = evaluation.baseline_metrics ?? {};
  const unavailable = Object.entries(evaluation.unavailable_metrics ?? {});

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <Badge tone="accent" glyph="★">
          primary: {metricLabel(evaluation.primary_metric)}
        </Badge>
        <Badge tone={evaluation.is_unbiased ? "good" : "warn"} glyph={evaluation.is_unbiased ? "✓" : "▲"}>
          {evaluation.is_unbiased
            ? "unbiased — measured once on unseen rows"
            : "optimistic — the test set informed selection"}
        </Badge>
        <span className="text-ink-500">
          {formatCount(evaluation.test_row_count)} test rows
        </span>
      </div>

      <DataTable
        caption="Metrics on the held-out test set, with the baseline for scale"
        head={
          <tr>
            <Th>Metric</Th>
            <Th>Direction</Th>
            <Th numeric>Model</Th>
            <Th numeric>Baseline</Th>
            <Th numeric>Difference</Th>
          </tr>
        }
      >
        {entries.map(([name, value]) => {
          const direction = metricDirection(
            name,
            name === evaluation.primary_metric
              ? evaluation.baseline_comparison?.direction
              : undefined,
          );
          const baselineValue = baselineMetrics[name] ?? null;
          const difference =
            value !== null && baselineValue !== null ? value - baselineValue : null;
          const better =
            difference === null
              ? null
              : direction === "higher_is_better"
                ? difference > 0
                : difference < 0;
          const isPrimary = name === evaluation.primary_metric;

          return (
            <tr key={name} className={isPrimary ? "bg-accent-50/60" : undefined}>
              <Th
                scope="row"
                className="normal-case tracking-normal text-ink-900"
              >
                {metricLabel(name)}
                {isPrimary && (
                  <span className="ml-2 font-sans">
                    <Badge tone="accent" glyph="★">
                      primary
                    </Badge>
                  </span>
                )}
              </Th>
              <Td className="text-xs text-ink-600">
                <span aria-hidden="true">
                  {direction === "higher_is_better" ? "↑ " : "↓ "}
                </span>
                {directionLabel(direction)}
              </Td>
              <Td numeric className="font-semibold">
                {formatMetric(value)}
              </Td>
              <Td numeric className="text-ink-600">
                {formatMetric(baselineValue)}
              </Td>
              <Td numeric>
                {difference === null ? (
                  "—"
                ) : (
                  <span
                    className={better ? "text-emerald-800" : "text-rose-800"}
                  >
                    {formatSigned(difference)}{" "}
                    <span className="text-xs">{better ? "better" : "worse"}</span>
                  </span>
                )}
              </Td>
            </tr>
          );
        })}
      </DataTable>

      {unavailable.length > 0 && (
        <div className="rounded-md border border-ink-200 bg-ink-50 px-3 py-2">
          <p className="text-xs font-medium text-ink-700">Not computed</p>
          <ul className="mt-1 space-y-0.5 text-xs text-ink-600">
            {unavailable.map(([name, reason]) => (
              <li key={name}>
                <span className="font-medium">{metricLabel(name)}</span>: {reason}
              </li>
            ))}
          </ul>
        </div>
      )}

      {evaluation.classification_details && (
        <div>
          <h4 className="mb-2 text-sm font-semibold text-ink-900">
            Confusion matrix
          </h4>
          <ConfusionMatrix details={evaluation.classification_details} />
        </div>
      )}
    </div>
  );
}
