import { Badge } from "@/components/common/Badge";
import { DataTable, Td, Th } from "@/components/common/DataTable";
import type {
  ExperimentEvaluation,
  ExperimentSelection,
} from "@/lib/api/types";
import {
  directionLabel,
  formatMetric,
  formatSigned,
  metricDirection,
  metricLabel,
} from "@/lib/format";

/**
 * The candidate models, and the one distinction this whole page turns on.
 *
 * **Cross-validated scores and the test score are not the same number, and
 * mixing them is the single easiest way to make a model look better than it
 * is.** Every candidate here was scored on folds of the *training* rows; the
 * winner was then retrained and measured **once** on rows no model had seen.
 * So the two live in visually separate column groups, each with its own
 * header saying what it is and whether the test set was touched, and only the
 * winner's row carries a test figure — because only the winner has one.
 *
 * The metric's direction is read from the backend, never assumed, and printed
 * in the header: an RMSE column that silently implied "higher is better"
 * would invert the reader's conclusion.
 */
export interface ModelComparisonTableProps {
  selection: ExperimentSelection;
  evaluation: ExperimentEvaluation;
}

export function ModelComparisonTable({
  selection,
  evaluation,
}: ModelComparisonTableProps) {
  const metric = selection.primary_metric;
  const direction = metricDirection(metric, selection.primary_metric_direction);
  const baseline = evaluation.baseline_comparison;
  const baselineValue =
    baseline?.baseline_value ??
    evaluation.baseline_metrics?.[metric] ??
    null;

  const strategyLabel =
    selection.strategy === "cross_validation"
      ? `${selection.folds ?? "k"}-fold cross-validation on training rows`
      : "holdout scoring";

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
        <Badge tone="accent">{metricLabel(metric)}</Badge>
        <Badge tone="neutral" glyph={direction === "higher_is_better" ? "↑" : "↓"}>
          {directionLabel(direction)}
        </Badge>
        <Badge tone={selection.uses_test_data ? "warn" : "good"} glyph={selection.uses_test_data ? "▲" : "✓"}>
          {selection.uses_test_data
            ? "selection used the test set"
            : "selection never saw the test set"}
        </Badge>
      </div>

      <DataTable
        caption={`Candidate models scored by ${strategyLabel}, and the winner's single measurement on the untouched test set`}
        head={
          <>
            <tr className="border-b border-ink-100">
              <Th scope="col" className="align-bottom">Model</Th>
              <Th scope="col" numeric className="bg-ink-50" colSpan={2}>
                Selection · {strategyLabel}
              </Th>
              <Th scope="col" numeric className="bg-accent-50" colSpan={3}>
                Final · untouched test set ({evaluation.test_row_count} rows)
              </Th>
              <Th scope="col" className="align-bottom">Status</Th>
            </tr>
            <tr>
              <Th />
              <Th scope="col" numeric className="bg-ink-50">
                CV mean
              </Th>
              <Th scope="col" numeric className="bg-ink-50">
                CV std
              </Th>
              <Th scope="col" numeric className="bg-accent-50">
                Test {metricLabel(metric)}
              </Th>
              <Th scope="col" numeric className="bg-accent-50">
                Baseline
              </Th>
              <Th scope="col" numeric className="bg-accent-50">
                Improvement
              </Th>
              <Th />
            </tr>
          </>
        }
      >
        {selection.candidates.map((candidate) => {
          const isWinner = candidate.model_name === selection.selected_model;
          return (
            <tr
              key={candidate.model_name}
              className={isWinner ? "bg-accent-50/60" : undefined}
            >
              <Th
                scope="row"
                className="normal-case tracking-normal text-ink-900"
              >
                <span className="font-medium">{candidate.display_name}</span>
                <span className="ml-2 font-mono text-xs font-normal text-ink-500">
                  {candidate.model_name}
                </span>
                {isWinner && (
                  <span className="ml-2 font-sans">
                    <Badge tone="accent" glyph="★">
                      selected
                    </Badge>
                  </span>
                )}
              </Th>
              <Td numeric className="bg-ink-50/60">
                {formatMetric(candidate.score)}
              </Td>
              <Td numeric className="bg-ink-50/60 text-xs text-ink-600">
                {candidate.score_std === null
                  ? "—"
                  : `± ${formatMetric(candidate.score_std)}`}
              </Td>
              <Td numeric className="bg-accent-50/40 font-semibold">
                {isWinner ? formatMetric(evaluation.primary_metric_value) : "—"}
              </Td>
              <Td numeric className="bg-accent-50/40 text-ink-600">
                {isWinner ? formatMetric(baselineValue) : "—"}
              </Td>
              <Td numeric className="bg-accent-50/40">
                {isWinner && baseline ? (
                  <span
                    className={
                      baseline.beats_baseline ? "text-emerald-800" : "text-rose-800"
                    }
                  >
                    {formatSigned(baseline.absolute_improvement)}
                  </span>
                ) : (
                  "—"
                )}
              </Td>
              <Td>
                {candidate.status === "succeeded" ? (
                  <Badge tone="good" glyph="✓">
                    succeeded
                  </Badge>
                ) : (
                  <Badge tone="bad" glyph="✕">
                    {candidate.status}
                  </Badge>
                )}
                {candidate.error && (
                  <span className="mt-1 block text-xs text-rose-700">
                    {candidate.error}
                  </span>
                )}
              </Td>
            </tr>
          );
        })}
      </DataTable>

      <p className="mt-3 text-xs text-ink-600">
        <span className="font-medium">Why two groups.</span> Every candidate was
        scored by {strategyLabel}. Only the winner was retrained and measured on
        the held-out rows, once — so the test column has one number in it by
        design, and the two groups are not comparable to each other.
        {baseline && (
          <>
            {" "}
            The baseline is{" "}
            <span className="font-mono">{evaluation.baseline_identifier}</span>,
            measured on the same held-out rows.
          </>
        )}
      </p>
    </div>
  );
}
