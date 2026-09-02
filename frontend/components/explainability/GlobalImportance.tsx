import { Badge } from "@/components/common/Badge";
import { Bar } from "@/components/common/Bar";
import { EmptyState } from "@/components/common/EmptyState";
import { CausationNote } from "./CausationNote";
import type { ExperimentExplainability, FeatureImportance } from "@/lib/api/types";
import { formatMetric } from "@/lib/format";

/**
 * Which features the model leaned on, ranked.
 *
 * Bars are scaled against the largest importance rather than against 1, so
 * the shape of the ranking is visible even when every value is small. The
 * number is always printed beside the bar — the bar is for scanning, the
 * number is the value.
 *
 * When the backend says an explanation is unavailable, that is rendered as
 * the honest fact it is, with the backend's own reason. Nothing is invented
 * to fill the space.
 */
export interface GlobalImportanceProps {
  explainability: ExperimentExplainability | null;
  /** How many features to list before truncating. */
  limit?: number;
}

export function GlobalImportance({
  explainability,
  limit = 15,
}: GlobalImportanceProps) {
  if (!explainability) {
    return (
      <EmptyState
        title="No explanation was produced"
        hint="This run did not include an explanation step."
      />
    );
  }

  const importances: FeatureImportance[] = explainability.feature_importances ?? [];

  if (explainability.status !== "available" || importances.length === 0) {
    return (
      <div>
        <EmptyState
          title="Explanation unavailable"
          hint={
            explainability.reason ??
            "The explainability layer could not explain this model."
          }
        />
        <CausationNote className="mt-3" />
      </div>
    );
  }

  const shown = importances.slice(0, limit);
  const largest = Math.max(...shown.map((entry) => Math.abs(entry.importance)), 0);

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="accent">{explainability.method}</Badge>
        {explainability.explainer && (
          <Badge tone="neutral">{explainability.explainer}</Badge>
        )}
        {explainability.sample_count !== undefined && (
          <span className="text-xs text-ink-500">
            {explainability.sample_count} rows explained
          </span>
        )}
      </div>

      {explainability.aggregation && (
        <p className="mt-2 text-xs text-ink-600">
          Values are the {explainability.aggregation}
          {explainability.explained_output
            ? `, for ${explainability.explained_output}`
            : ""}
          .
        </p>
      )}

      <ul className="mt-3 space-y-2">
        {shown.map((entry) => (
          <li key={entry.feature}>
            <div className="flex items-baseline justify-between gap-3">
              <span className="truncate font-mono text-xs text-ink-800">
                {entry.feature}
              </span>
              <span className="shrink-0 tabular-nums text-xs text-ink-700">
                {formatMetric(entry.importance)}
              </span>
            </div>
            <Bar
              fraction={largest > 0 ? Math.abs(entry.importance) / largest : 0}
              label={`${entry.feature}: importance ${formatMetric(entry.importance)}, rank ${entry.rank}`}
            />
          </li>
        ))}
      </ul>

      {importances.length > shown.length && (
        <p className="mt-2 text-xs text-ink-500">
          Showing the top {shown.length} of {importances.length} features.
        </p>
      )}

      {explainability.warnings && explainability.warnings.length > 0 && (
        <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-amber-800">
          {explainability.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}

      <CausationNote className="mt-3" />
    </div>
  );
}
