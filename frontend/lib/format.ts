/**
 * Presentation helpers — the small decisions about how a number reads.
 *
 * The one that matters is `metricDirection`. A dashboard that prints "0.94"
 * next to "1957.67" and calls both a score is lying by omission: for F1 larger
 * is better and for RMSE it is worse, and a reader comparing two runs needs to
 * know which. The backend already says so, per metric and per experiment, so
 * nothing here guesses — it reads the direction it was given and only falls
 * back to a table of well-known metric names when a caller has no direction to
 * hand.
 */

import type { MetricDirection } from "./api/types";

/** Metrics whose value should be as small as possible. */
const LOWER_IS_BETTER = new Set([
  "mae",
  "mse",
  "rmse",
  "median_absolute_error",
  "mean_absolute_error",
  "mean_squared_error",
  "root_mean_squared_error",
  "log_loss",
  "mape",
]);

/** Human labels for the metric identifiers the backend uses. */
const METRIC_LABELS: Record<string, string> = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1: "F1",
  roc_auc: "ROC-AUC",
  balanced_accuracy: "Balanced accuracy",
  mae: "MAE",
  mse: "MSE",
  rmse: "RMSE",
  r2: "R²",
  mape: "MAPE",
  log_loss: "Log loss",
};

/** The display label for a metric identifier. */
export function metricLabel(metric: string): string {
  return METRIC_LABELS[metric] ?? metric.replace(/_/g, " ");
}

/**
 * Whether a larger value of this metric is better.
 *
 * @param metric - The metric identifier, e.g. `f1` or `rmse`.
 * @param declared - The direction the backend reported, when there is one.
 *   Always preferred: the backend is the authority, and this function's own
 *   table is only for metrics reached without one.
 */
export function metricDirection(
  metric: string,
  declared?: MetricDirection | string | null,
): MetricDirection {
  if (declared === "higher_is_better" || declared === "lower_is_better") {
    return declared;
  }
  return LOWER_IS_BETTER.has(metric.toLowerCase())
    ? "lower_is_better"
    : "higher_is_better";
}

/** The short phrase shown beside a metric so its direction is never assumed. */
export function directionLabel(direction: MetricDirection): string {
  return direction === "higher_is_better"
    ? "Higher is better"
    : "Lower is better";
}

/** Format a metric value, or an em dash when there is none. */
export function formatMetric(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude !== 0 && magnitude < 0.001) return value.toExponential(2);
  if (magnitude >= 1000) {
    return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }
  return value.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
}

/** Format a count with thousands separators. */
export function formatCount(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return value.toLocaleString();
}

/** Format a percentage the backend already expressed as 0–100. */
export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return `${value.toFixed(value < 10 ? 2 : 1)}%`;
}

/** Format a signed contribution, so its direction is visible in the number. */
export function formatSigned(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  const rendered = formatMetric(Math.abs(value));
  return value < 0 ? `−${rendered}` : `+${rendered}`;
}

/** Format an ISO timestamp for a table, in the reader's own locale. */
export function formatTimestamp(iso: string): string {
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) return iso;
  return parsed.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Format a duration given in seconds. */
export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds - minutes * 60)}s`;
}

/** Format a byte count for the profile summary. */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

/** Title-case a snake_case identifier for display. */
export function humanise(value: string): string {
  if (!value) return "";
  const spaced = value.replace(/[_-]+/g, " ").trim();
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

/** The uppercase badge text for a source format. */
export function formatBadge(sourceFormat: string | null | undefined): string {
  if (!sourceFormat) return "—";
  return sourceFormat.toUpperCase();
}
