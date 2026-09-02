/**
 * A proportion drawn as a bar, with its number beside it.
 *
 * No charting library: a feature-importance list is a set of lengths, and a
 * div with a width is a length. The number is always rendered as text next to
 * the bar, so the bar is a convenience for scanning rather than the only way
 * to read the value — which is also what makes it usable without colour and
 * without sight.
 */
export interface BarProps {
  /** 0–1. Values outside the range are clamped. */
  fraction: number;
  /** Accessible description, e.g. "income: 0.84 mean absolute SHAP value". */
  label: string;
  /** A signed value colours the bar by sign; otherwise it is neutral. */
  signed?: boolean;
  value?: number;
}

export function Bar({ fraction, label, signed = false, value }: BarProps) {
  const width = Math.max(0, Math.min(1, Number.isFinite(fraction) ? fraction : 0));
  const negative = signed && typeof value === "number" && value < 0;
  return (
    <div
      className="h-2 w-full overflow-hidden rounded-full bg-ink-100"
      role="img"
      aria-label={label}
    >
      <div
        className={`h-full rounded-full ${negative ? "bg-rose-400" : "bg-accent-500"}`}
        style={{ width: `${(width * 100).toFixed(2)}%` }}
      />
    </div>
  );
}
