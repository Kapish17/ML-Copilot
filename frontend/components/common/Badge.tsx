import type { ReactNode } from "react";

/**
 * A short status label.
 *
 * Every tone pairs its colour with a word, never a colour alone: a reader who
 * cannot distinguish the hues still reads "critical" or "rejected". That rule
 * is the reason this component takes children rather than deriving text.
 */
export type BadgeTone = "neutral" | "info" | "good" | "warn" | "bad" | "accent";

const TONES: Record<BadgeTone, string> = {
  neutral: "bg-ink-100 text-ink-700 ring-ink-200",
  info: "bg-sky-50 text-sky-800 ring-sky-200",
  good: "bg-emerald-50 text-emerald-800 ring-emerald-200",
  warn: "bg-amber-50 text-amber-900 ring-amber-300",
  bad: "bg-rose-50 text-rose-800 ring-rose-200",
  accent: "bg-accent-50 text-accent-800 ring-accent-200",
};

export interface BadgeProps {
  tone?: BadgeTone;
  /** A glyph shown before the text, so tone is never carried by colour alone. */
  glyph?: string;
  className?: string;
  children: ReactNode;
}

export function Badge({
  tone = "neutral",
  glyph,
  className = "",
  children,
}: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${TONES[tone]} ${className}`}
    >
      {glyph && (
        <span aria-hidden="true" className="font-semibold">
          {glyph}
        </span>
      )}
      {children}
    </span>
  );
}
