import { Badge, type BadgeTone } from "@/components/common/Badge";
import { EmptyState } from "@/components/common/EmptyState";
import type { ExperimentDiagnostic } from "@/lib/api/types";
import { humanise } from "@/lib/format";

/**
 * Things about a finished run worth a second look.
 *
 * These are signals, not verdicts, and the wording matters more here than
 * anywhere else on the page: the backend says "potential overfitting signal"
 * precisely so that a reader is prompted to look rather than told a
 * conclusion. So the message is rendered exactly as it was sent — never
 * summarised, rephrased or given a headline of this component's own — and the
 * framing around it says what the list is and is not.
 *
 * A run with diagnostics is a run that completed. Nothing here failed.
 */
const SEVERITY: Record<string, { tone: BadgeTone; glyph: string; label: string }> = {
  warning: { tone: "warn", glyph: "▲", label: "worth checking" },
  info: { tone: "info", glyph: "●", label: "note" },
};

export function RunDiagnostics({
  diagnostics,
}: {
  diagnostics?: ExperimentDiagnostic[];
}) {
  const items = diagnostics ?? [];

  if (items.length === 0) {
    return (
      <EmptyState
        title="Nothing flagged on this run"
        hint="No threshold in the evaluation checks was crossed. That is not a guarantee the model is good — it means these particular checks found nothing to point at."
      />
    );
  }

  return (
    <div className="space-y-3">
      <p className="text-xs leading-relaxed text-ink-500">
        Signals worth a second look, read from this run&rsquo;s own recorded
        numbers. They are prompts to check something, not conclusions about the
        model, and none of them failed the run.
      </p>

      <ul className="space-y-2">
        {items.map((item, index) => {
          const severity = SEVERITY[item.severity] ?? SEVERITY.info;
          return (
            <li
              key={`${item.code}-${index}`}
              className="rounded-md border border-ink-200 bg-white p-3"
            >
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={severity.tone} glyph={severity.glyph}>
                  {severity.label}
                </Badge>
                <span className="font-mono text-xs text-ink-500">
                  {humanise(item.code)}
                </span>
              </div>
              <p className="mt-1.5 text-sm leading-relaxed text-ink-700">
                {item.message}
              </p>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
