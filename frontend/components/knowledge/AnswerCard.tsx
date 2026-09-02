import { Badge, type BadgeTone } from "@/components/common/Badge";
import { Card } from "@/components/common/Card";
import { CitationList } from "@/components/agent/CitationList";
import type { AskResponse } from "@/lib/api/types";
import { humanise } from "@/lib/format";

/**
 * A grounded answer from the Knowledge Assistant.
 *
 * Same rule as the agent's card: the status leads, because `grounded`,
 * `insufficient_evidence` and `grounding_failed` all arrive as 200 and only
 * the first is an answer. The response's `metadata` carries the provider and
 * model names — this card deliberately reads none of that; which vendor
 * answered is not the reader's business and is not shown.
 */
const STATUS_META: Record<
  string,
  { tone: BadgeTone; glyph: string; label: string; note: string }
> = {
  grounded: {
    tone: "good",
    glyph: "✓",
    label: "Grounded",
    note: "Every claim is supported by a retrieved passage, and every citation was checked against them.",
  },
  insufficient_evidence: {
    tone: "warn",
    glyph: "○",
    label: "Insufficient evidence",
    note: "Nothing retrieved was relevant enough to answer from, so the assistant declined rather than guessing.",
  },
  grounding_failed: {
    tone: "bad",
    glyph: "✕",
    label: "Not grounded",
    note: "The answer cited a source that was not retrieved. It was refused rather than repaired — do not treat the text below as a finding.",
  },
};

export function AnswerCard({ answer }: { answer: AskResponse }) {
  const meta = STATUS_META[answer.status] ?? {
    tone: "neutral" as BadgeTone,
    glyph: "•",
    label: humanise(answer.status),
    note: "",
  };

  return (
    <Card
      title="Knowledge Assistant"
      description={answer.question}
      aside={
        <Badge tone={meta.tone} glyph={meta.glyph}>
          {meta.label}
        </Badge>
      }
    >
      <div aria-live="polite">
        {meta.note && (
          <p
            className={`rounded-md px-3 py-2 text-xs ${
              answer.is_grounded
                ? "bg-emerald-50 text-emerald-900"
                : answer.status === "grounding_failed"
                  ? "bg-rose-50 text-rose-900"
                  : "bg-amber-50 text-amber-900"
            }`}
          >
            {meta.note}
          </p>
        )}
        <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-ink-900">
          {answer.answer}
        </p>
      </div>

      {answer.warnings.length > 0 && (
        <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-amber-800">
          {answer.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}

      <CitationList
        citations={answer.citations}
        rejected={answer.rejected_citations}
      />

      <p className="mt-3 text-xs text-ink-500">
        {answer.metadata.retrieved_count} passages retrieved ·{" "}
        {answer.metadata.context_count} used as context
        {answer.metadata.context_truncated && " · context truncated to fit"}
      </p>
    </Card>
  );
}
