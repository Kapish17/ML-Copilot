import Link from "next/link";
import { Badge } from "@/components/common/Badge";
import { resolveCitations } from "@/lib/citations";
import type { AgentCitation, AnswerCitation } from "@/lib/api/types";

/**
 * The sources an answer was built from.
 *
 * An experiment citation links to the run's own page in this application; a
 * documentation citation is shown with the file it came from and is **not**
 * linked, because this application does not serve that file and a link that
 * goes nowhere is worse than no link. The rule lives in `lib/citations.ts`;
 * this component only renders what it is told.
 *
 * Rejected citations get their own list. That is the point of the grounding
 * check being visible at all: a reader should be able to see that the model
 * cited something that was never retrieved and that the system refused it,
 * rather than the claim quietly disappearing.
 */
export interface CitationListProps {
  citations: Array<AgentCitation | AnswerCitation | string>;
  rejected?: string[];
}

export function CitationList({ citations, rejected = [] }: CitationListProps) {
  const resolved = resolveCitations(citations);

  if (resolved.length === 0 && rejected.length === 0) return null;

  return (
    <div className="mt-4 border-t border-ink-100 pt-3">
      {resolved.length > 0 && (
        <>
          <h4 className="text-xs font-semibold uppercase tracking-wide text-ink-600">
            Sources
          </h4>
          <ul className="mt-2 space-y-1.5">
            {resolved.map((citation) => (
              <li key={citation.id} className="flex flex-wrap items-baseline gap-2">
                <Badge
                  tone={citation.kind === "experiment" ? "accent" : "neutral"}
                  glyph={citation.kind === "experiment" ? "▤" : "▧"}
                >
                  {citation.kind === "experiment" ? "experiment" : "documentation"}
                </Badge>
                {citation.href ? (
                  <Link
                    href={citation.href}
                    className="text-sm font-medium text-accent-700 underline decoration-accent-300 underline-offset-2 hover:text-accent-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400"
                  >
                    {citation.label}
                  </Link>
                ) : (
                  <span className="text-sm text-ink-800">{citation.label}</span>
                )}
                {citation.reference && (
                  <span className="font-mono text-xs text-ink-500">
                    {citation.reference}
                  </span>
                )}
                <span className="break-id font-mono text-xs text-ink-400">
                  {citation.id}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}

      {rejected.length > 0 && (
        <div className="mt-3">
          <h4 className="text-xs font-semibold uppercase tracking-wide text-rose-800">
            Rejected citations
          </h4>
          <p className="mt-1 text-xs text-ink-600">
            The answer cited these, but they were not among the sources actually
            retrieved, so they were refused rather than repaired.
          </p>
          <ul className="mt-1.5 space-y-1">
            {rejected.map((id) => (
              <li key={id} className="break-id font-mono text-xs text-rose-800">
                {id}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
