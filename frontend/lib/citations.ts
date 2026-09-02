/**
 * Reading a citation id, and deciding whether it can be a link.
 *
 * The backend's citations are stable strings of two kinds:
 *
 *   `exp:exp_9f2c…`   an experiment this system actually ran
 *   `docs:readme#…`   a section of the project's own documentation
 *
 * An experiment citation names a page this application has — `/experiments/…`
 * — so it becomes a link. A documentation citation names a heading in a
 * README that this application does not serve, so it is shown as a labelled
 * source and **not** linked. Inventing a URL for it would be worse than
 * showing none: a citation whose link goes nowhere real is indistinguishable,
 * to a reader, from a fabricated citation.
 */

import type { AgentCitation, AnswerCitation } from "./api/types";

/** What a citation points at. */
export type CitationKind = "experiment" | "documentation" | "other";

/** One citation, resolved for display. */
export interface ResolvedCitation {
  /** The backend's stable identifier, shown verbatim. */
  id: string;
  kind: CitationKind;
  /** A short human label, e.g. "Experiment exp_9f2c…" or a heading. */
  label: string;
  /** Where the source lives, when the backend named it. */
  reference?: string;
  /** An in-app route, only when this application really serves one. */
  href?: string;
  score?: number | null;
}

/** The experiment id inside an `exp:` citation, when there is one. */
export function experimentIdFromCitation(citationId: string): string | null {
  const match = /^exp:([A-Za-z0-9_.-]+)/.exec(citationId.trim());
  return match ? match[1] : null;
}

/** The route for an experiment's detail page. */
export function experimentHref(experimentId: string): string {
  return `/experiments/${encodeURIComponent(experimentId)}`;
}

function labelFor(
  citationId: string,
  kind: CitationKind,
  title: string | undefined,
): string {
  if (kind === "experiment") {
    const id = experimentIdFromCitation(citationId);
    return title ? `${title}` : `Experiment ${id ?? citationId}`;
  }
  if (title) return title;
  const heading = citationId.split("#")[1];
  return heading ? heading.replace(/-/g, " ") : citationId;
}

/**
 * Resolve one citation for display.
 *
 * @param citation - A citation from the agent or from the ask endpoint.
 * @returns The label, kind and — only for experiments — an in-app route.
 */
export function resolveCitation(
  citation: AgentCitation | AnswerCitation | string,
): ResolvedCitation {
  const id = typeof citation === "string" ? citation : citation.citation_id;
  const title =
    typeof citation === "string" ? undefined : citation.source_title || undefined;
  const reference =
    typeof citation === "string"
      ? undefined
      : citation.source_reference || undefined;
  const score =
    typeof citation === "string"
      ? undefined
      : "relevance_score" in citation
        ? citation.relevance_score
        : citation.score;

  const experimentId = experimentIdFromCitation(id);
  const kind: CitationKind = experimentId
    ? "experiment"
    : id.startsWith("docs:")
      ? "documentation"
      : "other";

  return {
    id,
    kind,
    label: labelFor(id, kind, title),
    reference,
    // Only an experiment gets a link, and only to a route this app serves.
    href: experimentId ? experimentHref(experimentId) : undefined,
    score,
  };
}

/** Resolve a list of citations, preserving the backend's order. */
export function resolveCitations(
  citations: Array<AgentCitation | AnswerCitation | string>,
): ResolvedCitation[] {
  return citations.map(resolveCitation);
}
