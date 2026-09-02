import { Badge } from "@/components/common/Badge";
import { EmptyState } from "@/components/common/EmptyState";
import { resolveCitation } from "@/lib/citations";
import type { SearchResponse } from "@/lib/api/types";
import { formatMetric, humanise } from "@/lib/format";
import Link from "next/link";

/**
 * Retrieved passages, best first.
 *
 * The passage itself is the point, so it is shown in full rather than
 * truncated — a person judging whether an answer is well-founded needs to read
 * the evidence, not a preview of it. The score is shown as a number and as a
 * short bar; embeddings, vectors, chunk offsets and index internals are not
 * shown at all, because they tell a reader nothing about whether the passage
 * answers the question.
 */
export function SearchResults({ response }: { response: SearchResponse }) {
  if (response.results.length === 0) {
    return (
      <EmptyState
        title="Nothing matched that query"
        hint={`${response.candidate_count} passages were considered but none scored above the similarity threshold of ${formatMetric(response.similarity_threshold)}.`}
      />
    );
  }

  return (
    <div>
      <p className="text-xs text-ink-600">
        {response.result_count} of {response.candidate_count} passages scored
        above the threshold.
      </p>

      <ol className="mt-3 space-y-3">
        {response.results.map((result) => {
          const citation = resolveCitation({
            citation_id: result.citation_id,
            source_type: result.source_type,
            source_title: result.source_title,
            source_reference: result.source_reference,
            score: result.score,
          });
          return (
            <li
              key={result.chunk_id}
              className="rounded-md border border-ink-200 bg-white p-3"
            >
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge tone="neutral">#{result.rank}</Badge>
                  <Badge tone={citation.kind === "experiment" ? "accent" : "info"}>
                    {humanise(result.source_type)}
                  </Badge>
                  {citation.href ? (
                    <Link
                      href={citation.href}
                      className="text-sm font-medium text-accent-700 underline decoration-accent-300 underline-offset-2 hover:text-accent-900"
                    >
                      {result.source_title}
                    </Link>
                  ) : (
                    <span className="text-sm font-medium text-ink-900">
                      {result.source_title}
                    </span>
                  )}
                </div>
                <span className="text-xs tabular-nums text-ink-600">
                  score {formatMetric(result.score)}
                </span>
              </div>

              <div className="mt-1.5 flex items-center gap-2">
                <div
                  className="h-1.5 w-24 overflow-hidden rounded-full bg-ink-100"
                  role="img"
                  aria-label={`Relevance score ${formatMetric(result.score)}`}
                >
                  <div
                    className="h-full rounded-full bg-accent-500"
                    style={{
                      width: `${Math.max(0, Math.min(1, result.score)) * 100}%`,
                    }}
                  />
                </div>
                <span className="font-mono text-xs text-ink-500">
                  {result.source_reference}
                </span>
              </div>

              <p className="mt-2 whitespace-pre-wrap text-sm leading-relaxed text-ink-800">
                {result.content}
              </p>

              <p className="mt-2 break-id font-mono text-xs text-ink-400">
                {result.citation_id}
              </p>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
