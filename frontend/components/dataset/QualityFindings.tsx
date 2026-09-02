import { Badge, type BadgeTone } from "@/components/common/Badge";
import { DataTable, Td, Th } from "@/components/common/DataTable";
import { EmptyState } from "@/components/common/EmptyState";
import type { IssueSeverity, JsonObject, QualityIssue } from "@/lib/api/types";
import { humanise } from "@/lib/format";

/**
 * The data-quality findings, ordered worst first.
 *
 * Severity is carried by three things at once — a word, a glyph and a colour —
 * so the table is readable in greyscale and to a screen reader. Colour alone
 * would put the most important column of this table out of reach of a
 * noticeable share of readers.
 */
const SEVERITY: Record<IssueSeverity, { tone: BadgeTone; glyph: string; rank: number }> = {
  critical: { tone: "bad", glyph: "▲", rank: 0 },
  warning: { tone: "warn", glyph: "▲", rank: 1 },
  info: { tone: "info", glyph: "●", rank: 2 },
};

/** Render an issue's structured context as short key/value text. */
function renderDetails(details: JsonObject | undefined): string {
  if (!details) return "—";
  const parts = Object.entries(details)
    .filter(([, value]) => value !== null && typeof value !== "object")
    .map(([key, value]) => `${humanise(key)}: ${String(value)}`);
  return parts.length > 0 ? parts.join(" · ") : "—";
}

export function QualityFindings({ issues }: { issues: QualityIssue[] }) {
  if (issues.length === 0) {
    return (
      <EmptyState
        title="No data-quality findings"
        hint="Nothing in this dataset tripped a quality heuristic. Findings are prompts for a human check, not conclusions."
      />
    );
  }

  const ordered = [...issues].sort(
    (a, b) => (SEVERITY[a.severity]?.rank ?? 3) - (SEVERITY[b.severity]?.rank ?? 3),
  );

  return (
    <DataTable
      caption="Data-quality findings, most severe first"
      head={
        <tr>
          <Th>Severity</Th>
          <Th>Columns</Th>
          <Th>Finding</Th>
          <Th>Details</Th>
        </tr>
      }
    >
      {ordered.map((issue, index) => {
        const severity = SEVERITY[issue.severity] ?? SEVERITY.info;
        return (
          <tr key={`${issue.code}-${index}`}>
            <Td>
              <Badge tone={severity.tone} glyph={severity.glyph}>
                {issue.severity}
              </Badge>
            </Td>
            <Td className="font-mono text-xs">
              {issue.columns.length > 0 ? issue.columns.join(", ") : "whole dataset"}
            </Td>
            <Td>
              <span className="font-medium text-ink-900">{humanise(issue.code)}</span>
              <span className="mt-0.5 block text-xs text-ink-600">
                {issue.message}
              </span>
            </Td>
            <Td className="text-xs text-ink-600">{renderDetails(issue.details)}</Td>
          </tr>
        );
      })}
    </DataTable>
  );
}
