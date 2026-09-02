import { Badge } from "@/components/common/Badge";
import { Bar } from "@/components/common/Bar";
import { DataTable, Td, Th } from "@/components/common/DataTable";
import { CausationNote } from "./CausationNote";
import { formatMetric, formatSigned } from "@/lib/format";

/**
 * One prediction, and what moved it.
 *
 * Local contributions are signed, so direction is the point of the table: a
 * contribution is shown with its sign, a word ("increases" / "decreases") and
 * a bar coloured by sign — three signals for one fact, because the sign is the
 * fact a reader is most likely to misread.
 *
 * The backend supplies a value per feature only when it has one; the column
 * is dropped rather than filled with placeholders when it does not.
 */
export interface FeatureContribution {
  feature: string;
  contribution: number | null;
  direction?: string | null;
  rank?: number | null;
  /** The row's value for this feature, when the backend reported one. */
  value?: string | number | null;
}

export interface LocalExplanationProps {
  contributions: FeatureContribution[];
  prediction?: string | number | null;
  predictedClass?: string | null;
  baseValue?: number | null;
  method?: string | null;
  rowIndex?: number | null;
}

function directionWords(entry: FeatureContribution): string {
  if (entry.direction) return entry.direction.replace(/_/g, " ");
  if (entry.contribution === null || entry.contribution === undefined) return "—";
  return entry.contribution >= 0 ? "increases prediction" : "decreases prediction";
}

export function LocalExplanation({
  contributions,
  prediction,
  predictedClass,
  baseValue,
  method,
  rowIndex,
}: LocalExplanationProps) {
  const hasValues = contributions.some(
    (entry) => entry.value !== undefined && entry.value !== null,
  );
  const largest = Math.max(
    ...contributions.map((entry) => Math.abs(entry.contribution ?? 0)),
    0,
  );

  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        {method && <Badge tone="accent">{method}</Badge>}
        {rowIndex !== null && rowIndex !== undefined && (
          <Badge tone="neutral">row {rowIndex}</Badge>
        )}
        {predictedClass && (
          <span className="text-xs text-ink-600">
            predicted class{" "}
            <span className="font-mono text-ink-900">{predictedClass}</span>
          </span>
        )}
        {prediction !== undefined && prediction !== null && (
          <span className="text-xs text-ink-600">
            prediction{" "}
            <span className="font-mono text-ink-900">{String(prediction)}</span>
          </span>
        )}
        {baseValue !== undefined && baseValue !== null && (
          <span className="text-xs text-ink-600">
            base value{" "}
            <span className="font-mono text-ink-900">{formatMetric(baseValue)}</span>
          </span>
        )}
      </div>

      <div className="mt-3">
        <DataTable
          caption="Per-feature contributions to this single prediction"
          head={
            <tr>
              <Th>Feature</Th>
              {hasValues && <Th>Value</Th>}
              <Th numeric>Contribution</Th>
              <Th>Direction</Th>
              <Th>Magnitude</Th>
            </tr>
          }
        >
          {contributions.map((entry) => (
            <tr key={entry.feature}>
              <Th
                scope="row"
                className="font-mono text-xs normal-case tracking-normal text-ink-900"
              >
                {entry.feature}
              </Th>
              {hasValues && (
                <Td className="font-mono text-xs">
                  {entry.value === undefined || entry.value === null
                    ? "—"
                    : String(entry.value)}
                </Td>
              )}
              <Td numeric>{formatSigned(entry.contribution)}</Td>
              <Td className="text-xs text-ink-600">{directionWords(entry)}</Td>
              <Td className="w-32">
                <Bar
                  signed
                  value={entry.contribution ?? 0}
                  fraction={
                    largest > 0 ? Math.abs(entry.contribution ?? 0) / largest : 0
                  }
                  label={`${entry.feature}: ${formatSigned(entry.contribution)}, ${directionWords(entry)}`}
                />
              </Td>
            </tr>
          ))}
        </DataTable>
      </div>

      <CausationNote className="mt-3" />
    </div>
  );
}
