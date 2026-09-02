import { Badge } from "@/components/common/Badge";
import { DataTable, Td, Th } from "@/components/common/DataTable";
import type { ColumnProfile } from "@/lib/api/types";
import { formatCount, formatMetric, formatPercent } from "@/lib/format";

/**
 * One row per column: what it is, how complete it is, how varied it is.
 *
 * Summary statistics only — no cell values are shown here beyond the handful
 * of most frequent categories the backend already summarised, which is the
 * least a person needs to judge whether a categorical column is usable.
 */
export function ColumnTable({ columns }: { columns: ColumnProfile[] }) {
  return (
    <DataTable
      caption="Every column, with its inferred type, completeness and spread"
      head={
        <tr>
          <Th>Column</Th>
          <Th>Type</Th>
          <Th numeric>Missing</Th>
          <Th numeric>Unique</Th>
          <Th>Summary</Th>
        </tr>
      }
    >
      {columns.map((column) => (
        <tr key={column.name}>
          <Th scope="row" className="font-mono text-xs normal-case tracking-normal text-ink-900">
            {column.name}
            {column.is_constant && (
              <span className="ml-2 font-sans">
                <Badge tone="warn" glyph="▲">
                  constant
                </Badge>
              </span>
            )}
          </Th>
          <Td>
            <Badge tone="neutral">{column.inferred_type}</Badge>
            <span className="ml-2 font-mono text-xs text-ink-500">
              {column.dtype}
            </span>
          </Td>
          <Td numeric>
            {formatCount(column.missing_count)}
            <span className="block text-xs text-ink-500">
              {formatPercent(column.missing_percentage)}
            </span>
          </Td>
          <Td numeric>
            {formatCount(column.unique_count)}
            <span className="block text-xs text-ink-500">
              {formatPercent(column.unique_percentage)}
            </span>
          </Td>
          <Td className="text-xs text-ink-600">
            {column.numeric_stats ? (
              <span>
                mean {formatMetric(column.numeric_stats.mean)} · median{" "}
                {formatMetric(column.numeric_stats.median)} · min{" "}
                {formatMetric(column.numeric_stats.minimum)} · max{" "}
                {formatMetric(column.numeric_stats.maximum)}
              </span>
            ) : column.categorical_stats ? (
              <span>
                {column.categorical_stats.top_values
                  .slice(0, 3)
                  .map((entry) => `${entry.value} (${formatPercent(entry.percentage)})`)
                  .join(" · ")}
                {column.categorical_stats.truncated && " · …"}
              </span>
            ) : column.datetime_stats ? (
              <span>
                {column.datetime_stats.earliest ?? "—"} →{" "}
                {column.datetime_stats.latest ?? "—"}
              </span>
            ) : (
              "—"
            )}
          </Td>
        </tr>
      ))}
    </DataTable>
  );
}
