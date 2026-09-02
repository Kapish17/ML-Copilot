import { Badge } from "@/components/common/Badge";
import { Card } from "@/components/common/Card";
import { DataTable, Td, Th } from "@/components/common/DataTable";
import { StatRow, StatTile } from "@/components/common/StatTile";
import { Tabs } from "@/components/common/Tabs";
import { ColumnTable } from "./ColumnTable";
import { QualityFindings } from "./QualityFindings";
import type { DatasetProfile, TargetProfile } from "@/lib/api/types";
import {
  formatBadge,
  formatBytes,
  formatCount,
  formatMetric,
  formatPercent,
} from "@/lib/format";

/** The target column's own card: what it is and what it implies. */
function TargetCard({ target }: { target: TargetProfile }) {
  return (
    <Card
      title="Target column"
      headingLevel={3}
      aside={
        <Badge tone="accent">
          {target.task_suggestion === "undetermined"
            ? "task undetermined"
            : target.task_suggestion}
        </Badge>
      }
    >
      <p className="text-sm">
        <span className="font-mono font-medium text-ink-900">{target.name}</span>
        <span className="ml-2 text-ink-600">{target.task_reason}</span>
      </p>

      {target.class_balance && (
        <p className="mt-3 text-xs text-ink-600">
          {target.class_balance.class_count} classes · majority{" "}
          <span className="font-mono">{target.class_balance.majority_class}</span>{" "}
          at {formatPercent(target.class_balance.majority_percentage)}
          {target.class_balance.is_imbalanced && (
            <span className="ml-2">
              <Badge tone="warn" glyph="▲">
                imbalanced
              </Badge>
            </span>
          )}
        </p>
      )}

      {target.distribution && target.distribution.length > 0 && (
        <div className="mt-3">
          <DataTable
            caption={`Distribution of ${target.name}`}
            head={
              <tr>
                <Th>Value</Th>
                <Th numeric>Rows</Th>
                <Th numeric>Share</Th>
              </tr>
            }
          >
            {target.distribution.slice(0, 10).map((entry) => (
              <tr key={entry.value}>
                <Td className="font-mono text-xs">{entry.value}</Td>
                <Td numeric>{formatCount(entry.count)}</Td>
                <Td numeric>{formatPercent(entry.percentage)}</Td>
              </tr>
            ))}
          </DataTable>
        </div>
      )}

      {target.numeric_stats && (
        <p className="mt-3 text-xs text-ink-600">
          mean {formatMetric(target.numeric_stats.mean)} · median{" "}
          {formatMetric(target.numeric_stats.median)} · std{" "}
          {formatMetric(target.numeric_stats.std)} · range{" "}
          {formatMetric(target.numeric_stats.minimum)} –{" "}
          {formatMetric(target.numeric_stats.maximum)}
        </p>
      )}
    </Card>
  );
}

/**
 * The whole profile of one dataset.
 *
 * Structured as headline numbers first, then the two things a person actually
 * decides on — is the data clean enough, and what is being predicted — with
 * the full column list behind a tab because it is long and rarely the first
 * question.
 */
export function DatasetProfileView({ profile }: { profile: DatasetProfile }) {
  const { dataset, quality, target } = profile;
  const criticalCount = quality.issues.filter(
    (issue) => issue.severity === "critical",
  ).length;

  return (
    <div className="space-y-4">
      <Card
        title="Dataset"
        description={profile.filename}
        aside={
          <div className="flex items-center gap-2">
            <Badge tone="neutral">{formatBadge(profile.source_format)}</Badge>
            <Badge tone="neutral">{formatBytes(dataset.memory_usage_bytes)}</Badge>
          </div>
        }
      >
        <StatRow>
          <StatTile label="Rows" value={formatCount(dataset.row_count)} />
          <StatTile label="Columns" value={formatCount(dataset.column_count)} />
          <StatTile
            label="Task"
            value={target ? target.task_suggestion : "—"}
            hint={target ? undefined : "No target column selected"}
          />
          <StatTile label="Target" value={target ? target.name : "—"} />
          <StatTile
            label="Duplicate rows"
            value={formatCount(dataset.duplicate_row_count)}
            hint={formatPercent(dataset.duplicate_row_percentage)}
          />
          <StatTile
            label="Missing cells"
            value={formatCount(dataset.missing_cell_count)}
            hint={formatPercent(dataset.missing_cell_percentage)}
          />
          <StatTile
            label="Quality findings"
            value={formatCount(quality.issue_count)}
            hint={criticalCount > 0 ? `${criticalCount} critical` : undefined}
          />
          <StatTile
            label="Column types"
            value={
              <span className="text-sm font-medium">
                {Object.entries(dataset.column_type_counts)
                  .map(([type, count]) => `${count} ${type}`)
                  .join(", ") || "—"}
              </span>
            }
          />
        </StatRow>
      </Card>

      {target && <TargetCard target={target} />}

      <Card title="Detail" headingLevel={3}>
        <Tabs
          ariaLabel="Dataset profile sections"
          tabs={[
            {
              id: "quality",
              label: "Data quality",
              badge: quality.issue_count,
              content: <QualityFindings issues={quality.issues} />,
            },
            {
              id: "columns",
              label: "Columns",
              badge: profile.columns.length,
              content: <ColumnTable columns={profile.columns} />,
            },
          ]}
        />
      </Card>
    </div>
  );
}
