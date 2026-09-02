import { Badge } from "@/components/common/Badge";
import { Card } from "@/components/common/Card";
import { StatRow, StatTile } from "@/components/common/StatTile";
import type { ExperimentRecord, ExperimentExecution } from "@/lib/api/types";
import {
  formatBadge,
  formatCount,
  formatDuration,
  formatMetric,
  formatTimestamp,
  metricLabel,
} from "@/lib/format";

/**
 * The headline of one run: what won, on what data, measured how well.
 *
 * The two scores are labelled rather than placed side by side unqualified —
 * "CV mean" and "Test" — for the same reason the comparison table separates
 * them: they answer different questions and the test figure is the one that
 * counts.
 */
export function ExperimentSummary({
  record,
  execution,
}: {
  record: ExperimentRecord;
  execution?: ExperimentExecution;
}) {
  const { dataset, selection, evaluation } = record;

  return (
    <Card
      title={record.name}
      description={`${formatTimestamp(record.created_at)} · ${record.experiment_id}`}
      aside={
        <div className="flex flex-wrap items-center gap-2">
          <Badge tone="accent" glyph="★">
            {selection.selected_model}
          </Badge>
          <Badge tone="neutral">{dataset.task_type}</Badge>
          <Badge tone="neutral">{formatBadge(dataset.source_format)}</Badge>
        </div>
      }
    >
      <StatRow>
        <StatTile
          label={`Test ${metricLabel(evaluation.primary_metric)}`}
          value={formatMetric(evaluation.primary_metric_value)}
          hint="measured once, unseen rows"
        />
        <StatTile
          label="CV mean"
          value={formatMetric(selection.selection_score)}
          hint={
            selection.selection_score_std === null
              ? "training rows only"
              : `± ${formatMetric(selection.selection_score_std)} · training rows only`
          }
        />
        <StatTile label="Target" value={dataset.target_column} />
        <StatTile
          label="Rows × columns"
          value={`${formatCount(dataset.row_count)} × ${formatCount(dataset.column_count)}`}
        />
        <StatTile
          label="Strategy"
          value={selection.strategy.replace(/_/g, " ")}
          hint={selection.folds ? `${selection.folds} folds` : undefined}
        />
        <StatTile
          label="Candidates"
          value={formatCount(selection.candidates.length)}
        />
        <StatTile
          label="Fingerprint"
          value={
            <span className="break-id font-mono text-sm">
              {dataset.fingerprint}
            </span>
          }
          hint="identity is the data, not the file"
        />
        {execution && (
          <StatTile
            label="Duration"
            value={formatDuration(execution.duration_seconds)}
            hint={execution.stored ? "stored" : "not stored"}
          />
        )}
      </StatRow>

      {record.tags && record.tags.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {record.tags.map((tag) => (
            <Badge key={tag} tone="neutral">
              {tag}
            </Badge>
          ))}
        </div>
      )}
    </Card>
  );
}
