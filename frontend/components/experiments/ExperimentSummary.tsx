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
  const metric = metricLabel(evaluation.primary_metric);
  // Under the holdout strategy the selecting score is not cross-validated and
  // did not come from the training rows — it is the held-out score, used
  // twice. Labelling it "CV" and "training rows only" there would be the exact
  // conflation the rest of this page exists to prevent.
  const spread =
    selection.selection_score_std === null
      ? ""
      : `± ${formatMetric(selection.selection_score_std)} · `;
  const selectionLabel = selection.uses_test_data
    ? `Selection ${metric}`
    : `CV ${metric}`;
  const selectionHint = selection.uses_test_data
    ? `${spread}held-out rows — this score also chose the model`
    : `${spread}training rows only`;

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
          label={`Held-out ${metric}`}
          value={formatMetric(evaluation.primary_metric_value)}
          hint={
            evaluation.is_unbiased
              ? `measured once on ${formatCount(evaluation.test_row_count)} unseen rows`
              : `${formatCount(evaluation.test_row_count)} rows that also chose the model`
          }
        />
        <StatTile
          label={selectionLabel}
          value={formatMetric(selection.selection_score)}
          hint={selectionHint}
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
          label="Features"
          value={formatCount(record.preprocessing.transformed_feature_names.length)}
          hint="after encoding, as the model saw them"
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

      {/*
        Why this model won, in the backend's own words. Rendered as sent and
        never composed here: the sentence is built from the run's recorded
        numbers, and a second version assembled in the browser would be a
        second answer to the same question.
      */}
      {selection.rationale && (
        <p className="mt-4 border-l-2 border-accent-200 pl-3 text-sm leading-relaxed text-ink-700">
          {selection.rationale}
        </p>
      )}

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
