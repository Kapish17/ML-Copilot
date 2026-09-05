"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { Badge } from "@/components/common/Badge";
import { Card } from "@/components/common/Card";
import { DataTable, Td, Th } from "@/components/common/DataTable";
import { ErrorBanner } from "@/components/common/ErrorBanner";
import { Loading } from "@/components/common/Spinner";
import { Tabs } from "@/components/common/Tabs";
import { ExperimentSummary } from "@/components/experiments/ExperimentSummary";
import { MetricsPanel } from "@/components/experiments/MetricsPanel";
import { ModelComparisonTable } from "@/components/experiments/ModelComparisonTable";
import { PredictionPanel } from "@/components/experiments/PredictionPanel";
import { RunDiagnostics } from "@/components/experiments/RunDiagnostics";
import { GlobalImportance } from "@/components/explainability/GlobalImportance";
import { QualityFindings } from "@/components/dataset/QualityFindings";
import { getExperiment } from "@/lib/api/experiments";
import type { ExperimentRecord } from "@/lib/api/types";
import { formatCount, formatPercent } from "@/lib/format";

/**
 * One stored run in full — the page an experiment citation links to.
 *
 * Everything here comes from the stored record, which is what the backend
 * kept: the fingerprint, the preprocessing decisions, the scores and the
 * explanation. It contains no data and no fitted model, so this page shows
 * neither, and says so rather than leaving a reader to wonder.
 */
export default function ExperimentDetailPage() {
  const params = useParams<{ id: string }>();
  const experimentId = Array.isArray(params?.id) ? params.id[0] : params?.id;

  const [record, setRecord] = useState<ExperimentRecord | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    if (!experimentId) return;
    setLoading(true);
    setError(null);
    getExperiment(experimentId)
      .then(setRecord)
      .catch((cause: unknown) => {
        setRecord(null);
        setError(cause);
      })
      .finally(() => setLoading(false));
  }, [experimentId]);

  useEffect(load, [load]);

  return (
    <div className="space-y-6">
      <nav aria-label="Breadcrumb" className="text-sm">
        <Link
          href="/experiments"
          className="text-accent-700 underline decoration-accent-300 underline-offset-2 hover:text-accent-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400"
        >
          ← All experiments
        </Link>
      </nav>

      {loading && (
        <Card>
          <Loading label="Loading experiment…" />
        </Card>
      )}

      {!loading && error != null && (
        <ErrorBanner error={error} title="Could not load this experiment" onRetry={load} />
      )}

      {!loading && !error && record && (
        <>
          <h1 className="sr-only">Experiment {record.experiment_id}</h1>

          <ExperimentSummary record={record} />

          <Card title="Result" headingLevel={2}>
            <Tabs
              ariaLabel="Experiment detail sections"
              tabs={[
                {
                  id: "models",
                  label: "Model comparison",
                  badge: record.selection.candidates.length,
                  content: (
                    <ModelComparisonTable
                      selection={record.selection}
                      evaluation={record.evaluation}
                    />
                  ),
                },
                {
                  id: "metrics",
                  label: "Metrics",
                  content: <MetricsPanel evaluation={record.evaluation} />,
                },
                {
                  id: "diagnostics",
                  label: "Diagnostics",
                  // Hidden when nothing was flagged: a "0" beside the label
                  // reads as a score, not as an absence.
                  badge: record.evaluation.diagnostics?.length || undefined,
                  content: (
                    <RunDiagnostics
                      diagnostics={record.evaluation.diagnostics}
                    />
                  ),
                },
                {
                  id: "explain",
                  label: "Explainability",
                  content: (
                    <GlobalImportance explainability={record.explainability} />
                  ),
                },
                {
                  id: "predict",
                  label: "Predict",
                  content: <PredictionPanel experimentId={record.experiment_id} />,
                },
                {
                  id: "data",
                  label: "Data & preparation",
                  content: (
                    <div className="space-y-4">
                      <DataTable
                        caption="How the dataset was prepared for this run"
                        head={
                          <tr>
                            <Th>Step</Th>
                            <Th>Value</Th>
                          </tr>
                        }
                      >
                        <tr>
                          <Th scope="row" className="normal-case tracking-normal">
                            Train / test split
                          </Th>
                          <Td>
                            {formatCount(record.preprocessing.train_row_count)}{" "}
                            train ·{" "}
                            {formatCount(record.preprocessing.test_row_count)} test
                            (test size {formatPercent(record.preprocessing.test_size * 100)})
                            {record.preprocessing.stratified && (
                              <span className="ml-2">
                                <Badge tone="good" glyph="✓">
                                  stratified
                                </Badge>
                              </span>
                            )}
                          </Td>
                        </tr>
                        <tr>
                          <Th scope="row" className="normal-case tracking-normal">
                            Features used
                          </Th>
                          <Td className="font-mono text-xs">
                            {record.preprocessing.selected_columns.join(", ") || "—"}
                          </Td>
                        </tr>
                        <tr>
                          <Th scope="row" className="normal-case tracking-normal">
                            Excluded
                          </Th>
                          <Td className="font-mono text-xs">
                            {record.preprocessing.excluded_columns.join(", ") ||
                              "none"}
                          </Td>
                        </tr>
                        <tr>
                          <Th scope="row" className="normal-case tracking-normal">
                            Identifier columns
                          </Th>
                          <Td className="font-mono text-xs">
                            {record.preprocessing.identifier_columns.join(", ") ||
                              "none"}
                          </Td>
                        </tr>
                        <tr>
                          <Th scope="row" className="normal-case tracking-normal">
                            Rows dropped (missing target)
                          </Th>
                          <Td>
                            {formatCount(
                              record.preprocessing.rows_dropped_missing_target,
                            )}
                          </Td>
                        </tr>
                      </DataTable>

                      <div>
                        <h4 className="mb-2 text-sm font-semibold text-ink-900">
                          Data-quality findings recorded with this run
                        </h4>
                        <QualityFindings
                          issues={record.dataset.data_quality_issues ?? []}
                        />
                      </div>

                      <p className="text-xs text-ink-500">
                        The stored record holds the fingerprint, the decisions
                        and the scores. It contains no dataset rows: the
                        uploaded file was parsed in memory for one request and
                        released, and nothing here is a copy of it. The fitted
                        model is kept separately — see the Predict tab — and it
                        holds learned coefficients, not data.
                      </p>
                    </div>
                  ),
                },
              ]}
            />
          </Card>
        </>
      )}
    </div>
  );
}
