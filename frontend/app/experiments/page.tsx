"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/common/Button";
import { Card } from "@/components/common/Card";
import { ErrorBanner } from "@/components/common/ErrorBanner";
import { Loading } from "@/components/common/Spinner";
import { ExperimentComparisonView } from "@/components/experiments/ExperimentComparison";
import { ExperimentHistoryTable } from "@/components/experiments/ExperimentHistoryTable";
import { compareExperiments, listExperiments } from "@/lib/api/experiments";
import type {
  ExperimentComparison,
  ExperimentHeadline,
} from "@/lib/api/types";

/**
 * Every run this system has stored, and a way to rank a few against each other.
 *
 * Comparison is a backend operation, not a client-side sort: the backend
 * refuses to rank runs that do not share a task and a metric, which is a real
 * constraint a table sorted in the browser would silently ignore.
 */
export default function ExperimentsPage() {
  const [runs, setRuns] = useState<ExperimentHeadline[] | null>(null);
  const [listError, setListError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  const [selected, setSelected] = useState<string[]>([]);
  const [comparison, setComparison] = useState<ExperimentComparison | null>(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState<unknown>(null);

  const load = useCallback(() => {
    setLoading(true);
    setListError(null);
    listExperiments({ sort_by: "created_at", order: "desc", limit: 50 })
      .then((response) => setRuns(response.experiments))
      .catch((error: unknown) => {
        setRuns(null);
        setListError(error);
      })
      .finally(() => setLoading(false));
  }, []);

  useEffect(load, [load]);

  function toggle(experimentId: string) {
    setComparison(null);
    setCompareError(null);
    setSelected((current) =>
      current.includes(experimentId)
        ? current.filter((id) => id !== experimentId)
        : [...current, experimentId],
    );
  }

  async function onCompare() {
    setComparing(true);
    setCompareError(null);
    try {
      setComparison(await compareExperiments(selected));
    } catch (error) {
      setComparison(null);
      setCompareError(error);
    } finally {
      setComparing(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-ink-900">
          Experiment history
        </h1>
        <p className="mt-1 text-sm text-ink-600">
          Every stored run, identified by the content fingerprint of the data it
          ran on — so the same table uploaded as CSV, Excel or JSON appears as
          one dataset.
        </p>
      </div>

      <Card
        title="Runs"
        aside={
          <div className="flex items-center gap-2">
            <span className="text-xs text-ink-500">
              {selected.length} selected
            </span>
            <Button
              variant="secondary"
              disabled={selected.length < 2 || comparing}
              onClick={onCompare}
            >
              {comparing ? "Comparing…" : "Compare selected"}
            </Button>
          </div>
        }
      >
        {loading && <Loading label="Loading experiments…" />}
        {!loading && listError != null && (
          <ErrorBanner
            error={listError}
            title="Could not load the experiment history"
            onRetry={load}
          />
        )}
        {!loading && !listError && runs && (
          <ExperimentHistoryTable
            experiments={runs}
            selected={selected}
            onToggle={toggle}
          />
        )}
      </Card>

      {comparing && (
        <Card>
          <Loading label="Comparing models…" />
        </Card>
      )}

      {!comparing && compareError != null && (
        <ErrorBanner error={compareError} title="Those runs could not be compared" />
      )}

      {!comparing && comparison && (
        <Card title="Comparison" headingLevel={2}>
          <ExperimentComparisonView comparison={comparison} />
        </Card>
      )}
    </div>
  );
}
