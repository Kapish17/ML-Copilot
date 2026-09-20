"use client";

import { useCallback, useEffect, useState } from "react";
import { Button } from "@/components/common/Button";
import { Card } from "@/components/common/Card";
import { ErrorBanner } from "@/components/common/ErrorBanner";
import { Loading } from "@/components/common/Spinner";
import { ExperimentComparisonView } from "@/components/experiments/ExperimentComparison";
import { ExperimentHistoryTable } from "@/components/experiments/ExperimentHistoryTable";
import {
  clearExperiments,
  compareExperiments,
  deleteExperiment,
  listExperiments,
} from "@/lib/api/experiments";
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

  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<unknown>(null);
  const [clearing, setClearing] = useState(false);

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

  /**
   * Remove one stored run. This — never a reload, never a restart — is the
   * only thing that empties an entry out of history.
   */
  async function onDelete(experimentId: string) {
    if (!window.confirm("Delete this experiment? This cannot be undone.")) return;
    setDeleteError(null);
    setDeletingId(experimentId);
    try {
      await deleteExperiment(experimentId);
      setSelected((current) => current.filter((id) => id !== experimentId));
      setComparison(null);
      load();
    } catch (error) {
      setDeleteError(error);
    } finally {
      setDeletingId(null);
    }
  }

  /** Remove every stored run. Irreversible, and only ever explicit. */
  async function onClearAll() {
    if (
      !window.confirm(
        "Delete ALL stored experiment history? This cannot be undone.",
      )
    )
      return;
    setDeleteError(null);
    setClearing(true);
    try {
      await clearExperiments();
      setSelected([]);
      setComparison(null);
      load();
    } catch (error) {
      setDeleteError(error);
    } finally {
      setClearing(false);
    }
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
            <Button
              variant="ghost"
              disabled={!runs || runs.length === 0 || clearing}
              onClick={onClearAll}
              className="text-red-700 hover:bg-red-50"
            >
              {clearing ? "Clearing…" : "Clear all"}
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
        {!loading && deleteError != null && (
          <ErrorBanner
            error={deleteError}
            title="That did not delete"
            onRetry={() => setDeleteError(null)}
          />
        )}
        {!loading && !listError && runs && (
          <ExperimentHistoryTable
            experiments={runs}
            selected={selected}
            onToggle={toggle}
            onDelete={onDelete}
            deletingId={deletingId}
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
