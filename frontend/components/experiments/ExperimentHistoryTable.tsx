"use client";

import Link from "next/link";
import { Badge } from "@/components/common/Badge";
import { DataTable, Td, Th } from "@/components/common/DataTable";
import { EmptyState } from "@/components/common/EmptyState";
import { experimentHref } from "@/lib/citations";
import type { ExperimentHeadline } from "@/lib/api/types";
import {
  directionLabel,
  formatMetric,
  formatTimestamp,
  metricDirection,
  metricLabel,
} from "@/lib/format";

/**
 * Every stored run, with a checkbox per row for comparison.
 *
 * Each row's checkbox has its own accessible name naming the run, because
 * "select" repeated twenty times tells a screen-reader user nothing about
 * which run they are about to compare.
 */
export interface ExperimentHistoryTableProps {
  experiments: ExperimentHeadline[];
  selected: string[];
  onToggle: (experimentId: string) => void;
}

export function ExperimentHistoryTable({
  experiments,
  selected,
  onToggle,
}: ExperimentHistoryTableProps) {
  if (experiments.length === 0) {
    return (
      <EmptyState
        title="No experiments stored yet"
        hint="Run one from the dashboard — upload a dataset and either run an experiment directly or ask the AI Data Scientist to find the best model."
      />
    );
  }

  return (
    <DataTable
      caption="Stored experiments. Select two or more to compare them."
      head={
        <tr>
          <Th>
            <span className="sr-only">Select for comparison</span>
          </Th>
          <Th>Run</Th>
          <Th>Created</Th>
          <Th>Dataset</Th>
          <Th>Task</Th>
          <Th>Selected model</Th>
          {/*
            "Selection score", not "CV mean": this list mixes strategies, and
            a holdout run's selecting score is not cross-validated. One
            heading has to be true of every row under it.
          */}
          <Th numeric>Selection score</Th>
          <Th numeric>Held-out score</Th>
        </tr>
      }
    >
      {experiments.map((run) => {
        const direction = metricDirection(run.primary_metric);
        return (
          <tr key={run.experiment_id}>
            <Td>
              <input
                type="checkbox"
                checked={selected.includes(run.experiment_id)}
                onChange={() => onToggle(run.experiment_id)}
                aria-label={`Select ${run.name} for comparison`}
                className="h-4 w-4 rounded border-ink-300 text-accent-600 focus:ring-accent-400"
              />
            </Td>
            <Th scope="row" className="normal-case tracking-normal">
              <Link
                href={experimentHref(run.experiment_id)}
                className="font-medium text-accent-700 underline decoration-accent-300 underline-offset-2 hover:text-accent-900 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400"
              >
                {run.name}
              </Link>
              <span className="mt-0.5 block break-id font-mono text-xs font-normal text-ink-500">
                {run.experiment_id}
              </span>
            </Th>
            <Td className="whitespace-nowrap text-xs text-ink-600">
              {formatTimestamp(run.created_at)}
            </Td>
            <Td className="text-xs">
              <span className="break-id font-mono">{run.dataset_fingerprint}</span>
              <span className="mt-0.5 block text-ink-500">
                target {run.target_column}
              </span>
            </Td>
            <Td>
              <Badge tone="neutral">{run.task_type}</Badge>
            </Td>
            <Td className="font-mono text-xs">{run.selected_model}</Td>
            <Td numeric className="text-ink-600">
              {formatMetric(run.selection_score)}
              {run.selection_score_std != null && (
                <span className="block text-xs">
                  ± {formatMetric(run.selection_score_std)}
                </span>
              )}
            </Td>
            <Td numeric className="font-semibold">
              {formatMetric(run.test_score)}
              <span className="block text-xs font-normal text-ink-500">
                {metricLabel(run.primary_metric)} ·{" "}
                {directionLabel(direction).toLowerCase()}
              </span>
            </Td>
          </tr>
        );
      })}
    </DataTable>
  );
}
