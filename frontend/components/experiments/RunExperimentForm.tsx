"use client";

import { useId, useState } from "react";
import { Button } from "@/components/common/Button";

/**
 * The few experiment options worth putting in front of a person.
 *
 * The backend accepts around twenty form fields. Four are here: the target,
 * the fold count, a seed and whether to explain. The rest have sensible
 * inferred defaults, and exposing all of them would turn the main workflow
 * into a configuration screen. `GET /experiments/capabilities` is where the
 * full set lives for anyone who needs it.
 */
export interface RunExperimentFormProps {
  columns: string[];
  disabled: boolean;
  busyLabel?: string;
  /**
   * The target column, owned by the page.
   *
   * Lifted out of this form because profiling wants it too: a person who has
   * said what they are predicting should not have to say it twice, and a
   * profile computed without a target is missing the analysis they came for.
   */
  target: string;
  onTargetChange: (column: string) => void;
  onRun: (options: {
    target_column?: string;
    folds?: number;
    random_state?: number;
    explain: boolean;
  }) => void;
}

export function RunExperimentForm({
  columns,
  disabled,
  busyLabel,
  target,
  onTargetChange,
  onRun,
}: RunExperimentFormProps) {
  const base = useId();
  const [folds, setFolds] = useState(5);
  const [seed, setSeed] = useState(42);
  const [explain, setExplain] = useState(true);

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        onRun({
          target_column: target || undefined,
          folds,
          random_state: seed,
          explain,
        });
      }}
      className="space-y-4"
    >
      <div className="space-y-4">
        <div>
          <label
            htmlFor={`${base}-target`}
            className="block text-sm font-medium text-ink-800"
          >
            Target column
          </label>
          {columns.length > 0 ? (
            <select
              id={`${base}-target`}
              value={target}
              onChange={(event) => onTargetChange(event.target.value)}
              disabled={disabled}
              className="mt-1 w-full rounded-md border border-ink-300 bg-white px-3 py-2 text-sm focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 disabled:bg-ink-100"
            >
              <option value="">Infer from the data (last column)</option>
              {columns.map((column) => (
                <option key={column} value={column}>
                  {column}
                </option>
              ))}
            </select>
          ) : (
            <input
              id={`${base}-target`}
              type="text"
              value={target}
              onChange={(event) => onTargetChange(event.target.value)}
              disabled={disabled}
              placeholder="Leave empty to infer"
              className="mt-1 w-full rounded-md border border-ink-300 px-3 py-2 text-sm focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 disabled:bg-ink-100"
            />
          )}
          <p className="mt-1 text-xs text-ink-500">
            When omitted the last column is used by convention, and the run says
            so in its warnings.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label
              htmlFor={`${base}-folds`}
              className="block text-sm font-medium text-ink-800"
            >
              CV folds
            </label>
            <input
              id={`${base}-folds`}
              type="number"
              min={2}
              max={10}
              value={folds}
              onChange={(event) => setFolds(Number(event.target.value))}
              disabled={disabled}
              className="mt-1 w-full rounded-md border border-ink-300 px-3 py-2 text-sm focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 disabled:bg-ink-100"
            />
          </div>
          <div>
            <label
              htmlFor={`${base}-seed`}
              className="block text-sm font-medium text-ink-800"
            >
              Random seed
            </label>
            <input
              id={`${base}-seed`}
              type="number"
              min={0}
              value={seed}
              onChange={(event) => setSeed(Number(event.target.value))}
              disabled={disabled}
              className="mt-1 w-full rounded-md border border-ink-300 px-3 py-2 text-sm focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 disabled:bg-ink-100"
            />
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <label className="flex items-center gap-2 text-sm text-ink-800">
          <input
            type="checkbox"
            checked={explain}
            onChange={(event) => setExplain(event.target.checked)}
            disabled={disabled}
            className="h-4 w-4 rounded border-ink-300 text-accent-600 focus:ring-accent-400"
          />
          Explain the winning model with SHAP
        </label>
        <Button type="submit" disabled={disabled}>
          {disabled && busyLabel ? busyLabel : "Run experiment"}
        </Button>
      </div>
    </form>
  );
}
