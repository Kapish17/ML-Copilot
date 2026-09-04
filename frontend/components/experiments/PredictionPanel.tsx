"use client";

import { useCallback, useEffect, useId, useState } from "react";
import { Badge } from "@/components/common/Badge";
import { Bar } from "@/components/common/Bar";
import { Button } from "@/components/common/Button";
import { EmptyState } from "@/components/common/EmptyState";
import { ErrorBanner } from "@/components/common/ErrorBanner";
import { Loading } from "@/components/common/Spinner";
import { experimentModel, predictFromExperiment } from "@/lib/api/experiments";
import type {
  JsonObject,
  ModelAvailability,
  PredictedFeature,
  PredictionResponse,
} from "@/lib/api/types";
import { formatMetric } from "@/lib/format";

/**
 * Predicting from the model an experiment produced.
 *
 * The form is built from the model's **own declared schema**, fetched from
 * `GET /api/v1/experiments/{id}/model` rather than inferred from the record.
 * That matters twice over: the backend refuses a feature the model was not
 * trained on, so a form assembled from anything else would produce requests
 * that are rejected; and a model can be deleted after the run that made it, in
 * which case there is nothing to render a form for and this says so.
 *
 * One record at a time here. The endpoint takes a batch and the API client
 * sends one — a form for a thousand rows would be a spreadsheet, and that is
 * not what a dashboard is for.
 *
 * **This component holds no credential.** Like the rest of the dashboard it
 * calls the API from the browser; if the backend requires a key, the request
 * is refused and the error says so honestly rather than pretending a retry
 * would help.
 */
export interface PredictionPanelProps {
  experimentId: string;
}

/** What a blank form looks like: every declared feature, empty. */
function emptyValues(features: PredictedFeature[]): Record<string, string> {
  return Object.fromEntries(features.map((feature) => [feature.name, ""]));
}

/**
 * Turn the typed form into the record the API expects.
 *
 * An empty box becomes `null` rather than `""`. The backend treats both as a
 * missing value — its imputation was fitted for exactly that — but `null` is
 * what "I did not supply this" means in JSON, and sending it keeps the request
 * honest about what the person actually entered.
 *
 * Nothing else is converted. A numeric string is sent as a string and the
 * backend reads it as a number; guessing here would mean two places deciding
 * what a value is, and only one of them can see the model.
 */
function toRecord(values: Record<string, string>): JsonObject {
  return Object.fromEntries(
    Object.entries(values).map(([name, value]) => [
      name,
      value.trim() === "" ? null : value,
    ]),
  );
}

/** The input type that suits a feature's kind, without inventing validation. */
function inputTypeFor(kind: string): string {
  if (kind === "numeric") return "number";
  if (kind === "datetime") return "date";
  return "text";
}

export function PredictionPanel({ experimentId }: PredictionPanelProps) {
  const formId = useId();
  const [model, setModel] = useState<ModelAvailability | null>(null);
  const [loadError, setLoadError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);

  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [predictError, setPredictError] = useState<unknown>(null);
  const [predicting, setPredicting] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    setLoadError(null);
    experimentModel(experimentId)
      .then((availability) => {
        setModel(availability);
        setValues(emptyValues(availability.features));
      })
      .catch((cause: unknown) => {
        setModel(null);
        setLoadError(cause);
      })
      .finally(() => setLoading(false));
  }, [experimentId]);

  useEffect(load, [load]);

  const submit = useCallback(
    (event: React.FormEvent) => {
      event.preventDefault();
      setPredicting(true);
      setPredictError(null);
      setResult(null);
      predictFromExperiment(experimentId, [toRecord(values)])
        .then(setResult)
        .catch((cause: unknown) => setPredictError(cause))
        .finally(() => setPredicting(false));
    },
    [experimentId, values],
  );

  if (loading) return <Loading label="Checking for a stored model…" />;

  if (loadError != null) {
    return (
      <ErrorBanner
        error={loadError}
        title="Could not check for a stored model"
        onRetry={load}
      />
    );
  }

  if (!model?.available) {
    return (
      <EmptyState
        title="No model is stored for this experiment"
        hint={
          model?.reason ??
          "Runs recorded before model persistence was added cannot be predicted from."
        }
      />
    );
  }

  const prediction = result?.predictions[0];
  const probabilities = prediction?.probabilities ?? null;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone="good" glyph="●">
          <span className="font-semibold">Model stored</span>
          <span className="font-normal text-ink-600">{model.display_name}</span>
        </Badge>
        <Badge tone="neutral">
          Predicts <span className="font-mono">{model.target_column}</span>
        </Badge>
        <Badge tone="neutral">{model.task_type}</Badge>
      </div>

      <p className="text-xs text-ink-500">
        Values run through the same fitted preprocessing this experiment
        produced — nothing is re-fitted, which is what makes a prediction here
        comparable to the held-out score above. Leave a box empty for a missing
        value; the model&apos;s imputation was fitted for exactly that.
      </p>

      <form onSubmit={submit} className="space-y-4">
        <fieldset className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <legend className="sr-only">Feature values</legend>
          {model.features.map((feature) => (
            <div key={feature.name}>
              <label
                htmlFor={`${formId}-${feature.name}`}
                className="block text-xs font-medium text-ink-700"
              >
                {feature.name}
                <span className="ml-1 font-normal text-ink-400">
                  {feature.kind}
                </span>
              </label>
              <input
                id={`${formId}-${feature.name}`}
                name={feature.name}
                type={inputTypeFor(feature.kind)}
                step={feature.kind === "numeric" ? "any" : undefined}
                value={values[feature.name] ?? ""}
                onChange={(event) =>
                  setValues((current) => ({
                    ...current,
                    [feature.name]: event.target.value,
                  }))
                }
                className="mt-1 w-full rounded-md border border-ink-300 px-2 py-1.5 text-sm focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400"
              />
            </div>
          ))}
        </fieldset>

        <div className="flex items-center gap-3">
          <Button type="submit" disabled={predicting}>
            {predicting ? "Predicting…" : "Predict"}
          </Button>
          <Button
            type="button"
            variant="ghost"
            disabled={predicting}
            onClick={() => {
              setValues(emptyValues(model.features));
              setResult(null);
              setPredictError(null);
            }}
          >
            Clear
          </Button>
        </div>
      </form>

      {predicting && <Loading label="Predicting…" />}

      {predictError != null && (
        <ErrorBanner error={predictError} title="Could not predict" />
      )}

      {prediction && !predicting && (
        <div className="rounded-md border border-ink-200 bg-white p-4">
          <h4 className="text-xs font-medium uppercase tracking-widest text-ink-500">
            Predicted {result?.model.target_column}
          </h4>
          <p className="mt-1 text-2xl font-semibold tabular-nums text-ink-900">
            {typeof prediction.prediction === "number"
              ? formatMetric(prediction.prediction)
              : String(prediction.prediction)}
          </p>

          {probabilities && (
            <div className="mt-4">
              <h5 className="mb-2 text-xs font-medium uppercase tracking-widest text-ink-500">
                Class probability
              </h5>
              <ul className="space-y-1.5">
                {Object.entries(probabilities)
                  .sort(([, a], [, b]) => b - a)
                  .map(([label, probability]) => (
                    <li key={label} className="flex items-center gap-2">
                      <span className="w-24 shrink-0 truncate font-mono text-xs text-ink-700">
                        {label}
                      </span>
                      <Bar
                        fraction={probability}
                        label={`${label}: ${(probability * 100).toFixed(1)}%`}
                      />
                      <span className="w-14 shrink-0 text-right text-xs tabular-nums text-ink-600">
                        {(probability * 100).toFixed(1)}%
                      </span>
                    </li>
                  ))}
              </ul>
            </div>
          )}

          {result && (
            <p className="mt-4 text-xs text-ink-500">
              Produced by {result.model.display_name}, which scored{" "}
              {formatMetric(result.model.primary_metric_value)}{" "}
              {result.model.primary_metric} on the held-out test set. That is a
              measurement of the model, not a confidence in this prediction.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
