"use client";

import { useCallback, useEffect, useId, useMemo, useState } from "react";
import { Badge } from "@/components/common/Badge";
import { Bar } from "@/components/common/Bar";
import { Button } from "@/components/common/Button";
import { EmptyState } from "@/components/common/EmptyState";
import { ErrorBanner } from "@/components/common/ErrorBanner";
import { Loading } from "@/components/common/Spinner";
import { experimentModel, predictFromExperiment } from "@/lib/api/experiments";
import { ApiError } from "@/lib/api/errors";
import type {
  JsonObject,
  ModelAvailability,
  PredictedFeature,
  PredictionResponse,
} from "@/lib/api/types";
import { formatCount, formatMetric, metricLabel } from "@/lib/format";

/**
 * Predicting from the model an experiment produced.
 *
 * The form is built from the model's **own declared schema**, fetched from
 * `GET /api/v1/experiments/{id}/model` rather than inferred from the record.
 * That matters twice over: the backend refuses a feature the model was not
 * trained on, so a form assembled from anything else would produce requests
 * that are rejected; and a model can be deleted or damaged after the run that
 * made it, in which case there is nothing to render a form for and this says
 * so — differently in each case, because the fix differs.
 *
 * One record at a time here. The endpoint takes a batch and the API client
 * sends one — a form for five hundred rows would be a spreadsheet, and a
 * dashboard is not one. There is deliberately **no raw-JSON box**: it would
 * duplicate the form for the single-record case this screen exists to serve,
 * and anyone sending a real batch is doing it from a script against the
 * documented endpoint, not by pasting into a browser.
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

/** What to tell someone a box wants, in two words rather than a dtype. */
function hintFor(kind: string): string {
  if (kind === "numeric") return "number";
  if (kind === "datetime") return "date";
  if (kind === "boolean") return "true / false";
  return "text";
}

/**
 * The feature names a validation failure blamed, if it named any.
 *
 * The backend puts them in `details` as `missing_features` and
 * `unexpected_features`. Read out deliberately here rather than rendered as
 * prose by the error mapper: these are the only two fields worth showing, and
 * knowing which box is wrong is the difference between fixing it and guessing.
 */
function blamedFeatures(error: unknown): string[] {
  if (!(error instanceof ApiError)) return [];
  const named = [
    ...(Array.isArray(error.details.missing_features)
      ? error.details.missing_features
      : []),
    ...(Array.isArray(error.details.unexpected_features)
      ? error.details.unexpected_features
      : []),
  ];
  return named.filter((name): name is string => typeof name === "string");
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

  const blamed = useMemo(() => blamedFeatures(predictError), [predictError]);

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

  // The two unusable states are separated because what a person does next
  // differs: a run from before model persistence needs re-running, and a
  // damaged artifact is the server's own broken file. The backend's `reason`
  // already says which and what to do, so it is shown rather than reworded.
  if (!model || model.status !== "available") {
    const damaged = model?.status === "corrupted";
    return (
      <EmptyState
        title={
          damaged
            ? "This experiment's stored model cannot be used"
            : "No model is stored for this experiment"
        }
        hint={
          model?.reason ??
          "Runs recorded before model persistence was added cannot be predicted from."
        }
      />
    );
  }

  const prediction = result?.predictions[0];
  const probabilities = prediction?.probabilities ?? null;
  const scored = result?.model ?? model;

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
        comparable to the held-out score above. Every box is optional: leave one
        empty for a missing value, and the model&apos;s imputation handles it
        exactly as it was fitted to.
      </p>

      <form onSubmit={submit} className="space-y-4">
        <fieldset className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          <legend className="sr-only">Feature values</legend>
          {model.features.map((feature) => {
            const inputId = `${formId}-${feature.name}`;
            const wrong = blamed.includes(feature.name);
            // A box the backend named is outlined and announced, so a
            // rejected record points at itself instead of making someone
            // re-read the whole form against the message.
            const frame = wrong
              ? "border-rose-400 focus:border-rose-500 focus-visible:ring-rose-400"
              : "border-ink-300 focus:border-accent-500 focus-visible:ring-accent-400";
            const classes = `mt-1 w-full rounded-md border px-2 py-1.5 text-sm focus:outline-none focus-visible:ring-2 ${frame}`;
            return (
              <div key={feature.name}>
                <label
                  htmlFor={inputId}
                  className="block text-xs font-medium text-ink-700"
                >
                  {feature.name}
                  <span className="ml-1 font-normal text-ink-400">
                    {hintFor(feature.kind)}
                  </span>
                </label>
                {feature.kind === "boolean" ? (
                  // A column the model treats as true/false gets the two
                  // values it accepts and nothing else. Typing "yes" would
                  // work and typing "yep" would not, and a form should not
                  // make anyone find that out from a 422.
                  <select
                    id={inputId}
                    name={feature.name}
                    aria-invalid={wrong || undefined}
                    value={values[feature.name] ?? ""}
                    onChange={(event) =>
                      setValues((current) => ({
                        ...current,
                        [feature.name]: event.target.value,
                      }))
                    }
                    className={classes}
                  >
                    <option value="">— not supplied —</option>
                    <option value="true">true</option>
                    <option value="false">false</option>
                  </select>
                ) : (
                  <input
                    id={inputId}
                    name={feature.name}
                    type={inputTypeFor(feature.kind)}
                    step={feature.kind === "numeric" ? "any" : undefined}
                    aria-invalid={wrong || undefined}
                    value={values[feature.name] ?? ""}
                    onChange={(event) =>
                      setValues((current) => ({
                        ...current,
                        [feature.name]: event.target.value,
                      }))
                    }
                    className={classes}
                  />
                )}
              </div>
            );
          })}
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
            Predicted {scored.target_column}
          </h4>
          <p className="mt-1 text-3xl font-semibold tabular-nums text-ink-900">
            {typeof prediction.prediction === "number"
              ? formatMetric(prediction.prediction)
              : String(prediction.prediction)}
          </p>

          {probabilities && (
            <div className="mt-4">
              <h5 className="mb-1 text-xs font-medium uppercase tracking-widest text-ink-500">
                Class probability
              </h5>
              {/* Said once, next to the numbers it qualifies. A probability
                  here is what this estimator reports for this row — useful for
                  seeing how close the decision was, and not a calibrated claim
                  about how often the model is right. */}
              <p className="mb-2 text-xs text-ink-500">
                The model&apos;s own output for this record, not a measured
                real-world certainty.
              </p>
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

          {/* The model's score and this prediction are two different things,
              and this is the sentence that keeps them apart. The score is a
              measurement over held-out rows; nothing here scores this answer,
              and a UI that put a percentage beside a single prediction without
              saying so would be inviting exactly that misreading. */}
          <p className="mt-4 border-t border-ink-100 pt-3 text-xs text-ink-500">
            Produced by{" "}
            <span className="font-medium text-ink-700">
              {scored.display_name}
            </span>
            , which scored {formatMetric(scored.primary_metric_value)}{" "}
            {metricLabel(scored.primary_metric ?? "")}
            {scored.test_row_count
              ? ` on ${formatCount(scored.test_row_count)} held-out rows`
              : " on the held-out test set"}
            . That measures the model, not this prediction.
          </p>
        </div>
      )}
    </div>
  );
}
