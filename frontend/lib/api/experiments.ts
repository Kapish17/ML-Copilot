/** Experiment endpoints: running, listing, fetching and comparing. */

import { getJson, postForm, postJson, type RequestOptions } from "./client";
import type {
  ExperimentCapabilities,
  ExperimentComparison,
  ExperimentListResponse,
  ExperimentOptions,
  ExperimentRecord,
  ExperimentRunResponse,
  JsonObject,
  ModelAvailability,
  PredictionResponse,
} from "./types";

/** Render experiment options as the multipart fields the backend defines. */
export function toExperimentForm(
  file: File,
  options: ExperimentOptions,
): FormData {
  const form = new FormData();
  form.append("file", file);

  for (const [key, value] of Object.entries(options)) {
    if (value === undefined || value === null || value === "") continue;
    if (Array.isArray(value)) {
      for (const item of value) form.append(key, String(item));
    } else if (typeof value === "boolean") {
      form.append(key, value ? "true" : "false");
    } else {
      form.append(key, String(value));
    }
  }
  return form;
}

/**
 * Run a complete experiment on an uploaded dataset.
 *
 * Synchronous on the backend: the promise settles when the run has finished,
 * which is why every caller shows a progress state rather than a spinner.
 */
export function runExperiment(
  file: File,
  options: ExperimentOptions = {},
  request: RequestOptions = {},
): Promise<ExperimentRunResponse> {
  return postForm<ExperimentRunResponse>(
    "/api/v1/experiments/run",
    toExperimentForm(file, options),
    request,
  );
}

/** Filters the history listing accepts. */
export interface ExperimentQuery extends Record<string, unknown> {
  dataset_fingerprint?: string;
  target_column?: string;
  task_type?: string;
  model_name?: string;
  strategy?: string;
  primary_metric?: string;
  sort_by?: string;
  order?: string;
  limit?: number;
}

/** List stored experiments, newest or best first. */
export function listExperiments(
  query: ExperimentQuery = {},
  options: RequestOptions = {},
): Promise<ExperimentListResponse> {
  return getJson<ExperimentListResponse>("/api/v1/experiments", query, options);
}

/** Fetch one stored experiment in full. */
export function getExperiment(
  experimentId: string,
  options: RequestOptions = {},
): Promise<ExperimentRecord> {
  return getJson<ExperimentRecord>(
    `/api/v1/experiments/${encodeURIComponent(experimentId)}`,
    undefined,
    options,
  );
}

/** Rank two or more experiments that share a task and a metric. */
export function compareExperiments(
  experimentIds: string[],
  options: RequestOptions = {},
): Promise<ExperimentComparison> {
  return postJson<ExperimentComparison>(
    "/api/v1/experiments/compare",
    { experiment_ids: experimentIds },
    options,
  );
}

/** The models, metrics, strategies and limits a request may name. */
export function experimentCapabilities(
  options: RequestOptions = {},
): Promise<ExperimentCapabilities> {
  return getJson<ExperimentCapabilities>(
    "/api/v1/experiments/capabilities",
    undefined,
    options,
  );
}

/**
 * Whether an experiment can be predicted from, and with what.
 *
 * Read before rendering a prediction form, because the answer comes from the
 * artifact store rather than from the stored record: a run that reports a
 * model in its record may no longer have one on disk, and building a form from
 * the record would produce something that cannot work.
 */
export function experimentModel(
  experimentId: string,
  options: RequestOptions = {},
): Promise<ModelAvailability> {
  return getJson<ModelAvailability>(
    `/api/v1/experiments/${encodeURIComponent(experimentId)}/model`,
    undefined,
    options,
  );
}

/**
 * Predict from the model an experiment produced.
 *
 * One record or many take the same shape, and results come back in submission
 * order with the index of the record that produced each. The prediction runs
 * through the same fitted preprocessing the experiment used — none of that
 * happens here; this sends values and reads a result.
 */
export function predictFromExperiment(
  experimentId: string,
  records: JsonObject[],
  options: RequestOptions = {},
): Promise<PredictionResponse> {
  return postJson<PredictionResponse>(
    `/api/v1/experiments/${encodeURIComponent(experimentId)}/predict`,
    { records },
    options,
  );
}
