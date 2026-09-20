/** Experiment endpoints: running, listing, fetching and comparing. */

import { deleteJson, getJson, postForm, postJson, type RequestOptions } from "./client";
import { ApiError, CLIENT_ERROR_CODES } from "./errors";
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

/**
 * The sections the experiment page traverses rather than merely displays.
 *
 * A missing string renders as an empty string; a missing *section* is a
 * `.map` of undefined inside a render, which throws and takes the page with
 * it. These are the four the detail page iterates.
 */
const REQUIRED_RECORD_SECTIONS = [
  "dataset",
  "preprocessing",
  "selection",
  "evaluation",
] as const;

/**
 * Fetch one stored experiment in full.
 *
 * The response is checked for the sections the page will traverse. `requestJson`
 * already refuses a body that is not an object, which catches a proxy's HTML
 * error page — it cannot catch a *JSON* body of the wrong shape, and that is
 * the one a version skew or a partial deployment actually produces. Without
 * this the page threw mid-render; with it the caller gets the same
 * `malformed_response` every other unreadable answer produces, and the page
 * shows the error state it already has.
 */
export async function getExperiment(
  experimentId: string,
  options: RequestOptions = {},
): Promise<ExperimentRecord> {
  const record = await getJson<ExperimentRecord>(
    `/api/v1/experiments/${encodeURIComponent(experimentId)}`,
    undefined,
    options,
  );

  const missing = REQUIRED_RECORD_SECTIONS.filter((section) => {
    const value = (record as unknown as Record<string, unknown>)[section];
    return value === null || typeof value !== "object";
  });
  if (missing.length > 0) {
    throw new ApiError(
      CLIENT_ERROR_CODES.MALFORMED,
      "The backend returned an experiment this app could not read.",
      200,
      { missing_sections: missing },
    );
  }

  return record;
}

/** Confirmation of a single delete. */
export interface ExperimentDeleteResponse {
  experiment_id: string;
  deleted: boolean;
}

/** Confirmation of a bulk delete. */
export interface ExperimentClearResponse {
  deleted_count: number;
}

/**
 * Permanently remove one stored experiment and its fitted model.
 *
 * The only way a record leaves history: a page reload, a backend restart or
 * the passage of time never do this on their own.
 */
export function deleteExperiment(
  experimentId: string,
  options: RequestOptions = {},
): Promise<ExperimentDeleteResponse> {
  return deleteJson<ExperimentDeleteResponse>(
    `/api/v1/experiments/${encodeURIComponent(experimentId)}`,
    options,
  );
}

/** Permanently remove every stored experiment and fitted model. Irreversible. */
export function clearExperiments(
  options: RequestOptions = {},
): Promise<ExperimentClearResponse> {
  return deleteJson<ExperimentClearResponse>("/api/v1/experiments", options);
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
