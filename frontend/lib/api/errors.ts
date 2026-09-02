/**
 * Turning a backend failure into something worth showing a person.
 *
 * The backend already answers every failure with one envelope —
 * `{"error": {code, message, details}}` — and its messages are written for a
 * human. So why map them at all? Because the *code* is the stable thing and
 * the message is not, and because a few codes describe a situation the person
 * in front of this screen can act on more directly than the backend, which
 * does not know it is talking to a browser with an upload button in it.
 *
 * The rule for what gets shown is deliberately conservative. A mapped code
 * shows its own sentence. An unmapped code falls back to the backend's own
 * message, which is safe by construction: the backend never puts a traceback,
 * a filesystem path, a provider exception or a credential in one. Nothing
 * from `details` is rendered as prose — the few fields worth showing (limits,
 * available columns) are read out deliberately by the component that wants
 * them.
 */

import type { ErrorResponse, JsonObject } from "./types";

/** A failure from the backend, or from not reaching it at all. */
export class ApiError extends Error {
  /** The backend's stable machine-readable code, or a client-side one. */
  readonly code: string;
  /** The HTTP status, or 0 when the request never got a response. */
  readonly status: number;
  /** The backend's structured context, when it sent any. */
  readonly details: JsonObject;

  constructor(
    code: string,
    message: string,
    status = 0,
    details: JsonObject = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.details = details;
  }
}

/** Codes raised by the client itself, which never come from the backend. */
export const CLIENT_ERROR_CODES = {
  /** The request never reached the backend. */
  NETWORK: "network_unavailable",
  /** A response arrived, but it was not the shape the contract promises. */
  MALFORMED: "malformed_response",
} as const;

/**
 * Friendly text for the codes worth rewording, keyed by the backend's code.
 *
 * Everything here says what happened and, where there is one, what to do
 * about it. Nothing here repeats a limit or a column name — those live in
 * `details` and are rendered by the component that knows how to show them.
 */
const FRIENDLY_MESSAGES: Record<string, string> = {
  // Upload and ingestion
  unsupported_file_type:
    "That file type is not supported. Upload a CSV, an Excel workbook (.xlsx) or a JSON file.",
  file_too_large: "That file is larger than the upload limit.",
  empty_file: "That file is empty.",
  malformed_csv:
    "The CSV could not be read. Check that every row has the same number of fields as the header.",
  invalid_excel:
    "The uploaded Excel file could not be read. Save it as an .xlsx workbook and try again.",
  invalid_json:
    "The JSON file could not be read as a table. Upload an array of objects, one per row.",
  invalid_dataset_content:
    "The file's contents do not match the format it was sent as.",
  empty_dataset: "That file parsed, but it holds no rows to analyse.",
  dataset_too_large: "That dataset is larger than the configured limits.",
  missing_header: "The first row does not contain usable column names.",
  duplicate_columns: "The file repeats a column name, so it cannot be analysed.",
  target_column_not_found: "That column is not in the dataset.",

  // Experiments
  invalid_request: "The request could not be processed as written.",
  invalid_experiment_configuration:
    "That experiment configuration is not valid.",
  experiment_not_found: "No experiment is stored under that id.",
  invalid_experiment_id: "That experiment id is not valid.",
  incomparable_experiments:
    "Those experiments cannot be compared: they do not share a task and a metric.",

  // Knowledge and the language model
  retrieval_index_not_built:
    "Knowledge search is currently unavailable — nothing has been indexed yet.",
  retrieval_unavailable: "Knowledge search is currently unavailable.",
  embedding_provider_unavailable:
    "Knowledge search is currently unavailable — the embedding provider is not ready.",
  llm_not_configured:
    "The AI assistant is unavailable — no language-model credential is configured on the server.",
  llm_provider_error: "The AI provider is temporarily unavailable.",
  llm_timeout: "The AI provider took too long to respond. Try again.",
  llm_rate_limited: "The AI provider is rate limiting requests. Try again shortly.",

  // Agent
  agent_unavailable:
    "The AI Data Scientist is unavailable — the server has no language-model credential configured.",
  agent_provider_error: "The AI provider is temporarily unavailable.",
  agent_planner_error:
    "The AI Data Scientist could not plan a next step. Try rephrasing the question.",
  agent_run_failed: "The run did not produce an answer. Try again.",
  invalid_agent_budget:
    "That limit is higher than the server allows. Lower it and try again.",

  // Client-side
  [CLIENT_ERROR_CODES.NETWORK]:
    "The backend could not be reached. Check that it is running and that the API URL is correct.",
  [CLIENT_ERROR_CODES.MALFORMED]:
    "The backend returned a response this app could not read.",

  // Transport
  internal_error: "Something went wrong on the server.",
};

/** Whether a value is a usable error envelope. */
export function isErrorResponse(value: unknown): value is ErrorResponse {
  if (typeof value !== "object" || value === null) return false;
  const envelope = (value as { error?: unknown }).error;
  if (typeof envelope !== "object" || envelope === null) return false;
  const { code, message } = envelope as { code?: unknown; message?: unknown };
  return typeof code === "string" && typeof message === "string";
}

/**
 * The sentence to show for a failure.
 *
 * @param error - Anything thrown by the API client, or by fetch itself.
 * @returns A sentence safe to render. Never a traceback, a path or a key —
 *   an unmapped code falls back to the backend's own message, which the
 *   backend guarantees is free of all three.
 */
export function friendlyMessage(error: unknown): string {
  if (error instanceof ApiError) {
    return FRIENDLY_MESSAGES[error.code] ?? error.message;
  }
  return FRIENDLY_MESSAGES[CLIENT_ERROR_CODES.NETWORK];
}

/** The stable code for a failure, for tests and for conditional rendering. */
export function errorCode(error: unknown): string {
  return error instanceof ApiError ? error.code : CLIENT_ERROR_CODES.NETWORK;
}
