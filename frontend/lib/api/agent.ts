/** Agent endpoints: the AI Data Scientist, with and without a dataset. */

import { getJson, postForm, postJson, type RequestOptions } from "./client";
import type {
  AgentAnswer,
  AgentBudgets,
  AgentStatusResponse,
} from "./types";

/** Ask the agent a question with no dataset attached. */
export function askAgent(
  question: string,
  budgets: AgentBudgets = {},
  options: RequestOptions = {},
): Promise<AgentAnswer> {
  return postJson<AgentAnswer>(
    "/api/v1/agent/ask",
    { question, ...budgets },
    options,
  );
}

/**
 * Ask the agent a question about an uploaded dataset.
 *
 * One endpoint reads CSV, Excel and JSON — the backend resolves the format,
 * so nothing here branches on it. The file is sent to the configured backend
 * and nowhere else, and the backend holds it in memory for this one request:
 * the answer's `dataset.persisted` is always false.
 */
export function askAgentWithDataset(
  file: File,
  question: string,
  budgets: AgentBudgets = {},
  options: RequestOptions = {},
): Promise<AgentAnswer> {
  const form = new FormData();
  form.append("file", file);
  form.append("question", question);
  for (const [key, value] of Object.entries(budgets)) {
    if (value !== undefined) form.append(key, String(value));
  }
  return postForm<AgentAnswer>("/api/v1/agent/ask-with-dataset", form, options);
}

/** What the agent can currently do: its tools, formats and limits. */
export function agentStatus(
  options: RequestOptions = {},
): Promise<AgentStatusResponse> {
  return getJson<AgentStatusResponse>("/api/v1/agent/status", undefined, options);
}
