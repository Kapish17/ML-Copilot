/** Knowledge endpoints: retrieval, grounded answers, and availability. */

import { getJson, postJson, type RequestOptions } from "./client";
import type {
  AskResponse,
  KnowledgeFilters,
  KnowledgeStatus,
  SearchResponse,
} from "./types";

/** Search the indexed documentation and experiment history. */
export function searchKnowledge(
  query: string,
  topK?: number,
  filters?: KnowledgeFilters,
  options: RequestOptions = {},
): Promise<SearchResponse> {
  return postJson<SearchResponse>(
    "/api/v1/search",
    {
      query,
      ...(topK ? { top_k: topK } : {}),
      ...(filters ? { filters } : {}),
    },
    options,
  );
}

/**
 * Ask for a grounded answer over the same evidence.
 *
 * Distinct from the agent: this retrieves passages and answers from them. It
 * runs no tools, trains nothing and never sees a dataset — it can only
 * discuss a dataset once an experiment has run on it and been indexed, and
 * even then only through `filters`, never by being handed the data directly.
 */
export function askKnowledge(
  question: string,
  topK?: number,
  filters?: KnowledgeFilters,
  options: RequestOptions = {},
): Promise<AskResponse> {
  return postJson<AskResponse>(
    "/api/v1/ask",
    {
      question,
      ...(topK ? { top_k: topK } : {}),
      ...(filters ? { filters } : {}),
    },
    options,
  );
}

/** Whether search and answering are available, and their limits. */
export function knowledgeStatus(
  options: RequestOptions = {},
): Promise<KnowledgeStatus> {
  return getJson<KnowledgeStatus>("/api/v1/knowledge/status", undefined, options);
}
