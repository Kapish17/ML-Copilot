/** Knowledge endpoints: retrieval, grounded answers, and availability. */

import { getJson, postJson, type RequestOptions } from "./client";
import type { AskResponse, KnowledgeStatus, SearchResponse } from "./types";

/** Search the indexed documentation and experiment history. */
export function searchKnowledge(
  query: string,
  topK?: number,
  options: RequestOptions = {},
): Promise<SearchResponse> {
  return postJson<SearchResponse>(
    "/api/v1/search",
    topK ? { query, top_k: topK } : { query },
    options,
  );
}

/**
 * Ask for a grounded answer over the same evidence.
 *
 * Distinct from the agent: this retrieves passages and answers from them. It
 * runs no tools, trains nothing and never sees a dataset.
 */
export function askKnowledge(
  question: string,
  topK?: number,
  options: RequestOptions = {},
): Promise<AskResponse> {
  return postJson<AskResponse>(
    "/api/v1/ask",
    topK ? { question, top_k: topK } : { question },
    options,
  );
}

/** Whether search and answering are available, and their limits. */
export function knowledgeStatus(
  options: RequestOptions = {},
): Promise<KnowledgeStatus> {
  return getJson<KnowledgeStatus>("/api/v1/knowledge/status", undefined, options);
}
