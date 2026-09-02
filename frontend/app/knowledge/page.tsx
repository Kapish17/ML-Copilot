"use client";

import { useEffect, useId, useState } from "react";
import { Button } from "@/components/common/Button";
import { Card } from "@/components/common/Card";
import { EmptyState } from "@/components/common/EmptyState";
import { ErrorBanner } from "@/components/common/ErrorBanner";
import { Loading } from "@/components/common/Spinner";
import { AnswerCard } from "@/components/knowledge/AnswerCard";
import { SearchResults } from "@/components/knowledge/SearchResults";
import {
  askKnowledge,
  knowledgeStatus,
  searchKnowledge,
} from "@/lib/api/knowledge";
import type {
  AskResponse,
  KnowledgeStatus,
  SearchResponse,
} from "@/lib/api/types";

/**
 * The Knowledge Assistant — deliberately not the AI Data Scientist.
 *
 * The two are easy to confuse and behave very differently, so the page says
 * which is which before anything else. This one retrieves passages from the
 * project's own documentation and its experiment history and answers from
 * them. It runs no tools, trains nothing, and never sees a dataset. The AI
 * Data Scientist, on the dashboard, does all three.
 *
 * Search and answering are offered side by side because they answer different
 * needs: search shows the evidence, the assistant summarises it. Someone
 * checking a claim wants the first.
 */
const EXAMPLES = [
  "What is cross-validation?",
  "How does SHAP work in this project?",
  "How does ML Copilot prevent leakage?",
  "What dataset formats are supported?",
];

export default function KnowledgePage() {
  const inputId = useId();
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"search" | "ask">("search");

  const [status, setStatus] = useState<KnowledgeStatus | null>(null);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);

  useEffect(() => {
    const controller = new AbortController();
    knowledgeStatus({ signal: controller.signal })
      .then(setStatus)
      .catch(() => setStatus(null));
    return () => controller.abort();
  }, []);

  async function submit(text: string, requested: "search" | "ask") {
    const trimmed = text.trim();
    if (!trimmed || busy) return;

    setMode(requested);
    setBusy(true);
    setError(null);
    setResults(null);
    setAnswer(null);

    try {
      if (requested === "search") {
        setResults(await searchKnowledge(trimmed));
      } else {
        setAnswer(await askKnowledge(trimmed));
      }
    } catch (cause) {
      setError(cause);
    } finally {
      setBusy(false);
    }
  }

  const answeringOff = status !== null && !status.answering_available;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-ink-900">
          Knowledge Assistant
        </h1>
        <p className="mt-1 max-w-3xl text-sm text-ink-600">
          Searches this project&rsquo;s own documentation and its experiment
          history, and answers from what it finds. It runs no tools, trains
          nothing and never sees a dataset — for that, use the{" "}
          <span className="font-medium text-ink-800">AI Data Scientist</span> on
          the dashboard.
        </p>
      </div>

      <Card title="Ask or search">
        <form
          onSubmit={(event) => {
            event.preventDefault();
            submit(query, mode);
          }}
        >
          <label htmlFor={inputId} className="block text-sm font-medium text-ink-800">
            Your question
          </label>
          <input
            id={inputId}
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            disabled={busy}
            maxLength={status?.max_query_length}
            placeholder="What is cross-validation?"
            className="mt-1 w-full rounded-md border border-ink-300 px-3 py-2 text-sm focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 disabled:bg-ink-100"
          />

          <div className="mt-3 flex flex-wrap gap-2">
            <Button
              type="submit"
              variant="secondary"
              disabled={busy || query.trim().length === 0}
              onClick={() => setMode("search")}
            >
              Search passages
            </Button>
            <Button
              type="button"
              disabled={busy || query.trim().length === 0 || answeringOff}
              onClick={() => submit(query, "ask")}
            >
              Get a grounded answer
            </Button>
          </div>

          {answeringOff && (
            <p className="mt-2 text-xs text-ink-600">
              Grounded answers are unavailable — the server has no
              language-model credential configured. Search still works.
            </p>
          )}
        </form>

        <div className="mt-4 border-t border-ink-100 pt-3">
          <h2 className="text-xs font-semibold uppercase tracking-wide text-ink-600">
            Try
          </h2>
          <ul className="mt-2 flex flex-wrap gap-2">
            {EXAMPLES.map((example) => (
              <li key={example}>
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => {
                    setQuery(example);
                    submit(example, "search");
                  }}
                  className="rounded-full border border-ink-200 bg-white px-3 py-1 text-xs text-ink-700 hover:border-accent-300 hover:bg-accent-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 disabled:opacity-60"
                >
                  {example}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </Card>

      {busy && (
        <Card>
          <Loading
            label={mode === "search" ? "Searching knowledge…" : "Generating answer…"}
          />
        </Card>
      )}

      {!busy && error != null && (
        <ErrorBanner
          error={error}
          title={mode === "search" ? "Search failed" : "The assistant could not answer"}
        />
      )}

      {!busy && !error && answer && <AnswerCard answer={answer} />}

      {!busy && !error && results && (
        <Card title="Retrieved passages" headingLevel={2}>
          <SearchResults response={results} />
        </Card>
      )}

      {!busy && !error && !results && !answer && (
        <EmptyState
          title="Nothing searched yet"
          hint="Search returns the passages themselves, so you can read the evidence. A grounded answer summarises them and cites every source it used."
        />
      )}
    </div>
  );
}
