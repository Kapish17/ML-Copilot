"use client";

import { useId, useState } from "react";
import { Button } from "@/components/common/Button";
import { Card } from "@/components/common/Card";
import { EmptyState } from "@/components/common/EmptyState";
import { ErrorBanner } from "@/components/common/ErrorBanner";
import { Loading } from "@/components/common/Spinner";
import { AgentAnswerCard } from "./AgentAnswerCard";
import type { AgentAnswer } from "@/lib/api/types";

/**
 * The question box for the AI Data Scientist.
 *
 * Deliberately not a chat transcript. The agent has no conversation memory —
 * every question is answered from the dataset and the evidence, not from what
 * was asked before — so a threaded chat UI would imply a continuity that does
 * not exist, and would invite follow-ups like "and the other one?" that the
 * backend cannot honour. One question, one answer with its evidence, is the
 * honest shape.
 *
 * The suggestions are there because the useful questions are not obvious: a
 * person's first instinct with a text box is to ask something generic, and
 * this agent is at its best when asked something specific about the data.
 */
export interface AgentAskProps {
  /** Called with the question. The parent owns the request and the file. */
  onAsk: (question: string) => void;
  busy: boolean;
  busyLabel: string;
  answer: AgentAnswer | null;
  error: unknown;
  disabled?: boolean;
  disabledReason?: string;
  suggestions?: string[];
}

export const DATASET_SUGGESTIONS = [
  "What is the target column?",
  "Which model performs best and why?",
  "What are the most important features?",
  "Does my dataset have missing values?",
  "How reliable is this model?",
];

export function AgentAsk({
  onAsk,
  busy,
  busyLabel,
  answer,
  error,
  disabled = false,
  disabledReason,
  suggestions = DATASET_SUGGESTIONS,
}: AgentAskProps) {
  const inputId = useId();
  const [question, setQuestion] = useState("");
  const blocked = disabled || busy;

  function submit(text: string) {
    const trimmed = text.trim();
    if (!trimmed || blocked) return;
    onAsk(trimmed);
  }

  return (
    <div className="space-y-4">
      <Card
        title="Ask the AI Data Scientist"
        description="It chooses which of its own tools a question needs — profiling, an experiment, an explanation, a documentation search — and answers from what those steps returned."
      >
        <form
          onSubmit={(event) => {
            event.preventDefault();
            submit(question);
          }}
        >
          <label htmlFor={inputId} className="sr-only">
            Your question about the dataset
          </label>
          <textarea
            id={inputId}
            rows={3}
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            disabled={blocked}
            placeholder="Ask anything about your dataset…"
            aria-describedby={disabled && disabledReason ? `${inputId}-blocked` : undefined}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                submit(question);
              }
            }}
            className="w-full resize-y rounded-md border border-ink-300 px-3 py-2 text-sm leading-relaxed focus:border-accent-500 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 disabled:bg-ink-100"
          />

          <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs text-ink-500">
              Press <kbd className="rounded border border-ink-300 px-1">⌘</kbd> +{" "}
              <kbd className="rounded border border-ink-300 px-1">Enter</kbd> to
              ask.
            </p>
            <Button type="submit" disabled={blocked || question.trim().length === 0}>
              {busy ? busyLabel : "Ask"}
            </Button>
          </div>

          {disabled && disabledReason && (
            <p id={`${inputId}-blocked`} className="mt-2 text-xs text-ink-600">
              {disabledReason}
            </p>
          )}
        </form>

        {suggestions.length > 0 && (
          <div className="mt-4 border-t border-ink-100 pt-3">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-ink-600">
              Try asking
            </h3>
            <ul className="mt-2 flex flex-wrap gap-2">
              {suggestions.map((suggestion) => (
                <li key={suggestion}>
                  <button
                    type="button"
                    disabled={blocked}
                    onClick={() => {
                      setQuestion(suggestion);
                      submit(suggestion);
                    }}
                    className="rounded-full border border-ink-200 bg-white px-3 py-1 text-xs text-ink-700 hover:border-accent-300 hover:bg-accent-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {suggestion}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        )}
      </Card>

      {busy && (
        <Card>
          <Loading label={busyLabel} />
          <p className="mt-2 text-xs text-ink-500">
            Runs are synchronous — profiling and training happen while this
            request is open, so a question that trains a model takes as long as
            training one.
          </p>
        </Card>
      )}

      {!busy && error != null && (
        <ErrorBanner error={error} title="The AI Data Scientist could not answer" />
      )}

      {!busy && !error && answer && <AgentAnswerCard answer={answer} />}

      {!busy && !error && !answer && (
        <EmptyState
          title="No question asked yet"
          hint="Ask something specific about the data — the agent is at its most useful when it has a concrete question to plan against."
        />
      )}
    </div>
  );
}
