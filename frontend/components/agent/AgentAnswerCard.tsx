import Link from "next/link";
import { Badge, type BadgeTone } from "@/components/common/Badge";
import { Card } from "@/components/common/Card";
import { Tabs } from "@/components/common/Tabs";
import { CitationList } from "./CitationList";
import { ToolTrace } from "./ToolTrace";
import { WorkflowSteps } from "./WorkflowSteps";
import { GlobalImportance } from "@/components/explainability/GlobalImportance";
import {
  LocalExplanation,
  type FeatureContribution,
} from "@/components/explainability/LocalExplanation";
import { experimentHref } from "@/lib/citations";
import type {
  AgentAnswer,
  AgentObservation,
  ExperimentExplainability,
  JsonObject,
} from "@/lib/api/types";
import { formatBadge, formatCount, humanise } from "@/lib/format";

/**
 * One answer from the AI Data Scientist, with the evidence behind it.
 *
 * The status is the most important thing on this card, and it is rendered
 * before the answer text for exactly that reason. All four outcomes arrive as
 * HTTP 200 and only one of them — `completed` — is an answer a person should
 * act on. A dashboard that showed the prose first and the status as a footnote
 * would invite someone to read an ungrounded answer as a finding.
 *
 * Nothing here renders a system prompt, a provider name, a model name, a
 * credential or any reasoning: the backend does not return them, and this
 * card reads only the fields it names.
 *
 * When a run was planned, the plan is shown above the answer — what was going
 * to be done, in order, and how far it got. That is a list of short labels and
 * their outcomes, and deliberately not the arguments each step was called
 * with: those are in the tool trace, already reduced to names.
 */
const STATUS_META: Record<
  string,
  { tone: BadgeTone; glyph: string; label: string; note: string }
> = {
  completed: {
    tone: "good",
    glyph: "✓",
    label: "Grounded answer",
    note: "Every step finished and every citation checked out.",
  },
  partial: {
    tone: "warn",
    glyph: "▲",
    label: "Partial",
    note: "Some steps did not finish. Read the answer as incomplete, not wrong.",
  },
  insufficient_evidence: {
    tone: "warn",
    glyph: "○",
    label: "Insufficient evidence",
    note: "The run found nothing solid enough to answer from, and said so rather than guessing.",
  },
  grounding_failed: {
    tone: "bad",
    glyph: "✕",
    label: "Not grounded",
    note: "The answer cited a source that was not retrieved, so it was refused rather than repaired. Do not treat the text below as a finding.",
  },
};

/** Read a JSON object field as a list of records, when it is one. */
function recordList(value: unknown): JsonObject[] {
  return Array.isArray(value)
    ? value.filter(
        (entry): entry is JsonObject =>
          typeof entry === "object" && entry !== null && !Array.isArray(entry),
      )
    : [];
}

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asText(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

/**
 * Pull an explanation out of the agent's observations, when it produced one.
 *
 * The agent's `explain_experiment` tool is the only route to a *local*
 * explanation in this product — the experiment record carries global
 * importances only. So the observation is read for both shapes, and the card
 * shows whichever the run actually produced. Nothing is synthesised when the
 * run produced neither.
 */
export function explanationsFrom(observations: AgentObservation[]): {
  global: ExperimentExplainability | null;
  local: {
    contributions: FeatureContribution[];
    prediction: string | number | null;
    predictedClass: string | null;
    baseValue: number | null;
    method: string | null;
    rowIndex: number | null;
  } | null;
} {
  let globalExplanation: ExperimentExplainability | null = null;
  let local: ReturnType<typeof explanationsFrom>["local"] = null;

  for (const observation of observations) {
    if (observation.tool_name !== "explain_experiment") continue;
    const output = observation.output;
    if (!output || observation.status !== "ok") continue;

    const importances = recordList(output.feature_importances);
    if (importances.length > 0 && !globalExplanation) {
      globalExplanation = {
        status: "available",
        method: asText(output.method) ?? "shap",
        explainer: asText(output.explainer),
        aggregation: asText(output.aggregation),
        explained_output: asText(output.explained_output),
        feature_importances: importances.map((entry, index) => ({
          feature: asText(entry.feature) ?? `feature ${index + 1}`,
          importance: asNumber(entry.importance) ?? 0,
          rank: asNumber(entry.rank) ?? index + 1,
        })),
        reason: null,
      };
    }

    const contributions = recordList(output.feature_contributions);
    if (contributions.length > 0 && !local) {
      local = {
        contributions: contributions.map((entry, index) => ({
          feature: asText(entry.feature) ?? `feature ${index + 1}`,
          contribution: asNumber(entry.contribution),
          direction: asText(entry.direction),
          rank: asNumber(entry.rank) ?? index + 1,
          value:
            typeof entry.value === "string" || typeof entry.value === "number"
              ? entry.value
              : null,
        })),
        prediction:
          typeof output.prediction === "string" ||
          typeof output.prediction === "number"
            ? output.prediction
            : null,
        predictedClass: asText(output.predicted_class),
        baseValue: asNumber(output.base_value),
        method: asText(output.method),
        rowIndex: asNumber(output.row_index),
      };
    }
  }

  return { global: globalExplanation, local };
}

export function AgentAnswerCard({ answer }: { answer: AgentAnswer }) {
  const meta = STATUS_META[answer.status] ?? {
    tone: "neutral" as BadgeTone,
    glyph: "•",
    label: humanise(answer.status),
    note: "",
  };
  const { global: globalExplanation, local } = explanationsFrom(
    answer.observations ?? [],
  );
  const hasExplanation = Boolean(globalExplanation || local);

  return (
    <Card
      title="AI Data Scientist"
      description={answer.question}
      aside={
        <Badge tone={meta.tone} glyph={meta.glyph}>
          {meta.label}
        </Badge>
      }
    >
      <div aria-live="polite">
        {meta.note && (
          <p
            className={`rounded-md px-3 py-2 text-xs ${
              answer.status === "completed"
                ? "bg-emerald-50 text-emerald-900"
                : answer.status === "grounding_failed"
                  ? "bg-rose-50 text-rose-900"
                  : "bg-amber-50 text-amber-900"
            }`}
          >
            {meta.note}
          </p>
        )}

        <p className="mt-3 whitespace-pre-wrap text-sm leading-relaxed text-ink-900">
          {answer.final_answer || "The run produced no answer text."}
        </p>
      </div>

      {answer.workflow && <WorkflowSteps workflow={answer.workflow} />}

      {answer.warnings.length > 0 && (
        <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-amber-800">
          {answer.warnings.map((warning) => (
            <li key={warning}>{warning}</li>
          ))}
        </ul>
      )}

      <CitationList
        citations={answer.citations.length > 0 ? answer.citations : answer.citation_ids}
        rejected={answer.rejected_citations}
      />

      {answer.experiment_ids.length > 0 && (
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-ink-600">
            Experiments created
          </span>
          {answer.experiment_ids.map((id) => (
            <Link
              key={id}
              href={experimentHref(id)}
              className="break-id rounded border border-accent-200 bg-accent-50 px-2 py-0.5 font-mono text-xs text-accent-800 hover:bg-accent-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400"
            >
              {id}
            </Link>
          ))}
        </div>
      )}

      <div className="mt-4">
        <Tabs
          ariaLabel="Evidence behind this answer"
          tabs={[
            {
              id: "tools",
              label: "Tools used",
              badge: answer.tool_call_count,
              content: (
                <ToolTrace
                  toolCalls={answer.tool_calls}
                  observations={answer.observations}
                />
              ),
            },
            ...(hasExplanation
              ? [
                  {
                    id: "explanation",
                    label: "Explanation",
                    content: (
                      <div className="space-y-5">
                        {globalExplanation && (
                          <div>
                            <h4 className="mb-2 text-sm font-semibold text-ink-900">
                              Global feature importance
                            </h4>
                            <GlobalImportance explainability={globalExplanation} />
                          </div>
                        )}
                        {local && (
                          <div>
                            <h4 className="mb-2 text-sm font-semibold text-ink-900">
                              This prediction
                            </h4>
                            <LocalExplanation {...local} />
                          </div>
                        )}
                      </div>
                    ),
                  },
                ]
              : []),
            {
              id: "run",
              label: "Run",
              content: (
                <dl className="grid gap-3 text-xs sm:grid-cols-2 lg:grid-cols-3">
                  <div>
                    <dt className="font-semibold text-ink-600">Tools available</dt>
                    <dd className="mt-0.5 font-mono text-ink-800">
                      {answer.tools_available.join(", ") || "none"}
                    </dd>
                  </div>
                  <div>
                    <dt className="font-semibold text-ink-600">Steps</dt>
                    <dd className="mt-0.5 text-ink-800">
                      {answer.workflow
                        ? `${answer.workflow.completed_step_count} of ${answer.workflow.planned_step_count} planned steps`
                        : `${answer.iterations} planning steps`}{" "}
                      · {answer.tool_call_count} tool calls
                    </dd>
                  </div>
                  {answer.dataset && (
                    <>
                      <div>
                        <dt className="font-semibold text-ink-600">Dataset</dt>
                        <dd className="mt-0.5 text-ink-800">
                          {answer.dataset.filename}{" "}
                          <Badge tone="neutral">
                            {formatBadge(answer.dataset.source_format)}
                          </Badge>
                        </dd>
                      </div>
                      <div>
                        <dt className="font-semibold text-ink-600">Fingerprint</dt>
                        <dd className="mt-0.5 break-id font-mono text-ink-800">
                          {answer.dataset.fingerprint}
                        </dd>
                      </div>
                      <div>
                        <dt className="font-semibold text-ink-600">Shape</dt>
                        <dd className="mt-0.5 text-ink-800">
                          {formatCount(answer.dataset.row_count)} rows ×{" "}
                          {formatCount(answer.dataset.column_count)} columns
                        </dd>
                      </div>
                      <div>
                        <dt className="font-semibold text-ink-600">Storage</dt>
                        <dd className="mt-0.5 text-ink-800">
                          {answer.dataset.persisted
                            ? "persisted"
                            : "not persisted — held in memory for this request only"}
                        </dd>
                      </div>
                    </>
                  )}
                </dl>
              ),
            },
          ]}
        />
      </div>
    </Card>
  );
}
