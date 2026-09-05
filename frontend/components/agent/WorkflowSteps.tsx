import { Badge, type BadgeTone } from "@/components/common/Badge";
import type { AgentWorkflow } from "@/lib/api/types";

/**
 * The plan the agent made, and how far it got.
 *
 * **What was going to be done, not why.** Each line is the step's own short
 * label — "Profile the uploaded dataset" — which the backend returns as part
 * of the plan. There is no reasoning to show here because the backend returns
 * none, and this component asks for none.
 *
 * **No arguments, no observed values.** A step's arguments are the one place a
 * planner could put text of its own choosing into something a person reads, so
 * the API does not carry them on the plan at all. What each call actually
 * received is in the tool trace, already summarised and already limited to
 * names.
 *
 * The counter above the list is the point of the whole component: "2 of 3
 * steps" tells someone at a glance that an answer is covering less ground than
 * the question asked for, which is otherwise buried in a status word.
 */
const STEP_TONES: Record<string, { tone: BadgeTone; glyph: string; label: string }> = {
  ok: { tone: "good", glyph: "✓", label: "done" },
  unavailable: { tone: "warn", glyph: "○", label: "unavailable" },
  rejected: { tone: "bad", glyph: "✕", label: "refused" },
  failed: { tone: "bad", glyph: "✕", label: "failed" },
  skipped: { tone: "neutral", glyph: "—", label: "not run" },
};

export interface WorkflowStepsProps {
  workflow: AgentWorkflow;
}

export function WorkflowSteps({ workflow }: WorkflowStepsProps) {
  const { steps, completed_step_count: completed, is_complete: complete } = workflow;
  if (steps.length === 0) return null;

  return (
    <section
      aria-label="Planned workflow"
      className="mt-4 rounded-md border border-ink-200 bg-ink-50 p-3"
    >
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h4 className="text-xs font-semibold uppercase tracking-wide text-ink-600">
          Planned workflow
        </h4>
        <Badge
          tone={complete ? "good" : "warn"}
          glyph={complete ? "✓" : "▲"}
        >
          {completed} of {steps.length} steps
        </Badge>
      </div>

      {workflow.goal && (
        <p className="mt-1.5 text-xs text-ink-600">{workflow.goal}</p>
      )}

      <ol className="mt-2 space-y-1.5">
        {steps.map((step, index) => {
          const tone = STEP_TONES[step.status] ?? {
            tone: "neutral" as BadgeTone,
            glyph: "•",
            label: step.status,
          };
          const done = step.status === "ok";
          return (
            <li key={step.step} className="flex items-start gap-2 text-sm">
              <span className="mt-0.5 w-4 shrink-0 text-right text-xs tabular-nums text-ink-500">
                {index + 1}
              </span>
              <span className="min-w-0 flex-1">
                <span className={done ? "text-ink-900" : "text-ink-600"}>
                  {step.purpose}
                </span>
                <span className="ml-2 align-middle">
                  <Badge tone={tone.tone} glyph={tone.glyph}>
                    {tone.label}
                  </Badge>
                </span>
                <span className="ml-2 font-mono text-xs text-ink-400">
                  {step.tool}
                </span>
                {/* Only when a step did not work. An authored sentence from the
                    backend — never a stack trace, a path, or a tool's own
                    exception text. */}
                {!done && step.reason && (
                  <span className="mt-0.5 block text-xs text-ink-600">
                    {step.reason}
                  </span>
                )}
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
