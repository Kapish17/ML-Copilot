import { Badge, type BadgeTone } from "@/components/common/Badge";
import { DataTable, Td, Th } from "@/components/common/DataTable";
import type { AgentObservation, AgentToolCall } from "@/lib/api/types";
import { humanise } from "@/lib/format";

/**
 * What the agent actually did — which tools it ran and how each one went.
 *
 * **Which tool ran, not how it decided to run it.** The backend returns no
 * chain-of-thought and this component asks for none; what it shows is the
 * record of executed steps, which is the evidence a person needs to judge an
 * answer.
 *
 * Raw tool arguments are deliberately not rendered. They are the one place an
 * uploaded value could reach the screen unfiltered, and a reader gains almost
 * nothing from them beyond the argument *names*, which is what is shown.
 * A rejected call is worth surfacing loudly: it means the planner asked for
 * something the registry does not allow, and the system said no.
 */
const STATUS_TONES: Record<string, { tone: BadgeTone; glyph: string }> = {
  ok: { tone: "good", glyph: "✓" },
  unavailable: { tone: "warn", glyph: "○" },
  rejected: { tone: "bad", glyph: "✕" },
  failed: { tone: "bad", glyph: "✕" },
};

export interface ToolTraceProps {
  toolCalls: AgentToolCall[];
  observations?: AgentObservation[];
}

export function ToolTrace({ toolCalls, observations = [] }: ToolTraceProps) {
  if (toolCalls.length === 0) {
    return (
      <p className="text-sm text-ink-600">
        The agent answered without running a tool.
      </p>
    );
  }

  const errorsByCall = new Map(
    observations
      .filter((observation) => observation.error)
      .map((observation) => [observation.call_id, observation.error as string]),
  );

  return (
    <DataTable
      caption="Tools the agent ran, in order, and how each one finished"
      head={
        <tr>
          <Th>Step</Th>
          <Th>Tool</Th>
          <Th>Status</Th>
          <Th>Arguments given</Th>
        </tr>
      }
    >
      {toolCalls.map((call, index) => {
        const status = STATUS_TONES[call.status] ?? {
          tone: "neutral" as BadgeTone,
          glyph: "•",
        };
        const argumentNames = Object.keys(call.arguments ?? {});
        const error = errorsByCall.get(call.call_id);

        return (
          <tr key={call.call_id}>
            <Th scope="row" className="normal-case tracking-normal text-ink-600">
              {index + 1}
            </Th>
            <Td className="font-mono text-xs text-ink-900">{call.tool_name}</Td>
            <Td>
              <Badge tone={status.tone} glyph={status.glyph}>
                {humanise(call.status)}
              </Badge>
              {error && (
                <span className="mt-1 block text-xs text-ink-600">{error}</span>
              )}
            </Td>
            <Td className="text-xs text-ink-600">
              {argumentNames.length > 0 ? argumentNames.join(", ") : "none"}
            </Td>
          </tr>
        );
      })}
    </DataTable>
  );
}
