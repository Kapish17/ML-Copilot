"use client";

import { useEffect, useState } from "react";
import { Badge } from "@/components/common/Badge";
import { agentStatus } from "@/lib/api/agent";
import { knowledgeStatus } from "@/lib/api/knowledge";
import { serviceInfo } from "@/lib/api/system";
import type {
  AgentStatusResponse,
  KnowledgeStatus,
  ServiceInfo,
} from "@/lib/api/types";

/**
 * Whether the parts of the system a person is about to use are actually up.
 *
 * Worth its space because three of them can be down independently and for
 * ordinary reasons: nothing indexed yet, no language-model credential
 * configured on the server. Saying so here is much kinder than letting
 * someone type a question and get a 503.
 *
 * It reports availability only. It never shows a provider name, a model, a
 * key, a path or anything else about how the server is configured.
 *
 * The fourth thing it reports is authentication, and that one is not about
 * availability but about honesty. **This dashboard holds no API key and
 * cannot hold one** — it runs in a browser, so anything shipped with it is
 * readable by every visitor and would not be a secret. So when the backend
 * says it is protected, the header says so immediately, rather than letting
 * someone upload a dataset and meet a 401 they have no way to satisfy.
 */
export function SystemStatus() {
  const [agent, setAgent] = useState<AgentStatusResponse | null>(null);
  const [knowledge, setKnowledge] = useState<KnowledgeStatus | null>(null);
  const [service, setService] = useState<ServiceInfo | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    // All three endpoints are public on every deployment, so this check
    // itself never needs a credential — which is what lets it report that
    // one is needed.
    Promise.all([
      agentStatus({ signal: controller.signal }),
      knowledgeStatus({ signal: controller.signal }),
      serviceInfo({ signal: controller.signal }),
    ])
      .then(([agentValue, knowledgeValue, serviceValue]) => {
        setAgent(agentValue);
        setKnowledge(knowledgeValue);
        setService(serviceValue);
      })
      .catch(() => {
        if (!controller.signal.aborted) setFailed(true);
      });
    return () => controller.abort();
  }, []);

  if (failed) {
    return (
      <p className="text-xs text-ink-500" role="status">
        Backend unreachable
      </p>
    );
  }

  if (!agent || !knowledge || !service) {
    return (
      <p className="text-xs text-ink-400" role="status">
        Checking services…
      </p>
    );
  }

  const items: Array<{ label: string; ok: boolean; detail: string }> = [
    {
      label: "RAG",
      ok: knowledge.search_available && knowledge.index_built,
      detail: knowledge.index_built ? "ready" : "not indexed",
    },
    {
      label: "LLM",
      ok: knowledge.answering_available,
      detail: knowledge.answering_available ? "ready" : "not configured",
    },
    {
      label: "Agent",
      ok: agent.agent_available,
      detail: agent.agent_available ? "ready" : "not configured",
    },
  ];

  return (
    <div
      className="flex flex-wrap items-center gap-2"
      aria-label="System status"
      role="group"
    >
      {items.map((item) => (
        <Badge
          key={item.label}
          tone={item.ok ? "good" : "warn"}
          glyph={item.ok ? "●" : "○"}
        >
          <span className="font-semibold">{item.label}</span>
          <span className="font-normal text-ink-600">{item.detail}</span>
        </Badge>
      ))}
      <Badge tone="neutral">
        Formats{" "}
        <span className="font-normal">
          {(agent.supported_dataset_formats ?? []).join(" · ").toUpperCase() ||
            "—"}
        </span>
      </Badge>
      {service.authentication_required && (
        <Badge tone="warn" glyph="◆">
          <span className="font-semibold">API key required</span>
          <span className="font-normal text-ink-600">
            this dashboard cannot hold one
          </span>
        </Badge>
      )}
    </div>
  );
}
