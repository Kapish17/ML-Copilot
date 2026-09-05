/**
 * Tests for the AI Data Scientist's answer card and citations.
 *
 * Two claims are checked hardest. First, that an answer's *status* is visible
 * and worded — all four outcomes arrive as HTTP 200 and only `completed` is
 * safe to act on, so a card that showed the prose without the verdict would be
 * actively misleading. Second, that nothing the backend refuses to send is
 * reconstructed here: no chain-of-thought, no prompt, no provider, no raw
 * tool arguments.
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { AgentAnswerCard } from "@/components/agent/AgentAnswerCard";
import { AgentAsk } from "@/components/agent/AgentAsk";
import { CitationList } from "@/components/agent/CitationList";
import { ToolTrace } from "@/components/agent/ToolTrace";
import { resolveCitation, experimentIdFromCitation } from "@/lib/citations";
import {
  AGENT_COMPLETED,
  AGENT_GROUNDING_FAILED,
  AGENT_INSUFFICIENT,
  AGENT_LOCAL_EXPLANATION,
  AGENT_PARTIAL,
  AGENT_PLANNED,
  AGENT_PLANNED_PARTIAL,
  AGENT_REJECTED_TOOL,
} from "./fixtures";

describe("citation resolution", () => {
  it("extracts an experiment id and builds an in-app route", () => {
    expect(experimentIdFromCitation("exp:exp_abc123")).toBe("exp_abc123");
    const resolved = resolveCitation("exp:exp_abc123");
    expect(resolved.kind).toBe("experiment");
    expect(resolved.href).toBe("/experiments/exp_abc123");
  });

  it("never invents a URL for a documentation citation", () => {
    const resolved = resolveCitation("docs:readme#not-implemented");
    expect(resolved.kind).toBe("documentation");
    expect(resolved.href).toBeUndefined();
  });
});

describe("citation list", () => {
  it("links an experiment citation and labels a documentation one", () => {
    render(<CitationList citations={AGENT_COMPLETED.citations} />);

    expect(
      screen.getByRole("link", { name: /customers.csv · renewed/ }),
    ).toHaveAttribute(
      "href",
      "/experiments/exp_e36e7bbf5267_20260902T054517Z_503e",
    );
    expect(screen.getByText("ML Copilot — ML Layer")).toBeInTheDocument();
    expect(screen.getByText("ml/README.md")).toBeInTheDocument();
    // The documentation source is text, not a dead link.
    expect(
      screen.queryByRole("link", { name: "ML Copilot — ML Layer" }),
    ).not.toBeInTheDocument();
  });

  it("shows refused citations and says why they were refused", () => {
    render(
      <CitationList citations={[]} rejected={["docs:secret-internal#nope"]} />,
    );

    expect(screen.getByText("Rejected citations")).toBeInTheDocument();
    expect(screen.getByText("docs:secret-internal#nope")).toBeInTheDocument();
    expect(
      screen.getByText(/refused rather than repaired/i),
    ).toBeInTheDocument();
  });
});

describe("tool trace", () => {
  it("names each tool and how it finished", () => {
    render(<ToolTrace toolCalls={AGENT_COMPLETED.tool_calls} />);

    expect(screen.getByText("dataset_profile")).toBeInTheDocument();
    expect(screen.getByText("run_experiment")).toBeInTheDocument();
    expect(screen.getAllByText("Ok").length).toBe(3);
  });

  it("shows a rejected tool call rather than hiding it", () => {
    render(<ToolTrace toolCalls={AGENT_REJECTED_TOOL.tool_calls} />);

    expect(screen.getByText("shell")).toBeInTheDocument();
    expect(screen.getByText("Rejected")).toBeInTheDocument();
  });

  it("lists argument names but never their values", () => {
    render(<ToolTrace toolCalls={AGENT_COMPLETED.tool_calls} />);

    expect(screen.getAllByText(/dataset, target_column/).length).toBeGreaterThan(0);
    // The value "uploaded_dataset" is an argument value, not a name.
    expect(screen.queryByText(/uploaded_dataset/)).not.toBeInTheDocument();
  });
});

describe("agent answer card", () => {
  it("leads with a grounded verdict and shows the answer", () => {
    render(<AgentAnswerCard answer={AGENT_COMPLETED} />);

    expect(screen.getByText("Grounded answer")).toBeInTheDocument();
    expect(
      screen.getByText(/every citation checked out/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/Logistic regression was selected/)).toBeInTheDocument();
  });

  it("warns that a partial answer is incomplete", () => {
    render(<AgentAnswerCard answer={AGENT_PARTIAL} />);

    expect(screen.getByText("Partial")).toBeInTheDocument();
    expect(screen.getByText(/read the answer as incomplete/i)).toBeInTheDocument();
    expect(
      screen.getByText(/fitted model for that experiment is not persisted/i),
    ).toBeInTheDocument();
  });

  it("says plainly when there was not enough evidence", () => {
    render(<AgentAnswerCard answer={AGENT_INSUFFICIENT} />);

    expect(screen.getByText("Insufficient evidence")).toBeInTheDocument();
    expect(screen.getByText(/rather than guessing/i)).toBeInTheDocument();
  });

  it("tells the reader not to trust an answer that failed grounding", () => {
    render(<AgentAnswerCard answer={AGENT_GROUNDING_FAILED} />);

    expect(screen.getByText("Not grounded")).toBeInTheDocument();
    expect(
      screen.getByText(/Do not treat the text below as a finding/i),
    ).toBeInTheDocument();
    expect(screen.getByText("docs:secret-internal#nope")).toBeInTheDocument();
  });

  it("links every experiment the run created", () => {
    render(<AgentAnswerCard answer={AGENT_COMPLETED} />);

    expect(
      screen.getByRole("link", {
        name: "exp_e36e7bbf5267_20260902T054517Z_503e",
      }),
    ).toHaveAttribute(
      "href",
      "/experiments/exp_e36e7bbf5267_20260902T054517Z_503e",
    );
  });

  it("reports the dataset's facts and that it was not persisted", async () => {
    render(<AgentAnswerCard answer={AGENT_COMPLETED} />);

    await userEvent.click(screen.getByRole("tab", { name: /Run/ }));

    expect(screen.getByText("customers.csv")).toBeInTheDocument();
    expect(screen.getByText("9d610b7e1abef86c")).toBeInTheDocument();
    expect(screen.getByText("180 rows × 4 columns")).toBeInTheDocument();
    expect(
      screen.getByText(/not persisted — held in memory for this request only/i),
    ).toBeInTheDocument();
  });

  it("renders a global explanation the agent produced", async () => {
    render(<AgentAnswerCard answer={AGENT_COMPLETED} />);

    await userEvent.click(screen.getByRole("tab", { name: /Explanation/ }));

    expect(screen.getByText("Global feature importance")).toBeInTheDocument();
    expect(screen.getByText("income")).toBeInTheDocument();
    expect(
      screen.getByText(/not causation/i),
    ).toBeInTheDocument();
  });

  it("renders a local explanation when the run produced one", async () => {
    render(<AgentAnswerCard answer={AGENT_LOCAL_EXPLANATION} />);

    await userEvent.click(screen.getByRole("tab", { name: /Explanation/ }));

    expect(screen.getByText("This prediction")).toBeInTheDocument();
    const row = screen
      .getByRole("rowheader", { name: "income" })
      .closest("tr") as HTMLElement;
    expect(within(row).getByText("+0.31")).toBeInTheDocument();
    expect(within(row).getByText("increases prediction")).toBeInTheDocument();
  });

  it("exposes no reasoning, prompt, provider or credential", () => {
    const { container } = render(<AgentAnswerCard answer={AGENT_COMPLETED} />);
    const text = container.textContent ?? "";

    for (const forbidden of [
      "chain_of_thought",
      "chain of thought",
      "system prompt",
      "system_prompt",
      "provider",
      "api_key",
      "sk-",
    ]) {
      expect(text.toLowerCase()).not.toContain(forbidden.toLowerCase());
    }
  });
});

describe("asking a question", () => {
  it("submits the typed question", async () => {
    const onAsk = vi.fn();
    render(
      <AgentAsk
        onAsk={onAsk}
        busy={false}
        busyLabel="Thinking…"
        answer={null}
        error={null}
      />,
    );

    await userEvent.type(
      screen.getByLabelText(/your question about the dataset/i),
      "Which model is best?",
    );
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(onAsk).toHaveBeenCalledWith("Which model is best?");
  });

  it("offers concrete suggestions and asks one on click", async () => {
    const onAsk = vi.fn();
    render(
      <AgentAsk
        onAsk={onAsk}
        busy={false}
        busyLabel="Thinking…"
        answer={null}
        error={null}
      />,
    );

    await userEvent.click(
      screen.getByRole("button", { name: "What is the target column?" }),
    );
    expect(onAsk).toHaveBeenCalledWith("What is the target column?");
  });

  it("blocks and explains itself when there is no dataset", () => {
    render(
      <AgentAsk
        onAsk={vi.fn()}
        busy={false}
        busyLabel="Thinking…"
        answer={null}
        error={null}
        disabled
        disabledReason="Upload a dataset to ask about it."
      />,
    );

    expect(screen.getByLabelText(/your question/i)).toBeDisabled();
    expect(
      screen.getByText("Upload a dataset to ask about it."),
    ).toBeInTheDocument();
  });

  it("announces the stage it is at while working", () => {
    render(
      <AgentAsk
        onAsk={vi.fn()}
        busy
        busyLabel="Thinking… the agent may profile, train and explain"
        answer={null}
        error={null}
      />,
    );

    const status = screen.getByRole("status");
    expect(status).toHaveTextContent(/Thinking…/);
    expect(status).toHaveAttribute("aria-live", "polite");
    expect(screen.getByText(/Runs are synchronous/i)).toBeInTheDocument();
  });

  it("shows an empty state before anything is asked", () => {
    render(
      <AgentAsk
        onAsk={vi.fn()}
        busy={false}
        busyLabel="Thinking…"
        answer={null}
        error={null}
      />,
    );
    expect(screen.getByText(/no question asked yet/i)).toBeInTheDocument();
  });
});

describe("the planned workflow", () => {
  it("shows what the agent was going to do, in order", () => {
    render(<AgentAnswerCard answer={AGENT_PLANNED} />);

    const plan = screen.getByRole("region", { name: /planned workflow/i });
    const items = within(plan).getAllByRole("listitem");

    expect(items.map((item) => item.textContent)).toEqual([
      expect.stringContaining("Profile the uploaded dataset"),
      expect.stringContaining("Compare models"),
      expect.stringContaining("Explain the winning model"),
    ]);
  });

  it("says how far a completed plan got", () => {
    render(<AgentAnswerCard answer={AGENT_PLANNED} />);

    expect(screen.getByText("3 of 3 steps")).toBeInTheDocument();
  });

  it("says how far a half-finished plan got, and why it stopped", () => {
    // The point of showing a plan at all: an answer covering less ground than
    // the question asked for should say so before anyone reads the prose.
    render(<AgentAnswerCard answer={AGENT_PLANNED_PARTIAL} />);

    expect(screen.getByText("1 of 3 steps")).toBeInTheDocument();
    expect(screen.getByText(/not run/i)).toBeInTheDocument();
    expect(
      screen.getByText(/needed the result of step-2/i),
    ).toBeInTheDocument();
  });

  it("shows nothing about planning when the run was not planned", () => {
    // Every run looked like this before plans existed, and a client that meets
    // one must not render an empty plan.
    render(<AgentAnswerCard answer={AGENT_COMPLETED} />);

    expect(screen.queryByText(/planned workflow/i)).toBeNull();
  });

  it("renders no step arguments and no reasoning", () => {
    // The API does not send a step's arguments, and this asserts the card does
    // not find them somewhere else either. `uploaded_dataset` appears in the
    // tool trace's argument *names* — the plan section must not carry values.
    render(<AgentAnswerCard answer={AGENT_PLANNED} />);
    const plan = screen.getByRole("region", { name: /planned workflow/i });

    expect(plan.textContent).not.toMatch(/renewed/);
    expect(plan.textContent).not.toMatch(/chain of thought|reasoning|because/i);
  });

  it("counts planned steps rather than planning turns on the run tab", async () => {
    render(<AgentAnswerCard answer={AGENT_PLANNED} />);

    await userEvent.click(screen.getByRole("tab", { name: /run/i }));

    expect(screen.getByText(/3 of 3 planned steps/i)).toBeInTheDocument();
  });
});
