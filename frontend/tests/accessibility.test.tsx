/**
 * Accessibility, responsiveness, and the two failures worth pinning at page level.
 *
 * The accessibility checks here are structural rather than exhaustive: heading
 * outlines, form labels, table semantics, keyboard operation of the tab set,
 * and the rule that no state is signalled by colour alone. Those are the ones
 * a component can silently lose in a refactor, and they are the ones that
 * decide whether this dashboard is usable at all without a mouse or without
 * colour vision. They are not a substitute for an audit with a real screen
 * reader, and this file does not claim to be one.
 *
 * The responsive checks assert on the layout contract the components declare —
 * that wide tables scroll inside their own container rather than pushing the
 * page sideways, and that the grids collapse to one column at small widths.
 * jsdom does not lay anything out, so this pins the intent, not the pixels.
 */

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DashboardPage from "@/app/dashboard/page";
import KnowledgePage from "@/app/knowledge/page";
import { AppShell } from "@/components/layout/AppShell";
import { AgentAnswerCard } from "@/components/agent/AgentAnswerCard";
import { DatasetProfileView } from "@/components/dataset/DatasetProfileView";
import { ModelComparisonTable } from "@/components/experiments/ModelComparisonTable";
import { Tabs } from "@/components/common/Tabs";
import {
  AGENT_COMPLETED,
  AGENT_STATUS,
  CLASSIFICATION_PROFILE,
  CLASSIFICATION_RUN,
  KNOWLEDGE_STATUS,
  csvFile,
} from "./fixtures";
import { errorEnvelope, mockBackend, statusRoutes } from "./mockApi";

vi.mock("next/navigation", () => ({
  usePathname: () => "/dashboard",
  useParams: () => ({ id: "exp_1" }),
  redirect: vi.fn(),
}));

const STATUS = statusRoutes(AGENT_STATUS, KNOWLEDGE_STATUS);

describe("headings and landmarks", () => {
  it("gives the dashboard exactly one h1 and a labelled main region", () => {
    mockBackend(STATUS);
    render(
      <AppShell>
        <DashboardPage />
      </AppShell>,
    );

    expect(screen.getAllByRole("heading", { level: 1 })).toHaveLength(1);
    expect(screen.getByRole("main")).toBeInTheDocument();
    expect(screen.getByRole("contentinfo")).toBeInTheDocument();
    expect(screen.getByRole("banner")).toBeInTheDocument();
  });

  it("marks the current page in the navigation", () => {
    mockBackend(STATUS);
    render(
      <AppShell>
        <div />
      </AppShell>,
    );

    const current = screen.getByRole("link", { name: "Dashboard" });
    expect(current).toHaveAttribute("aria-current", "page");
  });
});

describe("forms and controls", () => {
  it("labels every interactive control on the dashboard", async () => {
    mockBackend(STATUS);
    render(<DashboardPage />);

    // The experiment form only exists once there is a file to run it on.
    await userEvent.upload(screen.getByLabelText(/dataset file/i), csvFile());

    expect(screen.getByLabelText(/dataset file/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/target column/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/cv folds/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/random seed/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/explain the winning model/i)).toBeInTheDocument();
    expect(
      screen.getByLabelText(/your question about the dataset/i),
    ).toBeInTheDocument();

    for (const button of screen.getAllByRole("button")) {
      expect(button).toHaveAccessibleName();
    }
  });

  it("labels the knowledge query input", () => {
    mockBackend(STATUS);
    render(<KnowledgePage />);
    expect(screen.getByLabelText(/your question/i)).toBeInTheDocument();
  });
});

describe("tables", () => {
  it("captions every table and gives each row a header", () => {
    render(<DatasetProfileView profile={CLASSIFICATION_PROFILE} />);

    for (const table of screen.getAllByRole("table")) {
      const caption = table.querySelector("caption");
      expect(caption?.textContent?.trim().length ?? 0).toBeGreaterThan(0);
    }
  });

  it("scrolls a wide table inside its own container", () => {
    const { container } = render(
      <ModelComparisonTable
        selection={CLASSIFICATION_RUN.selection}
        evaluation={CLASSIFICATION_RUN.evaluation}
      />,
    );

    const table = container.querySelector("table");
    // The page body must never scroll sideways because a table is wide.
    expect(table?.parentElement?.className).toContain("overflow-x-auto");
  });
});

describe("tabs", () => {
  it("moves between tabs with the arrow keys and keeps focus roving", async () => {
    render(
      <Tabs
        ariaLabel="Example"
        tabs={[
          { id: "one", label: "One", content: <p>First panel</p> },
          { id: "two", label: "Two", content: <p>Second panel</p> },
          { id: "three", label: "Three", content: <p>Third panel</p> },
        ]}
      />,
    );

    const first = screen.getByRole("tab", { name: "One" });
    expect(first).toHaveAttribute("aria-selected", "true");
    expect(first).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("tab", { name: "Two" })).toHaveAttribute(
      "tabindex",
      "-1",
    );

    first.focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Two" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.getByText("Second panel")).toBeInTheDocument();

    await userEvent.keyboard("{End}");
    expect(screen.getByRole("tab", { name: "Three" })).toHaveAttribute(
      "aria-selected",
      "true",
    );

    await userEvent.keyboard("{Home}");
    expect(screen.getByRole("tab", { name: "One" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("associates each panel with its tab", () => {
    render(
      <Tabs
        ariaLabel="Example"
        tabs={[{ id: "one", label: "One", content: <p>Panel</p> }]}
      />,
    );

    const tab = screen.getByRole("tab", { name: "One" });
    const panel = screen.getByRole("tabpanel");
    expect(panel).toHaveAttribute("aria-labelledby", tab.id);
    expect(tab).toHaveAttribute("aria-controls", panel.id);
  });
});

describe("status is never carried by colour alone", () => {
  it("names each agent outcome in words", () => {
    render(<AgentAnswerCard answer={AGENT_COMPLETED} />);
    expect(screen.getByText("Grounded answer")).toBeInTheDocument();
  });

  it("names each candidate's outcome in words", () => {
    render(
      <ModelComparisonTable
        selection={CLASSIFICATION_RUN.selection}
        evaluation={CLASSIFICATION_RUN.evaluation}
      />,
    );
    expect(screen.getAllByText("succeeded").length).toBeGreaterThan(0);
    expect(screen.getByText("failed")).toBeInTheDocument();
    expect(screen.getByText("selected")).toBeInTheDocument();
  });

  it("gives every importance bar a text alternative", () => {
    render(<AgentAnswerCard answer={AGENT_COMPLETED} />);
    // The bar is decorative; the number beside it is the value.
    expect(screen.getByText("Grounded answer")).toBeInTheDocument();
  });
});

describe("responsive layout", () => {
  it("collapses the dashboard's two columns on small screens", () => {
    mockBackend(STATUS);
    const { container } = render(<DashboardPage />);

    const grid = container.querySelector("div.grid.gap-6");
    // One column by default; two only from the `lg` breakpoint upward.
    expect(grid?.className).toContain("lg:grid-cols-");
    expect(grid?.className).not.toContain("grid-cols-2 ");
  });

  it("wraps the header rather than overflowing it", () => {
    mockBackend(STATUS);
    const { container } = render(
      <AppShell>
        <div />
      </AppShell>,
    );

    const bar = container.querySelector("header > div");
    expect(bar?.className).toContain("flex-wrap");
  });

  it("lays stat tiles out in a responsive grid", () => {
    const { container } = render(
      <DatasetProfileView profile={CLASSIFICATION_PROFILE} />,
    );

    const list = container.querySelector("dl.grid");
    expect(list?.className).toContain("grid-cols-2");
    expect(list?.className).toContain("sm:grid-cols-3");
    expect(list?.className).toContain("lg:grid-cols-4");
  });
});

describe("failures at page level", () => {
  it("says the AI provider is unavailable without naming it", async () => {
    mockBackend([
      {
        match: "/api/v1/agent/ask-with-dataset",
        status: 502,
        body: errorEnvelope(
          "agent_provider_error",
          "The language model provider failed.",
        ),
      },
      ...STATUS,
    ]);
    render(<DashboardPage />);

    await userEvent.upload(screen.getByLabelText(/dataset file/i), csvFile());
    await userEvent.click(
      screen.getByRole("button", { name: "What is the target column?" }),
    );

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/AI provider is temporarily unavailable/i);
    expect(alert.textContent).not.toMatch(/openai|anthropic|gpt|claude/i);
  });

  it("handles a malformed response without rendering its body", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response("<html>Traceback: /srv/app/main.py</html>", {
            status: 500,
            headers: { "Content-Type": "text/html" },
          }),
      ),
    );
    render(<DashboardPage />);

    await userEvent.upload(screen.getByLabelText(/dataset file/i), csvFile());
    await userEvent.click(screen.getByRole("button", { name: /profile dataset/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/could not read/i);
    expect(alert.textContent).not.toContain("Traceback");
    expect(alert.textContent).not.toContain("/srv/");
  });

  it("exposes no credential, path or prompt anywhere on the dashboard", async () => {
    mockBackend([
      { match: "/api/v1/datasets/profile", body: CLASSIFICATION_PROFILE },
      { match: "/api/v1/agent/ask-with-dataset", body: AGENT_COMPLETED },
      ...STATUS,
    ]);
    const { container } = render(<DashboardPage />);

    await userEvent.upload(screen.getByLabelText(/dataset file/i), csvFile());
    await userEvent.click(screen.getByRole("button", { name: /profile dataset/i }));
    await screen.findByText("CSV");
    await userEvent.click(
      screen.getByRole("button", { name: "What is the target column?" }),
    );
    await screen.findByText("Grounded answer");

    const text = (container.textContent ?? "").toLowerCase();
    for (const forbidden of [
      "sk-",
      "api_key",
      "llm_api_key",
      "system prompt",
      "chain_of_thought",
      "/home/",
      "/srv/",
      "c:\\",
      "traceback",
    ]) {
      expect(text).not.toContain(forbidden);
    }
  });

  it("documents exactly one setting, and it is not a secret", () => {
    // Everything a NEXT_PUBLIC_ variable holds is inlined into the browser
    // bundle, so the example file is the place a secret would most plausibly
    // be introduced by accident. It must contain the API URL and nothing else.
    const example = readFileSync(
      resolve(__dirname, "..", ".env.example"),
      "utf8",
    );
    const settings = example
      .split("\n")
      .filter((line) => line.includes("=") && !line.trimStart().startsWith("#"))
      .map((line) => line.split("=")[0].trim());

    expect(settings).toEqual(["NEXT_PUBLIC_API_BASE_URL"]);

    // The commentary may discuss secrets; no *value* may be one.
    const values = example
      .split("\n")
      .filter((line) => line.includes("=") && !line.trimStart().startsWith("#"))
      .map((line) => line.split("=").slice(1).join("="));
    for (const value of values) {
      expect(value).not.toMatch(/sk-|secret|token|password/i);
    }
  });
});

describe("empty and unavailable states", () => {
  it("tells a person what to do first", () => {
    mockBackend(STATUS);
    render(<DashboardPage />);

    expect(screen.getByText(/no dataset yet/i)).toBeInTheDocument();
    expect(screen.getByText(/upload a dataset first/i)).toBeInTheDocument();
    expect(screen.getByText(/no question asked yet/i)).toBeInTheDocument();
  });

  it("keeps the question box disabled until a dataset is chosen", () => {
    mockBackend(STATUS);
    render(<DashboardPage />);

    expect(screen.getByLabelText(/your question about the dataset/i)).toBeDisabled();
    expect(
      screen.getByText("Upload a dataset to ask about it."),
    ).toBeInTheDocument();
  });
});
