/**
 * End-to-end tests of the pages, against a mocked backend.
 *
 * These exercise the workflow a person actually performs — pick a file,
 * profile it, ask a question, read the answer — and the parts of it that only
 * appear when the pieces are wired together: that the loading state announces
 * itself, that a failure is shown in mapped language, that choosing a new file
 * clears results derived from the old one, and that nothing about the dataset
 * is written to browser storage on the way through.
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import DashboardPage from "@/app/dashboard/page";
import ExperimentsPage from "@/app/experiments/page";
import KnowledgePage from "@/app/knowledge/page";
import ExperimentDetailPage from "@/app/experiments/[id]/page";
import { AppShell } from "@/components/layout/AppShell";
import {
  AGENT_COMPLETED,
  AGENT_STATUS,
  ASK_GROUNDED,
  CLASSIFICATION_PROFILE,
  CLASSIFICATION_RUN,
  COMPARISON,
  EXPERIMENT_LIST,
  JSON_PROFILE,
  KNOWLEDGE_STATUS,
  SEARCH_RESPONSE,
  SERVICE_INFO_AUTHENTICATED,
  XLSX_PROFILE,
  csvFile,
  jsonFile,
  xlsxFile,
} from "./fixtures";
import { errorEnvelope, mockBackend, statusRoutes } from "./mockApi";

// Next's navigation hooks are not available outside a Next runtime, so they
// are stubbed at the module boundary rather than by rendering a whole router.
const routeParams = { id: CLASSIFICATION_RUN.experiment_id };
vi.mock("next/navigation", () => ({
  usePathname: () => "/dashboard",
  useParams: () => routeParams,
  redirect: vi.fn(),
}));

const STATUS = statusRoutes(AGENT_STATUS, KNOWLEDGE_STATUS);

describe("application shell", () => {
  it("renders the product name, navigation and a main landmark", async () => {
    mockBackend(STATUS);
    render(
      <AppShell>
        <h1>Content</h1>
      </AppShell>,
    );

    expect(screen.getByText("ML Copilot")).toBeInTheDocument();
    expect(screen.getByText("AI Data Scientist")).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Primary" })).toBeInTheDocument();
    expect(screen.getByRole("main")).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /skip to main content/i }),
    ).toBeInTheDocument();
    // The causation disclaimer is in the footer of every page.
    expect(screen.getByText(/not causation/i)).toBeInTheDocument();
  });

  it("shows RAG, LLM, agent and format availability", async () => {
    mockBackend(STATUS);
    render(
      <AppShell>
        <div />
      </AppShell>,
    );

    const group = await screen.findByRole("group", { name: /system status/i });
    expect(within(group).getByText("RAG")).toBeInTheDocument();
    expect(within(group).getByText("LLM")).toBeInTheDocument();
    expect(within(group).getByText("Agent")).toBeInTheDocument();
    expect(within(group).getByText("CSV · XLSX · JSON")).toBeInTheDocument();
  });

  it("says a capability is unavailable in words, not only in colour", async () => {
    mockBackend(
      statusRoutes(
        { ...AGENT_STATUS, agent_available: false },
        { ...KNOWLEDGE_STATUS, index_built: false, answering_available: false },
      ),
    );
    render(
      <AppShell>
        <div />
      </AppShell>,
    );

    const group = await screen.findByRole("group", { name: /system status/i });
    expect(within(group).getByText("not indexed")).toBeInTheDocument();
    expect(within(group).getAllByText("not configured")).toHaveLength(2);
  });

  it("says nothing about authentication when the backend needs none", async () => {
    mockBackend(statusRoutes(AGENT_STATUS, KNOWLEDGE_STATUS));
    render(
      <AppShell>
        <div />
      </AppShell>,
    );

    const group = await screen.findByRole("group", { name: /system status/i });
    expect(within(group).queryByText(/API key required/i)).toBeNull();
  });

  it("admits it cannot use a protected backend rather than pretending", async () => {
    // The honest state. This dashboard holds no API key and must not — a
    // browser bundle is readable by every visitor, so anything shipped in it
    // would not be a secret. Saying so in the header is much better than
    // letting someone upload a dataset and meet a 401 they cannot satisfy.
    mockBackend(
      statusRoutes(AGENT_STATUS, KNOWLEDGE_STATUS, SERVICE_INFO_AUTHENTICATED),
    );
    render(
      <AppShell>
        <div />
      </AppShell>,
    );

    const group = await screen.findByRole("group", { name: /system status/i });
    expect(within(group).getByText(/API key required/i)).toBeInTheDocument();
    expect(
      within(group).getByText(/this dashboard cannot hold one/i),
    ).toBeInTheDocument();
  });

  it("reports an unreachable backend rather than pretending", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("Failed to fetch");
      }),
    );
    render(
      <AppShell>
        <div />
      </AppShell>,
    );

    expect(await screen.findByText(/backend unreachable/i)).toBeInTheDocument();
  });
});

describe("dashboard", () => {
  it("introduces the workflow and starts from an empty state", () => {
    mockBackend(STATUS);
    render(<DashboardPage />);

    expect(
      screen.getByRole("heading", { level: 1, name: /AI Data Scientist/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Upload → Analyse → Experiment → Explain/),
    ).toBeInTheDocument();
    expect(screen.getByText(/no dataset yet/i)).toBeInTheDocument();
  });

  it.each([
    ["CSV", csvFile(), CLASSIFICATION_PROFILE, "CSV"],
    ["Excel", xlsxFile(), XLSX_PROFILE, "XLSX"],
    ["JSON", jsonFile(), JSON_PROFILE, "JSON"],
  ])("profiles a %s upload through the one endpoint", async (
    _label,
    file,
    profile,
    badge,
  ) => {
    const backend = mockBackend([
      { match: "/api/v1/datasets/profile", body: profile },
      ...STATUS,
    ]);
    render(<DashboardPage />);

    await userEvent.upload(screen.getByLabelText(/dataset file/i), file);
    await userEvent.click(screen.getByRole("button", { name: /profile dataset/i }));

    expect(await screen.findByText(badge)).toBeInTheDocument();
    expect(screen.getByText("Rows", { selector: "dt" })).toBeInTheDocument();

    // One endpoint for every format — no per-format route.
    const profileCalls = backend.requests.filter((request) =>
      request.url.includes("/api/v1/datasets/profile"),
    );
    expect(profileCalls).toHaveLength(1);
  });

  it("announces each stage while a request is open", async () => {
    mockBackend([
      { match: "/api/v1/datasets/profile", body: CLASSIFICATION_PROFILE, delayMs: 40 },
      ...STATUS,
    ]);
    render(<DashboardPage />);

    await userEvent.upload(screen.getByLabelText(/dataset file/i), csvFile());
    await userEvent.click(screen.getByRole("button", { name: /profile dataset/i }));

    const status = await screen.findByText(/profiling dataset/i);
    expect(status.closest("[role='status']")).toHaveAttribute(
      "aria-live",
      "polite",
    );
    await screen.findByText("CSV");
  });

  it("maps a backend failure to friendly language and offers a retry", async () => {
    mockBackend([
      {
        match: "/api/v1/datasets/profile",
        status: 422,
        body: errorEnvelope(
          "invalid_excel",
          "The workbook could not be opened. It may be corrupted.",
        ),
      },
      ...STATUS,
    ]);
    render(<DashboardPage />);

    await userEvent.upload(screen.getByLabelText(/dataset file/i), xlsxFile());
    await userEvent.click(screen.getByRole("button", { name: /profile dataset/i }));

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/Excel file could not be read/i);
    expect(alert).toHaveTextContent("invalid_excel");
    expect(
      within(alert).getByRole("button", { name: /try again/i }),
    ).toBeInTheDocument();
  });

  it("runs the agent and shows the answer with its citations", async () => {
    mockBackend([
      { match: "/api/v1/agent/ask-with-dataset", body: AGENT_COMPLETED },
      ...STATUS,
    ]);
    render(<DashboardPage />);

    await userEvent.upload(screen.getByLabelText(/dataset file/i), csvFile());
    await userEvent.click(
      screen.getByRole("button", { name: "Which model performs best and why?" }),
    );

    expect(await screen.findByText("Grounded answer")).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /customers.csv · renewed/ }),
    ).toHaveAttribute(
      "href",
      "/experiments/exp_e36e7bbf5267_20260902T054517Z_503e",
    );
  });

  it("runs an experiment and separates CV from the test measurement", async () => {
    mockBackend([
      { match: "/api/v1/experiments/run", body: CLASSIFICATION_RUN },
      ...STATUS,
    ]);
    render(<DashboardPage />);

    await userEvent.upload(screen.getByLabelText(/dataset file/i), csvFile());
    await userEvent.click(screen.getByRole("button", { name: /run experiment/i }));

    expect(await screen.findByText(/4 · Experiment result/)).toBeInTheDocument();
    expect(
      screen.getByRole("columnheader", { name: /Final · untouched test set/i }),
    ).toBeInTheDocument();
    expect(screen.getByText("selected")).toBeInTheDocument();
  });

  it("clears results derived from the previous file when a new one is chosen", async () => {
    mockBackend([
      { match: "/api/v1/datasets/profile", body: CLASSIFICATION_PROFILE },
      ...STATUS,
    ]);
    render(<DashboardPage />);

    const input = screen.getByLabelText(/dataset file/i);
    await userEvent.upload(input, csvFile());
    await userEvent.click(screen.getByRole("button", { name: /profile dataset/i }));
    await screen.findByText("CSV");

    await userEvent.upload(input, jsonFile());

    // A stale profile beside a new upload is how someone reads the wrong result.
    await waitFor(() =>
      expect(screen.getByText(/not profiled yet/i)).toBeInTheDocument(),
    );
  });

  it("writes nothing about the dataset to browser storage", async () => {
    mockBackend([
      { match: "/api/v1/datasets/profile", body: CLASSIFICATION_PROFILE },
      { match: "/api/v1/agent/ask-with-dataset", body: AGENT_COMPLETED },
      ...STATUS,
    ]);
    render(<DashboardPage />);

    await userEvent.upload(screen.getByLabelText(/dataset file/i), csvFile());
    await userEvent.click(screen.getByRole("button", { name: /profile dataset/i }));
    await screen.findByText("CSV");

    expect(window.localStorage.length).toBe(0);
    expect(window.sessionStorage.length).toBe(0);
  });

  it("keeps the file out of the URL", async () => {
    mockBackend([
      { match: "/api/v1/datasets/profile", body: CLASSIFICATION_PROFILE },
      ...STATUS,
    ]);
    render(<DashboardPage />);

    await userEvent.upload(screen.getByLabelText(/dataset file/i), csvFile());
    await userEvent.click(screen.getByRole("button", { name: /profile dataset/i }));
    await screen.findByText("CSV");

    expect(window.location.search).toBe("");
    expect(window.location.hash).toBe("");
  });
});

describe("experiments page", () => {
  it("lists stored runs and links each to its detail page", async () => {
    mockBackend([{ match: "/api/v1/experiments", body: EXPERIMENT_LIST }, ...STATUS]);
    render(<ExperimentsPage />);

    expect(
      await screen.findByRole("link", { name: /customers.csv · renewed/ }),
    ).toHaveAttribute(
      "href",
      "/experiments/exp_e36e7bbf5267_20260902T054517Z_503e",
    );
  });

  it("shows an empty state when nothing has been run", async () => {
    mockBackend([
      { match: "/api/v1/experiments", body: { count: 0, limit: 50, experiments: [] } },
      ...STATUS,
    ]);
    render(<ExperimentsPage />);

    expect(
      await screen.findByText(/no experiments stored yet/i),
    ).toBeInTheDocument();
  });

  it("compares selected runs through the backend", async () => {
    mockBackend([
      { match: "/api/v1/experiments/compare", body: COMPARISON },
      { match: "/api/v1/experiments", body: EXPERIMENT_LIST },
      ...STATUS,
    ]);
    render(<ExperimentsPage />);

    const boxes = await screen.findAllByRole("checkbox");
    await userEvent.click(boxes[0]);
    await userEvent.click(boxes[1]);
    await userEvent.click(screen.getByRole("button", { name: /compare selected/i }));

    expect(await screen.findByText("best")).toBeInTheDocument();
    expect(screen.getByText("Higher is better")).toBeInTheDocument();
  });

  it("refuses to enable comparison for a single run", async () => {
    mockBackend([{ match: "/api/v1/experiments", body: EXPERIMENT_LIST }, ...STATUS]);
    render(<ExperimentsPage />);

    const boxes = await screen.findAllByRole("checkbox");
    await userEvent.click(boxes[0]);
    expect(screen.getByRole("button", { name: /compare selected/i })).toBeDisabled();
  });

  it("explains a comparison the backend rejected", async () => {
    mockBackend([
      {
        match: "/api/v1/experiments/compare",
        status: 409,
        body: errorEnvelope("incomparable_experiments", "Different metrics."),
      },
      { match: "/api/v1/experiments", body: EXPERIMENT_LIST },
      ...STATUS,
    ]);
    render(<ExperimentsPage />);

    const boxes = await screen.findAllByRole("checkbox");
    await userEvent.click(boxes[0]);
    await userEvent.click(boxes[1]);
    await userEvent.click(screen.getByRole("button", { name: /compare selected/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /do not share a task and a metric/i,
    );
  });

  it("reports a failure to load the history and offers a retry", async () => {
    mockBackend([
      {
        match: "/api/v1/experiments",
        status: 500,
        body: errorEnvelope("internal_error", "…"),
      },
      ...STATUS,
    ]);
    render(<ExperimentsPage />);

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent(/went wrong on the server/i);
  });
});

describe("experiment detail page", () => {
  it("renders the stored record in full", async () => {
    mockBackend([
      { match: "/api/v1/experiments/exp_", body: CLASSIFICATION_RUN },
      ...STATUS,
    ]);
    render(<ExperimentDetailPage />);

    expect(await screen.findByText("customers.csv · renewed")).toBeInTheDocument();
    expect(screen.getByText("9d610b7e1abef86c")).toBeInTheDocument();
    expect(
      screen.getByRole("link", { name: /all experiments/i }),
    ).toHaveAttribute("href", "/experiments");
  });

  it("says an experiment was not found in plain language", async () => {
    mockBackend([
      {
        match: "/api/v1/experiments/exp_",
        status: 404,
        body: errorEnvelope("experiment_not_found", "No such run."),
      },
      ...STATUS,
    ]);
    render(<ExperimentDetailPage />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /No experiment is stored under that id/i,
    );
  });

  it("shows the preparation and says no model was persisted", async () => {
    mockBackend([
      { match: "/api/v1/experiments/exp_", body: CLASSIFICATION_RUN },
      ...STATUS,
    ]);
    render(<ExperimentDetailPage />);

    await screen.findByText("customers.csv · renewed");
    await userEvent.click(screen.getByRole("tab", { name: /Data & preparation/ }));

    expect(screen.getByText(/144 train · 36 test/)).toBeInTheDocument();
    // The record still holds no data. What changed in Commit 22 is that the
    // fitted model is now kept — separately, and holding coefficients rather
    // than rows — so the page says that instead of the old blanket claim.
    expect(screen.getByText(/contains no dataset rows/i)).toBeInTheDocument();
    expect(
      screen.getByText(/learned coefficients, not data/i),
    ).toBeInTheDocument();
  });
});

describe("knowledge page", () => {
  it("distinguishes itself from the AI Data Scientist", async () => {
    mockBackend(STATUS);
    render(<KnowledgePage />);

    expect(
      screen.getByRole("heading", { level: 1, name: /knowledge assistant/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/runs no tools, trains\s+nothing and never sees a dataset/i),
    ).toBeInTheDocument();
  });

  it("searches and shows passages with their scores and sources", async () => {
    mockBackend([{ match: "/api/v1/search", body: SEARCH_RESPONSE }, ...STATUS]);
    render(<KnowledgePage />);

    await userEvent.click(
      screen.getByRole("button", { name: "What is cross-validation?" }),
    );

    expect(await screen.findByText(/naive baseline plays no part/)).toBeInTheDocument();
    expect(screen.getByText("ML Copilot — ML Layer")).toBeInTheDocument();
    expect(screen.getByText(/score 0.3817/)).toBeInTheDocument();
  });

  it("never exposes embedding or vector internals", async () => {
    mockBackend([{ match: "/api/v1/search", body: SEARCH_RESPONSE }, ...STATUS]);
    const { container } = render(<KnowledgePage />);

    await userEvent.click(
      screen.getByRole("button", { name: "What is cross-validation?" }),
    );
    await screen.findByText(/naive baseline plays no part/);

    const text = (container.textContent ?? "").toLowerCase();
    for (const forbidden of ["embedding", "vector", "dimension", "hashingvectorizer"]) {
      expect(text).not.toContain(forbidden);
    }
  });

  it("returns a grounded answer with its citations", async () => {
    mockBackend([{ match: "/api/v1/ask", body: ASK_GROUNDED }, ...STATUS]);
    render(<KnowledgePage />);

    await userEvent.type(screen.getByLabelText(/your question/i), "What is CV?");
    await userEvent.click(
      screen.getByRole("button", { name: /get a grounded answer/i }),
    );

    expect(await screen.findByText("Grounded")).toBeInTheDocument();
    expect(screen.getByText("ML Copilot — ML Layer")).toBeInTheDocument();
    expect(screen.getByText(/5 passages retrieved/)).toBeInTheDocument();
  });

  it("disables answering when the server has no credential", async () => {
    mockBackend(
      statusRoutes(AGENT_STATUS, {
        ...KNOWLEDGE_STATUS,
        answering_available: false,
      }),
    );
    render(<KnowledgePage />);

    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /get a grounded answer/i }),
      ).toBeDisabled(),
    );
    expect(
      screen.getByText(/the server has no language-model credential configured/i),
    ).toBeInTheDocument();
  });

  it("explains an unbuilt index rather than showing a raw code", async () => {
    mockBackend([
      {
        match: "/api/v1/search",
        status: 503,
        body: errorEnvelope("retrieval_index_not_built", "Index missing."),
      },
      ...STATUS,
    ]);
    render(<KnowledgePage />);

    await userEvent.click(
      screen.getByRole("button", { name: "What is cross-validation?" }),
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(
      /Knowledge search is currently unavailable/i,
    );
  });

  it("shows an empty state before anything is searched", () => {
    mockBackend(STATUS);
    render(<KnowledgePage />);
    expect(screen.getByText(/nothing searched yet/i)).toBeInTheDocument();
  });
});
