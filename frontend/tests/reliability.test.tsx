/**
 * What the dashboard does when the backend does not cooperate.
 *
 * The happy paths and the documented error codes are covered elsewhere. This
 * file is about the responses nobody designed for: a body of the wrong shape,
 * a field that is a string where a number belongs, a status this build has
 * never seen. Those arrive in production — from a proxy, from a version skew,
 * from a partial deployment — and the difference between handling them and
 * not is the difference between a message and a blank white page.
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import ExperimentDetailPage from "@/app/experiments/[id]/page";
import RouteError from "@/app/error";
import { ExperimentSummary } from "@/components/experiments/ExperimentSummary";
import { MetricsPanel } from "@/components/experiments/MetricsPanel";
import {
  formatCount,
  formatMetric,
  formatPercent,
  formatSigned,
} from "@/lib/format";
import { CLASSIFICATION_RUN, KNOWLEDGE_STATUS, SERVICE_INFO } from "./fixtures";
import { errorEnvelope, mockBackend } from "./mockApi";

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "exp_e36e7bbf5267_20260902T054517Z_503e" }),
}));

const STATUS = [
  { exactPath: "/", body: SERVICE_INFO },
  { match: "/api/v1/knowledge/status", body: KNOWLEDGE_STATUS },
  { match: "/api/v1/agent/status", body: { available: false, reason: "no key" } },
  { match: "/model", body: { experiment_id: "x", status: "not_available", available: false, max_records: 500 } },
];

describe("formatting values that are not numbers", () => {
  /**
   * These helpers are called on values that came over the network. A field
   * that arrives as `"0.84"` or `null` type-checks clean at compile time and
   * has no `.toFixed`, and the throw happens inside a render — which takes
   * the page with it.
   */
  it.each([
    ["a string", "0.84"],
    ["null", null],
    ["undefined", undefined],
    ["an object", { value: 1 }],
    ["NaN", Number.NaN],
    ["Infinity", Number.POSITIVE_INFINITY],
  ])("renders an em dash rather than throwing for %s", (_label, value) => {
    expect(formatMetric(value as never)).toBe("—");
    expect(formatCount(value as never)).toBe("—");
    expect(formatPercent(value as never)).toBe("—");
    expect(formatSigned(value as never)).toBe("—");
  });

  it("still formats real numbers", () => {
    expect(formatMetric(0.8659)).toBe("0.8659");
    expect(formatCount(1234)).toBe("1,234");
    expect(formatSigned(-0.5)).toBe("−0.5");
  });
});

describe("a record whose fields are the wrong type", () => {
  it("renders the summary without crashing", () => {
    const damaged = {
      ...CLASSIFICATION_RUN,
      selection: {
        ...CLASSIFICATION_RUN.selection,
        selection_score: "0.85" as never,
        selection_score_std: null,
      },
      evaluation: {
        ...CLASSIFICATION_RUN.evaluation,
        primary_metric_value: null,
        test_row_count: "36" as never,
      },
    };

    render(<ExperimentSummary record={damaged} />);

    expect(screen.getByText(/Held-out/)).toBeInTheDocument();
    expect(screen.getAllByText("—").length).toBeGreaterThan(0);
  });

  it("renders the metrics panel when a metric value is a string", () => {
    const damaged = {
      ...CLASSIFICATION_RUN,
      evaluation: {
        ...CLASSIFICATION_RUN.evaluation,
        metrics: { accuracy: "high" as never, f1: 0.94 },
        baseline_metrics: {},
        baseline_comparison: {} as never,
      },
    };

    render(<MetricsPanel evaluation={damaged.evaluation} />);

    expect(screen.getByRole("rowheader", { name: /F1/ })).toBeInTheDocument();
  });
});

describe("a response that is not the shape the client expects", () => {
  it("reports a failure rather than rendering nothing", async () => {
    mockBackend([
      { match: "/api/v1/experiments/exp_", body: { unexpected: true } },
      ...STATUS,
    ]);

    render(<ExperimentDetailPage />);

    // Either an alert or the record's own heading would be acceptable; a page
    // with neither is the failure this asserts against.
    const alert = await screen.findByRole("alert");
    expect(alert).toBeInTheDocument();
  });

  it("handles an error body that is not the envelope", async () => {
    mockBackend([
      {
        match: "/api/v1/experiments/exp_",
        status: 500,
        body: "<html>502 Bad Gateway</html>",
      },
      ...STATUS,
    ]);

    render(<ExperimentDetailPage />);

    expect(await screen.findByRole("alert")).toBeInTheDocument();
  });

  it("explains a conflict distinctly from a not-found", async () => {
    mockBackend([
      {
        match: "/api/v1/experiments/exp_",
        status: 409,
        body: errorEnvelope(
          "model_not_available",
          "This run has no usable model.",
        ),
      },
      ...STATUS,
    ]);

    render(<ExperimentDetailPage />);

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).not.toMatch(/not found/i);
  });

  it("explains an oversized request rather than showing a raw status", async () => {
    mockBackend([
      {
        match: "/api/v1/experiments/exp_",
        status: 413,
        body: errorEnvelope(
          "request_body_too_large",
          "The request body is larger than this service accepts.",
        ),
      },
      ...STATUS,
    ]);

    render(<ExperimentDetailPage />);

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toBeTruthy();
    expect(alert.textContent).not.toMatch(/undefined|\[object Object\]/);
  });
});

describe("the route error boundary", () => {
  /**
   * The boundary is the last line: if a component throws anyway, this is what
   * the reader sees instead of nothing.
   */
  it("shows an authored message and never the thrown text", () => {
    const error = Object.assign(
      new Error("TypeError: cannot read property 'toFixed' of undefined"),
      { digest: "abc123" },
    );
    const reset = vi.fn();
    vi.spyOn(console, "error").mockImplementation(() => undefined);

    render(<RouteError error={error} reset={reset} />);

    expect(
      screen.getByText(/this page could not be displayed/i),
    ).toBeInTheDocument();
    expect(screen.getByText(/abc123/)).toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/toFixed/);
    expect(
      screen.getByRole("button", { name: /try again/i }),
    ).toBeInTheDocument();
  });
});
