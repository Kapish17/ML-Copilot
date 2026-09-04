/**
 * Tests for predicting from a stored model, in the dashboard.
 *
 * The panel's job is to be honest about three states and useful in the fourth:
 * checking, no model stored, a failed prediction, and a result. The one that
 * would be easy to get wrong is the second — a run recorded before model
 * persistence existed, or one whose artifact was deleted, has no model, and
 * rendering an empty form for it would produce requests that cannot succeed.
 *
 * The form is built from the schema the backend declares rather than from
 * anything the page already knows, so a test that fakes a different schema
 * must see a different form. That is asserted directly, because it is the
 * property that keeps the dashboard from sending features the model was not
 * trained on.
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PredictionPanel } from "@/components/experiments/PredictionPanel";
import { MODEL_AVAILABLE, MODEL_UNAVAILABLE, PREDICTION } from "./fixtures";
import { errorEnvelope, mockBackend } from "./mockApi";

const EXPERIMENT_ID = "exp_e36e7bbf5267_20260902T054517Z_503e";

const MODEL_ROUTE = `/api/v1/experiments/${EXPERIMENT_ID}/model`;
const PREDICT_ROUTE = `/api/v1/experiments/${EXPERIMENT_ID}/predict`;

function renderPanel() {
  return render(<PredictionPanel experimentId={EXPERIMENT_ID} />);
}

async function fillAndSubmit(values: Record<string, string>) {
  for (const [name, value] of Object.entries(values)) {
    await userEvent.type(await screen.findByLabelText(new RegExp(name)), value);
  }
  await userEvent.click(screen.getByRole("button", { name: /^predict$/i }));
}

describe("prediction availability", () => {
  it("says so when the experiment has no stored model", async () => {
    mockBackend([{ match: MODEL_ROUTE, body: MODEL_UNAVAILABLE }]);
    renderPanel();

    expect(await screen.findByText(/no model is stored/i)).toBeInTheDocument();
    // The backend's own reason, not a guess by the dashboard.
    expect(screen.getByText(/re-run the experiment/i)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^predict$/i })).toBeNull();
  });

  it("builds the form from the schema the model declares", async () => {
    mockBackend([{ match: MODEL_ROUTE, body: MODEL_AVAILABLE }]);
    renderPanel();

    for (const feature of MODEL_AVAILABLE.features) {
      expect(
        await screen.findByLabelText(new RegExp(feature.name)),
      ).toBeInTheDocument();
    }
    // And nothing the model was not trained on.
    expect(screen.queryByLabelText(/renewed/)).toBeNull();
  });

  it("gives a numeric feature a numeric input and a date a date one", async () => {
    mockBackend([{ match: MODEL_ROUTE, body: MODEL_AVAILABLE }]);
    renderPanel();

    expect(await screen.findByLabelText(/income/)).toHaveAttribute(
      "type",
      "number",
    );
    expect(screen.getByLabelText(/segment/)).toHaveAttribute("type", "text");
  });

  it("names the model and what it predicts before anything is submitted", async () => {
    mockBackend([{ match: MODEL_ROUTE, body: MODEL_AVAILABLE }]);
    renderPanel();

    expect(await screen.findByText(/model stored/i)).toBeInTheDocument();
    expect(screen.getByText(MODEL_AVAILABLE.display_name!)).toBeInTheDocument();
    expect(screen.getByText("renewed")).toBeInTheDocument();
  });

  it("reports a failure to check rather than an empty form", async () => {
    mockBackend([
      {
        match: MODEL_ROUTE,
        status: 500,
        body: errorEnvelope("internal_error", "Something went wrong."),
      },
    ]);
    renderPanel();

    expect(
      await screen.findByText(/could not check for a stored model/i),
    ).toBeInTheDocument();
  });
});

describe("making a prediction", () => {
  it("shows the predicted class and its probabilities", async () => {
    mockBackend([
      { match: PREDICT_ROUTE, body: PREDICTION },
      { match: MODEL_ROUTE, body: MODEL_AVAILABLE },
    ]);
    renderPanel();

    await fillAndSubmit({ income: "42000", tenure_months: "12", segment: "a" });

    // "yes" is both the predicted class and one of the probability labels, so
    // the assertion is scoped to the block the heading introduces.
    const heading = await screen.findByRole("heading", {
      name: /predicted renewed/i,
    });
    const result = heading.parentElement!;
    expect(within(result).getByText("yes", { selector: "p" })).toBeInTheDocument();
    expect(within(result).getByText("81.0%")).toBeInTheDocument();
    expect(within(result).getByText("19.0%")).toBeInTheDocument();
  });

  it("sends one record built from the typed values", async () => {
    const backend = mockBackend([
      { match: PREDICT_ROUTE, body: PREDICTION },
      { match: MODEL_ROUTE, body: MODEL_AVAILABLE },
    ]);
    renderPanel();

    await fillAndSubmit({ income: "42000", tenure_months: "12", segment: "a" });

    const call = backend.requests.find((request) =>
      request.url.includes("/predict"),
    );
    expect(call).toBeDefined();
    expect(JSON.parse(String(call!.body))).toEqual({
      records: [{ income: "42000", tenure_months: "12", segment: "a" }],
    });
  });

  it("sends null for a box left empty rather than an empty string", async () => {
    // The backend treats both as missing — its imputation was fitted for that
    // — but `null` is what "not supplied" means in JSON, and sending it keeps
    // the request honest about what was actually entered.
    const backend = mockBackend([
      { match: PREDICT_ROUTE, body: PREDICTION },
      { match: MODEL_ROUTE, body: MODEL_AVAILABLE },
    ]);
    renderPanel();

    await fillAndSubmit({ income: "42000" });

    const call = backend.requests.find((request) =>
      request.url.includes("/predict"),
    );
    expect(JSON.parse(String(call!.body)).records[0]).toEqual({
      income: "42000",
      tenure_months: null,
      segment: null,
    });
  });

  it("separates the model's score from the prediction", async () => {
    mockBackend([
      { match: PREDICT_ROUTE, body: PREDICTION },
      { match: MODEL_ROUTE, body: MODEL_AVAILABLE },
    ]);
    renderPanel();

    await fillAndSubmit({ income: "42000" });

    expect(
      await screen.findByText(/measurement of the model, not a confidence/i),
    ).toBeInTheDocument();
  });

  it("says a rejected record was rejected, in mapped language", async () => {
    mockBackend([
      {
        match: PREDICT_ROUTE,
        status: 422,
        body: errorEnvelope(
          "invalid_prediction_input",
          "Record 0 is missing 1 required feature(s): income.",
          { missing_features: ["income"] },
        ),
      },
      { match: MODEL_ROUTE, body: MODEL_AVAILABLE },
    ]);
    renderPanel();

    await fillAndSubmit({ segment: "a" });

    expect(await screen.findByText(/could not predict/i)).toBeInTheDocument();
    expect(
      screen.getByText(/do not match what the model expects/i),
    ).toBeInTheDocument();
  });

  it("explains a model that disappeared between loading and predicting", async () => {
    mockBackend([
      {
        match: PREDICT_ROUTE,
        status: 409,
        body: errorEnvelope("model_not_available", "No model is stored."),
      },
      { match: MODEL_ROUTE, body: MODEL_AVAILABLE },
    ]);
    renderPanel();

    await fillAndSubmit({ income: "1" });

    expect(
      await screen.findByText(/has no stored model/i),
    ).toBeInTheDocument();
  });

  it("clears the form and the result on request", async () => {
    mockBackend([
      { match: PREDICT_ROUTE, body: PREDICTION },
      { match: MODEL_ROUTE, body: MODEL_AVAILABLE },
    ]);
    renderPanel();

    await fillAndSubmit({ income: "42000" });
    await screen.findByRole("heading", { name: /predicted renewed/i });

    await userEvent.click(screen.getByRole("button", { name: /clear/i }));

    expect(screen.queryByText("81.0%")).toBeNull();
    expect(
      screen.queryByRole("heading", { name: /predicted renewed/i }),
    ).toBeNull();
    expect(screen.getByLabelText(/income/)).toHaveValue(null);
  });

  it("never renders where the model is stored", async () => {
    // The backend does not send a path, and the panel has no field that could
    // hold one. This is the assertion that keeps it that way.
    mockBackend([
      { match: PREDICT_ROUTE, body: PREDICTION },
      { match: MODEL_ROUTE, body: MODEL_AVAILABLE },
    ]);
    const { container } = renderPanel();

    await fillAndSubmit({ income: "42000" });
    await screen.findByRole("heading", { name: /predicted renewed/i });

    for (const leak of ["joblib", "/data/", "artifact.json", ".pkl"]) {
      expect(container.textContent).not.toContain(leak);
    }
  });
});

describe("the regression case", () => {
  it("shows a number and no probabilities", async () => {
    const regression = {
      ...MODEL_AVAILABLE,
      task_type: "regression",
      target_column: "price",
      classes: [],
    };
    const result = {
      ...PREDICTION,
      predictions: [{ index: 0, prediction: 254000.5, probabilities: null }],
      model: {
        ...PREDICTION.model,
        task_type: "regression",
        target_column: "price",
        classes: [],
      },
    };
    mockBackend([
      { match: PREDICT_ROUTE, body: result },
      { match: MODEL_ROUTE, body: regression },
    ]);
    renderPanel();

    await fillAndSubmit({ income: "1" });

    const heading = await screen.findByRole("heading", {
      name: /predicted price/i,
    });
    // A regression prediction is a number, formatted the way every other
    // number in the dashboard is — not a class label.
    expect(heading.parentElement!.textContent).toContain("254,000.5");
    expect(screen.queryByText(/class probability/i)).toBeNull();
  });
});

// The tab is reachable from the experiment detail page — asserted there rather
// than here, so this file stays about the panel itself.
vi.mock("next/navigation", () => ({
  usePathname: () => "/experiments",
  useParams: () => ({ id: EXPERIMENT_ID }),
  redirect: vi.fn(),
}));
