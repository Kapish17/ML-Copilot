/**
 * Tests for the experiment views.
 *
 * The load-bearing one is the CV-versus-test distinction. A model comparison
 * table that let those two numbers sit in one undifferentiated column would
 * make an optimistic score look like a measured one, so several tests below
 * check that the separation is present in the markup and in words, not only
 * in a shade of background colour.
 *
 * The other is metric direction: the same components render F1 (higher wins)
 * and RMSE (lower wins), and both are checked.
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ModelComparisonTable } from "@/components/experiments/ModelComparisonTable";
import { MetricsPanel } from "@/components/experiments/MetricsPanel";
import { ConfusionMatrix } from "@/components/experiments/ConfusionMatrix";
import { ExperimentSummary } from "@/components/experiments/ExperimentSummary";
import { ExperimentHistoryTable } from "@/components/experiments/ExperimentHistoryTable";
import { ExperimentComparisonView } from "@/components/experiments/ExperimentComparison";
import { RunDiagnostics } from "@/components/experiments/RunDiagnostics";
import { GlobalImportance } from "@/components/explainability/GlobalImportance";
import { LocalExplanation } from "@/components/explainability/LocalExplanation";
import {
  CLASSIFICATION_RUN,
  COMPARISON,
  EXPERIMENT_LIST,
  REGRESSION_RUN,
} from "./fixtures";

describe("model comparison", () => {
  it("separates cross-validated scores from the final test measurement", () => {
    render(
      <ModelComparisonTable
        selection={CLASSIFICATION_RUN.selection}
        evaluation={CLASSIFICATION_RUN.evaluation}
      />,
    );

    // The two groups are named, not merely coloured.
    expect(
      screen.getByRole("columnheader", { name: /Selection · 3-fold cross-validation/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("columnheader", { name: /Final · untouched test set/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "CV mean" })).toBeInTheDocument();
    expect(
      screen.getByRole("columnheader", { name: /Held-out F1/ }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/the two groups are not comparable to each other/i),
    ).toBeInTheDocument();
  });

  it("says in words whether selection touched the test set", () => {
    render(
      <ModelComparisonTable
        selection={CLASSIFICATION_RUN.selection}
        evaluation={CLASSIFICATION_RUN.evaluation}
      />,
    );
    expect(
      screen.getByText(/selection never saw the test set/i),
    ).toBeInTheDocument();
  });

  it("does not label a holdout selection score as cross-validated", () => {
    render(
      <ModelComparisonTable
        selection={{ ...CLASSIFICATION_RUN.selection, strategy: "holdout", uses_test_data: true }}
        evaluation={{ ...CLASSIFICATION_RUN.evaluation, is_unbiased: false }}
      />,
    );

    expect(
      screen.getByRole("columnheader", { name: "Selection F1" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("columnheader", { name: "CV mean" }),
    ).not.toBeInTheDocument();
    expect(screen.getByText(/selection used the test set/i)).toBeInTheDocument();
    expect(document.body.textContent ?? "").toMatch(
      /both columns were computed on the same rows/i,
    );
  });

  it("marks the selected model and gives only it a test score", () => {
    render(
      <ModelComparisonTable
        selection={CLASSIFICATION_RUN.selection}
        evaluation={CLASSIFICATION_RUN.evaluation}
      />,
    );

    const winnerRow = screen
      .getByRole("rowheader", { name: /Logistic Regression/ })
      .closest("tr") as HTMLElement;
    expect(within(winnerRow).getByText("selected")).toBeInTheDocument();
    expect(within(winnerRow).getByText("0.9444")).toBeInTheDocument();

    const loserRow = screen
      .getByRole("rowheader", { name: /Random Forest Classifier/ })
      .closest("tr") as HTMLElement;
    // A candidate that was not selected has no held-out measurement at all.
    expect(within(loserRow).getAllByText("—").length).toBeGreaterThanOrEqual(3);
  });

  it("reports a candidate that failed rather than hiding it", () => {
    render(
      <ModelComparisonTable
        selection={CLASSIFICATION_RUN.selection}
        evaluation={CLASSIFICATION_RUN.evaluation}
      />,
    );
    expect(screen.getByText("failed")).toBeInTheDocument();
    expect(
      screen.getByText(/could not be fitted on this data/i),
    ).toBeInTheDocument();
  });

  it.each([
    [CLASSIFICATION_RUN, "Higher is better"],
    [REGRESSION_RUN, "Lower is better"],
  ])("states the metric's direction rather than assuming it (%#)", (run, label) => {
    render(
      <ModelComparisonTable
        selection={run.selection}
        evaluation={run.evaluation}
      />,
    );
    expect(screen.getAllByText(label).length).toBeGreaterThan(0);
  });
});

describe("metrics", () => {
  it("renders the classification metric set", () => {
    render(<MetricsPanel evaluation={CLASSIFICATION_RUN.evaluation} />);

    for (const label of ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]) {
      expect(
        screen.getByRole("rowheader", { name: new RegExp(label) }),
      ).toBeInTheDocument();
    }
  });

  it("renders the regression metric set with per-metric directions", () => {
    render(<MetricsPanel evaluation={REGRESSION_RUN.evaluation} />);

    // Exact names: "MSE" is a substring of "RMSE", so a loose match is
    // ambiguous — which is itself worth pinning, since both must be present.
    for (const label of ["MAE", "MSE", "RMSE", "R²"]) {
      expect(
        screen.getByRole("rowheader", { name: new RegExp(`^${label}`) }),
      ).toBeInTheDocument();
    }

    const rmseRow = screen
      .getByRole("rowheader", { name: /^RMSE/ })
      .closest("tr") as HTMLElement;
    expect(within(rmseRow).getByText(/Lower is better/)).toBeInTheDocument();

    const r2Row = screen
      .getByRole("rowheader", { name: /^R²/ })
      .closest("tr") as HTMLElement;
    expect(within(r2Row).getByText(/Higher is better/)).toBeInTheDocument();
  });

  it("judges a lower-is-better metric correctly against the baseline", () => {
    render(<MetricsPanel evaluation={REGRESSION_RUN.evaluation} />);

    const rmseRow = screen
      .getByRole("rowheader", { name: /^RMSE/ })
      .closest("tr") as HTMLElement;
    // RMSE fell from 6421 to 1958: a negative difference, which is better.
    expect(within(rmseRow).getByText("better")).toBeInTheDocument();
  });

  it("lists metrics the backend could not compute, with its reason", () => {
    render(<MetricsPanel evaluation={REGRESSION_RUN.evaluation} />);

    expect(screen.getByText("Not computed")).toBeInTheDocument();
    expect(screen.getByText(/target contains zeros/i)).toBeInTheDocument();
  });

  it("says whether the measurement is unbiased", () => {
    render(<MetricsPanel evaluation={CLASSIFICATION_RUN.evaluation} />);
    expect(
      screen.getByText(/unbiased — measured once on unseen rows/i),
    ).toBeInTheDocument();
  });
});

describe("confusion matrix", () => {
  it("is a real table with actual as row headers and predicted as columns", () => {
    render(
      <ConfusionMatrix
        details={CLASSIFICATION_RUN.evaluation.classification_details!}
      />,
    );

    expect(screen.getByRole("columnheader", { name: "yes" })).toBeInTheDocument();
    const actualNo = screen
      .getByRole("rowheader", { name: "no" })
      .closest("tr") as HTMLElement;
    const cells = within(actualNo).getAllByRole("cell");
    expect(cells[0]).toHaveTextContent("17");
    expect(cells[1]).toHaveTextContent("1");
    expect(screen.getByText(/Cells on the diagonal are correct/i)).toBeInTheDocument();
  });
});

describe("explainability", () => {
  it("ranks global feature importance with values beside the bars", () => {
    render(<GlobalImportance explainability={CLASSIFICATION_RUN.explainability} />);

    expect(screen.getByText("income")).toBeInTheDocument();
    expect(screen.getByText("0.8356")).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: /income: importance 0.8356, rank 1/ }),
    ).toBeInTheDocument();
  });

  it("always carries the causation disclaimer", () => {
    render(<GlobalImportance explainability={CLASSIFICATION_RUN.explainability} />);
    expect(
      screen.getByText(/describes model behaviour and association, not causation/i),
    ).toBeInTheDocument();
  });

  it("says an explanation is unavailable instead of inventing one", () => {
    render(<GlobalImportance explainability={REGRESSION_RUN.explainability} />);

    expect(screen.getByText(/Explanation unavailable/i)).toBeInTheDocument();
    expect(screen.getByText(/fitted_model_not_persisted/)).toBeInTheDocument();
    expect(
      screen.getByText(/not causation/i),
    ).toBeInTheDocument();
  });

  it("renders a local explanation with sign, words and magnitude", () => {
    render(
      <LocalExplanation
        method="shap"
        rowIndex={7}
        prediction="yes"
        predictedClass="yes"
        baseValue={0.02}
        contributions={[
          {
            feature: "monthly_charges",
            value: 112,
            contribution: 0.31,
            direction: "increases prediction",
          },
          {
            feature: "tenure_months",
            value: 4,
            contribution: -0.12,
            direction: "decreases prediction",
          },
        ]}
      />,
    );

    const row = screen
      .getByRole("rowheader", { name: "monthly_charges" })
      .closest("tr") as HTMLElement;
    expect(within(row).getByText("112")).toBeInTheDocument();
    expect(within(row).getByText("+0.31")).toBeInTheDocument();
    expect(within(row).getByText("increases prediction")).toBeInTheDocument();

    const negative = screen
      .getByRole("rowheader", { name: "tenure_months" })
      .closest("tr") as HTMLElement;
    expect(within(negative).getByText("−0.12")).toBeInTheDocument();
    expect(within(negative).getByText("decreases prediction")).toBeInTheDocument();
  });
});

describe("experiment summary", () => {
  it("labels the two scores so they cannot be confused", () => {
    render(
      <ExperimentSummary
        record={CLASSIFICATION_RUN}
        execution={CLASSIFICATION_RUN.execution}
      />,
    );

    expect(screen.getByText("Held-out F1")).toBeInTheDocument();
    expect(screen.getByText(/measured once on 36 unseen rows/i)).toBeInTheDocument();
    expect(screen.getByText("CV F1")).toBeInTheDocument();
    expect(screen.getByText(/training rows only/i)).toBeInTheDocument();
  });

  it("does not call a holdout selection score cross-validated", () => {
    const holdout = {
      ...CLASSIFICATION_RUN,
      selection: { ...CLASSIFICATION_RUN.selection, uses_test_data: true },
      evaluation: { ...CLASSIFICATION_RUN.evaluation, is_unbiased: false },
    };
    render(<ExperimentSummary record={holdout} />);

    expect(screen.getByText("Selection F1")).toBeInTheDocument();
    expect(screen.queryByText("CV F1")).not.toBeInTheDocument();
    expect(
      screen.getByText(/held-out rows — this score also chose the model/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/36 rows that also chose the model/i),
    ).toBeInTheDocument();
    expect(screen.queryByText(/training rows only/i)).not.toBeInTheDocument();
  });

  it("shows the dataset fingerprint as the run's identity", () => {
    render(<ExperimentSummary record={CLASSIFICATION_RUN} />);
    expect(screen.getByText("9d610b7e1abef86c")).toBeInTheDocument();
    expect(screen.getByText(/identity is the data, not the file/i)).toBeInTheDocument();
  });

  it("says why the winning model won, in the backend's words", () => {
    render(<ExperimentSummary record={CLASSIFICATION_RUN} />);

    expect(
      screen.getByText(
        /selected because it achieved the best cross-validation F1 over 3 folds/i,
      ),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/independent measurement taken after this choice/i),
    ).toBeInTheDocument();
  });

  it("shows nothing where a record has no rationale", () => {
    const older = {
      ...CLASSIFICATION_RUN,
      selection: { ...CLASSIFICATION_RUN.selection, rationale: undefined },
    };
    render(<ExperimentSummary record={older} />);

    expect(screen.queryByText(/selected because/i)).not.toBeInTheDocument();
  });
});

describe("run diagnostics", () => {
  it("shows each signal in the words it was written in", () => {
    render(
      <RunDiagnostics diagnostics={CLASSIFICATION_RUN.evaluation.diagnostics} />,
    );

    expect(
      screen.getByText(/Small held-out set: the final measurement was taken on 36 rows/i),
    ).toBeInTheDocument();
    expect(screen.getByText("Small test set")).toBeInTheDocument();
  });

  it("separates a warning from a note by word and glyph, not colour alone", () => {
    render(
      <RunDiagnostics diagnostics={CLASSIFICATION_RUN.evaluation.diagnostics} />,
    );

    expect(screen.getByText("worth checking")).toBeInTheDocument();
    expect(screen.getByText("note")).toBeInTheDocument();
  });

  it("frames the list as signals rather than conclusions", () => {
    render(
      <RunDiagnostics diagnostics={CLASSIFICATION_RUN.evaluation.diagnostics} />,
    );
    const text = document.body.textContent ?? "";

    expect(text).toMatch(/prompts to check something, not conclusions/i);
    expect(text).not.toMatch(/is overfit|unusable|do not use/i);
  });

  it("says nothing was flagged without calling the model sound", () => {
    render(<RunDiagnostics diagnostics={[]} />);
    const text = document.body.textContent ?? "";

    expect(text).toMatch(/nothing flagged on this run/i);
    expect(text).toMatch(/not a guarantee the model is good/i);
  });
});

describe("experiment history", () => {
  it("links each run to its detail page", () => {
    render(
      <ExperimentHistoryTable
        experiments={EXPERIMENT_LIST.experiments}
        selected={[]}
        onToggle={vi.fn()}
      />,
    );

    const link = screen.getByRole("link", { name: /customers.csv · renewed/ });
    expect(link).toHaveAttribute(
      "href",
      "/experiments/exp_e36e7bbf5267_20260902T054517Z_503e",
    );
  });

  it("names each comparison checkbox after its run", async () => {
    const onToggle = vi.fn();
    render(
      <ExperimentHistoryTable
        experiments={EXPERIMENT_LIST.experiments}
        selected={[]}
        onToggle={onToggle}
      />,
    );

    const checkbox = screen.getByRole("checkbox", {
      name: /Select customers.csv · renewed for comparison/,
    });
    await userEvent.click(checkbox);
    expect(onToggle).toHaveBeenCalledWith(
      "exp_e36e7bbf5267_20260902T054517Z_503e",
    );
  });

  it("shows an empty state rather than an empty table", () => {
    render(
      <ExperimentHistoryTable experiments={[]} selected={[]} onToggle={vi.fn()} />,
    );
    expect(screen.getByText(/no experiments stored yet/i)).toBeInTheDocument();
  });
});

describe("experiment comparison", () => {
  it("marks the run the backend called best, and does not re-rank", () => {
    render(<ExperimentComparisonView comparison={COMPARISON} />);

    const rows = screen.getAllByRole("row").slice(1);
    // The backend's order is preserved verbatim.
    expect(rows[0]).toHaveTextContent("customers.csv · renewed");
    expect(within(rows[0]).getByText("best")).toBeInTheDocument();
    expect(screen.getByText("Higher is better")).toBeInTheDocument();
  });

  it("labels the two score columns with the metric and its provenance", () => {
    render(<ExperimentComparisonView comparison={COMPARISON} />);

    expect(
      screen.getByRole("columnheader", { name: "CV F1" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("columnheader", { name: "Held-out F1" }),
    ).toBeInTheDocument();
    expect(document.body.textContent ?? "").toMatch(
      /how much they disagreed, not a confidence interval/i,
    );
  });

  it("shows the cross-validation spread beside its mean", () => {
    render(<ExperimentComparisonView comparison={COMPARISON} />);
    const rows = screen.getAllByRole("row").slice(1);

    expect(within(rows[0]).getByText(/± 0\.0161/)).toBeInTheDocument();
  });

  it("shows how much data was behind each score", () => {
    render(<ExperimentComparisonView comparison={COMPARISON} />);
    const rows = screen.getAllByRole("row").slice(1);

    expect(
      screen.getByRole("columnheader", { name: "Train rows" }),
    ).toBeInTheDocument();
    expect(within(rows[0]).getByText("192")).toBeInTheDocument();
    expect(within(rows[0]).getAllByText("7").length).toBeGreaterThan(0);
  });

  it("marks a run that raised signals beside its name, not in a far column", () => {
    render(<ExperimentComparisonView comparison={COMPARISON} />);
    const flagged = screen
      .getByRole("rowheader", { name: /customers\.json · renewed/ });

    expect(within(flagged).getByText("2 to review")).toBeInTheDocument();
    // The unflagged run is not labelled "clean" or "good" — only unmarked.
    const clean = screen.getByRole("rowheader", { name: /customers\.csv · renewed/ });
    expect(within(clean).queryByText(/review/i)).not.toBeInTheDocument();
  });

  it("does not head the column CV when a run did not cross-validate", () => {
    const mixed = {
      ...COMPARISON,
      runs: [
        { ...COMPARISON.runs[0], strategy: "holdout" },
        COMPARISON.runs[1],
      ],
    };
    render(<ExperimentComparisonView comparison={mixed} />);

    expect(
      screen.getByRole("columnheader", { name: "Selection F1" }),
    ).toBeInTheDocument();
    expect(
      screen.queryByRole("columnheader", { name: "CV F1" }),
    ).not.toBeInTheDocument();
    expect(document.body.textContent ?? "").toMatch(
      /one measurement used twice/i,
    );
  });

  it("shows the baseline beside every score", () => {
    render(<ExperimentComparisonView comparison={COMPARISON} />);

    expect(
      screen.getByRole("columnheader", { name: "Baseline" }),
    ).toBeInTheDocument();
  });

  it("renders a missing score as an em dash rather than a zero", () => {
    const withGap = {
      ...COMPARISON,
      runs: [
        { ...COMPARISON.runs[0], test_score: null, selection_score_std: null },
        COMPARISON.runs[1],
      ],
    };
    render(<ExperimentComparisonView comparison={withGap} />);
    const rows = screen.getAllByRole("row").slice(1);

    expect(within(rows[0]).getAllByText("—").length).toBeGreaterThan(0);
    expect(within(rows[0]).queryByText(/± /)).not.toBeInTheDocument();
  });
});
