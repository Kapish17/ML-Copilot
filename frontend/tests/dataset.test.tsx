/**
 * Tests for the upload control and the profile view.
 *
 * The profile view is the first thing a person reads about their data, so
 * these check the things that would mislead if they were wrong: that the
 * headline numbers are the backend's, that a critical finding is not carried
 * by colour alone, and that the same dataset uploaded three ways renders the
 * same profile with only the format label differing.
 */

import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { DatasetUpload } from "@/components/dataset/DatasetUpload";
import { DatasetProfileView } from "@/components/dataset/DatasetProfileView";
import { QualityFindings } from "@/components/dataset/QualityFindings";
import { ColumnTable } from "@/components/dataset/ColumnTable";
import {
  CLASSIFICATION_PROFILE,
  JSON_PROFILE,
  XLSX_PROFILE,
  csvFile,
  jsonFile,
  xlsxFile,
} from "./fixtures";

describe("dataset upload", () => {
  it("labels its file input and names the three supported formats", () => {
    render(<DatasetUpload file={null} onSelect={vi.fn()} />);

    expect(screen.getByLabelText(/dataset file/i)).toBeInTheDocument();
    expect(screen.getByText(/CSV · Excel \(\.xlsx\) · JSON/)).toBeInTheDocument();
  });

  it.each([
    ["CSV", csvFile()],
    ["Excel", xlsxFile()],
    ["JSON", jsonFile()],
  ])("accepts a %s file", async (_label, file) => {
    const onSelect = vi.fn();
    render(<DatasetUpload file={null} onSelect={onSelect} />);

    await userEvent.upload(screen.getByLabelText(/dataset file/i), file);

    expect(onSelect).toHaveBeenCalledWith(file);
  });

  it("refuses an obviously unsupported file without a round trip", () => {
    const onSelect = vi.fn();
    const { container } = render(
      <DatasetUpload file={null} onSelect={onSelect} />,
    );

    // Dropped rather than picked: the file input's `accept` already filters
    // the picker, so the drop target is the path where the guard has to work.
    const dropZone = container.querySelector("div.border-dashed") as HTMLElement;
    fireEvent.drop(dropZone, {
      dataTransfer: {
        files: [
          new File(["x"], "model.parquet", { type: "application/octet-stream" }),
        ],
      },
    });

    expect(onSelect).toHaveBeenCalledWith(null);
    expect(screen.getByRole("alert")).toHaveTextContent(
      /not supported.*CSV.*Excel.*JSON/i,
    );
  });

  it("shows the chosen file and offers to remove it", async () => {
    const onSelect = vi.fn();
    render(<DatasetUpload file={csvFile()} onSelect={onSelect} />);

    expect(screen.getByText("customers.csv")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /remove/i }));
    expect(onSelect).toHaveBeenCalledWith(null);
  });

  it("says the file is not kept in the browser", () => {
    render(<DatasetUpload file={null} onSelect={vi.fn()} />);
    expect(
      screen.getByText(/never stored in your browser/i),
    ).toBeInTheDocument();
  });
});

describe("dataset profile", () => {
  it("renders the headline numbers the backend reported", () => {
    render(<DatasetProfileView profile={CLASSIFICATION_PROFILE} />);

    // Each tile is a term/definition pair, so read the value beside its
    // label. Scoped to <dt> because "Rows" is also a column header elsewhere.
    function tileValue(label: string): string {
      const term = screen.getByText(label, { selector: "dt" });
      return term.nextElementSibling?.textContent ?? "";
    }

    expect(tileValue("Rows")).toBe("180");
    expect(tileValue("Columns")).toBe("4");
    expect(tileValue("Duplicate rows")).toBe("48");
    expect(tileValue("Missing cells")).toBe("0");
    expect(tileValue("Quality findings")).toBe("2");
    expect(tileValue("Target")).toBe("renewed");
    expect(tileValue("Task")).toBe("classification");
  });

  it("shows the target column, the inferred task and the reason", () => {
    render(<DatasetProfileView profile={CLASSIFICATION_PROFILE} />);

    const targetCard = screen
      .getByRole("heading", { name: /target column/i })
      .closest("section") as HTMLElement;

    expect(within(targetCard).getByText("renewed")).toBeInTheDocument();
    expect(
      within(targetCard).getByText(/categorical with 2 distinct values/i),
    ).toBeInTheDocument();
  });

  it.each([
    [CLASSIFICATION_PROFILE, "CSV"],
    [XLSX_PROFILE, "XLSX"],
    [JSON_PROFILE, "JSON"],
  ])("reports the format the file was read as (%#)", (profile, badge) => {
    render(<DatasetProfileView profile={profile} />);
    expect(screen.getByText(badge)).toBeInTheDocument();
  });

  it("renders identically across formats apart from the format label", () => {
    const { container: csv, unmount } = render(
      <DatasetProfileView profile={CLASSIFICATION_PROFILE} />,
    );
    const csvText = csv.textContent?.replace(/customers\.csv|CSV/g, "");
    unmount();

    const { container: json } = render(
      <DatasetProfileView profile={JSON_PROFILE} />,
    );
    const jsonText = json.textContent?.replace(/customers\.json|JSON/g, "");

    expect(jsonText).toBe(csvText);
  });
});

describe("data-quality findings", () => {
  it("orders findings worst first and names each severity in words", () => {
    render(<QualityFindings issues={CLASSIFICATION_PROFILE.quality.issues} />);

    const rows = screen.getAllByRole("row").slice(1);
    expect(within(rows[0]).getByText("critical")).toBeInTheDocument();
    expect(within(rows[1]).getByText("warning")).toBeInTheDocument();
  });

  it("shows the column, the reason and the structured details", () => {
    render(<QualityFindings issues={CLASSIFICATION_PROFILE.quality.issues} />);

    expect(screen.getByText("notes")).toBeInTheDocument();
    expect(screen.getByText(/82% missing/)).toBeInTheDocument();
    expect(screen.getByText(/Duplicate row count: 48/)).toBeInTheDocument();
  });

  it("shows an empty state rather than a blank table", () => {
    render(<QualityFindings issues={[]} />);
    expect(screen.getByText(/no data-quality findings/i)).toBeInTheDocument();
  });
});

describe("column table", () => {
  it("gives every column a row header and a summary", () => {
    render(<ColumnTable columns={CLASSIFICATION_PROFILE.columns} />);

    for (const column of CLASSIFICATION_PROFILE.columns) {
      expect(
        screen.getByRole("rowheader", { name: new RegExp(column.name) }),
      ).toBeInTheDocument();
    }
    expect(screen.getByText(/mean 43,355/)).toBeInTheDocument();
    expect(screen.getByText(/retail \(66.7%\)/)).toBeInTheDocument();
  });
});
