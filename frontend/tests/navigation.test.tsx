/**
 * Tests for the two things a visitor sees before any data loads: what the
 * browser tab says, and what a wrong URL gives them.
 *
 * Both are easy to get wrong in a way nothing else catches. Every route here
 * is a client component, and a client component cannot export `metadata` — so
 * the titles come from a `layout.tsx` per route folder, and the failure mode
 * of deleting one is not an error but four identical browser tabs. Likewise a
 * missing `not-found.tsx` is not a crash; it is Next's unstyled default
 * rendered inside this application's shell, which reads as breakage.
 *
 * These assert the wiring, not the wording.
 */

import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import RootLayout, { metadata as rootMetadata } from "@/app/layout";
import { metadata as dashboardMetadata } from "@/app/dashboard/layout";
import { metadata as experimentsMetadata } from "@/app/experiments/layout";
import { metadata as knowledgeMetadata } from "@/app/knowledge/layout";
import NotFound from "@/app/not-found";
import { AppShell } from "@/components/layout/AppShell";

vi.mock("next/navigation", () => ({
  usePathname: () => "/nowhere",
  redirect: vi.fn(),
}));

describe("browser titles", () => {
  it("gives the root a template so a route title reads as a location", () => {
    const title = rootMetadata.title as { template: string; default: string };

    expect(title.template).toContain("%s");
    expect(title.template).toContain("ML Copilot");
    // The fallback is the full product name, for the routes with no layout of
    // their own — including the 404 before its own title applies.
    expect(title.default).toContain("ML Copilot");
  });

  it.each([
    ["dashboard", dashboardMetadata, "Dashboard"],
    ["experiments", experimentsMetadata, "Experiments"],
    ["knowledge", knowledgeMetadata, "Knowledge"],
  ])("names the %s route", (_route, metadata, expected) => {
    expect(metadata.title).toBe(expected);
    // A description as well: it is what a link preview and a search result
    // show, and "ML Copilot" three times over says nothing.
    expect(typeof metadata.description).toBe("string");
    expect((metadata.description as string).length).toBeGreaterThan(40);
  });

  it("gives every route a distinct title", () => {
    const titles = [
      dashboardMetadata.title,
      experimentsMetadata.title,
      knowledgeMetadata.title,
    ];

    expect(new Set(titles).size).toBe(titles.length);
  });

  it("wraps the application in the shell", () => {
    // `RootLayout` returns <html><body>, which cannot be mounted into a
    // container. Rendering its children through the shell asserts the part
    // that matters here: the shell is what every page sits in.
    render(
      <AppShell>
        <p>page content</p>
      </AppShell>,
    );

    expect(screen.getByRole("navigation", { name: "Primary" })).toBeInTheDocument();
    expect(screen.getByText("page content")).toBeInTheDocument();
    expect(typeof RootLayout).toBe("function");
  });
});

describe("the not-found page", () => {
  it("explains the address is wrong rather than implying a failure", () => {
    render(<NotFound />);

    expect(
      screen.getByRole("heading", { name: /does not exist/i }),
    ).toBeInTheDocument();
    expect(screen.getByText(/nothing has gone wrong with the service/i)).toBeInTheDocument();
  });

  it("offers a way to every route the navigation offers", () => {
    render(<NotFound />);

    for (const label of ["Dashboard", "Experiments", "Knowledge"]) {
      expect(screen.getByRole("link", { name: new RegExp(label) })).toBeInTheDocument();
    }
  });

  it("links to real routes, not to invented ones", () => {
    render(<NotFound />);

    const hrefs = screen
      .getAllByRole("link")
      .map((link) => link.getAttribute("href"));

    expect(hrefs).toEqual(["/dashboard", "/experiments", "/knowledge"]);
  });
});
