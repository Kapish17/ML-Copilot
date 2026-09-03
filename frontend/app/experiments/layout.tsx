import type { Metadata } from "next";
import type { ReactNode } from "react";

/**
 * Titles for the experiments routes.
 *
 * This covers the detail page too. A per-run title would need the run's name,
 * which means fetching it on the server before the page renders — a second
 * request, and a server-side dependency on the API, to change a browser tab.
 * The static title is the better trade.
 */
export const metadata: Metadata = {
  title: "Experiments",
  description:
    "Browse stored experiment runs, compare them on a shared metric, and read one run in full with its model comparison and SHAP explanation.",
};

export default function ExperimentsLayout({ children }: { children: ReactNode }) {
  return children;
}
