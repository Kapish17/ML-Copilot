import type { Metadata } from "next";
import type { ReactNode } from "react";

/**
 * Titles for the dashboard route.
 *
 * A layout, not a page export, because `page.tsx` is a client component and a
 * client component cannot export `metadata`. This adds no markup — it wraps
 * the page in nothing — and exists only so the browser tab says where you are.
 */
export const metadata: Metadata = {
  title: "Dashboard",
  description:
    "Upload a dataset, read its profile and quality findings, ask the AI Data Scientist, and run a cross-validated experiment.",
};

export default function DashboardLayout({ children }: { children: ReactNode }) {
  return children;
}
