import type { Metadata } from "next";
import type { ReactNode } from "react";

/** Titles for the knowledge route. See `app/dashboard/layout.tsx` for why. */
export const metadata: Metadata = {
  title: "Knowledge",
  description:
    "Search this project's own documentation and its experiment history, and read the cited passages an answer is allowed to draw on.",
};

export default function KnowledgeLayout({ children }: { children: ReactNode }) {
  return children;
}
