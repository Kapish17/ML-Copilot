import type { Metadata } from "next";
import { AppShell } from "@/components/layout/AppShell";
import "./globals.css";

/**
 * The template is what makes a browser tab useful.
 *
 * Every route is a client component — they all fetch — and a client component
 * cannot export `metadata`, so each route folder carries a two-line
 * `layout.tsx` that supplies its own title. Those fill `%s` here, giving
 * "Experiments · ML Copilot" rather than four identical tabs. `default` covers
 * the routes that have no layout of their own, including the 404.
 */
export const metadata: Metadata = {
  title: {
    template: "%s · ML Copilot",
    default: "ML Copilot — AI Data Scientist",
  },
  description:
    "Upload a dataset, profile it, run a cross-validated experiment, explain the winner, and ask the AI Data Scientist about the result.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
