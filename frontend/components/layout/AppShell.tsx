import type { ReactNode } from "react";
import { NavBar } from "./NavBar";
import { SystemStatus } from "./SystemStatus";

/** The frame every page sits in: skip link, header, main landmark, footer. */
export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col">
      <a
        href="#main"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-50 focus:rounded focus:bg-ink-900 focus:px-3 focus:py-2 focus:text-sm focus:text-white"
      >
        Skip to main content
      </a>

      <header className="border-b border-ink-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
          <NavBar />
          <SystemStatus />
        </div>
      </header>

      <main id="main" className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6">
        {children}
      </main>

      <footer className="border-t border-ink-200 bg-white">
        <div className="mx-auto max-w-7xl px-4 py-3 text-xs text-ink-500 sm:px-6">
          ML Copilot · every result on this page is produced by the backend.
          Feature importance describes model behaviour and association, not
          causation.
        </div>
      </footer>
    </div>
  );
}
