"use client";

import { useId, useState, type ReactNode } from "react";

/**
 * A tab set with real tab semantics.
 *
 * Roving focus, arrow-key movement, Home and End: a person navigating by
 * keyboard moves between tabs the way they expect, and only the selected tab
 * is in the tab order. Panels are mounted only when selected, which also
 * keeps a large experiment record from rendering four times over.
 */
export interface TabDefinition {
  id: string;
  label: string;
  /** Rendered after the label, e.g. a count. */
  badge?: ReactNode;
  content: ReactNode;
}

export function Tabs({
  tabs,
  ariaLabel,
}: {
  tabs: TabDefinition[];
  ariaLabel: string;
}) {
  const base = useId();
  const [active, setActive] = useState(tabs[0]?.id);
  if (tabs.length === 0) return null;

  const index = Math.max(
    0,
    tabs.findIndex((tab) => tab.id === active),
  );

  function onKeyDown(event: React.KeyboardEvent<HTMLButtonElement>) {
    const moves: Record<string, number> = { ArrowRight: 1, ArrowLeft: -1 };
    let next: number | null = null;
    if (event.key in moves) {
      next = (index + moves[event.key] + tabs.length) % tabs.length;
    } else if (event.key === "Home") {
      next = 0;
    } else if (event.key === "End") {
      next = tabs.length - 1;
    }
    if (next === null) return;
    event.preventDefault();
    setActive(tabs[next].id);
    document.getElementById(`${base}-tab-${tabs[next].id}`)?.focus();
  }

  const current = tabs[index];

  return (
    <div>
      <div
        role="tablist"
        aria-label={ariaLabel}
        className="flex flex-wrap gap-1 border-b border-ink-200"
      >
        {tabs.map((tab) => {
          const selected = tab.id === current.id;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              id={`${base}-tab-${tab.id}`}
              aria-selected={selected}
              aria-controls={`${base}-panel-${tab.id}`}
              tabIndex={selected ? 0 : -1}
              onClick={() => setActive(tab.id)}
              onKeyDown={onKeyDown}
              className={`-mb-px rounded-t border-b-2 px-3 py-2 text-sm font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 ${
                selected
                  ? "border-accent-600 text-accent-800"
                  : "border-transparent text-ink-600 hover:text-ink-900"
              }`}
            >
              {tab.label}
              {tab.badge !== undefined && (
                <span className="ml-1.5 text-xs text-ink-500">{tab.badge}</span>
              )}
            </button>
          );
        })}
      </div>
      <div
        role="tabpanel"
        id={`${base}-panel-${current.id}`}
        aria-labelledby={`${base}-tab-${current.id}`}
        tabIndex={0}
        className="pt-4 focus:outline-none"
      >
        {current.content}
      </div>
    </div>
  );
}
