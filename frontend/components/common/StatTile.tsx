import type { ReactNode } from "react";

/** One headline number with its label — the row across the top of a card. */
export interface StatTileProps {
  label: string;
  value: ReactNode;
  hint?: string;
}

export function StatTile({ label, value, hint }: StatTileProps) {
  return (
    <div className="rounded-md border border-ink-200 bg-white px-3 py-2.5">
      <dt className="text-xs font-medium uppercase tracking-wide text-ink-500">
        {label}
      </dt>
      <dd className="mt-1 text-lg font-semibold tabular-nums text-ink-900">
        {value}
      </dd>
      {hint && <p className="mt-0.5 text-xs text-ink-500">{hint}</p>}
    </div>
  );
}

/** A responsive row of stat tiles, as a description list. */
export function StatRow({ children }: { children: ReactNode }) {
  return (
    <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
      {children}
    </dl>
  );
}
