import type { ReactNode } from "react";

/** What a region shows before it has anything to show. */
export interface EmptyStateProps {
  title: string;
  hint?: string;
  action?: ReactNode;
}

export function EmptyState({ title, hint, action }: EmptyStateProps) {
  return (
    <div className="rounded-md border border-dashed border-ink-200 bg-ink-50 px-4 py-8 text-center">
      <p className="text-sm font-medium text-ink-700">{title}</p>
      {hint && <p className="mx-auto mt-1 max-w-md text-xs text-ink-500">{hint}</p>}
      {action && <div className="mt-3">{action}</div>}
    </div>
  );
}
