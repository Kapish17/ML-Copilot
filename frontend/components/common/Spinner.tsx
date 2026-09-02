/**
 * The progress indicator used for every request.
 *
 * It carries a *label*, not just a spinning shape, because the backend is
 * synchronous and some of these waits are long: "Running experiment…" tells a
 * person the page is not stuck in a way a bare spinner cannot. The label is
 * announced politely, so a screen reader hears the stage change without the
 * page stealing focus.
 */
export interface LoadingProps {
  label: string;
  className?: string;
}

export function Loading({ label, className = "" }: LoadingProps) {
  return (
    <div
      role="status"
      aria-live="polite"
      className={`flex items-center gap-3 text-sm text-ink-600 ${className}`}
    >
      <span
        aria-hidden="true"
        className="h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-ink-200 border-t-accent-600"
      />
      <span>{label}</span>
    </div>
  );
}
