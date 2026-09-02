import { friendlyMessage, errorCode } from "@/lib/api/errors";

/**
 * A failed request, said plainly.
 *
 * The text comes from the error mapper, which never renders a traceback, a
 * provider exception, a filesystem path or a credential. The backend's stable
 * code is shown in small type beside it — useful when reporting a problem,
 * and safe by construction since a code is a fixed identifier.
 */
export interface ErrorBannerProps {
  error: unknown;
  /** What the user was trying to do, e.g. "Profiling failed". */
  title?: string;
  onRetry?: () => void;
}

export function ErrorBanner({ error, title, onRetry }: ErrorBannerProps) {
  if (!error) return null;
  return (
    <div
      role="alert"
      className="rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-900"
    >
      <div className="flex items-start gap-2">
        <span aria-hidden="true" className="mt-0.5 font-semibold">
          !
        </span>
        <div className="min-w-0 flex-1">
          {title && <p className="font-semibold">{title}</p>}
          <p className="mt-0.5">{friendlyMessage(error)}</p>
          <p className="mt-1 font-mono text-xs text-rose-700">
            {errorCode(error)}
          </p>
        </div>
        {onRetry && (
          <button
            type="button"
            onClick={onRetry}
            className="shrink-0 rounded border border-rose-300 bg-white px-2.5 py-1 text-xs font-medium text-rose-800 hover:bg-rose-100 focus:outline-none focus-visible:ring-2 focus-visible:ring-rose-400"
          >
            Try again
          </button>
        )}
      </div>
    </div>
  );
}
