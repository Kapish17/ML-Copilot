"use client";

/**
 * The boundary for a failure in the root layout itself.
 *
 * `app/error.tsx` is rendered *inside* the layout, so it cannot catch a
 * failure in the layout that would contain it. This one replaces the whole
 * document, which is why it has to supply its own `<html>` and `<body>` — and
 * why it uses inline styles: a layout that failed may never have loaded the
 * stylesheet.
 *
 * It should never be seen. It exists so that if it ever is, it is a sentence
 * rather than a blank page.
 */
export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body
        style={{
          fontFamily: "system-ui, sans-serif",
          margin: 0,
          padding: "4rem 1.5rem",
          textAlign: "center",
          color: "#1f2937",
          background: "#f8fafc",
        }}
      >
        <h1 style={{ fontSize: "1.125rem", fontWeight: 600 }}>
          ML Copilot could not start this page
        </h1>
        <p style={{ fontSize: "0.875rem", color: "#4b5563" }}>
          The application failed while loading. Reloading usually resolves it.
        </p>
        {error.digest && (
          <p
            style={{
              fontFamily: "ui-monospace, monospace",
              fontSize: "0.75rem",
              color: "#6b7280",
            }}
          >
            reference {error.digest}
          </p>
        )}
        <button
          type="button"
          onClick={reset}
          style={{
            marginTop: "1rem",
            borderRadius: "0.375rem",
            border: "none",
            background: "#1d4ed8",
            color: "#fff",
            padding: "0.5rem 0.875rem",
            fontSize: "0.875rem",
            cursor: "pointer",
          }}
        >
          Try again
        </button>
      </body>
    </html>
  );
}
