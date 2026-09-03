import type { Metadata } from "next";
import Link from "next/link";

/**
 * The page for a URL that is not a route.
 *
 * Next.js ships a default 404, and it is unstyled system text. Inside this
 * application's shell — its header, its navigation, its footer — that reads as
 * something broken rather than as a wrong address, which is the wrong
 * impression for the one page a visitor reaches by mistyping a URL.
 *
 * A server component, so it is fully static and costs nothing to serve.
 */
export const metadata: Metadata = {
  title: "Page not found",
};

const DESTINATIONS = [
  {
    href: "/dashboard",
    label: "Dashboard",
    hint: "Upload a dataset, profile it, and run an experiment",
  },
  {
    href: "/experiments",
    label: "Experiments",
    hint: "Every stored run, with model comparison and SHAP",
  },
  {
    href: "/knowledge",
    label: "Knowledge",
    hint: "Search this project's documentation and run history",
  },
];

export default function NotFound() {
  return (
    <section className="mx-auto max-w-2xl py-10">
      <p className="text-xs font-medium uppercase tracking-widest text-ink-500">
        404
      </p>
      <h1 className="mt-2 text-2xl font-semibold tracking-tight text-ink-900">
        That page does not exist
      </h1>
      <p className="mt-2 text-sm text-ink-600">
        The address may be mistyped, or it may point at an experiment that is no
        longer stored. Nothing has gone wrong with the service.
      </p>

      <ul className="mt-6 space-y-2">
        {DESTINATIONS.map((destination) => (
          <li key={destination.href}>
            <Link
              href={destination.href}
              className="block rounded-md border border-ink-200 bg-white px-4 py-3 transition-colors hover:border-ink-300 hover:bg-ink-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400"
            >
              <span className="text-sm font-medium text-ink-900">
                {destination.label}
              </span>
              <span className="mt-0.5 block text-xs text-ink-500">
                {destination.hint}
              </span>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}
