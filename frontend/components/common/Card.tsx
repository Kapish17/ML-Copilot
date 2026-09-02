import type { ReactNode } from "react";

/**
 * The one container the whole dashboard is built from.
 *
 * A card is a titled region with a heading level the caller chooses, so the
 * page's heading outline stays correct wherever a card is placed. That is the
 * only reason `headingLevel` exists — nesting a card inside a section must not
 * produce an h2 under an h3.
 */
export interface CardProps {
  title?: ReactNode;
  description?: ReactNode;
  /** Rendered at the top right — a badge, a count, a small control. */
  aside?: ReactNode;
  headingLevel?: 2 | 3 | 4;
  className?: string;
  children: ReactNode;
}

export function Card({
  title,
  description,
  aside,
  headingLevel = 2,
  className = "",
  children,
}: CardProps) {
  const Heading = `h${headingLevel}` as "h2" | "h3" | "h4";
  return (
    <section
      className={`rounded-lg border border-ink-200 bg-white shadow-sm ${className}`}
    >
      {(title || aside) && (
        <div className="flex flex-wrap items-start justify-between gap-3 border-b border-ink-100 px-4 py-3 sm:px-5">
          <div className="min-w-0">
            {title && (
              <Heading className="text-sm font-semibold tracking-tight text-ink-900">
                {title}
              </Heading>
            )}
            {description && (
              <p className="mt-1 text-xs leading-relaxed text-ink-500">
                {description}
              </p>
            )}
          </div>
          {aside && <div className="shrink-0">{aside}</div>}
        </div>
      )}
      <div className="px-4 py-4 sm:px-5">{children}</div>
    </section>
  );
}
