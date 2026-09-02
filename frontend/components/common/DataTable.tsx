import type { ReactNode } from "react";

/**
 * The table used everywhere data is tabular.
 *
 * Two things it insists on. A `caption`, always — a screen reader reaching a
 * table out of context needs to know what it is, and several tables here sit
 * side by side. And a horizontal scroll container of its own, so a wide table
 * scrolls inside its card on a phone instead of pushing the page sideways.
 */
export interface DataTableProps {
  caption: string;
  /** Show the caption visually as well as to assistive technology. */
  visibleCaption?: boolean;
  head: ReactNode;
  children: ReactNode;
  className?: string;
}

export function DataTable({
  caption,
  visibleCaption = false,
  head,
  children,
  className = "",
}: DataTableProps) {
  return (
    <div className={`-mx-4 overflow-x-auto sm:mx-0 ${className}`}>
      <table className="w-full min-w-full border-collapse text-sm">
        <caption
          className={
            visibleCaption
              ? "px-4 pb-2 text-left text-xs text-ink-500 sm:px-0"
              : "sr-only"
          }
        >
          {caption}
        </caption>
        <thead className="border-b border-ink-200 text-left">{head}</thead>
        <tbody className="divide-y divide-ink-100">{children}</tbody>
      </table>
    </div>
  );
}

/** A header cell. `numeric` right-aligns it, matching its column's values. */
export function Th({
  children,
  numeric = false,
  className = "",
  scope,
  colSpan,
}: {
  children?: ReactNode;
  numeric?: boolean;
  className?: string;
  scope?: "col" | "row";
  colSpan?: number;
}) {
  // An empty header cell is a spacer: it heads nothing, so it gets no scope.
  // Anything with content is a column header unless the caller says otherwise.
  const resolvedScope = scope ?? (children === undefined ? undefined : "col");
  return (
    <th
      scope={resolvedScope}
      colSpan={colSpan}
      className={`whitespace-nowrap px-3 py-2 text-xs font-semibold uppercase tracking-wide text-ink-600 ${
        numeric ? "text-right" : "text-left"
      } ${className}`}
    >
      {children}
    </th>
  );
}

/** A body cell. */
export function Td({
  children,
  numeric = false,
  className = "",
}: {
  children: ReactNode;
  numeric?: boolean;
  className?: string;
}) {
  return (
    <td
      className={`px-3 py-2 align-top text-ink-800 ${
        numeric ? "text-right tabular-nums" : "text-left"
      } ${className}`}
    >
      {children}
    </td>
  );
}
