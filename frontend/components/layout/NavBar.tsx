"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/dashboard", label: "Dashboard" },
  { href: "/experiments", label: "Experiments" },
  { href: "/knowledge", label: "Knowledge" },
];

/** The product name and the three places a person can be. */
export function NavBar() {
  const pathname = usePathname() ?? "";

  return (
    <nav aria-label="Primary" className="flex flex-wrap items-center gap-x-6 gap-y-2">
      <Link
        href="/dashboard"
        className="flex items-baseline gap-2 rounded focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400"
      >
        <span className="text-base font-semibold tracking-tight text-ink-900">
          ML Copilot
        </span>
        <span className="text-xs font-medium uppercase tracking-widest text-ink-500">
          AI Data Scientist
        </span>
      </Link>

      <ul className="flex items-center gap-1">
        {LINKS.map((link) => {
          const active =
            pathname === link.href || pathname.startsWith(`${link.href}/`);
          return (
            <li key={link.href}>
              <Link
                href={link.href}
                aria-current={active ? "page" : undefined}
                className={`rounded-md px-2.5 py-1.5 text-sm font-medium focus:outline-none focus-visible:ring-2 focus-visible:ring-accent-400 ${
                  active
                    ? "bg-ink-900 text-white"
                    : "text-ink-600 hover:bg-ink-100 hover:text-ink-900"
                }`}
              >
                {link.label}
              </Link>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
