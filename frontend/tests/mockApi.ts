/**
 * The fake backend the component tests run against.
 *
 * `fetch` is the seam, deliberately. Stubbing the client modules instead would
 * leave the URL building, the error-envelope parsing and the error mapper
 * untested, and those are exactly the parts a component depends on being
 * right. Mocking the transport keeps the whole real path under test and still
 * needs no server, no index and no credential.
 */

import { vi } from "vitest";
import type { ErrorResponse } from "@/lib/api/types";
import { SERVICE_INFO } from "./fixtures";

/** One rule: a path fragment, and what the backend answers for it. */
export interface Route {
  /** Matched with `includes` against the request URL. */
  match?: string;
  /**
   * Matched against the URL's pathname exactly.
   *
   * `match` cannot express the service-info endpoint: its path is `/`, and
   * `includes("/")` is true of every URL there has ever been. This is the
   * escape hatch for a path that is a prefix of all the others.
   */
  exactPath?: string;
  status?: number;
  body: unknown;
  /** Delay the response, so a test can observe the loading state. */
  delayMs?: number;
}

/** The requests a test's fake backend received, in order. */
export interface RecordedRequest {
  url: string;
  method: string;
  body: BodyInit | null | undefined;
}

export interface MockBackend {
  requests: RecordedRequest[];
  /** The `FormData` of the most recent multipart request, if there was one. */
  lastForm(): FormData | null;
}

/** Build the error envelope the backend really returns. */
export function errorEnvelope(
  code: string,
  message: string,
  details: Record<string, unknown> = {},
): ErrorResponse {
  return { error: { code, message, details: details as never } };
}

/**
 * Install a fake backend for the duration of one test.
 *
 * Routes are tried in order, so a specific path can be listed before a more
 * general one. An unmatched request fails the test loudly rather than
 * returning something plausible — a component quietly calling an endpoint
 * nobody expected is worth knowing about.
 */
export function mockBackend(routes: Route[]): MockBackend {
  const requests: RecordedRequest[] = [];

  const fetchMock = vi.fn(
    async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
      const url = typeof input === "string" ? input : input.toString();
      requests.push({
        url,
        method: init?.method ?? "GET",
        body: init?.body,
      });

      const route = routes.find((candidate) =>
        candidate.exactPath !== undefined
          ? new URL(url).pathname === candidate.exactPath
          : candidate.match !== undefined && url.includes(candidate.match),
      );
      if (!route) {
        throw new Error(`No mock route matched ${init?.method ?? "GET"} ${url}`);
      }

      if (route.delayMs) {
        await new Promise((resolve) => setTimeout(resolve, route.delayMs));
      }

      const status = route.status ?? 200;
      return new Response(JSON.stringify(route.body), {
        status,
        headers: { "Content-Type": "application/json" },
      });
    },
  );

  vi.stubGlobal("fetch", fetchMock);

  return {
    requests,
    lastForm() {
      for (let index = requests.length - 1; index >= 0; index -= 1) {
        const body = requests[index].body;
        if (body instanceof FormData) return body;
      }
      return null;
    },
  };
}

/** Make every request fail the way an unreachable backend does. */
export function mockNetworkFailure(): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    }),
  );
}

/** Respond with a body that is not JSON at all. */
export function mockNonJsonResponse(): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response("<html><body>502 Bad Gateway</body></html>", {
          status: 502,
          headers: { "Content-Type": "text/html" },
        }),
    ),
  );
}

/**
 * The routes that keep the app shell's status indicator quiet.
 *
 * Three, since the header also asks the service whether it requires an API
 * key. `serviceBody` defaults to an unauthenticated deployment, which is what
 * every test that does not care about authentication should see.
 */
export function statusRoutes(
  agentBody: unknown,
  knowledgeBody: unknown,
  serviceBody: unknown = SERVICE_INFO,
): Route[] {
  return [
    { match: "/api/v1/agent/status", body: agentBody },
    { match: "/api/v1/knowledge/status", body: knowledgeBody },
    { exactPath: "/", body: serviceBody },
  ];
}
