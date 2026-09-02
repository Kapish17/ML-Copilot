/**
 * The one place this application talks to the network.
 *
 * Every request in the dashboard goes through `requestJson`, and it does three
 * things nothing else has to repeat: it resolves the base URL from
 * configuration rather than a hard-coded host, it turns a failed response into
 * a typed `ApiError` carrying the backend's own stable code, and it refuses to
 * hand back a body that is not JSON.
 *
 * The base URL is read at module scope from `NEXT_PUBLIC_API_BASE_URL`. Next
 * inlines `NEXT_PUBLIC_` variables at build time, which is exactly right here:
 * the browser makes these calls, so the value has to be public, and that is
 * also why nothing secret may ever be configured this way. The frontend holds
 * no credential of any kind — the language-model key lives on the server and
 * is never sent to, requested by, or storable in this code.
 */

import { ApiError, CLIENT_ERROR_CODES, isErrorResponse } from "./errors";
import type { JsonObject } from "./types";

/** Where the backend runs when nothing is configured. */
export const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

/**
 * The configured backend origin, with any trailing slash removed.
 *
 * Read through a function rather than exported as a constant so a test can
 * assert on the resolution rules themselves.
 */
export function apiBaseUrl(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  const base = configured && configured.length > 0 ? configured : DEFAULT_API_BASE_URL;
  return base.replace(/\/+$/, "");
}

/** Build an absolute URL for a backend path. */
export function apiUrl(path: string, query?: Record<string, unknown>): string {
  const suffix = path.startsWith("/") ? path : `/${path}`;
  const url = new URL(`${apiBaseUrl()}${suffix}`);

  for (const [key, value] of Object.entries(query ?? {})) {
    if (value === undefined || value === null || value === "") continue;
    if (Array.isArray(value)) {
      for (const item of value) url.searchParams.append(key, String(item));
    } else {
      url.searchParams.set(key, String(value));
    }
  }
  return url.toString();
}

/** Options every request may carry. */
export interface RequestOptions {
  /** Aborts the request — wired to component unmounts and to Cancel. */
  signal?: AbortSignal;
}

async function readJson(response: Response): Promise<unknown> {
  const text = await response.text();
  if (!text) return null;
  try {
    return JSON.parse(text) as unknown;
  } catch {
    // A non-JSON body means something other than the backend answered —
    // a proxy, a gateway, an HTML error page. Its content is not shown.
    throw new ApiError(
      CLIENT_ERROR_CODES.MALFORMED,
      "The backend returned a response this app could not read.",
      response.status,
    );
  }
}

/**
 * Perform a request and return its parsed body.
 *
 * @param path - Path on the backend, e.g. `/api/v1/experiments`.
 * @param init - Standard fetch options. A `body` of `FormData` is sent as
 *   multipart; a plain object should be pre-serialised by the caller.
 * @returns The parsed response body, typed by the caller.
 * @throws ApiError - On any non-2xx response, on a body that is not JSON, and
 *   on a transport failure. The thrown error always carries a stable `code`.
 */
export async function requestJson<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(apiUrl(path), init);
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "AbortError") throw cause;
    throw new ApiError(
      CLIENT_ERROR_CODES.NETWORK,
      "The backend could not be reached.",
      0,
    );
  }

  const body = await readJson(response);

  if (!response.ok) {
    if (isErrorResponse(body)) {
      throw new ApiError(
        body.error.code,
        body.error.message,
        response.status,
        (body.error.details ?? {}) as JsonObject,
      );
    }
    throw new ApiError(
      CLIENT_ERROR_CODES.MALFORMED,
      "The backend returned an error this app could not read.",
      response.status,
    );
  }

  if (body === null || typeof body !== "object") {
    throw new ApiError(
      CLIENT_ERROR_CODES.MALFORMED,
      "The backend returned a response this app could not read.",
      response.status,
    );
  }

  return body as T;
}

/** GET a JSON resource. */
export function getJson<T>(
  path: string,
  query?: Record<string, unknown>,
  options: RequestOptions = {},
): Promise<T> {
  const url = apiUrl(path, query);
  const relative = url.slice(apiBaseUrl().length);
  return requestJson<T>(relative, { method: "GET", signal: options.signal });
}

/** POST a JSON body and read a JSON response. */
export function postJson<T>(
  path: string,
  payload: unknown,
  options: RequestOptions = {},
): Promise<T> {
  return requestJson<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: options.signal,
  });
}

/**
 * POST a multipart form — the only way a file reaches the backend.
 *
 * The destination is always the configured backend and nothing else. There is
 * no third-party upload target anywhere in this application.
 */
export function postForm<T>(
  path: string,
  form: FormData,
  options: RequestOptions = {},
): Promise<T> {
  // Content-Type is deliberately unset: the browser must add the multipart
  // boundary itself, and setting the header by hand strips it.
  return requestJson<T>(path, {
    method: "POST",
    body: form,
    signal: options.signal,
  });
}
