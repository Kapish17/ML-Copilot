/**
 * Service-level endpoints: identity, liveness, and whether the API is locked.
 *
 * Both are public on every deployment — a healthcheck cannot carry a
 * credential, and a client has to be able to ask "do I need one?" before it
 * has one.
 */

import { getJson, type RequestOptions } from "./client";
import type { HealthStatus, ServiceInfo } from "./types";

/**
 * Identify the running service, and learn whether it requires an API key.
 *
 * `authentication_required` is the reason the dashboard calls this at all.
 * **The dashboard holds no key and cannot hold one** — it is a browser
 * application, so anything it shipped would be readable by every visitor —
 * so when the backend is protected, the honest thing is to say so in the
 * header rather than to let every action fail with a 401 the user cannot
 * act on.
 */
export function serviceInfo(options: RequestOptions = {}): Promise<ServiceInfo> {
  return getJson<ServiceInfo>("/", undefined, options);
}

/** Liveness. Public, and the same check both containers' healthchecks run. */
export function health(options: RequestOptions = {}): Promise<HealthStatus> {
  return getJson<HealthStatus>("/health", undefined, options);
}
