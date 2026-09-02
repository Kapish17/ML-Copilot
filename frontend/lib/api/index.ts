/**
 * The typed client for the ML Copilot backend.
 *
 * One module per area of the API, all over the same transport in `client.ts`,
 * all typed against the contracts in `types.ts`. Components import from here
 * and never call `fetch` themselves.
 */

export * from "./client";
export * from "./errors";
export * from "./types";
export * as datasets from "./datasets";
export * as experiments from "./experiments";
export * as agent from "./agent";
export * as knowledge from "./knowledge";
