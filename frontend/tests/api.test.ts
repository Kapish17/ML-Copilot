/**
 * Tests for the transport, the configuration and the error mapper.
 *
 * These are the parts every component depends on and none of them can see: if
 * the base URL resolves wrongly, or a 422 envelope is not turned into a coded
 * error, every screen in the app misbehaves in the same way at once.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_API_BASE_URL,
  apiBaseUrl,
  apiUrl,
  requestJson,
} from "@/lib/api/client";
import {
  ApiError,
  CLIENT_ERROR_CODES,
  errorCode,
  friendlyMessage,
  isErrorResponse,
} from "@/lib/api/errors";
import { listExperiments } from "@/lib/api/experiments";
import { askAgentWithDataset } from "@/lib/api/agent";
import { profileDataset } from "@/lib/api/datasets";
import {
  errorEnvelope,
  mockBackend,
  mockNetworkFailure,
  mockNonJsonResponse,
} from "./mockApi";
import { CLASSIFICATION_PROFILE, csvFile, xlsxFile } from "./fixtures";

beforeEach(() => {
  // Each test states its own API URL, so start every one from "unconfigured".
  vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "http://backend.test");
});

describe("API base URL configuration", () => {
  it("uses the configured origin", () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "https://api.example.test");
    expect(apiBaseUrl()).toBe("https://api.example.test");
    expect(apiUrl("/api/v1/experiments")).toBe(
      "https://api.example.test/api/v1/experiments",
    );
  });

  it("falls back to a documented default rather than a hard-coded host", () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "");
    expect(apiBaseUrl()).toBe(DEFAULT_API_BASE_URL);
  });

  it("tolerates a trailing slash, so both spellings work", () => {
    vi.stubEnv("NEXT_PUBLIC_API_BASE_URL", "https://api.example.test/");
    expect(apiUrl("/health")).toBe("https://api.example.test/health");
  });

  it("appends query parameters and repeats arrays", () => {
    const url = apiUrl("/api/v1/experiments", {
      limit: 5,
      tags: ["a", "b"],
      empty: "",
      missing: undefined,
    });
    expect(url).toContain("limit=5");
    expect(url).toContain("tags=a&tags=b");
    expect(url).not.toContain("empty");
    expect(url).not.toContain("missing");
  });

  it("sends every request to the configured backend and nowhere else", async () => {
    const backend = mockBackend([
      { match: "/api/v1/datasets/profile", body: CLASSIFICATION_PROFILE },
    ]);

    await profileDataset(csvFile());

    expect(backend.requests).toHaveLength(1);
    expect(backend.requests[0].url.startsWith("http://backend.test/")).toBe(true);
  });
});

describe("error envelope handling", () => {
  it("recognises the backend's envelope", () => {
    expect(isErrorResponse(errorEnvelope("invalid_excel", "…"))).toBe(true);
    expect(isErrorResponse({ error: "nope" })).toBe(false);
    expect(isErrorResponse(null)).toBe(false);
  });

  it("turns a failure into a typed error carrying the backend's code", async () => {
    mockBackend([
      {
        match: "/api/v1/datasets/profile",
        status: 422,
        body: errorEnvelope(
          "invalid_excel",
          "The workbook could not be opened.",
          { filename: "book.xlsx" },
        ),
      },
    ]);

    const failure = await profileDataset(xlsxFile()).catch(
      (error: unknown) => error,
    );

    expect(failure).toBeInstanceOf(ApiError);
    const apiError = failure as ApiError;
    expect(apiError.code).toBe("invalid_excel");
    expect(apiError.status).toBe(422);
    expect(apiError.details.filename).toBe("book.xlsx");
  });

  it("reports an unreachable backend as a network failure", async () => {
    mockNetworkFailure();
    const failure = await listExperiments().catch((error: unknown) => error);

    expect(errorCode(failure)).toBe(CLIENT_ERROR_CODES.NETWORK);
    expect(friendlyMessage(failure)).toContain("could not be reached");
  });

  it("refuses a response that is not JSON instead of rendering it", async () => {
    mockNonJsonResponse();
    const failure = await listExperiments().catch((error: unknown) => error);

    expect(errorCode(failure)).toBe(CLIENT_ERROR_CODES.MALFORMED);
    // The HTML body must not leak into what a person sees.
    expect(friendlyMessage(failure)).not.toContain("Bad Gateway");
  });

  it("refuses a 200 whose body is not an object", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(JSON.stringify("just a string"), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          }),
      ),
    );

    const failure = await requestJson("/api/v1/experiments").catch(
      (error: unknown) => error,
    );
    expect(errorCode(failure)).toBe(CLIENT_ERROR_CODES.MALFORMED);
  });
});

describe("friendly error messages", () => {
  it.each([
    ["invalid_excel", "Excel file could not be read"],
    ["invalid_json", "could not be read as a table"],
    ["agent_provider_error", "temporarily unavailable"],
    ["retrieval_index_not_built", "currently unavailable"],
    ["unsupported_file_type", "CSV, an Excel workbook"],
    ["file_too_large", "larger than the upload limit"],
  ])("maps %s to something a person can act on", (code, fragment) => {
    const message = friendlyMessage(new ApiError(code, "raw backend text", 422));
    expect(message).toContain(fragment);
    expect(message).not.toContain("raw backend text");
  });

  it("falls back to the backend's own message for an unmapped code", () => {
    const message = friendlyMessage(
      new ApiError("some_new_code", "A future error, explained plainly.", 400),
    );
    expect(message).toBe("A future error, explained plainly.");
  });

  it("never renders a traceback, a path or a key even if one arrives", () => {
    // The backend guarantees it does not send these. If one ever did, the
    // mapper must not be the thing that puts it on screen unchallenged, so
    // this pins the mapper's own output for the codes it owns.
    const message = friendlyMessage(
      new ApiError(
        "llm_provider_error",
        'Traceback: /home/app/llm.py, key sk-live-123',
        502,
      ),
    );
    expect(message).not.toContain("Traceback");
    expect(message).not.toContain("/home/");
    expect(message).not.toContain("sk-");
  });
});

describe("multipart uploads", () => {
  it("posts the file and question as form fields", async () => {
    const backend = mockBackend([
      { match: "/api/v1/agent/ask-with-dataset", body: { status: "completed" } },
    ]);

    await askAgentWithDataset(csvFile(), "Which model is best?", {
      max_tool_calls: 4,
    });

    const form = backend.lastForm();
    expect(form).not.toBeNull();
    expect((form?.get("file") as File).name).toBe("customers.csv");
    expect(form?.get("question")).toBe("Which model is best?");
    expect(form?.get("max_tool_calls")).toBe("4");
  });

  it("does not set Content-Type, so the browser adds the boundary", async () => {
    const fetchSpy = vi.fn(
      async (_input: RequestInfo | URL, _init?: RequestInit) =>
        new Response(JSON.stringify(CLASSIFICATION_PROFILE), { status: 200 }),
    );
    vi.stubGlobal("fetch", fetchSpy);

    await profileDataset(csvFile());

    const init = fetchSpy.mock.calls[0][1];
    expect(init?.headers).toBeUndefined();
  });
});
