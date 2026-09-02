import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tsconfigPaths from "vite-tsconfig-paths";

/**
 * Component tests in jsdom, with the API mocked at `fetch`.
 *
 * Mocking the transport rather than the client modules means the typed client,
 * the error mapper and every component are all under test on the real path a
 * request takes. No test needs a running backend, and none needs a credential.
 */
export default defineConfig({
  plugins: [tsconfigPaths(), react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
    css: false,
    restoreMocks: true,
    // A stubbed `fetch` or a changed API URL must not leak into the next test.
    unstubGlobals: true,
    unstubEnvs: true,
  },
});
