import type { Config } from "tailwindcss";

/**
 * The design tokens the whole dashboard is built from.
 *
 * A restrained palette on purpose: one neutral scale carries the layout, and
 * colour is reserved for the few things that genuinely mean something — the
 * selected model, a severity, a metric direction. Nothing here is decorative.
 */
const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: {
          50: "#f6f7f9",
          100: "#eceef2",
          200: "#d6dae2",
          300: "#b2b9c7",
          400: "#8792a6",
          500: "#67728a",
          600: "#525b71",
          700: "#434a5c",
          800: "#3a3f4e",
          900: "#2f3340",
          950: "#1c1f28",
        },
        accent: {
          50: "#eef4ff",
          100: "#dbe6fe",
          200: "#bfd3fe",
          300: "#93b4fd",
          400: "#608cfa",
          500: "#3b66f6",
          600: "#2547eb",
          700: "#1d35d8",
          800: "#1e2faf",
          900: "#1e2d8a",
        },
      },
      fontFamily: {
        sans: [
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Segoe UI",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: [
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "Consolas",
          "Liberation Mono",
          "monospace",
        ],
      },
    },
  },
  plugins: [],
};

export default config;
