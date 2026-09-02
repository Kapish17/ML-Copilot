import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { FlatCompat } from "@eslint/eslintrc";

const compat = new FlatCompat({
  baseDirectory: dirname(fileURLToPath(import.meta.url)),
});

/** Next's recommended rules plus its TypeScript rules, and nothing else. */
const config = [
  ...compat.extends("next/core-web-vitals", "next/typescript"),
  {
    ignores: [".next/**", "node_modules/**", "next-env.d.ts", "coverage/**"],
  },
  {
    rules: {
      // API responses are typed; `any` should never be needed for them.
      "@typescript-eslint/no-explicit-any": "error",
      // A leading underscore marks a parameter kept only to satisfy a
      // signature — a typed test double, an ignored callback argument.
      "@typescript-eslint/no-unused-vars": [
        "warn",
        { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
      ],
    },
  },
];

export default config;
