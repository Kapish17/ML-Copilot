/**
 * Next.js configuration.
 *
 * Deliberately close to empty. The backend is reached over HTTP at the URL in
 * NEXT_PUBLIC_API_BASE_URL, not through a rewrite, so there is no build-time
 * coupling between the two services and the same bundle runs against a local
 * backend or a deployed one.
 *
 * `output: "standalone"` is what lets the container image ship a runtime
 * stage without the build tooling: the build emits a self-contained server
 * and only the `node_modules` the application actually reaches. It is purely
 * additive — `next dev` and `next start` behave exactly as before.
 *
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  output: "standalone",
};

export default nextConfig;
