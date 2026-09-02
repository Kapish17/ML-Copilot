/**
 * Next.js configuration.
 *
 * Deliberately close to empty. The backend is reached over HTTP at the URL in
 * NEXT_PUBLIC_API_BASE_URL, not through a rewrite, so there is no build-time
 * coupling between the two services and the same bundle runs against a local
 * backend or a deployed one.
 *
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
};

export default nextConfig;
