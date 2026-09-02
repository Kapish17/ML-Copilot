import { redirect } from "next/navigation";

/**
 * The root, which is the dashboard.
 *
 * A separate marketing landing page would be a page a person passes through
 * on the way to the thing they came for. The dashboard *is* the product, and
 * it introduces itself, so `/` goes straight there.
 */
export default function HomePage() {
  redirect("/dashboard");
}
