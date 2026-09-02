"""Container healthcheck: does the API actually answer?

A process-liveness check would pass while the application was still importing
scikit-learn, or after it had wedged. This asks ``/health`` — the endpoint that
has existed since Commit 1 — and exits non-zero on anything but a 200. No new
API surface was added for Docker's benefit.

Run as the image's ``HEALTHCHECK`` and as the Compose healthcheck, so the two
can never drift apart.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

TIMEOUT_SECONDS = 4


def main() -> int:
    """Return 0 when the API reports itself healthy, 1 otherwise."""
    port = os.environ.get("API_PORT", "8000")
    url = f"http://127.0.0.1:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_SECONDS) as response:
            return 0 if response.status == 200 else 1
    except (urllib.error.URLError, OSError, ValueError):
        return 1


if __name__ == "__main__":
    sys.exit(main())
