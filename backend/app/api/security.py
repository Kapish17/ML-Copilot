"""API-key authentication.

**What this is.** One static bearer token, held on the server, checked in
constant time, required on the endpoints that cost something or touch stored
state. It is off unless switched on.

**What this is not.** It is not identity. There are no users, no roles, no
sessions, no expiry and no revocation short of changing the key and
restarting. Everyone who holds the key is the same caller, and the log cannot
tell them apart. That is a real limitation, written down here and in
`docs/PRODUCTION_READINESS.md` rather than left for someone to discover.

**Why a static key is nonetheless the right answer for this commit.** The
threat it addresses is precise: a service with no authentication accepts file
uploads and runs cross-validated model training synchronously, so anyone who
can reach the port can spend the host's CPU. A shared key stops that. It does
not stop an authorised caller from doing the same thing, which is why the
budgets that bound every expensive operation matter as much as this file does.

**Why no library.** JWT, OAuth and session management all bring key rotation,
clock skew, revocation lists and a dependency tree, and none of them makes a
single-tenant local tool safer than comparing one string in constant time.
FastAPI's own `HTTPBearer` supplies the OpenAPI security scheme, and
`secrets` supplies the comparison. Nothing else is needed.

---

**The comparison.** `secrets.compare_digest` over the SHA-256 digests of the
two values rather than over the values themselves. Two reasons: the digests
are fixed-length, so the comparison leaks nothing about the *length* of the
configured key (`compare_digest` is documented as not hiding it), and they are
bytes, so a key containing a non-ASCII character cannot raise `TypeError` on
the authentication path.

**What a failure returns.** 401, the project's standard error envelope, a
`WWW-Authenticate: Bearer` header, and a message that is the same for every
wrong key. Nothing distinguishes a near miss from a random guess, nothing
echoes what was sent, and nothing carries a traceback or a path.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from typing import Annotated

from fastapi import Depends, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.api.dependencies import SettingsDep
from app.core.config import Settings
from app.core.errors import AuthenticationRequiredError, InvalidCredentialsError
from app.schemas.errors import ErrorResponse

logger = logging.getLogger(__name__)

#: The scheme FastAPI documents in the OpenAPI schema. `auto_error=False` is
#: what makes the failures this project's own: left on, `HTTPBearer` raises its
#: own `HTTPException` with FastAPI's `{"detail": ...}` body and a 403 for a
#: missing header, which would be a second error format and the wrong status.
bearer_scheme = HTTPBearer(
    scheme_name="API key",
    # An opaque string, not a JWT. Saying so in the schema stops a reader
    # assuming there is a payload to decode or an expiry to check.
    bearerFormat="opaque",
    description=(
        "The API key configured on the server as `API_AUTH_KEY`, sent as "
        "`Authorization: Bearer <key>`. Required only when the deployment sets "
        "`API_AUTH_ENABLED=true`; the local and demo configuration leaves it "
        "off and these endpoints need no credential."
    ),
    auto_error=False,
)

CredentialsDep = Annotated[
    HTTPAuthorizationCredentials | None, Security(bearer_scheme)
]

#: The documented 401, attached to every protected route so the generated
#: documentation shows it rather than leaving a reader to discover it.
UNAUTHORIZED_RESPONSE: dict[int | str, dict[str, object]] = {
    status.HTTP_401_UNAUTHORIZED: {
        "model": ErrorResponse,
        "description": (
            "Authentication is enabled on this deployment and the request "
            "carried no usable credential. Send `Authorization: Bearer <key>`."
        ),
    }
}


def _digest(value: str) -> bytes:
    """Return a fixed-length digest of a credential, for constant-time comparison."""
    return hashlib.sha256(value.encode("utf-8")).digest()


def verify_api_key(settings: Settings, credentials: object | None) -> None:
    """Check one request's credential against the configured key.

    Separated from the dependency so the rule can be exercised directly, with
    no HTTP and no application, by the tests that care about the comparison
    rather than about routing.

    Args:
        settings: The active settings, carrying the switch and the key.
        credentials: What the bearer scheme parsed out of the request, or
            ``None`` when there was no usable ``Authorization`` header.

    Raises:
        AuthenticationRequiredError: If nothing usable was presented.
        InvalidCredentialsError: If what was presented is not the key.
    """
    if not settings.api_auth_enabled:
        return

    token = getattr(credentials, "credentials", "") or ""
    if not token.strip():
        # No header, a header that is not `Bearer`, or `Bearer` with nothing
        # after it. All three are "you have not authenticated", not "your key
        # is wrong" — and the log line says so without quoting the request.
        logger.warning("Authentication failed: no credential presented")
        raise AuthenticationRequiredError(
            "Authentication required. Send an API key as "
            "'Authorization: Bearer <key>'."
        )

    if not secrets.compare_digest(_digest(token), _digest(settings.api_auth_key)):
        # **The supplied value is not logged.** A wrong key is very often the
        # right key for somewhere else, or the right key with a character
        # missing; either way, writing it into a log file turns a failed
        # request into a stored credential.
        logger.warning("Authentication failed: credential rejected")
        raise InvalidCredentialsError("Invalid authentication credentials.")


def require_api_key(settings: SettingsDep, credentials: CredentialsDep) -> None:
    """FastAPI dependency guarding one protected route.

    Returns nothing: a route does not need to know who called it, only that
    the call was allowed. Failure raises, and the application's existing
    handler turns it into the standard envelope — so a 401 is the same shape
    as every other error this API produces.

    When authentication is disabled this is a boolean check and a return,
    which is what keeps the local and demo experience exactly as it was.
    """
    verify_api_key(settings, credentials)


#: What a protected route declares. One name, so adding a route means adding
#: one thing rather than remembering three, and so a reader can grep for it.
Protected = Depends(require_api_key)


__all__ = [
    "CredentialsDep",
    "Protected",
    "UNAUTHORIZED_RESPONSE",
    "bearer_scheme",
    "require_api_key",
    "verify_api_key",
]
