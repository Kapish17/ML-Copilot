"""Tests for API-key authentication, and for the key never getting out.

Two halves, and the second is the one that would be missed.

**Does the lock work?** Every failure mode returns 401 in the project's own
error envelope, the comparison is constant-time, and the classification of
routes into public and protected is computed from the running application
rather than trusted to a list somebody maintained by hand — so a new POST route
added later fails these tests until someone decides what it is.

**Does the key stay in?** A shared secret has more ways out than in. It can be
echoed in an error, written to a log, published in the OpenAPI schema, passed
to the frontend service by Compose, baked into an image, committed to
`.env.example`, or quoted in documentation. Each of those is a separate test,
because each is a separate mistake, and the last five are the ones no amount of
careful backend code prevents.

The key used throughout is obviously fake and obviously test-only. One test
searches the whole repository for it, which is what makes running this suite
unable to leave a credential behind.
"""

from __future__ import annotations

import io
import json
import logging
import re
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.security import require_api_key, verify_api_key
from app.core.config import MIN_API_AUTH_KEY_LENGTH, Settings, _env_bool
from app.core.errors import AuthenticationRequiredError, InvalidCredentialsError
from app.core.logging import REQUEST_ID_HEADER
from app.main import create_app

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = REPOSITORY_ROOT / "frontend"

#: The credential these tests configure. Unmistakably fake, unmistakably
#: test-only, and long enough to satisfy the minimum length so the tests
#: exercise the real path rather than the start-up guard.
TEST_KEY = "test-key-not-a-real-secret-000000000000"

#: A wrong key that is *nearly* right. Nothing in any response or log may
#: distinguish it from a random guess.
NEAR_MISS = TEST_KEY[:-1] + "1"

PROFILE_URL = "/api/v1/datasets/profile"
SEARCH_URL = "/api/v1/search"

#: The endpoints that stay open, and why each one has to.
PUBLIC_ROUTES = {
    ("GET", "/"),  # identity, and whether a key is required at all
    ("GET", "/health"),  # both container healthchecks call this
    ("GET", "/api/v1/experiments/capabilities"),  # models, metrics, limits
    ("GET", "/api/v1/knowledge/status"),  # availability booleans
    ("GET", "/api/v1/agent/status"),  # availability booleans
}

#: Everything that costs CPU, touches stored state, accepts an upload, or
#: reaches a language model.
PROTECTED_ROUTES = {
    ("POST", "/api/v1/datasets/profile"),
    ("POST", "/api/v1/experiments/run"),
    ("GET", "/api/v1/experiments"),
    ("GET", "/api/v1/experiments/{experiment_id}"),
    ("POST", "/api/v1/experiments/compare"),
    ("POST", "/api/v1/search"),
    ("POST", "/api/v1/ask"),
    ("POST", "/api/v1/agent/ask"),
    ("POST", "/api/v1/agent/ask-with-dataset"),
}


@pytest.fixture(scope="module")
def secured_client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    """A client for an application with authentication switched on."""
    settings = Settings(
        api_auth_enabled=True,
        api_auth_key=TEST_KEY,
        experiment_store_dir=tmp_path_factory.mktemp("secured-store"),
    )
    with TestClient(create_app(settings)) as client:
        yield client


def auth(key: str = TEST_KEY) -> dict[str, str]:
    """Build the Authorization header."""
    return {"Authorization": f"Bearer {key}"}


def upload() -> dict[str, tuple]:
    """A small valid CSV upload."""
    content = b"a,b,label\n1,2,yes\n3,4,no\n5,6,yes\n"
    return {"file": ("data.csv", io.BytesIO(content), "text/csv")}


def routes_of(client: TestClient) -> dict[tuple[str, str], dict]:
    """Every documented operation, keyed by method and path."""
    schema = client.get("/openapi.json").json()
    return {
        (method.upper(), path): operation
        for path, operations in schema["paths"].items()
        for method, operation in operations.items()
    }


# ---------------------------------------------------------------------------
# Disabled: the local and demo experience, unchanged
# ---------------------------------------------------------------------------


def test_authentication_is_off_unless_it_is_turned_on() -> None:
    """The default, and the reason `docker compose up` needs no secret."""
    assert Settings().api_auth_enabled is False


def test_no_default_key_exists() -> None:
    """None is shipped, and none is generated.

    A generated key would be a secret nobody knows that changes on every
    restart; a shipped one would be a password published in a public
    repository. Both are worse than refusing to start.
    """
    assert Settings().api_auth_key == ""

    # And the source says so in both places it could say otherwise: the field's
    # default is the empty string, and the environment lookup falls back to the
    # empty string rather than to a value.
    config = (REPOSITORY_ROOT / "backend/app/core/config.py").read_text("utf-8")

    assert 'api_auth_key: str = ""' in config
    assert 'os.getenv("API_AUTH_KEY", "")' in config
    # Nothing invents one, either. A generated key would differ on every
    # restart and appear in no operator's notes — and generating one needs
    # `secrets`, which this module does not import. (It names
    # `secrets.token_urlsafe` in a comment and in the error message, as advice
    # to a human; that is why this checks the import rather than the text.)
    assert not re.search(r"^\s*(import secrets|from secrets import)", config, re.M)


@pytest.mark.parametrize(
    "method,path",
    [("POST", PROFILE_URL), ("GET", "/api/v1/experiments")],
)
def test_protected_routes_are_open_when_authentication_is_disabled(
    client: TestClient, method: str, path: str
) -> None:
    """The regression that matters: the demo must behave exactly as before."""
    response = (
        client.post(path, files=upload()) if method == "POST" else client.get(path)
    )

    assert response.status_code != 401


# ---------------------------------------------------------------------------
# Enabled: the four outcomes
# ---------------------------------------------------------------------------


def test_a_correct_key_is_admitted(secured_client: TestClient) -> None:
    """The endpoint then behaves exactly as it does without authentication."""
    response = secured_client.post(PROFILE_URL, files=upload(), headers=auth())

    assert response.status_code == 200
    assert response.json()["dataset"]["row_count"] == 3


def test_a_missing_key_is_refused(secured_client: TestClient) -> None:
    """No `Authorization` header at all."""
    response = secured_client.post(PROFILE_URL, files=upload())

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_required"


def test_a_wrong_key_is_refused(secured_client: TestClient) -> None:
    """A well-formed credential that is not the configured one."""
    response = secured_client.post(
        PROFILE_URL, files=upload(), headers=auth("completely-wrong-key-0000000000000")
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_credentials"


@pytest.mark.parametrize(
    "header",
    [
        "",  # present but empty
        "Bearer",  # scheme with no token
        "Bearer ",  # scheme with whitespace
        f"Basic {TEST_KEY}",  # right key, wrong scheme
        TEST_KEY,  # right key, no scheme
        f"bearer{TEST_KEY}",  # no separator
    ],
)
def test_a_malformed_authorization_header_is_refused(
    secured_client: TestClient, header: str
) -> None:
    """Every shape of "not a bearer credential" is 401, never 403 and never 500.

    `Basic <key>` is the interesting one: the value is correct and the scheme
    is not, and accepting it would mean the scheme was decorative.
    """
    response = secured_client.post(
        PROFILE_URL, files=upload(), headers={"Authorization": header}
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_required"


def test_a_near_miss_is_refused_exactly_like_a_random_guess(
    secured_client: TestClient,
) -> None:
    """No oracle. One character out must be indistinguishable from nonsense."""
    near = secured_client.post(PROFILE_URL, files=upload(), headers=auth(NEAR_MISS))
    random = secured_client.post(
        PROFILE_URL, files=upload(), headers=auth("x" * len(TEST_KEY))
    )
    shorter = secured_client.post(PROFILE_URL, files=upload(), headers=auth("x"))

    bodies = {near.text, random.text, shorter.text}
    assert {near.status_code, random.status_code, shorter.status_code} == {401}
    assert len(bodies) == 1, "the response distinguishes between wrong keys"


def test_the_comparison_is_constant_time_and_length_blind() -> None:
    """Asserted on the implementation, because timing is not testable here.

    A wall-clock test on a shared runner measures the runner. What is checkable
    is that the code compares fixed-length digests with `secrets.compare_digest`
    and never with `==` — `compare_digest` is documented as not hiding the
    length of its operands, which is why the digests exist.
    """
    source = (REPOSITORY_ROOT / "backend/app/api/security.py").read_text("utf-8")

    # The call itself, not merely the name — the name also appears in the
    # module docstring, so a mutation that swapped the comparison for `==`
    # would leave a substring check passing.
    assert re.search(
        r"secrets\.compare_digest\(\s*_digest\(\s*token\s*\)\s*,\s*"
        r"_digest\(\s*settings\.api_auth_key\s*\)\s*\)",
        source,
    ), "the credential is not compared with compare_digest over digests"
    assert "hashlib.sha256" in source

    # And no direct comparison of the credential anywhere, in either direction.
    assert not re.search(r"token\s*[=!]=|[=!]=\s*settings\.api_auth_key", source)


def test_the_rule_can_be_checked_without_an_application() -> None:
    """The same decision, exercised directly."""
    enabled = Settings(api_auth_enabled=True, api_auth_key=TEST_KEY)
    disabled = Settings()

    class Credential:
        def __init__(self, value: str) -> None:
            self.credentials = value

    verify_api_key(enabled, Credential(TEST_KEY))  # does not raise
    verify_api_key(disabled, None)  # disabled: anything goes

    with pytest.raises(AuthenticationRequiredError):
        verify_api_key(enabled, None)
    with pytest.raises(InvalidCredentialsError):
        verify_api_key(enabled, Credential(NEAR_MISS))


# ---------------------------------------------------------------------------
# The refusal itself
# ---------------------------------------------------------------------------


def test_a_401_uses_the_existing_error_envelope(secured_client: TestClient) -> None:
    """One error format for the whole API, authentication included."""
    body = secured_client.post(PROFILE_URL, files=upload()).json()

    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "details"}
    assert isinstance(body["error"]["message"], str) and body["error"]["message"]


def test_a_401_says_how_to_authenticate(secured_client: TestClient) -> None:
    """`WWW-Authenticate`, which HTTP requires on a 401."""
    response = secured_client.post(PROFILE_URL, files=upload())

    assert response.headers["www-authenticate"] == "Bearer"


def test_a_401_carries_a_request_id(secured_client: TestClient) -> None:
    """A refused request is exactly the one someone will ask about."""
    response = secured_client.post(
        PROFILE_URL, files=upload(), headers={REQUEST_ID_HEADER: "auth-trace-1"}
    )

    assert response.status_code == 401
    assert response.headers[REQUEST_ID_HEADER] == "auth-trace-1"


def test_a_401_carries_no_traceback_path_or_internal_detail(
    secured_client: TestClient,
) -> None:
    """It is a refusal, not a crash."""
    text = secured_client.post(PROFILE_URL, files=upload(), headers=auth("nope")).text

    for leak in ("Traceback", "/home/", "C:\\", "app.api.security", "compare_digest"):
        assert leak not in text


def test_the_supplied_credential_is_never_echoed(secured_client: TestClient) -> None:
    """Not in the body, and not in any header.

    Echoing a rejected credential puts it into the caller's logs, into every
    proxy in between and into a browser's network panel — and tells whoever
    sent it that it arrived intact.
    """
    for candidate in (TEST_KEY, NEAR_MISS, "guessed-secret-value"):
        response = secured_client.post(PROFILE_URL, files=upload(), headers=auth(candidate))
        rendered = response.text + json.dumps(dict(response.headers))
        if candidate == TEST_KEY:
            continue  # the correct key succeeds; nothing to echo
        assert candidate not in rendered


def test_a_failure_is_logged_without_the_credential(
    secured_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """"Authentication failed" is useful. The value that failed is a liability."""
    with caplog.at_level(logging.DEBUG):
        secured_client.post(PROFILE_URL, files=upload(), headers=auth(NEAR_MISS))
        secured_client.post(PROFILE_URL, files=upload())

    assert "Authentication failed" in caplog.text
    assert NEAR_MISS not in caplog.text
    assert TEST_KEY not in caplog.text
    # And the header itself is never written, whole or in part.
    assert "Authorization" not in caplog.text
    assert "Bearer" not in caplog.text


# ---------------------------------------------------------------------------
# Which routes are which
# ---------------------------------------------------------------------------


def test_the_public_and_protected_sets_are_exactly_as_declared(
    client: TestClient,
) -> None:
    """Computed from the running application, not from a hand-kept list."""
    routes = routes_of(client)
    protected = {key for key, operation in routes.items() if operation.get("security")}
    public = set(routes) - protected

    assert protected == PROTECTED_ROUTES
    assert public == PUBLIC_ROUTES


def test_no_route_that_changes_or_costs_anything_is_public(
    client: TestClient,
) -> None:
    """The rule, so a route added later fails until someone classifies it.

    Every POST either uploads a file, trains a model, or reaches a language
    model. There is no cheap POST in this API and there is unlikely ever to be
    one, so "POST implies protected" is a rule rather than a coincidence.
    """
    routes = routes_of(client)

    unprotected_writes = [
        key
        for key, operation in routes.items()
        if key[0] != "GET" and not operation.get("security")
    ]

    assert not unprotected_writes, f"expensive routes left public: {unprotected_writes}"


def test_every_public_route_answers_without_a_credential(
    secured_client: TestClient,
) -> None:
    """On a *secured* deployment, which is the case that matters."""
    for method, path in sorted(PUBLIC_ROUTES):
        assert method == "GET", path
        response = secured_client.get(path)
        assert response.status_code == 200, f"{path} -> {response.status_code}"


def test_health_stays_public_so_a_healthcheck_needs_no_secret(
    secured_client: TestClient,
) -> None:
    """A healthcheck carrying the key would put it in every host process list."""
    assert secured_client.get("/health").json()["status"] == "ok"

    healthcheck = (REPOSITORY_ROOT / "backend/healthcheck.py").read_text("utf-8")
    assert "/health" in healthcheck
    assert "API_AUTH_KEY" not in healthcheck
    assert "Authorization" not in healthcheck


def test_the_service_reports_whether_a_key_is_required(
    client: TestClient, secured_client: TestClient
) -> None:
    """A boolean about configuration, so a client can say so before failing.

    Not a disclosure: the same fact is one unauthenticated request away.
    """
    assert client.get("/").json()["authentication_required"] is False

    body = secured_client.get("/").json()
    assert body["authentication_required"] is True
    assert TEST_KEY not in json.dumps(body)


# ---------------------------------------------------------------------------
# Configuration fails safely
# ---------------------------------------------------------------------------


def test_enabling_authentication_without_a_key_refuses_to_start() -> None:
    """The dangerous state is "protected" that is not. It is not reachable.

    Raised by `Settings` itself rather than by `get_settings`, so it holds for
    every construction — including the ones a test writes by hand.
    """
    with pytest.raises(ValueError, match="API_AUTH_KEY is empty"):
        Settings(api_auth_enabled=True)

    with pytest.raises(ValueError, match="API_AUTH_KEY is empty"):
        Settings(api_auth_enabled=True, api_auth_key="   ")


def test_a_short_key_is_refused() -> None:
    """The failure mode that actually happens: `API_AUTH_KEY=test`."""
    with pytest.raises(ValueError, match=str(MIN_API_AUTH_KEY_LENGTH)):
        Settings(api_auth_enabled=True, api_auth_key="short")


def test_the_error_explains_how_to_generate_one_without_generating_one() -> None:
    """Help, not a fallback."""
    with pytest.raises(ValueError) as raised:
        Settings(api_auth_enabled=True, api_auth_key="short")

    assert "token_urlsafe" in str(raised.value)


@pytest.mark.parametrize("value", ["ture", "yes please", "maybe", "1.0"])
def test_an_unreadable_switch_value_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """`API_AUTH_ENABLED=ture` must not quietly mean "off".

    A security switch decided by a guess is worse than no switch, because the
    operator believes they set it.
    """
    monkeypatch.setenv("API_AUTH_ENABLED", value)

    with pytest.raises(ValueError, match="API_AUTH_ENABLED"):
        _env_bool("API_AUTH_ENABLED", False)


@pytest.mark.parametrize(
    "value,expected",
    [("true", True), ("TRUE", True), ("1", True), ("on", True),
     ("false", False), ("0", False), ("off", False), ("", False)],
)
def test_the_switch_reads_the_spellings_people_actually_write(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: bool
) -> None:
    """Including the empty string, which is what an unset `.env` line leaves."""
    monkeypatch.setenv("API_AUTH_ENABLED", value)

    assert _env_bool("API_AUTH_ENABLED", False) is expected


# ---------------------------------------------------------------------------
# The key does not get out
# ---------------------------------------------------------------------------


def test_the_openapi_schema_documents_the_scheme_and_not_the_key(
    secured_client: TestClient,
) -> None:
    """A reader should learn *that* a bearer token is needed, and nothing more."""
    schema = secured_client.get("/openapi.json").json()
    rendered = json.dumps(schema)

    schemes = schema["components"]["securitySchemes"]
    assert any(
        scheme.get("type") == "http" and scheme.get("scheme") == "bearer"
        for scheme in schemes.values()
    )
    assert TEST_KEY not in rendered
    assert "API_AUTH_KEY=" not in rendered


def test_the_frontend_never_learns_the_key_exists() -> None:
    """No `NEXT_PUBLIC_` secret, and not even the variable's name.

    The rule is not "do not put the key in the bundle" — it is that a browser
    application cannot hold a shared secret at all, so the frontend has no
    business knowing the name of the variable that holds one.
    """
    sources = [
        path
        for pattern in ("*.ts", "*.tsx", "*.mjs", "*.json")
        for path in FRONTEND.rglob(pattern)
        if "node_modules" not in path.parts and ".next" not in path.parts
    ]
    assert sources, "the frontend source walk found nothing"

    offenders: list[str] = []
    for path in sources:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "API_AUTH_KEY" in text or re.search(
            r"NEXT_PUBLIC_\w*(KEY|SECRET|TOKEN|PASSWORD)", text
        ):
            offenders.append(str(path.relative_to(REPOSITORY_ROOT)))

    assert not offenders, f"the frontend references a server secret: {offenders}"


def test_the_built_frontend_bundle_carries_no_credential() -> None:
    """Checked against the real build output when there is one.

    Skipped rather than faked when `.next` is absent: the backend CI job does
    not build the frontend, and a test that silently passes on a missing
    directory is worse than one that says it did not run. The frontend job and
    the Docker smoke test both cover the built artefact.
    """
    build = FRONTEND / ".next"
    if not build.is_dir():
        pytest.skip("no production build present; run `npm run build` in frontend/")

    offenders: list[str] = []
    for path in build.rglob("*.js"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if TEST_KEY in text or "API_AUTH_KEY" in text:
            offenders.append(str(path.relative_to(REPOSITORY_ROOT)))

    assert not offenders, f"the browser bundle carries a server secret: {offenders}"


def test_compose_gives_the_key_to_the_backend_and_only_the_backend() -> None:
    """Resolved by Compose itself, so this is what would actually be passed."""
    if not shutil_which("docker"):
        pytest.skip("docker is not installed")

    result = subprocess.run(
        ["docker", "compose", "config", "--format", "json"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip(f"docker compose config failed: {result.stderr.strip()[:200]}")

    services = json.loads(result.stdout)["services"]

    backend = services["backend"]["environment"]
    assert "API_AUTH_ENABLED" in backend and "API_AUTH_KEY" in backend
    assert backend["API_AUTH_KEY"] == "", "a key is committed in the compose file"

    frontend = services["frontend"]
    rendered = json.dumps(frontend)
    assert "API_AUTH_KEY" not in rendered, "Compose passes the key to the frontend"
    assert "API_AUTH_ENABLED" not in rendered


def shutil_which(name: str) -> str | None:
    """`shutil.which`, imported lazily to keep the import block small."""
    import shutil

    return shutil.which(name)


def test_no_dockerfile_mentions_the_key() -> None:
    """A secret in an image layer is in every copy of that image, forever."""
    for dockerfile in (
        REPOSITORY_ROOT / "backend/Dockerfile",
        REPOSITORY_ROOT / "frontend/Dockerfile",
    ):
        text = dockerfile.read_text(encoding="utf-8")
        assert "API_AUTH_KEY" not in text, dockerfile.name


def test_the_env_example_documents_the_variable_with_no_value() -> None:
    """Documentation, not a credential."""
    text = (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8")

    assert re.search(r"^API_AUTH_ENABLED=false$", text, re.M)
    assert re.search(r"^API_AUTH_KEY=$", text, re.M), "API_AUTH_KEY has a value"


def test_the_test_key_exists_nowhere_but_this_file() -> None:
    """The search the whole exercise depends on.

    Running this suite puts the key into a live application's settings, into
    request headers, into captured logs and into rendered responses. If any of
    that reached disk, it would be here. Everything is searched — source,
    configuration, documentation, the demo data, the build output — except
    this file, which is where the key is supposed to be.
    """
    ignored = {
        ".git",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".venv",
        "venv",
    }
    this_file = Path(__file__).resolve()

    offenders: list[str] = []
    for path in REPOSITORY_ROOT.rglob("*"):
        if not path.is_file() or path.resolve() == this_file:
            continue
        if ignored.intersection(path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:  # pragma: no cover - unreadable file
            continue
        if TEST_KEY in text:
            offenders.append(str(path.relative_to(REPOSITORY_ROOT)))

    assert not offenders, f"the test key was written to: {offenders}"


def test_the_documentation_never_quotes_a_usable_key() -> None:
    """An example key in a README is a key somebody will paste into production."""
    pattern = re.compile(r"API_AUTH_KEY\s*[=:]\s*([^\s`\"']+)")

    offenders: list[str] = []
    for path in REPOSITORY_ROOT.rglob("*.md"):
        if {"node_modules", ".next"}.intersection(path.parts):
            continue
        for value in pattern.findall(path.read_text(encoding="utf-8")):
            # A placeholder is fine; a string long enough to be a real key is not.
            if len(value) >= MIN_API_AUTH_KEY_LENGTH and not value.startswith("<"):
                offenders.append(f"{path.relative_to(REPOSITORY_ROOT)}: {value[:12]}…")

    assert not offenders, f"documentation quotes a usable key: {offenders}"


# ---------------------------------------------------------------------------
# The dependency is wired, not merely defined
# ---------------------------------------------------------------------------


def test_every_protected_route_actually_calls_the_dependency(
    secured_client: TestClient,
) -> None:
    """Declaring `security` in the schema and enforcing it are different things.

    A route could document a padlock and check nothing. Each protected route is
    called without a credential and must refuse — which is the only way to know
    the dependency is attached rather than described.
    """
    calls = {
        ("POST", "/api/v1/datasets/profile"): lambda: secured_client.post(
            PROFILE_URL, files=upload()
        ),
        ("POST", "/api/v1/experiments/run"): lambda: secured_client.post(
            "/api/v1/experiments/run", files=upload()
        ),
        ("GET", "/api/v1/experiments"): lambda: secured_client.get(
            "/api/v1/experiments"
        ),
        ("GET", "/api/v1/experiments/{experiment_id}"): lambda: secured_client.get(
            "/api/v1/experiments/exp_whatever"
        ),
        ("POST", "/api/v1/experiments/compare"): lambda: secured_client.post(
            "/api/v1/experiments/compare", json={"experiment_ids": ["a", "b"]}
        ),
        ("POST", "/api/v1/search"): lambda: secured_client.post(
            SEARCH_URL, json={"query": "x"}
        ),
        ("POST", "/api/v1/ask"): lambda: secured_client.post(
            "/api/v1/ask", json={"question": "x"}
        ),
        ("POST", "/api/v1/agent/ask"): lambda: secured_client.post(
            "/api/v1/agent/ask", json={"question": "x"}
        ),
        ("POST", "/api/v1/agent/ask-with-dataset"): lambda: secured_client.post(
            "/api/v1/agent/ask-with-dataset",
            files=upload(),
            data={"question": "x"},
        ),
    }

    assert set(calls) == PROTECTED_ROUTES, "a protected route has no call here"

    for route, call in calls.items():
        response = call()
        assert response.status_code == 401, f"{route} answered {response.status_code}"
        assert response.json()["error"]["code"] == "authentication_required", route


def test_the_dependency_is_a_module_level_object_routes_can_share() -> None:
    """One name to add to a route, so there is one thing to forget rather than three."""
    assert callable(require_api_key)


# ---------------------------------------------------------------------------
# CORS, after authentication
# ---------------------------------------------------------------------------


def test_cors_still_permits_no_wildcard_and_no_ambient_credentials(
    secured_client: TestClient,
) -> None:
    """Enabling authentication must not have loosened anything.

    `allow_credentials` in particular stays off: a bearer token is a header the
    caller sets, never a cookie a browser attaches by itself, so nothing here
    needs ambient credentials — and turning them on would forbid the origin
    list from ever being widened safely.
    """
    response = secured_client.options(
        PROFILE_URL,
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "access-control-allow-credentials" not in response.headers


def test_a_browser_cannot_send_the_api_key_cross_origin() -> None:
    """`Authorization` is not in `allow_headers`, and that is the decision.

    The rest of this commit argues that a browser must not hold this key.
    Permitting the header cross-origin would serve no legitimate caller — a
    server-to-server client is not subject to CORS at all, and the supported
    browser path is a server-side proxy that adds the header, whose requests
    are not cross-origin browser requests either — while inviting exactly the
    anti-pattern the documentation warns about.

    So the refusal is the guardrail: putting the key into JavaScript fails
    visibly at the first request instead of quietly working with a leaked
    credential.
    """
    application = create_app(Settings())
    cors = next(
        middleware
        for middleware in application.user_middleware
        if middleware.cls.__name__ == "CORSMiddleware"
    )

    permitted = {header.lower() for header in cors.kwargs["allow_headers"]}
    assert "authorization" not in permitted
    assert cors.kwargs["allow_credentials"] is False
    assert "*" not in cors.kwargs["allow_origins"]
