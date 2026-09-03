"""Tests for the container and Compose configuration.

Configuration is where a containerised stack usually goes wrong, and it fails
quietly: the browser is handed a hostname only the Compose network can
resolve, a wildcard creeps into CORS, a volume is added that turns a
request-scoped upload into permanent storage, or a `COPY` names a path
`.dockerignore` excludes. None of those break a unit test; all of them break
the product.

So these tests read the real files rather than describing them. Where Docker
is installed, the Compose configuration is validated by **Docker's own
parser** — `docker compose config` — rather than by a hand-rolled YAML reader,
because the question worth answering is "does Docker agree this is valid", not
"does my parser like it". Those tests skip when Docker is absent; everything
else runs everywhere.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPOSITORY_ROOT / "docker-compose.yml"
BACKEND_DOCKERFILE = REPOSITORY_ROOT / "backend" / "Dockerfile"
FRONTEND_DOCKERFILE = REPOSITORY_ROOT / "frontend" / "Dockerfile"
ROOT_DOCKERIGNORE = REPOSITORY_ROOT / ".dockerignore"
FRONTEND_DOCKERIGNORE = REPOSITORY_ROOT / "frontend" / ".dockerignore"
ENV_EXAMPLE = REPOSITORY_ROOT / ".env.example"

#: The published frontend origin. The browser loads the dashboard from here,
#: so this is what the backend has to permit and what the frontend's API URL
#: has to agree with.
FRONTEND_ORIGIN = "http://localhost:3000"
#: The published backend origin, as the browser must be able to resolve it.
BACKEND_ORIGIN = "http://localhost:8000"


@lru_cache(maxsize=1)
def compose_config() -> dict:
    """Return the Compose configuration as Docker itself resolves it.

    Returns:
        dict: The fully interpolated configuration.

    Raises:
        pytest.skip.Exception: If the Docker CLI is not installed, or the
            Compose plugin is missing. The configuration is still worth
            testing on a machine that has them, and not worth faking on one
            that does not.
    """
    if shutil.which("docker") is None:
        pytest.skip("Docker is not installed; Compose validation needs its parser")

    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "config", "--format", "json"],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
        timeout=120,
    )
    if result.returncode != 0:
        if "compose" in result.stderr and "is not a docker command" in result.stderr:
            pytest.skip("The Docker Compose plugin is not installed")
        pytest.fail(f"`docker compose config` rejected the file:\n{result.stderr}")

    return json.loads(result.stdout)


def dockerfile_instructions(path: Path) -> list[tuple[str, str]]:
    """Parse a Dockerfile into (instruction, arguments) pairs.

    Comments, blank lines and line continuations are folded away, so a test
    can ask about the instructions rather than about the formatting.

    Args:
        path: The Dockerfile to read.

    Returns:
        list[tuple[str, str]]: One entry per instruction, uppercased keyword
        first.
    """
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\\\s*\n", " ", text)  # join continuations

    instructions: list[tuple[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        keyword, _, arguments = stripped.partition(" ")
        instructions.append((keyword.upper(), arguments.strip()))
    return instructions


# ---------------------------------------------------------------------------
# The Compose file, validated by Docker
# ---------------------------------------------------------------------------


def test_the_compose_file_parses() -> None:
    """Docker's own parser accepts the file and resolves every variable."""
    config = compose_config()

    assert config["name"] == "ml-copilot"
    assert set(config["services"]) == {"backend", "frontend"}


def test_no_service_beyond_the_two_the_product_needs() -> None:
    """No database, queue, cache or proxy was added to make Docker work."""
    services = compose_config()["services"]

    for unwanted in ("redis", "celery", "postgres", "qdrant", "nginx", "worker"):
        assert unwanted not in services

    # And nothing sneaks in as an image on one of the two services.
    for service in services.values():
        image = service.get("image", "")
        assert not any(
            name in image for name in ("redis", "postgres", "qdrant", "nginx")
        )


def test_the_backend_builds_from_the_repository_root() -> None:
    """Its context must include ml, rag, llm, agent and the READMEs.

    The API imports all four packages and the retrieval layer indexes the
    project's own documentation, so a context rooted at `backend/` would
    produce an image that cannot start.
    """
    build = compose_config()["services"]["backend"]["build"]

    assert Path(build["context"]).resolve() == REPOSITORY_ROOT
    assert build["dockerfile"].endswith("backend/Dockerfile")


def test_the_browser_is_given_a_url_it_can_resolve() -> None:
    """The frontend's API URL must not be a Compose service name.

    The dashboard is a client-side application: the *browser* makes every API
    call. `http://backend:8000` resolves inside the Compose network and
    nowhere else, so a build carrying it would fail every request from the
    first page load — while looking perfectly correct in this file.
    """
    args = compose_config()["services"]["frontend"]["build"]["args"]
    api_url = args["NEXT_PUBLIC_API_BASE_URL"]

    assert api_url == BACKEND_ORIGIN
    assert "backend:" not in api_url
    assert not re.match(r"^https?://(backend|frontend|api|web)\b", api_url)


def test_the_backend_is_published_because_the_browser_calls_it() -> None:
    """The API port is published, and on the port the frontend was built for."""
    ports = compose_config()["services"]["backend"]["ports"]
    published = {str(entry["published"]) for entry in ports}

    assert "8000" in published
    assert BACKEND_ORIGIN.endswith(":8000")


def test_only_the_two_ports_a_person_uses_are_published() -> None:
    """No incidental port is exposed to the host."""
    services = compose_config()["services"]
    published = {
        str(entry["published"])
        for service in services.values()
        for entry in service.get("ports", [])
    }

    assert published == {"3000", "8000"}


def test_both_published_ports_are_bound_to_loopback_by_default() -> None:
    """`127.0.0.1`, not Docker's default of every interface.

    This API has no authentication — deliberately, it is a local analysis tool
    — and it accepts file uploads and runs training synchronously. Published on
    `0.0.0.0` it offers all of that to every machine on whatever network the
    host is attached to, and it does so past a host firewall rather than
    through it, because publishing a port installs a DNAT rule the firewall
    never sees.

    Serving other machines stays possible: both bindings interpolate
    `BIND_ADDRESS`, so `BIND_ADDRESS=0.0.0.0` is one line in `.env`. What this
    test fixes is which of the two you get by not deciding.
    """
    services = compose_config()["services"]
    bindings = {
        (name, str(entry["published"])): entry.get("host_ip")
        for name, service in services.items()
        for entry in service.get("ports", [])
    }

    assert bindings, "no port is published at all"
    for (service, port), host_ip in bindings.items():
        assert host_ip == "127.0.0.1", (
            f"{service}:{port} is published on {host_ip or '0.0.0.0'}, which "
            "offers an unauthenticated API to the whole network"
        )

    # The escape hatch exists, so the default is a choice rather than a limit.
    compose = COMPOSE_FILE.read_text(encoding="utf-8")
    assert "${BIND_ADDRESS:-127.0.0.1}" in compose


def test_cors_permits_the_dashboard_and_is_never_a_wildcard() -> None:
    """An explicit list. `*` would let any page on the internet call this API."""
    environment = compose_config()["services"]["backend"]["environment"]
    origins = environment["CORS_ALLOW_ORIGINS"]

    assert FRONTEND_ORIGIN in origins
    assert "*" not in origins
    # Every entry is a real origin, not a pattern or a bare hostname.
    for origin in origins.split(","):
        assert re.match(r"^https?://[^/*\s]+(:\d+)?$", origin.strip()), origin


def test_the_frontend_origin_and_the_cors_allowlist_agree() -> None:
    """The port the dashboard is published on is the one CORS permits.

    These are configured independently and a mismatch produces the single most
    confusing failure mode in the stack: the page loads, and every request on
    it is blocked by the browser with nothing in the server log.
    """
    config = compose_config()
    frontend_ports = {
        str(entry["published"]) for entry in config["services"]["frontend"]["ports"]
    }
    origins = config["services"]["backend"]["environment"]["CORS_ALLOW_ORIGINS"]

    permitted_ports = {origin.rsplit(":", 1)[-1] for origin in origins.split(",")}
    assert frontend_ports <= permitted_ports


def test_the_container_writes_its_state_to_the_volumes() -> None:
    """The two data paths point at the mounts, not at the source tree."""
    backend = compose_config()["services"]["backend"]
    environment = backend["environment"]
    targets = {mount["target"] for mount in backend["volumes"]}

    assert environment["EXPERIMENT_STORE_DIR"] == "/data/experiments"
    assert environment["RAG_INDEX_DIR"] == "/data/rag-index"
    assert targets == {"/data/experiments", "/data/rag-index"}


def test_no_volume_could_persist_an_uploaded_dataset() -> None:
    """Containerising must not quietly undo the Commit 14 guarantee.

    An uploaded dataset is parsed in memory for one request and released. A
    mount at an upload path — or a bind mount of the host's data directories —
    would turn that loan into permanent storage without a line of application
    code changing.
    """
    services = compose_config()["services"]

    for service in services.values():
        for mount in service.get("volumes", []):
            target = mount["target"]
            assert mount["type"] == "volume", f"bind mount at {target}"
            assert not re.search(r"upload|dataset|/data/raw|/tmp", target), target


def test_no_service_is_privileged_or_mounts_the_host() -> None:
    """Nothing here needs elevated capabilities or the host filesystem."""
    services = compose_config()["services"]

    for name, service in services.items():
        assert not service.get("privileged"), name
        assert not service.get("cap_add"), name
        assert service.get("network_mode") != "host", name
        assert not service.get("pid"), name


def test_the_frontend_waits_for_a_healthy_backend() -> None:
    """Ordering is on the healthcheck, not on the process having started."""
    depends = compose_config()["services"]["frontend"]["depends_on"]

    assert depends["backend"]["condition"] == "service_healthy"


def test_the_stack_restarts_itself_for_a_local_demo() -> None:
    """A crashed container comes back, but a stopped stack stays stopped."""
    for service in compose_config()["services"].values():
        assert service["restart"] == "unless-stopped"


def test_no_credential_is_written_into_the_compose_file() -> None:
    """The key is read from `.env`; nothing resembling one is committed here."""
    text = COMPOSE_FILE.read_text(encoding="utf-8")

    assert not re.search(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]", text)
    # The key is referenced by name, with an empty default, and never assigned.
    assert "${LLM_API_KEY:-}" in text
    assert not re.search(r"LLM_API_KEY:\s*[\"']?[A-Za-z0-9]", text)


def test_no_backend_secret_reaches_the_frontend_image() -> None:
    """Build args are baked into a browser bundle; only public values belong.

    Anything named `NEXT_PUBLIC_` is served to every visitor, so this is the
    one place where a leaked credential would be published rather than merely
    exposed.
    """
    frontend = compose_config()["services"]["frontend"]
    args = frontend["build"].get("args", {})
    environment = frontend.get("environment", {})

    assert set(args) == {"NEXT_PUBLIC_API_BASE_URL"}
    for name, value in {**args, **environment}.items():
        assert not re.search(r"key|secret|token|password", name, re.I), name
        assert not re.search(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]", str(value))
    assert "LLM_API_KEY" not in json.dumps(frontend)


# ---------------------------------------------------------------------------
# The Dockerfiles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dockerfile", [BACKEND_DOCKERFILE, FRONTEND_DOCKERFILE], ids=["backend", "frontend"]
)
def test_the_image_drops_root(dockerfile: Path) -> None:
    """Neither service runs as root, and the switch is the last USER given."""
    users = [
        arguments
        for keyword, arguments in dockerfile_instructions(dockerfile)
        if keyword == "USER"
    ]

    assert users, f"{dockerfile.name} never leaves root"
    assert users[-1] not in ("root", "0")


@pytest.mark.parametrize(
    "dockerfile", [BACKEND_DOCKERFILE, FRONTEND_DOCKERFILE], ids=["backend", "frontend"]
)
def test_the_image_checks_that_the_application_answers(dockerfile: Path) -> None:
    """A healthcheck that asks the service a question, not `ps`."""
    checks = [
        arguments
        for keyword, arguments in dockerfile_instructions(dockerfile)
        if keyword == "HEALTHCHECK"
    ]

    assert len(checks) == 1, f"{dockerfile.name} should define exactly one"
    # It performs a request rather than inspecting a process table.
    assert re.search(r"healthcheck\.py|http", checks[0], re.I)
    assert not re.search(r"\bps\b|pgrep|pidof", checks[0])


@pytest.mark.parametrize(
    "dockerfile", [BACKEND_DOCKERFILE, FRONTEND_DOCKERFILE], ids=["backend", "frontend"]
)
def test_no_secret_is_baked_into_an_image(dockerfile: Path) -> None:
    """No credential, and no ARG or ENV that looks like one."""
    text = dockerfile.read_text(encoding="utf-8")

    assert not re.search(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]", text)
    for keyword, arguments in dockerfile_instructions(dockerfile):
        if keyword not in ("ENV", "ARG"):
            continue
        for assignment in re.findall(r"([A-Z_][A-Z0-9_]*)=", arguments):
            assert not re.search(
                r"SECRET|PASSWORD|CREDENTIAL|_TOKEN|API_KEY", assignment
            ), assignment


def test_the_backend_image_installs_no_test_dependency() -> None:
    """`requirements-dev.txt` is for a developer, not for the runtime image."""
    text = BACKEND_DOCKERFILE.read_text(encoding="utf-8")
    installed = [
        arguments
        for keyword, arguments in dockerfile_instructions(BACKEND_DOCKERFILE)
        if keyword == "RUN" and "pip install" in arguments
    ]

    assert installed, "the image should install its dependencies"
    for command in installed:
        assert "requirements-dev" not in command
        assert "pytest" not in command
        assert "httpx" not in command

    # The runtime stage never runs pip at all — the venv arrives by COPY.
    _, _, runtime_stage = text.partition("FROM python:3.11-slim AS runtime")
    assert "pip install" not in runtime_stage


def test_the_backend_runtime_stage_carries_no_compiler() -> None:
    """`build-essential` belongs to the builder stage and stays there."""
    text = BACKEND_DOCKERFILE.read_text(encoding="utf-8")
    _, _, runtime_stage = text.partition("FROM python:3.11-slim AS runtime")

    assert "build-essential" in text, "the builder should still install it"
    assert "build-essential" not in runtime_stage
    assert "apt-get install" not in runtime_stage


def test_the_frontend_runtime_stage_carries_no_build_tooling() -> None:
    """The final stage copies the standalone output and installs nothing."""
    text = FRONTEND_DOCKERFILE.read_text(encoding="utf-8")
    _, _, runtime_stage = text.partition("AS runtime")

    assert "npm ci" not in runtime_stage
    assert "npm run build" not in runtime_stage
    assert ".next/standalone" in runtime_stage


def test_the_frontend_build_runs_the_projects_own_gates() -> None:
    """Lint, typecheck and build, so a broken image fails at build time."""
    text = FRONTEND_DOCKERFILE.read_text(encoding="utf-8")

    for script in ("npm run lint", "npm run typecheck", "npm run build"):
        assert script in text


def test_the_frontend_config_emits_a_standalone_server() -> None:
    """Without this the runtime stage would have nothing to run.

    Checked against the config file rather than the Dockerfile, because this
    is the setting the Dockerfile depends on and the one a future edit would
    remove without noticing.
    """
    config = (REPOSITORY_ROOT / "frontend" / "next.config.mjs").read_text("utf-8")

    assert re.search(r"output:\s*[\"']standalone[\"']", config)


def test_the_backend_binds_to_every_interface() -> None:
    """A container that binds to 127.0.0.1 is unreachable from the host."""
    text = BACKEND_DOCKERFILE.read_text(encoding="utf-8")
    entrypoint = (REPOSITORY_ROOT / "backend" / "docker-entrypoint.sh").read_text(
        "utf-8"
    )

    assert "API_HOST=0.0.0.0" in text
    assert "--host" in entrypoint
    assert "127.0.0.1" not in entrypoint


def test_the_entrypoint_starts_the_existing_application() -> None:
    """No new entry point was invented; it runs `app.main:app` as before."""
    entrypoint = (REPOSITORY_ROOT / "backend" / "docker-entrypoint.sh").read_text(
        "utf-8"
    )

    assert "uvicorn app.main:app" in entrypoint
    # `exec` so uvicorn is PID 1 and receives Docker's stop signal directly.
    assert "exec uvicorn" in entrypoint


# ---------------------------------------------------------------------------
# The build contexts
# ---------------------------------------------------------------------------


def ignored_prefixes(dockerignore: Path) -> list[str]:
    """Directory prefixes a `.dockerignore` excludes.

    Only the plain directory rules are returned — the ones that could hide a
    path a `COPY` names. Glob rules are matched by Docker itself and are not
    re-implemented here.
    """
    prefixes: list[str] = []
    for line in dockerignore.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            continue
        if "*" in stripped or "?" in stripped:
            continue
        prefixes.append(stripped.rstrip("/"))
    return prefixes


def test_every_path_the_backend_copies_exists_and_is_not_excluded() -> None:
    """The strongest check available without a registry: does the build have
    the files it names?

    A `COPY` of a path `.dockerignore` excludes fails the build; a `COPY` of a
    path that does not exist fails it too. Both are cheap to introduce and
    invisible until someone runs a build.
    """
    excluded = ignored_prefixes(ROOT_DOCKERIGNORE)

    sources: list[str] = []
    for keyword, arguments in dockerfile_instructions(BACKEND_DOCKERFILE):
        if keyword != "COPY" or arguments.startswith("--from="):
            continue
        parts = arguments.split()
        sources.extend(parts[:-1])  # everything but the destination

    assert sources, "the Dockerfile should copy the source in"

    for source in sources:
        path = REPOSITORY_ROOT / source.rstrip("/")
        assert path.exists(), f"COPY names a path that does not exist: {source}"

        normalised = source.rstrip("/")
        for prefix in excluded:
            assert normalised != prefix and not normalised.startswith(
                f"{prefix}/"
            ), f"COPY {source} is excluded by .dockerignore rule {prefix!r}"


def test_the_backend_context_excludes_the_frontend_and_local_state() -> None:
    """The dashboard has its own image; local state belongs in a volume."""
    excluded = ignored_prefixes(ROOT_DOCKERIGNORE)

    for path in ("frontend", ".git", "rag/index", "ml/experiments/runs", ".venv"):
        assert path in excluded, f"{path} should be excluded from the context"


def test_the_frontend_context_excludes_its_build_output() -> None:
    """`node_modules` and `.next` are rebuilt in the image, never copied."""
    excluded = ignored_prefixes(FRONTEND_DOCKERIGNORE)

    for path in ("node_modules", ".next", "coverage"):
        assert path in excluded


@pytest.mark.parametrize(
    "dockerignore", [ROOT_DOCKERIGNORE, FRONTEND_DOCKERIGNORE], ids=["root", "frontend"]
)
def test_a_local_env_file_never_enters_a_build_context(dockerignore: Path) -> None:
    """A developer's real configuration must not be copied into a layer."""
    rules = [
        line.strip()
        for line in dockerignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert ".env" in rules
    assert ".env.*" in rules
    # The example is documentation and is deliberately kept.
    assert "!.env.example" in rules


# ---------------------------------------------------------------------------
# The documented configuration
# ---------------------------------------------------------------------------


def env_example_settings() -> dict[str, str]:
    """The assignments in `.env.example`, ignoring commentary."""
    settings: dict[str, str] = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        settings[name.strip()] = value.strip()
    return settings


def test_the_example_documents_what_docker_needs() -> None:
    """Someone copying this file gets a working stack."""
    settings = env_example_settings()

    for name in (
        "CORS_ALLOW_ORIGINS",
        "NEXT_PUBLIC_API_BASE_URL",
        "FRONTEND_PORT",
        "BACKEND_PORT",
        "LLM_API_KEY",
    ):
        assert name in settings, f"{name} is undocumented"

    assert FRONTEND_ORIGIN in settings["CORS_ALLOW_ORIGINS"]
    assert settings["NEXT_PUBLIC_API_BASE_URL"] == BACKEND_ORIGIN


def test_the_example_holds_no_secret() -> None:
    """Every documented value is a default, a limit or a URL — never a key."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    settings = env_example_settings()

    assert settings["LLM_API_KEY"] == "", "the example must ship an empty key"
    assert not re.search(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]", text)

    # Credential-shaped names only. `LLM_MAX_OUTPUT_TOKENS` is a limit, and a
    # rule loose enough to catch it would be a rule nobody trusts.
    credential = re.compile(r"(API_KEY|_SECRET|PASSWORD|ACCESS_TOKEN|AUTH_TOKEN)$")
    for name, value in settings.items():
        if credential.search(name):
            assert value == "", f"{name} should be empty in the example"


def test_the_example_never_suggests_a_wildcard_origin() -> None:
    """The documented CORS value is a list of real origins."""
    settings = env_example_settings()

    assert "*" not in settings["CORS_ALLOW_ORIGINS"]
    for origin in settings["CORS_ALLOW_ORIGINS"].split(","):
        assert re.match(r"^https?://[^/*\s]+(:\d+)?$", origin.strip())


def test_the_documented_api_url_is_browser_resolvable() -> None:
    """`.env.example` must not suggest a Compose service name either."""
    api_url = env_example_settings()["NEXT_PUBLIC_API_BASE_URL"]

    assert not re.match(r"^https?://(backend|frontend|api|web)\b", api_url)
    assert "localhost" in api_url or "127.0.0.1" in api_url


def test_the_compose_defaults_match_the_documented_ones() -> None:
    """`.env.example` and `docker-compose.yml` cannot drift apart.

    Compose falls back to its own defaults when a variable is unset, so a
    person running without a `.env` and a person copying the example must end
    up with the same stack.
    """
    documented = env_example_settings()
    resolved = compose_config()["services"]

    assert (
        resolved["backend"]["environment"]["CORS_ALLOW_ORIGINS"]
        == documented["CORS_ALLOW_ORIGINS"]
    )
    assert (
        resolved["frontend"]["build"]["args"]["NEXT_PUBLIC_API_BASE_URL"]
        == documented["NEXT_PUBLIC_API_BASE_URL"]
    )
    published = {
        str(entry["published"]) for entry in resolved["frontend"]["ports"]
    }
    assert published == {documented["FRONTEND_PORT"]}
