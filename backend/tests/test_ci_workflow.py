"""Tests for the CI workflow and the smoke test it runs.

A workflow file is configuration that nothing else validates: it is not
imported, not type-checked, and its only feedback loop is a push to GitHub.
That makes it easy for it to drift away from the project — a renamed npm
script, a requirements file that moved, a cleanup step deleted during a
refactor — and easy for the drift to go unnoticed until CI is quietly testing
less than it claims to.

So these tests read the real YAML with a real parser and assert on its
*structure*: which jobs exist, what each one runs, that the commands it runs
are commands this project actually has, that the token is read-only and that
cleanup happens even on failure. Matching arbitrary strings would pass just as
happily against a workflow that ran nothing.

Two of them are worth calling out because they check agreement between files
rather than the contents of one: every `npm run` the workflow invokes must
exist in `package.json`, and every requirements file it installs must exist on
disk. Those are the two ways this workflow can rot without anyone editing it.
"""

from __future__ import annotations

import json
import re
import stat
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
SMOKE_TEST = REPOSITORY_ROOT / "scripts" / "smoke-test.sh"
FRONTEND_PACKAGE = REPOSITORY_ROOT / "frontend" / "package.json"

#: The published origins the stack uses. The smoke test must check the browser
#: is pointed at the first and never at a Compose service name.
BACKEND_ORIGIN = "http://localhost:8000"
INTERNAL_HOST = "backend:8000"


def workflow() -> dict:
    """Parse the workflow file.

    Returns:
        dict: The workflow as PyYAML reads it.
    """
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def steps_of(job: str) -> list[dict]:
    """Every step of one job."""
    return workflow()["jobs"][job]["steps"]


def commands_of(job: str) -> str:
    """Every shell command a job runs, joined into one searchable string."""
    return "\n".join(step.get("run", "") for step in steps_of(job))


def uses_of(job: str) -> list[str]:
    """Every action a job uses."""
    return [step["uses"] for step in steps_of(job) if "uses" in step]


# ---------------------------------------------------------------------------
# The workflow exists and triggers on the right events
# ---------------------------------------------------------------------------


def test_the_workflow_file_exists_and_parses() -> None:
    """A YAML error here is invisible until a push; parse it in the suite."""
    assert WORKFLOW.exists(), "there is no CI workflow"

    config = workflow()
    assert config["name"] == "CI"
    assert isinstance(config["jobs"], dict)


def test_it_runs_on_pushes_and_pull_requests_to_main() -> None:
    """Both, and only for `main`.

    PyYAML reads a bare `on:` key as the boolean True — the well-known YAML
    1.1 quirk — so the trigger block is looked up under either spelling.
    """
    config = workflow()
    triggers = config.get("on", config.get(True))

    assert set(triggers) == {"push", "pull_request"}
    assert triggers["push"]["branches"] == ["main"]
    assert triggers["pull_request"]["branches"] == ["main"]


def test_the_three_jobs_exist_and_no_more() -> None:
    """Backend, frontend, Docker. Nothing speculative."""
    assert set(workflow()["jobs"]) == {"backend", "frontend", "docker"}


# ---------------------------------------------------------------------------
# Permissions, concurrency and pinning
# ---------------------------------------------------------------------------


def test_the_token_can_only_read_the_repository() -> None:
    """The workflow validates and publishes nothing, so read is all it needs."""
    config = workflow()

    assert config["permissions"] == {"contents": "read"}
    # And no job quietly widens it.
    for name, job in config["jobs"].items():
        assert "permissions" not in job or job["permissions"] == {
            "contents": "read"
        }, name


def test_no_job_asks_for_a_secret() -> None:
    """The normal path must work on a fork, where no secret is available."""
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "secrets." not in text
    assert "${{ secrets" not in text
    # And nothing credential-shaped is written in by hand.
    assert not re.search(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]", text)
    for forbidden in ("LLM_API_KEY:", "DOCKERHUB", "AWS_", "GCP_", "AZURE_"):
        assert forbidden not in text, forbidden


def test_a_newer_commit_cancels_an_older_run() -> None:
    """Three pushes in a row should cost one CI run, not three."""
    concurrency = workflow()["concurrency"]

    assert "github.ref" in concurrency["group"]
    assert concurrency["cancel-in-progress"] is True


def test_every_action_is_an_official_one_at_a_pinned_major() -> None:
    """No third-party action, and no floating `@main`."""
    config = workflow()
    used = [
        step["uses"]
        for job in config["jobs"].values()
        for step in job["steps"]
        if "uses" in step
    ]

    assert used, "the workflow should use the setup actions"
    for action in used:
        name, _, version = action.partition("@")
        assert name.startswith(("actions/", "docker/")), name
        assert re.fullmatch(r"v\d+", version), f"{action} is not pinned to a major"


def test_the_runner_and_toolchain_are_pinned() -> None:
    """A silent Ubuntu or Python bump makes a red build ambiguous."""
    config = workflow()

    for name, job in config["jobs"].items():
        assert re.fullmatch(r"ubuntu-\d+\.\d+", job["runs-on"]), name
        assert "timeout-minutes" in job, f"{name} could hang forever"

    assert re.fullmatch(r"3\.\d+", str(config["env"]["PYTHON_VERSION"]))
    assert re.fullmatch(r"\d+", str(config["env"]["NODE_VERSION"]))


# ---------------------------------------------------------------------------
# The backend job runs this project's own checks
# ---------------------------------------------------------------------------


def test_the_backend_job_installs_and_tests() -> None:
    """Checkout, Python, install, compile, pytest."""
    commands = commands_of("backend")
    actions = uses_of("backend")

    assert any(action.startswith("actions/checkout@") for action in actions)
    assert any(action.startswith("actions/setup-python@") for action in actions)
    assert "pip install" in commands
    assert "compileall" in commands
    assert re.search(r"\bpytest\b", commands), "the suite is never run"


def test_the_backend_job_installs_files_that_exist() -> None:
    """A requirements file that moved would silently narrow the install."""
    commands = commands_of("backend")
    referenced = re.findall(r"-r\s+(\S+requirements[\w.-]*\.txt)", commands)

    assert referenced, "no requirements file is installed"
    for path in referenced:
        assert (REPOSITORY_ROOT / path).exists(), f"{path} does not exist"

    # The dev file, because CI needs pytest; it pulls the runtime file in.
    assert any("requirements-dev.txt" in path for path in referenced)


def test_the_backend_job_caches_on_every_pin_that_matters() -> None:
    """A cache keyed on one file would go stale when another pin changed."""
    setup = next(
        step for step in steps_of("backend") if "setup-python" in step.get("uses", "")
    )
    assert setup["with"]["cache"] == "pip"

    cached = setup["with"]["cache-dependency-path"].split()
    for path in cached:
        assert (REPOSITORY_ROOT / path).exists(), path
    # Every requirements file in the project, so no pin is missed.
    on_disk = {
        str(path.relative_to(REPOSITORY_ROOT))
        for path in REPOSITORY_ROOT.glob("*/requirements*.txt")
    }
    assert set(cached) == on_disk


def test_the_backend_job_needs_no_credential() -> None:
    """The suite runs on the deterministic fake provider throughout."""
    job = workflow()["jobs"]["backend"]
    rendered = json.dumps(job)

    assert "LLM_API_KEY" not in rendered
    assert "secrets" not in rendered


# ---------------------------------------------------------------------------
# The frontend job runs the four gates
# ---------------------------------------------------------------------------


def test_the_frontend_job_runs_every_gate() -> None:
    """Install from the lockfile, then lint, typecheck, test and build."""
    commands = commands_of("frontend")
    actions = uses_of("frontend")

    assert any(action.startswith("actions/setup-node@") for action in actions)
    assert "npm ci" in commands, "CI must install from the lockfile"
    assert "npm install" not in commands, "`npm install` may drift from the lockfile"

    for script in ("lint", "typecheck", "test", "build"):
        assert re.search(rf"npm (run )?{script}\b", commands), script


def test_every_npm_script_the_workflow_runs_exists() -> None:
    """The check that catches a renamed script before a push does."""
    commands = commands_of("frontend")
    package = json.loads(FRONTEND_PACKAGE.read_text(encoding="utf-8"))

    invoked = set(re.findall(r"npm run ([\w:-]+)", commands))
    assert invoked, "the job should run npm scripts"
    for script in invoked:
        assert script in package["scripts"], f"package.json has no `{script}` script"

    # `npm test` is the shorthand for the `test` script.
    if re.search(r"npm test\b", commands):
        assert "test" in package["scripts"]


def test_the_frontend_job_builds_against_the_published_backend_url() -> None:
    """The URL is inlined at build time, so CI must build with the real one."""
    build = next(
        step
        for step in steps_of("frontend")
        if "npm run build" in step.get("run", "")
    )
    assert build["env"]["NEXT_PUBLIC_API_BASE_URL"] == BACKEND_ORIGIN


def test_the_frontend_job_works_in_the_frontend_directory() -> None:
    """Otherwise every command would run against the repository root."""
    job = workflow()["jobs"]["frontend"]

    assert job["defaults"]["run"]["working-directory"] == "frontend"


def test_the_frontend_cache_is_keyed_on_the_lockfile() -> None:
    """The lockfile is what decides the installed tree."""
    setup = next(
        step for step in steps_of("frontend") if "setup-node" in step.get("uses", "")
    )

    assert setup["with"]["cache"] == "npm"
    path = setup["with"]["cache-dependency-path"]
    assert (REPOSITORY_ROOT / path).exists(), path


# ---------------------------------------------------------------------------
# The Docker job actually starts the stack
# ---------------------------------------------------------------------------


def test_the_docker_job_validates_builds_starts_and_smoke_tests() -> None:
    """The four steps that make this job mean something, in order."""
    commands = commands_of("docker")

    assert "docker compose config" in commands
    assert "docker compose build" in commands
    assert "docker compose up" in commands
    assert "smoke-test.sh" in commands


def test_the_docker_job_waits_for_health_rather_than_sleeping() -> None:
    """`--wait` blocks on the images' own healthchecks.

    A `sleep` would pass on a fast runner and fail on a slow one, which is the
    worst kind of CI flake: one that looks like a real failure.
    """
    commands = commands_of("docker")

    assert re.search(r"docker compose up[^\n]*--wait", commands)
    assert not re.search(r"\bsleep \d+", commands)


def test_the_docker_job_runs_the_smoke_test_that_exists() -> None:
    """And it is executable, so the runner can invoke it directly."""
    commands = commands_of("docker")
    referenced = re.findall(r"(\./scripts/[\w.-]+\.sh)", commands)

    assert referenced, "no script is invoked"
    for path in referenced:
        script = REPOSITORY_ROOT / path.lstrip("./")
        assert script.exists(), path
        assert script.stat().st_mode & stat.S_IXUSR, f"{path} is not executable"


def test_the_stack_is_torn_down_even_when_a_check_fails() -> None:
    """A failed smoke test must not leave containers or volumes behind."""
    cleanup = [
        step
        for step in steps_of("docker")
        if "docker compose down" in step.get("run", "")
    ]

    assert cleanup, "nothing tears the stack down"
    assert any(step.get("if") == "always()" for step in cleanup)
    # `-v` because the volumes hold only this run's throwaway data.
    assert any("-v" in step["run"] for step in cleanup)


def test_a_failed_docker_job_shows_the_container_logs() -> None:
    """Otherwise a red build says only that something failed."""
    logs = [
        step
        for step in steps_of("docker")
        if "docker compose logs" in step.get("run", "")
    ]

    assert logs, "a failure would leave no diagnostic"
    assert any(step.get("if") == "failure()" for step in logs)


def test_the_docker_job_writes_no_env_file() -> None:
    """The stack must work on its documented defaults, with no credential."""
    commands = commands_of("docker")

    assert not re.search(r">\s*\.env\b", commands)
    assert "LLM_API_KEY" not in commands


# ---------------------------------------------------------------------------
# The smoke test checks the things that fail quietly
# ---------------------------------------------------------------------------


def test_the_smoke_test_is_a_runnable_script() -> None:
    """Executable, and strict about errors."""
    assert SMOKE_TEST.exists()
    assert SMOKE_TEST.stat().st_mode & stat.S_IXUSR

    text = SMOKE_TEST.read_text(encoding="utf-8")
    assert text.startswith("#!")
    # Without this a failing curl would be ignored and the test would "pass".
    assert "set -euo pipefail" in text


def test_the_smoke_test_checks_the_browser_facing_url() -> None:
    """The mistake that looks healthy from outside and breaks every visitor."""
    text = SMOKE_TEST.read_text(encoding="utf-8")

    assert INTERNAL_HOST in text, "it should name the hostname it rejects"
    assert BACKEND_ORIGIN in text, "it should name the URL it requires"


def test_the_smoke_test_covers_both_services_and_every_route() -> None:
    """Backend health and status, and all four dashboard routes."""
    text = SMOKE_TEST.read_text(encoding="utf-8")

    for path in ("/health", "/api/v1/knowledge/status", "/api/v1/agent/status"):
        assert f'BACKEND_URL{path}"' in text, path
    for route in ("/", "/dashboard", "/experiments", "/knowledge"):
        assert f'FRONTEND_URL{route}"' in text, route


def test_the_smoke_test_exercises_a_real_workflow() -> None:
    """Profile, train, explain, remember — not just a 200 from `/health`."""
    text = SMOKE_TEST.read_text(encoding="utf-8")

    for endpoint in (
        "/api/v1/datasets/profile",
        "/api/v1/experiments/run",
        "/api/v1/search",
    ):
        assert endpoint in text, endpoint
    assert "explainability" in text, "the SHAP step is not checked"
    assert "fingerprint" in text, "the run is not looked up in the history"


def test_the_smoke_test_checks_cors_and_secrets() -> None:
    """Both are container-only failure modes, and both are cheap to check."""
    text = SMOKE_TEST.read_text(encoding="utf-8")

    assert "Access-Control-Request-Method" in text, "no preflight is sent"
    # It must reject a wildcard, not merely accept a permitted origin.
    assert r"access-control-allow-origin: \*" in text, "a wildcard would pass"
    # And it must check that a foreign origin gets nothing.
    assert "not-the-dashboard.example" in text, "no foreign origin is tried"
    # `sk-` at a word boundary, so this endpoint's own path does not trip it.
    assert r"sk-[A-Za-z0-9]" in text, "nothing checks for a credential"


def test_the_smoke_test_needs_no_credential() -> None:
    """It runs on a fork, on a clean clone, with nothing configured."""
    text = SMOKE_TEST.read_text(encoding="utf-8")

    assert not re.search(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{6,}", text)
    assert "LLM_API_KEY=" not in text


@pytest.mark.parametrize("url_name", ["BACKEND_URL", "FRONTEND_URL"])
def test_the_smoke_test_targets_are_configurable(url_name: str) -> None:
    """So the same script serves CI, a local stack and a different port."""
    text = SMOKE_TEST.read_text(encoding="utf-8")

    assert re.search(rf'{url_name}="\$\{{{url_name}:-', text), url_name
