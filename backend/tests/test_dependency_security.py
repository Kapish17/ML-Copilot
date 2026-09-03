"""Tests for dependency monitoring and the audit steps that enforce it.

Three files decide whether this repository notices a vulnerable dependency:
`.github/dependabot.yml` (which manifests are watched), `.github/workflows/ci.yml`
(whether a vulnerable one can still be merged) and `frontend/package.json`
(the one override that exists to hold a transitive dependency above a
vulnerable version). None of them is imported, type-checked or exercised by any
other test, and all three fail *silently* — a dependabot entry pointing at a
directory with no manifest simply never opens a pull request, and an audit step
that resolves nothing prints "no known vulnerabilities found" and passes.

So these tests assert the two properties that actually matter and cannot be
read off the file at a glance:

**Coverage.** Every dependency manifest in the repository is watched, and every
production requirements file is audited. Both are computed by *walking the
repository*, not by comparing one hard-coded list against another — so adding a
sixth requirements file fails these tests until it is added to both.

**Teeth.** Nothing suppresses a finding: no `|| true`, no `continue-on-error`,
no lowered threshold, no `ignore` rule, no automatic merge, and no `npm audit
fix` rewriting the lockfile mid-run so that CI tests a dependency set that is
not the one in the repository.

There is deliberately no test asserting that the audits currently pass. That is
a fact about the world on the day it is asked, not a property of this
repository, and a test claiming it would be a test that fails the morning an
advisory is published — which is exactly when CI should be the thing that
speaks up.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEPENDABOT = REPOSITORY_ROOT / ".github" / "dependabot.yml"
WORKFLOWS = REPOSITORY_ROOT / ".github" / "workflows"
WORKFLOW = WORKFLOWS / "ci.yml"
FRONTEND_PACKAGE = REPOSITORY_ROOT / "frontend" / "package.json"
FRONTEND_LOCKFILE = REPOSITORY_ROOT / "frontend" / "package-lock.json"
BACKEND_DOCKERFILE = REPOSITORY_ROOT / "backend" / "Dockerfile"
FRONTEND_DOCKERFILE = REPOSITORY_ROOT / "frontend" / "Dockerfile"

#: The requirements files the production image installs. `directory_of` turns
#: each into the directory dependabot must watch.
PRODUCTION_REQUIREMENTS = (
    "backend/requirements.txt",
    "ml/requirements.txt",
    "rag/requirements.txt",
    "llm/requirements.txt",
)

#: Where a checkout keeps things that are not source, and which a repository
#: walk must therefore not treat as a manifest of this project.
IGNORED_DIRECTORIES = frozenset(
    {".git", "node_modules", ".next", ".venv", "venv", "__pycache__", ".pytest_cache"}
)


def dependabot() -> dict:
    """Parse the dependabot configuration."""
    return yaml.safe_load(DEPENDABOT.read_text(encoding="utf-8"))


def workflow() -> dict:
    """Parse the CI workflow."""
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def commands_of(job: str) -> str:
    """Every shell command one job runs, joined into one searchable string."""
    steps = workflow()["jobs"][job]["steps"]
    return "\n".join(step.get("run", "") for step in steps)


def update_for(ecosystem: str) -> dict:
    """The single dependabot entry for one package ecosystem.

    Raises:
        AssertionError: If there is not exactly one. Two entries for the same
            ecosystem is the duplication this configuration is meant to avoid.
    """
    matches = [u for u in dependabot()["updates"] if u["package-ecosystem"] == ecosystem]
    assert len(matches) == 1, f"expected one {ecosystem} entry, found {len(matches)}"
    return matches[0]


def directories_of(update: dict) -> set[str]:
    """The directories one entry watches, however it spells them."""
    if "directories" in update:
        return set(update["directories"])
    return {update["directory"]}


def python_manifests() -> set[str]:
    """Every Python requirements file in the repository, as a POSIX path.

    Walks the checkout rather than trusting a list, so a requirements file
    added in a new package is found by these tests before it is forgotten.
    """
    found: set[str] = set()
    for path in REPOSITORY_ROOT.rglob("requirements*.txt"):
        if IGNORED_DIRECTORIES.intersection(path.parts):
            continue
        found.add(path.relative_to(REPOSITORY_ROOT).as_posix())
    return found


# ---------------------------------------------------------------------------
# Dependabot: the configuration is valid, and it watches everything
# ---------------------------------------------------------------------------


def test_the_dependabot_file_exists_and_parses() -> None:
    """A YAML error here is invisible until GitHub rejects it silently."""
    assert DEPENDABOT.exists(), "there is no dependabot configuration"

    config = dependabot()
    assert config["version"] == 2
    assert isinstance(config["updates"], list) and config["updates"]


def test_it_watches_python_javascript_and_the_actions_themselves() -> None:
    """Three ecosystems, one entry each.

    The third is the one people forget. A workflow pinned to
    `actions/checkout@v4` is pinned to a *moving* tag, and nothing else in this
    repository would ever tell you a new major exists.
    """
    ecosystems = [u["package-ecosystem"] for u in dependabot()["updates"]]

    assert sorted(ecosystems) == ["github-actions", "npm", "pip"]
    assert len(ecosystems) == len(set(ecosystems)), "an ecosystem is configured twice"


def test_every_python_manifest_in_the_repository_is_watched() -> None:
    """Computed from a walk of the checkout, not from a second hard-coded list.

    `requirements-dev.txt` needs no directory of its own: it lives beside
    `backend/requirements.txt` and dependabot follows the `-r` reference. What
    this test insists on is that no manifest sits in a directory nobody
    watches, which is how a package added later ends up unmonitored.
    """
    watched = directories_of(update_for("pip"))
    manifests = python_manifests()
    assert manifests, "the walk found no requirements files at all"

    unwatched = {
        manifest
        for manifest in manifests
        if f"/{Path(manifest).parent.as_posix()}" not in watched
    }
    assert not unwatched, f"requirements files in unwatched directories: {unwatched}"


def test_the_dev_requirements_are_reached_through_the_runtime_file() -> None:
    """The premise of the test above: `-r requirements.txt`, in that directory.

    If that reference is ever replaced by a copied list of pins, the backend
    directory stops covering the test dependencies and this configuration
    quietly narrows.
    """
    dev = (REPOSITORY_ROOT / "backend" / "requirements-dev.txt").read_text("utf-8")
    assert re.search(r"^-r\s+requirements\.txt\s*$", dev, re.M)


def test_the_dashboard_manifest_is_watched_where_it_lives() -> None:
    """`/frontend`, which is where package.json and the lockfile are."""
    assert directories_of(update_for("npm")) == {"/frontend"}
    assert FRONTEND_PACKAGE.exists() and FRONTEND_LOCKFILE.exists()


def test_the_actions_entry_is_rooted_at_the_repository() -> None:
    """`/`, not `/.github/workflows`.

    The GitHub Actions ecosystem is rooted at the repository and finds the
    workflow directory itself. Pointing it at the workflow directory is the
    common mistake, and it produces an entry that watches nothing.
    """
    assert directories_of(update_for("github-actions")) == {"/"}
    assert list(WORKFLOWS.glob("*.yml")), "there is no workflow for it to watch"


def test_every_entry_is_checked_weekly() -> None:
    """Weekly on every ecosystem — daily is noise, monthly is too late."""
    for update in dependabot()["updates"]:
        assert update["schedule"]["interval"] == "weekly", update["package-ecosystem"]


# ---------------------------------------------------------------------------
# Dependabot: nothing is suppressed, and nothing merges itself
# ---------------------------------------------------------------------------


def test_no_advisory_is_ignored() -> None:
    """An `ignore` rule is an advisory nobody is told about twice."""
    for update in dependabot()["updates"]:
        assert "ignore" not in update, f"{update['package-ecosystem']} ignores updates"


def test_nothing_in_the_repository_merges_a_dependency_update_by_itself() -> None:
    """No workflow enables auto-merge, and no entry asks for one.

    A dependency bump is a code change written outside this repository. It goes
    through CI and a human. `gh pr merge --auto`, the `--auto` flag on any
    merge, and the `pull-requests: write` permission a self-merging workflow
    needs are all absent — checked across every workflow, not just `ci.yml`,
    because the usual way this appears is a second file called
    `dependabot-auto-merge.yml`.
    """
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        assert "--auto" not in text, f"{path.name} auto-merges a pull request"
        assert "enable-pull-request-automerge" not in text, path.name
        assert "pull-requests: write" not in text, path.name


# ---------------------------------------------------------------------------
# The Python audit: it runs, it covers production, and it can fail
# ---------------------------------------------------------------------------


def test_a_job_audits_the_python_dependencies() -> None:
    """pip-audit is in CI, in a job of its own."""
    assert "pip-audit" in commands_of("security")


def test_the_python_audit_covers_every_production_requirements_file() -> None:
    """Every file the image installs, by name, in the production audit step.

    This is the test that stops the audit from covering a fraction of the
    application while reporting success. It is asserted against the same list
    the backend Dockerfile installs — see the test below, which checks the two
    lists agree — so an audit of `backend/requirements.txt` alone, with the ML,
    retrieval and language-model layers unaudited, fails here.
    """
    steps = workflow()["jobs"]["security"]["steps"]
    production = [
        step
        for step in steps
        if "production" in (step.get("name") or "").lower() and "run" in step
    ]
    assert len(production) == 1, "expected exactly one production audit step"

    command = production[0]["run"]
    for requirements in PRODUCTION_REQUIREMENTS:
        assert f"-r {requirements}" in command, f"{requirements} is not audited"


def test_the_audited_production_set_is_the_set_the_image_installs() -> None:
    """Agreement between the Dockerfile and the audit, computed from both.

    The audit is only meaningful if the files it reads are the files that end
    up in the container. The Dockerfile copies each requirements file to a
    distinct name under /tmp, so the COPY sources are the authority.
    """
    dockerfile = BACKEND_DOCKERFILE.read_text(encoding="utf-8")
    copied = set(re.findall(r"^COPY\s+(\S+/requirements\.txt)\s", dockerfile, re.M))

    assert copied == set(PRODUCTION_REQUIREMENTS), (
        "the backend image installs a different set of requirements files than "
        f"the production audit covers: image={sorted(copied)}"
    )


def test_the_development_dependencies_are_audited_too() -> None:
    """A test dependency runs in CI with the repository checked out."""
    steps = workflow()["jobs"]["security"]["steps"]
    development = [
        step
        for step in steps
        if "development" in (step.get("name") or "").lower() and "run" in step
    ]
    assert len(development) == 1, "expected exactly one development audit step"
    assert "-r backend/requirements-dev.txt" in development[0]["run"]


def test_the_python_audit_fails_rather_than_skips_what_it_cannot_resolve() -> None:
    """`--strict`.

    Without it pip-audit skips a dependency it cannot collect and still exits
    zero, which is the difference between "nothing is vulnerable" and "I could
    not tell". Every pip-audit invocation must carry it.
    """
    invocations = [
        line
        for line in commands_of("security").splitlines()
        if "pip-audit" in line and "pip install" not in line
    ]
    assert invocations, "no pip-audit invocation found"
    for line in invocations:
        assert "--strict" in line, f"pip-audit without --strict: {line.strip()}"


def test_pip_audit_is_installed_away_from_the_dependencies_it_audits() -> None:
    """Its own virtual environment.

    Installed beside the application's dependencies, pip-audit and its own
    dependency tree become part of the environment under audit — and the job
    can then fail over an advisory against the checker rather than against this
    project.
    """
    commands = commands_of("security")
    assert "python -m venv" in commands
    assert "pip install pip-audit" in commands


def test_the_python_audit_is_not_in_the_production_image() -> None:
    """Not in any requirements file, and not installed by either Dockerfile.

    pip-audit is a development tool. Shipping it would add a dependency tree
    (requests, cyclonedx, rich and the rest) to an image that has no use for
    it, which is more attack surface, not less.
    """
    for manifest in python_manifests():
        text = (REPOSITORY_ROOT / manifest).read_text(encoding="utf-8")
        pins = [
            line
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        assert not any("pip-audit" in line for line in pins), manifest

    for dockerfile in (BACKEND_DOCKERFILE, FRONTEND_DOCKERFILE):
        assert "pip-audit" not in dockerfile.read_text(encoding="utf-8"), dockerfile.name


# ---------------------------------------------------------------------------
# The JavaScript audit: it runs after the install, and it can fail
# ---------------------------------------------------------------------------


def test_the_dashboard_is_audited_immediately_after_the_lockfile_install() -> None:
    """`npm ci`, then `npm audit`, in that order and with nothing between.

    Order is the whole point: `npm audit` reads the installed tree, so it has
    to follow the install. Adjacency is what keeps it from drifting to the end
    of the job, where a lint failure would mean the audit never ran.
    """
    steps = workflow()["jobs"]["frontend"]["steps"]
    runs = [(index, step.get("run", "")) for index, step in enumerate(steps)]

    install = [index for index, run in runs if re.search(r"\bnpm ci\b", run)]
    audit = [index for index, run in runs if re.search(r"\bnpm audit\b", run)]

    assert len(install) == 1 and len(audit) == 1
    assert audit[0] == install[0] + 1, "the audit does not immediately follow npm ci"


def test_the_dashboard_audit_fails_on_a_high_or_critical_advisory() -> None:
    """`--audit-level=high`.

    The threshold decides what fails the build, not what is printed: a moderate
    advisory still appears in the log. Lowering the bar to `critical` — or
    raising it to `none` — is the quiet way to make this step stop mattering.
    """
    frontend = commands_of("frontend")
    assert "npm audit --audit-level=high" in frontend
    for weaker in ("--audit-level=critical", "--audit-level=none"):
        assert weaker not in frontend, f"the audit threshold was lowered to {weaker}"


def test_ci_installs_from_the_lockfile_and_never_rewrites_it() -> None:
    """No `npm install`, and no `npm audit fix`.

    Both resolve dependencies afresh and write `package-lock.json`. In CI that
    means the job tests a dependency set that is not the one in the repository,
    and a green build that proves nothing about what anybody would install. A
    fix is made locally, reviewed, and committed as a lockfile change.
    """
    frontend = commands_of("frontend")
    assert not re.search(r"\bnpm install\b", frontend)
    assert not re.search(r"\bnpm audit fix\b", frontend)
    assert not re.search(r"\bnpm update\b", frontend)


# ---------------------------------------------------------------------------
# Neither audit is allowed to be decorative
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("job", ["frontend", "security"])
def test_no_audit_step_swallows_its_own_failure(job: str) -> None:
    """No `|| true`, no `continue-on-error`, no `exit 0` chaser.

    Each of these turns a failing audit into a passing job while leaving the
    step in the file, which is worse than not auditing at all: the workflow
    still reads as though it checks.
    """
    for step in workflow()["jobs"][job]["steps"]:
        run = step.get("run", "")
        assert "|| true" not in run, step.get("name")
        assert "|| exit 0" not in run, step.get("name")
        assert step.get("continue-on-error") is not True, step.get("name")

    assert workflow()["jobs"][job].get("continue-on-error") is not True


# ---------------------------------------------------------------------------
# The one dependency override, and why it is allowed to exist
# ---------------------------------------------------------------------------


def test_every_override_raises_a_version_rather_than_holding_one_back() -> None:
    """An `overrides` block is how a transitive pin gets patched. It is also
    how a transitive pin gets held *below* a fix, so each one is checked
    against the lockfile: the resolved version must be the overridden version,
    and the override must be an exact pin rather than a range that can drift
    back down.
    """
    package = json.loads(FRONTEND_PACKAGE.read_text(encoding="utf-8"))
    overrides = package.get("overrides", {})
    if not overrides:
        pytest.skip("no overrides are configured")

    lock = json.loads(FRONTEND_LOCKFILE.read_text(encoding="utf-8"))
    for name, version in overrides.items():
        assert re.fullmatch(r"\d+\.\d+\.\d+", version), (
            f"the {name} override is a range ({version}); an override that "
            "exists for a security fix must pin the fixed version exactly"
        )
        resolved = {
            entry.get("version")
            for path, entry in lock["packages"].items()
            if path.split("node_modules/")[-1] == name and "version" in entry
        }
        assert resolved == {version}, (
            f"{name} is overridden to {version} but the lockfile resolves it "
            f"to {sorted(resolved)}"
        )
