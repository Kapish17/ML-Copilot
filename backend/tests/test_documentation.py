"""Tests that keep the user-facing documentation honest.

Documentation is the one part of a project that no other test touches, so it
is the part that drifts. A feature ships and the README still says it is
planned; a variable is renamed and `.env.example` still documents the old one;
a development note written for one commit survives into a file a stranger
reads first.

These tests read the real files. They do not check prose quality — they check
the three things that can be checked mechanically and that were actually wrong
at some point in this project's history:

* every variable `.env.example` assigns is read by something,
* no user-facing document narrates the commit history it came from,
* no user-facing document claims the project is "production-ready".

The last one is a wording rule with a reason. This project is production
*oriented* — it has been hardened, bounded and tested — and it has never been
run in production by anyone. "Production-ready" is a claim about operational
history that no test in this repository can support, so it is not made.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ENV_EXAMPLE = REPOSITORY_ROOT / ".env.example"

#: Documents a stranger reads. `docs/DECISIONS.md` and `CHANGELOG.md` are
#: deliberately excluded: a decision log and a changelog are *supposed* to be
#: written in terms of when things happened.
USER_FACING_DOCS = (
    "README.md",
    "backend/README.md",
    "ml/README.md",
    "rag/README.md",
    "llm/README.md",
    "agent/README.md",
    "frontend/README.md",
    "docs/ARCHITECTURE.md",
    "docs/API.md",
    "docs/PRODUCTION_READINESS.md",
    "docs/RELEASE_CHECKLIST.md",
    "docs/SECURITY.md",
    "docs/DEMO.md",
    ".env.example",
)

#: Where a variable may legitimately be read from: application code, the
#: container and Compose configuration, the scripts, and CI.
SEARCHED_FOR_USAGE = (
    "backend",
    "ml",
    "rag",
    "llm",
    "agent",
    "frontend/app",
    "frontend/lib",
    "frontend/components",
    "scripts",
    ".github",
    "docker-compose.yml",
    "backend/Dockerfile",
    "frontend/Dockerfile",
    "frontend/next.config.ts",
    "frontend/next.config.mjs",
    "frontend/next.config.js",
)

_SKIPPED_DIRECTORIES = {
    "node_modules",
    ".next",
    "__pycache__",
    ".git",
    ".venv",
    "dist",
    "build",
}


def _existing_docs() -> list[Path]:
    return [
        REPOSITORY_ROOT / name
        for name in USER_FACING_DOCS
        if (REPOSITORY_ROOT / name).is_file()
    ]


def _searchable_text() -> str:
    """Every byte of source, configuration and CI, concatenated once.

    Concatenating is crude and it is also the right shape for the question:
    "is this name mentioned anywhere that could read it". A per-file walk
    would answer the same question more slowly and no more precisely.
    """
    chunks: list[str] = []
    for entry in SEARCHED_FOR_USAGE:
        path = REPOSITORY_ROOT / entry
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
            continue
        if not path.is_dir():
            continue
        for candidate in path.rglob("*"):
            if not candidate.is_file():
                continue
            if _SKIPPED_DIRECTORIES & set(candidate.parts):
                continue
            if candidate.suffix not in {
                ".py",
                ".ts",
                ".tsx",
                ".js",
                ".mjs",
                ".yml",
                ".yaml",
                ".sh",
                "",
            }:
                continue
            chunks.append(candidate.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(chunks)


def env_example_names() -> list[str]:
    """The variables `.env.example` assigns, ignoring commentary."""
    names: list[str] = []
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        names.append(stripped.partition("=")[0].strip())
    return names


def test_every_documented_variable_is_read_by_something() -> None:
    """`.env.example` describes the real configuration surface, not an intended one.

    A placeholder for a feature that does not exist is worse than no
    documentation: someone sets it, nothing happens, and they have no way to
    tell whether they made a mistake or the project did.
    """
    haystack = _searchable_text()

    unread = [name for name in env_example_names() if name not in haystack]

    assert not unread, (
        "these variables are documented in .env.example but nothing reads "
        f"them: {sorted(unread)}"
    )


def test_the_example_documents_no_technology_the_project_does_not_have() -> None:
    """No placeholder invites configuration of something that is not built."""
    text = ENV_EXAMPLE.read_text(encoding="utf-8")

    for name in ("POSTGRES_URL", "QDRANT_URL", "MLFLOW_TRACKING_URI", "REDIS_URL"):
        assert not re.search(rf"^\s*#?\s*{name}\s*=", text, re.MULTILINE), (
            f"{name} is offered as a setting, but the technology behind it is "
            "not implemented"
        )


@pytest.mark.parametrize(
    "document",
    _existing_docs(),
    ids=lambda p: str(p.relative_to(REPOSITORY_ROOT)),
)
def test_no_user_facing_document_narrates_the_commit_history(document: Path) -> None:
    """A reader arriving at this project has no idea what "Commit 7" was.

    Development narration is fine in a decision log, where the sequence is the
    subject. In a document that explains what the system *is*, it dates the
    prose and explains nothing.
    """
    text = document.read_text(encoding="utf-8")

    narration = re.findall(r"\bCommit \d+\b(?:'s)?", text)

    assert not narration, (
        f"{document.relative_to(REPOSITORY_ROOT)} still narrates development "
        f"history: {sorted(set(narration))}. Describe the system instead."
    )


@pytest.mark.parametrize(
    "document",
    _existing_docs(),
    ids=lambda p: str(p.relative_to(REPOSITORY_ROOT)),
)
def test_no_user_facing_document_claims_the_project_is_production_ready(
    document: Path,
) -> None:
    """The claim this project can support is "hardened and tested", not "proven".

    "production-oriented", "production-readiness hardened" and
    "locally deployable" are all accurate. "Production-ready" asserts operating
    history that does not exist.
    """
    text = document.read_text(encoding="utf-8")

    # "production-readiness", "production-ready-ish" and the document title
    # `PRODUCTION_READINESS.md` are all fine; the banned phrase is the claim.
    claims = re.findall(r"production[- ]ready\b(?!.{0,3}ness)", text, re.IGNORECASE)

    assert not claims, (
        f"{document.relative_to(REPOSITORY_ROOT)} claims the project is "
        "production-ready. Use 'production-oriented' or "
        "'production-readiness hardened' instead."
    )
