"""Tests for the documentation and the demo assets.

Documentation rots silently. A link that stops resolving still renders as a
link; a README that names a script nobody kept still reads as instructions; a
demo dataset excluded by a stray `.gitignore` rule is simply absent from every
checkout but present on the machine that wrote it. None of that fails a build,
and all of it is what a stranger meets first.

So these tests are about *agreement between files*, computed by walking the
repository rather than by comparing one hard-coded list against another:

- every relative link in every Markdown file points at something that exists,
  and every in-page anchor points at a heading that exists;
- the demo dataset is present in all three formats, holds the same table in
  each, and is small;
- the committed files and their generator still agree;
- nothing in `.gitignore` quietly excludes any of it.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pandas as pd
import pytest

from app.core.config import Settings
from app.services.datasets.ingestion import default_registry, detect_format
from ml.experiments.fingerprint import fingerprint_dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY_ROOT / "examples"
GENERATOR = EXAMPLES / "generate_demo_dataset.py"

#: The demo table, in each format the project reads.
DEMO_FILES = (
    "customer_churn.csv",
    "customer_churn.json",
    "customer_churn.xlsx",
)

#: Small enough that the demo is a demo. All three together, in bytes.
MAX_DEMO_BYTES = 400_000

#: Directories a repository walk must not treat as this project's own files.
IGNORED = frozenset(
    {".git", "node_modules", ".next", ".venv", "venv", "__pycache__", ".pytest_cache"}
)

#: `[text](target)`, with the target captured. Bare autolinks in angle brackets
#: are left alone: they are always absolute URLs here.
MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$", re.M)


def markdown_files() -> list[Path]:
    """Every Markdown file this project owns."""
    return sorted(
        path
        for path in REPOSITORY_ROOT.rglob("*.md")
        if not IGNORED.intersection(path.parts)
    )


def anchor_for(heading: str) -> str:
    """Return the anchor GitHub generates for a heading.

    Lowercased, punctuation dropped, spaces to hyphens. Close enough for the
    headings this project writes, and the rule is applied identically to both
    sides of the comparison.
    """
    text = re.sub(r"`([^`]*)`", r"\1", heading)
    text = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[\s]+", "-", text.strip())


def load_generator():
    """Import the demo-data generator by path.

    `examples/` is not a package — it holds data and one script — so it is
    loaded directly rather than added to the import path for one test.
    """
    spec = importlib.util.spec_from_file_location("demo_generator", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


def test_every_relative_link_in_every_markdown_file_resolves() -> None:
    """Computed from a walk, so a new document is checked the day it is added."""
    broken: list[str] = []

    for path in markdown_files():
        for target in MARKDOWN_LINK.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            if not resolved.exists():
                broken.append(f"{path.relative_to(REPOSITORY_ROOT)} -> {target}")

    assert not broken, "links pointing at nothing:\n  " + "\n  ".join(broken)


def test_every_in_page_anchor_points_at_a_heading_that_exists() -> None:
    """The other half of a link: `#limitations` with no Limitations heading."""
    broken: list[str] = []

    for path in markdown_files():
        text = path.read_text(encoding="utf-8")
        anchors = {anchor_for(heading) for heading in HEADING.findall(text)}
        for target in MARKDOWN_LINK.findall(text):
            if not target.startswith("#"):
                continue
            if target[1:] not in anchors:
                broken.append(f"{path.relative_to(REPOSITORY_ROOT)} -> {target}")

    assert not broken, "anchors pointing at nothing:\n  " + "\n  ".join(broken)


def test_the_readme_links_to_the_documents_it_promises() -> None:
    """The navigation line is the entry point; every destination must be real."""
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")

    for target in (
        "docs/ARCHITECTURE.md",
        "docs/API.md",
        "docs/PRODUCTION_READINESS.md",
        "examples/README.md",
        ".github/workflows/ci.yml",
    ):
        assert f"({target})" in readme, f"the README does not link to {target}"
        assert (REPOSITORY_ROOT / target).exists()


def test_every_script_the_documentation_names_exists_and_runs() -> None:
    """A named script that is not there is worse than no instructions."""
    for name in ("demo.sh", "smoke-test.sh"):
        script = REPOSITORY_ROOT / "scripts" / name
        assert script.exists(), f"scripts/{name} is referenced but missing"
        assert script.read_text(encoding="utf-8").startswith("#!")


def test_no_document_still_calls_this_project_unfinished() -> None:
    """The status line said "no agent" for seven commits after the agent landed.

    A stale status claim is the most damaging kind of documentation rot in a
    portfolio project, because the reader believes it.
    """
    stale = re.compile(
        r"status:\s*early development|will act as|planned technology stack",
        re.I,
    )

    offenders = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path in markdown_files()
        if stale.search(path.read_text(encoding="utf-8"))
    ]

    assert not offenders, f"stale status claims in: {offenders}"


# ---------------------------------------------------------------------------
# The demo data
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", DEMO_FILES)
def test_the_demo_dataset_is_committed_in_every_format(name: str) -> None:
    """All three, so the format-agnostic claim can be checked rather than read."""
    assert (EXAMPLES / name).is_file(), f"examples/{name} is missing"


def test_the_demo_dataset_is_small_enough_to_be_a_demo() -> None:
    """A demo dataset that takes a minute to train on is not a demo."""
    total = sum((EXAMPLES / name).stat().st_size for name in DEMO_FILES)

    assert total < MAX_DEMO_BYTES, f"the demo data is {total:,} bytes"


def test_all_three_formats_hold_the_same_table() -> None:
    """One fingerprint from three files.

    This is the property the README advertises and the demo demonstrates: a
    dataset is identified by its contents, so the spreadsheet and the CSV land
    in one experiment history. If a writer in the generator ever starts
    emitting a format-native date or an Excel serial number, the fingerprints
    diverge and this fails.
    """
    settings = Settings()
    registry = default_registry()

    fingerprints = set()
    for name in DEMO_FILES:
        path = EXAMPLES / name
        ingested = registry.load(
            path.read_bytes(),
            detect_format(name, settings),
            settings,
            filename=name,
        )
        fingerprints.add(fingerprint_dataset(ingested.frame).value)

    assert len(fingerprints) == 1, f"three formats, {len(fingerprints)} identities"


def test_the_committed_files_still_match_their_generator() -> None:
    """Provenance, checked rather than asserted.

    The generator's docstring says running it reproduces these files. This
    compares the table it builds against the committed CSV, so an edit to
    either that is not made to the other fails here — without the test writing
    anything into the working tree.
    """
    generated = load_generator().build_frame()

    # Rendered with the writer's own options and compared as text, so the
    # assertion is on the bytes a checkout receives rather than on a DataFrame
    # comparison that would have to be told how to treat every dtype.
    rendered = generated.to_csv(index=False, lineterminator="\n")
    committed = (EXAMPLES / "customer_churn.csv").read_text(encoding="utf-8")

    assert rendered == committed, (
        "examples/customer_churn.csv and generate_demo_dataset.py disagree. "
        "Run: python examples/generate_demo_dataset.py"
    )


def test_the_demo_data_carries_the_flaws_it_is_meant_to_demonstrate() -> None:
    """A clean table proves nothing about a tool that finds what is wrong.

    An identifier column, real missingness and a mildly imbalanced target are
    the three the profiler should report, so the demo's second step has
    something to show.
    """
    frame = pd.read_csv(EXAMPLES / "customer_churn.csv")

    assert frame["customer_id"].is_unique, "no identifier column to flag"
    assert frame["satisfaction_score"].isna().sum() > 0, "no missing values"

    majority = frame["renewed"].value_counts(normalize=True).max()
    assert 0.6 < majority < 0.9, "the target is either balanced or degenerate"


def test_the_demo_data_holds_no_personal_information() -> None:
    """Synthetic by construction; this asserts the columns stayed that way."""
    frame = pd.read_csv(EXAMPLES / "customer_churn.csv")

    forbidden = re.compile(r"name|email|phone|address|ssn|dob|birth", re.I)
    offenders = [column for column in frame.columns if forbidden.search(column)]

    assert not offenders, f"columns that look personal: {offenders}"


def test_nothing_excludes_the_demo_data_from_a_checkout() -> None:
    """The trap this repository actually had.

    `.gitignore` carries a blanket `*.csv`, which is right for the local
    datasets it exists to keep out and silently wrong for the demo. Without the
    negation the demo is present on the machine that wrote it and absent
    everywhere else — and no test that reads the file would ever notice,
    because the file is right there.
    """
    ignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "*.csv" in ignore, "the blanket rule this exception exists for is gone"
    assert "!examples/*.csv" in ignore
