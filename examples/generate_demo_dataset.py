"""Generate the demo dataset, in all three formats ML Copilot reads.

The three files beside this script are **committed**, so nothing has to be
generated to run the demo and CI never downloads anything. This script is here
so the data's provenance is checkable rather than asserted: run it and the
files are reproduced byte for byte.

    python examples/generate_demo_dataset.py

Why generate rather than ship a public dataset? Three reasons, and they are the
same three that make this data safe to commit:

**It is synthetic.** Every value comes from the generator below, seeded once.
No person, company, address or account exists anywhere in it, so there is
nothing to anonymise and no licence to honour.

**It is deterministic.** One seed, no wall-clock, no environment lookup. The
same three files on every machine, which is what lets the demo promise a
specific fingerprint and lets a test assert one.

**It has the flaws worth showing.** A churn table where every column is clean
and predictive demonstrates nothing. This one carries an identifier column, a
column with real missingness, a mildly imbalanced target and one genuinely
useless column, because the profiler's job is to find exactly those and the
demo is more honest when it does.

Written as one table into three files so the same data can be uploaded as CSV,
as an Excel workbook and as JSON. Identity here comes from the *contents*, so
all three produce the same dataset fingerprint and land in the same experiment
history — which is the point being demonstrated.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

#: One seed for everything below. Changing it changes the fingerprint the
#: README and the demo script quote, so it is not a knob to turn casually.
SEED = 20240517

#: Small on purpose: a full six-model cross-validated run with SHAP finishes in
#: seconds, which is what makes it a demo rather than a wait.
ROW_COUNT = 300

#: The reference point for `signup_date`. A fixed date, not `date.today()` —
#: the whole file has to be reproducible next year too.
EPOCH = date(2022, 1, 1)

#: Stamped into the workbook's document properties and every zip entry, in
#: place of the clock. Arbitrary, and that is the point: it never changes.
FIXED_TIMESTAMP = datetime(2024, 5, 17, 12, 0, 0)
FIXED_ZIP_DATE = (2024, 5, 17, 12, 0, 0)

PLANS = ("basic", "standard", "premium")
REGIONS = ("north", "south", "east", "west")
CHANNELS = ("organic", "referral", "paid_search", "partner")


class Random:
    """A tiny linear congruential generator, so the data needs no dependency.

    ``numpy.random`` would do this better, but its stream is a compatibility
    guarantee this file should not lean on: a NumPy upgrade must never change
    what these committed files should contain. The constants are the ones from
    Numerical Recipes; the quality bar here is "varied and reproducible", not
    "cryptographic".
    """

    def __init__(self, seed: int) -> None:
        self._state = seed & 0xFFFFFFFF

    def next_float(self) -> float:
        """Return the next value in [0, 1)."""
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return self._state / 0x100000000

    def integer(self, low: int, high: int) -> int:
        """Return an integer in [low, high]."""
        return low + int(self.next_float() * (high - low + 1))

    def choice(self, options: tuple[str, ...]) -> str:
        """Return one option, uniformly."""
        return options[self.integer(0, len(options) - 1)]

    def normal(self, mean: float, spread: float) -> float:
        """Return a roughly normal value, by summing six uniforms."""
        total = sum(self.next_float() for _ in range(6)) - 3.0
        return mean + spread * total


def build_frame() -> pd.DataFrame:
    """Build the demo table.

    The target is generated from a real logistic relationship over three of the
    columns, so the models have something to find and SHAP has something to
    attribute. ``region`` and ``signup_channel`` are deliberately outside it:
    a demo where every feature matters teaches nothing about reading an
    importance chart.

    Returns:
        pandas.DataFrame: 300 rows, 10 columns.
    """
    rng = Random(SEED)
    rows: list[dict[str, object]] = []

    for index in range(ROW_COUNT):
        tenure = rng.integer(1, 60)
        plan = rng.choice(PLANS)
        base_spend = {"basic": 19.0, "standard": 49.0, "premium": 99.0}[plan]
        spend = round(max(5.0, rng.normal(base_spend, base_spend * 0.18)), 2)
        tickets = max(0, int(rng.normal(2.2, 1.9)))
        logins = max(0, int(rng.normal(24 - tickets * 2 + tenure / 6, 7)))

        # The relationship the models are meant to recover: long tenure and
        # frequent logins push towards renewal, support tickets away from it.
        score = -0.85 + 0.045 * tenure + 0.055 * logins - 0.38 * tickets
        renewed = int(rng.next_float() < 1.0 / (1.0 + math.exp(-score)))

        # About one row in eight has no satisfaction score, which is what a
        # survey response rate looks like and what makes the missing-value
        # finding and the imputation step worth watching.
        satisfaction: float | None = None
        if rng.next_float() > 0.12:
            satisfaction = round(
                min(10.0, max(1.0, rng.normal(7.4 - tickets * 0.5, 1.4))), 1
            )

        rows.append(
            {
                # A unique identifier. The profiler should flag it and the
                # pipeline should keep it out of the feature set — leaving it
                # in is one of the most common ways to fake a good score.
                "customer_id": f"CUS-{10_000 + index}",
                "signup_date": (EPOCH + timedelta(days=rng.integer(0, 900))).isoformat(),
                "plan": plan,
                "region": rng.choice(REGIONS),
                "signup_channel": rng.choice(CHANNELS),
                "tenure_months": tenure,
                "monthly_spend": spend,
                "support_tickets": tickets,
                "logins_last_30d": logins,
                "satisfaction_score": satisfaction,
                "renewed": renewed,
            }
        )

    return pd.DataFrame(rows)


def write_all(frame: pd.DataFrame) -> list[Path]:
    """Write the table as CSV, JSON and .xlsx.

    Every column is written as text or as a plain number in all three files —
    no format-native date type, no Excel serial number — so the three parse
    back to the same table and therefore to the same content fingerprint. That
    equality is the property the demo advertises, so it is worth this small
    constraint on the writer.

    Args:
        frame: The table to write.

    Returns:
        list[pathlib.Path]: The files written, in the order written.
    """
    csv_path = HERE / "customer_churn.csv"
    json_path = HERE / "customer_churn.json"
    excel_path = HERE / "customer_churn.xlsx"

    frame.to_csv(csv_path, index=False, lineterminator="\n")

    # An array of objects — the shape the JSON adapter reads first, and the one
    # anybody would guess. `null` for a missing satisfaction score, not the
    # string "NaN".
    records = json.loads(frame.to_json(orient="records"))
    json_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # openpyxl stamps the workbook with the current time, and a zip entry
    # carries a modification date, so a plain `to_excel` produces different
    # bytes on every run. Both are overwritten with a fixed instant, which is
    # what makes "run this and the committed files come back" true for the
    # workbook as well as for the two text files.
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="customers")
        properties = writer.book.properties
        properties.created = FIXED_TIMESTAMP
        properties.modified = FIXED_TIMESTAMP
        properties.creator = "ML Copilot demo data generator"
        properties.lastModifiedBy = properties.creator

    _make_workbook_reproducible(excel_path)

    return [csv_path, json_path, excel_path]


#: openpyxl writes this element from the clock as it saves, overriding whatever
#: `properties.modified` was set to, so it has to be corrected afterwards.
_MODIFIED_ELEMENT = re.compile(
    r"<dcterms:modified[^>]*>[^<]*</dcterms:modified>", re.ASCII
)


def _make_workbook_reproducible(archive: Path) -> None:
    """Rewrite an .xlsx so two runs produce identical bytes.

    Two clock readings survive an ordinary save. Each zip entry stores a
    modification time, and ``docProps/core.xml`` carries a ``dcterms:modified``
    element that openpyxl stamps as it writes, after any property set on the
    workbook. Both are replaced with the same fixed instant.

    Args:
        archive: The workbook to rewrite in place.
    """
    with zipfile.ZipFile(archive) as source:
        entries = [(item, source.read(item.filename)) for item in source.infolist()]

    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
        for item, payload in entries:
            if item.filename == "docProps/core.xml":
                payload = _MODIFIED_ELEMENT.sub(
                    '<dcterms:modified xsi:type="dcterms:W3CDTF">'
                    f"{FIXED_TIMESTAMP.strftime('%Y-%m-%dT%H:%M:%SZ')}"
                    "</dcterms:modified>",
                    payload.decode("utf-8"),
                ).encode("utf-8")
            frozen = zipfile.ZipInfo(item.filename, date_time=FIXED_ZIP_DATE)
            frozen.compress_type = item.compress_type
            frozen.external_attr = item.external_attr
            target.writestr(frozen, payload)


def main() -> None:
    """Write all three files and report what was written."""
    frame = build_frame()
    for path in write_all(frame):
        print(f"wrote {path.relative_to(HERE.parent)} ({path.stat().st_size:,} bytes)")
    print(f"{len(frame)} rows x {len(frame.columns)} columns")
    print(f"renewed: {int(frame['renewed'].sum())} of {len(frame)}")
    print(f"satisfaction_score missing: {int(frame['satisfaction_score'].isna().sum())}")


if __name__ == "__main__":
    main()
