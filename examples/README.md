# Demo data

**Synthetic. Generated. Not real data about real people.**

Three files, one table, 300 rows:

| File | Format | Size |
| --- | --- | --- |
| `customer_churn.csv` | CSV | ~19 KB |
| `customer_churn.json` | JSON, an array of objects | ~90 KB |
| `customer_churn.xlsx` | Excel workbook, sheet `customers` | ~21 KB |

They exist so the five-minute demo has something to run on without anybody
downloading anything, and so CI never reaches the network for data.

## The same data three times, on purpose

All three files hold the identical table, so uploading any of them produces the
same content fingerprint:

```
60502bb371071023
```

That is the point being demonstrated. A dataset in this project is identified
by a hash of its **normalised contents**, never by its filename or its format,
so a run on the `.xlsx` lands in the same experiment history as a run on the
CSV. Upload two of them and watch the history agree.

## What is in it

A fictional subscription business. One row per customer, one column to predict.

| Column | Type | Notes |
| --- | --- | --- |
| `customer_id` | text | Unique per row — the profiler should flag it as a likely identifier and the pipeline should keep it out of the feature set |
| `signup_date` | date, as text | Expanded into date parts by preprocessing |
| `plan` | categorical | `basic`, `standard`, `premium` |
| `region` | categorical | Four values, and **no relationship to the target** |
| `signup_channel` | categorical | Four values, also unrelated to the target |
| `tenure_months` | integer | 1–60. Real signal |
| `monthly_spend` | float | Follows the plan |
| `support_tickets` | integer | Real signal, negative |
| `logins_last_30d` | integer | Real signal, positive |
| `satisfaction_score` | float | 1–10, **missing for 34 of 300 rows** |
| `renewed` | 0 / 1 | **The target.** 218 of 300 renewed |

`renewed` is generated from a genuine logistic relationship over
`tenure_months`, `logins_last_30d` and `support_tickets`. So there is something
real for the models to find, SHAP has something true to attribute, and the
three columns it ranks highest are the three the generator actually used —
which is what makes the explanation worth looking at rather than taking on
trust.

**The flaws are deliberate.** A table where every column is clean, complete and
predictive demonstrates nothing about a tool whose first job is to tell you
what is wrong with your data. This one carries an identifier column, real
missingness, a mildly imbalanced target and two columns that mean nothing, so
the profiler's quality findings have something to report and the importance
chart has something to rank near zero.

## Why it is safe to commit

- **Synthetic.** Every value comes from the generator, seeded once. No person,
  company, address or account exists anywhere in it, so there is nothing to
  anonymise and no licence to honour.
- **Small.** All three files together are under 140 KB.
- **Deterministic.** One seed, no wall-clock, no environment lookup — including
  in the workbook, whose document properties and zip entry timestamps are
  frozen so two runs produce identical bytes.

## Reproducing it

```bash
python examples/generate_demo_dataset.py
```

Rewrites all three files. They are committed, so this is never necessary — it
is here so the data's provenance is checkable rather than asserted. If the
files change, the generator changed; running it should leave the working tree
clean.

## Using it

```bash
docker compose up -d --wait
./scripts/demo.sh
```

Or upload `customer_churn.csv` at <http://localhost:3000/dashboard> and follow
the [five-minute demo](../README.md#the-five-minute-demo).
