# ML Copilot — API reference

Sixteen endpoints under `http://localhost:8000`. This is the readable
companion to the generated schema; the authoritative, always-current version is
the interactive documentation the running service serves at **`/docs`**, and
the raw schema at **`/openapi.json`**.

Everything here exists. No endpoint on this page is planned.

---

## Conventions

**Base path.** Everything except `/` and `/health` lives under `/api/v1`.

**Authentication.** Optional, and **off by default** — which is what lets the
local and demo stack run with no secret at all. A deployment that sets
`API_AUTH_ENABLED=true` (and a key in `API_AUTH_KEY`) requires a bearer token
on every protected endpoint:

```
Authorization: Bearer <API_AUTH_KEY>
```

| Public — never needs a credential | Protected — needs one when enabled |
| --- | --- |
| `GET /` | `POST /api/v1/datasets/profile` |
| `GET /health` | `POST /api/v1/experiments/run` |
| `GET /api/v1/experiments/capabilities` | `GET /api/v1/experiments` |
| `GET /api/v1/knowledge/status` | `GET /api/v1/experiments/{id}` |
| `GET /api/v1/agent/status` | `POST /api/v1/experiments/compare` |
| | `GET /api/v1/experiments/{id}/model` |
| | `POST /api/v1/experiments/{id}/predict` |
| | `POST /api/v1/search` |
| | `POST /api/v1/ask` |
| | `POST /api/v1/agent/ask` |
| | `POST /api/v1/agent/ask-with-dataset` |

The public five are liveness and capability information: booleans, limits and
the service's own version. A container healthcheck cannot carry a credential,
and a client has to be able to ask *whether* one is needed before it has one —
`GET /` answers that with `authentication_required`.

**401 behaviour.** Every failure is the same envelope as any other error, with
`WWW-Authenticate: Bearer` and an `X-Request-ID`:

| Situation | `code` |
| --- | --- |
| No `Authorization` header | `authentication_required` |
| A header that is not `Bearer <token>` | `authentication_required` |
| `Bearer` with an empty token | `authentication_required` |
| A bearer token that is not the configured key | `invalid_credentials` |

The comparison is constant-time over SHA-256 digests, so neither the response
nor its timing distinguishes a near miss from a random guess. **Nothing echoes
the credential** — not the message, not the details, not a header, not a log
line. A rejected request logs `Authentication failed` and the value that
failed appears nowhere.

**From a browser, deliberately not.** `Authorization` is not among the CORS
allowed request headers, so a page on an allowed origin cannot send the key
cross-origin. A browser cannot hold a shared secret safely, so the supported
way to reach a protected deployment from one is a server-side proxy that adds
the header — whose own requests are not cross-origin browser requests, and are
unaffected. Server-to-server callers are not subject to CORS at all.

This is a single shared key, not identity: no users, no roles, no expiry. And
it is a password sent on every request, so a remote deployment must terminate
TLS in front of it — see
[PRODUCTION_READINESS.md](PRODUCTION_READINESS.md).

**Uploads** are `multipart/form-data`; everything else is JSON. Three formats
are accepted wherever a dataset is: **CSV**, **Excel (`.xlsx`, first
worksheet)** and **JSON** (an array of objects, or one object whose record list
is under `data`, `records`, `rows`, `items` or `results`).

**One error envelope.** Every failure, from every layer, answers:

```json
{ "error": { "code": "target_column_not_found",
             "message": "Column 'churn' is not in the dataset.",
             "details": { "target_column": "churn",
                          "available_columns": ["renewed", "tenure_months"] } } }
```

`code` is stable and safe to branch on. `message` is safe to show a user.
`details` carries structured context — a limit, a column list, a set of
available options — and never a filesystem path, a credential, a system prompt
or a provider internal. A 5xx returns a generic message with the details
dropped and the real cause logged server-side.

**Correlation.** Every response carries an `X-Request-ID` header. Send your own
in the request header to have it used instead — it must be 1–64 characters of
`A–Z a–z 0–9 _ -`, or it is replaced. The id appears on every log line the
request produces and is the fastest way to find one request in a log.

**A result is not a failure.** An answer that could not be grounded, or an
agent run that ran out of budget, returns **200** with a status field saying
so. Only a genuine provider or server failure is 5xx.

---

## System

### `GET /`

Describe the running service. **Public.**

**Returns** `name`, `version`, `environment`, `docs_url`, and
`authentication_required` — whether the protected endpoints need a bearer
token on this deployment. That last one is a fact about the *configuration*,
never about the key: it says a credential is required, not what it is. It is
also not a disclosure, since one unauthenticated request and a 401 reveals the
same thing; publishing it lets a client say so up front instead of failing on
the user's first action.

### `GET /health`

Liveness. This is what both container healthchecks call, and it is **public on
every deployment** — a healthcheck that carried the API key would put the
credential into an image layer, into `docker inspect` and into every process
list on the host.

**Returns** `{"status": "ok", "version": "...", "environment": "..."}`.

---

## Datasets

### `POST /api/v1/datasets/profile`

Profile an uploaded dataset: structure, per-column statistics, data-quality
findings and — optionally — an analysis of the column you intend to predict.
Nothing is trained and nothing is stored.

| Field | | Notes |
| --- | --- | --- |
| `file` | **required** | CSV, `.xlsx` or JSON |
| `target_column` | optional | Adds a target analysis with a task suggestion |

**Returns** `DatasetProfileResponse`:

- `dataset` — row and column counts, duplicate rows, missing cells, a count per
  column type.
- `columns[]` — dtype, inferred type, missing and unique counts, and numeric,
  datetime or categorical statistics as appropriate.
- `quality` — findings with a `code`, a `severity`, a message and the columns
  involved: high missingness, constant columns, high cardinality, likely
  identifiers, duplicate rows, class imbalance.
- `target` — present only when `target_column` was given: distribution, class
  balance, and `task_suggestion` with the reason for it.
- `source_format` — which adapter read the file.

**Errors.** `415 unsupported_file_type` · `413 file_too_large` ·
`400 empty_file` / `malformed_csv` / `invalid_excel` / `invalid_json` /
`invalid_dataset_content` / `empty_dataset` / `dataset_too_large` /
`missing_header` / `duplicate_columns` / `target_column_not_found`.

---

## Experiments

### `POST /api/v1/experiments/run`

Run a complete experiment and store it. Synchronous: the response arrives when
the run is finished.

The pipeline is profile → preprocessing → leakage-safe split →
cross-validation on the training rows → retrain the winner → **one** measurement
on the untouched test set → SHAP → store.

| Field | | Notes |
| --- | --- | --- |
| `file` | **required** | The dataset |
| `target_column` | optional | Defaults to the last column; the response says which was used |
| `models` | optional, repeatable | Registry identifiers; defaults to every model valid for the task |
| `primary_metric` | optional | Defaults to the task's own default |
| `strategy` | optional | `cross_validation` (default) or `holdout` |
| `folds`, `test_size`, `random_state` | optional | Bounded by the server's configured limits |
| `excluded_columns`, `identifier_columns` | optional, repeatable | Kept out of the feature set |
| `scaling_strategy`, `numeric_imputation`, `categorical_imputation`, `add_missing_indicators`, `max_categorical_cardinality` | optional | Preprocessing overrides |
| `explain` | optional, default `true` | Run SHAP on the winner |
| `name`, `description`, `tags` | optional | Labels for later retrieval |

**Returns** `ExperimentRunResponse` — the full stored record plus `execution`
and `warnings`. The parts worth knowing:

- `dataset.fingerprint` — the run's identity. Computed from the normalised
  data, so the same table as CSV, `.xlsx` and JSON gives one fingerprint.
- `preprocessing` — every decision made, per column, plus the split sizes and
  whether it was stratified.
- `selection` — `candidates[]` with each model's CV score and spread,
  `selected_model`, `selection_score`, `scored_on`, and
  `uses_test_data: false`.
- `evaluation` — `primary_metric_value` and `metrics` **on the held-out test
  set**, the baseline comparison, `test_row_count` and `is_unbiased`.
- `explainability` — `status`, `method` (`shap` or the permutation fallback)
  and ranked `feature_importances`.
- `model_artifact` — present when the winning model was persisted: `stored`,
  `model_name`, `task_type`, `target_column`, `feature_names`, `feature_count`,
  `class_labels`, `artifact_schema_version` and `created_at`. Column *names*
  only; no cell and no row. Absent on a run recorded before persistence
  existed, and absent when persistence failed — a failed write is a warning,
  never a failed experiment.

> `selection_score` is cross-validated on training rows.
> `evaluation.primary_metric_value` is the single held-out measurement. They
> are different numbers answering different questions and must not be compared
> as though they were the same one.

**Errors.** Everything the profile endpoint can return, plus
`400 invalid_experiment_configuration` (a bad fold count, an unknown model, a
metric that does not fit the task) and `409` when the dataset cannot support
the requested run.

### `GET /api/v1/experiments`

List stored runs, newest first.

**Query** — all optional: `dataset_fingerprint`, `target_column`, `task_type`,
`model_name`, `strategy`, `primary_metric`, `tags` (repeatable), `sort_by`,
`order`, `limit`.

**Returns** `count`, `limit` and `experiments[]` of headlines — id, timestamp,
name, fingerprint, task, target, selected model, strategy, primary metric,
`selection_score` and `test_score`.

**Errors.** `400 invalid_request` for an unknown sort key, order or filter
value — the `details` name what is accepted.

### `GET /api/v1/experiments/{experiment_id}`

Fetch one stored run in full — the same record `run` returned.

**Errors.** `404 experiment_not_found` · `400 invalid_experiment_id` (the id is
validated and the resolved path re-checked against the store root, so it can
never address a file outside it).

### `POST /api/v1/experiments/compare`

Rank several stored runs against each other.

**Body** `{"experiment_ids": ["exp_...", "exp_..."]}` — at least two.

**Returns** the shared `task_type`, `primary_metric`, `direction` and
`higher_is_better`; `best_experiment_id`; and a `table[]` row per run with its
selection score, test score, baseline and improvement.

Ranking respects metric direction — for RMSE and MAE the best run is the lowest
one, and the response says which direction was applied.

**Errors.** `404 experiment_not_found` · `409 incomparable_experiments` when
the runs do not share one task and one primary metric.

### `GET /api/v1/experiments/capabilities`

What an experiment may ask for on this deployment: `models[]` (identifier,
display name, task type, whether it supports probabilities and a random state),
`metrics`, `strategies`, `sort_keys`, `limits` and
`supported_dataset_extensions`.

The dashboard builds its run form from this, so the form can never offer a
model the server does not have.

---

## Prediction

An experiment that finished successfully leaves its winning model behind. These
two endpoints ask what that model expects, and then use it.

**The same fitted objects, not a rebuild.** What is stored is the complete
sklearn `Pipeline` the experiment produced — the preprocessing *as fitted on
the training rows*, and the estimator retrained on them. A prediction runs
`pipeline.predict(...)` on that object. **Nothing is re-fitted on a prediction
request**, and nothing is re-derived from the record; a prediction made here
uses exactly the transformation the held-out score was measured through, which
is what makes the two comparable.

**No path is ever accepted.** A request carries feature values and nothing
else. The model is addressed by the experiment id in the URL, which is
validated as an id before it is used to address anything, and the resolved
location is re-checked against the artifact root. There is no field, anywhere,
that names a file — see [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) for
the deserialisation trust boundary.

### `GET /api/v1/experiments/{experiment_id}/model`

Whether this experiment can be predicted from, and with what.

Answered from the **artifact store**, not from the experiment record: the
record says a model was written when the run finished; this says whether one is
readable now.

**Returns** `ModelAvailability`:

- `experiment_id`, `available`, `max_records`, and `reason` — why not, when
  `available` is false.
- When available: `model_name`, `display_name`, `task_type`, `target_column`,
  `classes` (empty for regression), `created_at`, `train_row_count`,
  `primary_metric`, `primary_metric_value`, and `features[]` of
  `{name, kind, dtype}` — `kind` being which branch of the fitted preprocessing
  handles the column (`numeric`, `categorical`, `boolean`, `datetime`).
- When unavailable, every descriptive field is `null` and `features` is empty —
  deliberately not a placeholder schema, so a client that builds a form from
  `features` builds nothing rather than a form whose requests cannot succeed.

**`available: false` is a 200.** "This run has no model" is an answer, not a
failure. A 404 here means the experiment itself does not exist.

**Errors.** `404 experiment_not_found` · `400 invalid_experiment_id`.

### `POST /api/v1/experiments/{experiment_id}/predict`

Predict with the model this experiment produced.

```json
{ "records": [ { "tenure_months": 30, "monthly_spend": 34.89,
                 "support_tickets": 2, "satisfaction_score": 6.7 } ] }
```

One shape for one record and for a batch: `records` is always a list, and the
response's `predictions` is always a list of the same length in the same order.
The list must hold at least one record and at most the server's
`max_records` (default 500; the schema itself refuses more than 1000 before a
list that size is ever built).

**Every key is a feature name.** Every feature the model was trained on must be
present. `null` is a valid value for a missing one — the model's imputation was
fitted for exactly that. **A feature the model was not trained on is refused
rather than ignored**, because the fitted `ColumnTransformer` drops an unknown
column rather than using it, so a misspelt name would otherwise produce a
confident prediction made without that value.

**Returns** `PredictionResponse`:

- `predictions[]` — `index` (position in the submitted batch), `prediction`
  (the class label, or the number), and `probabilities` — one entry per class
  label for a classifier whose estimator provides them, `null` for regression
  and for a classifier that does not.
- `prediction_count`.
- `model` — `experiment_id`, `created_at`, `model_name`, `display_name`,
  `task_type`, `target_column`, `classes`, `features[]`, `train_row_count`,
  `primary_metric` and `primary_metric_value`.

> `model.primary_metric_value` is the winner's score on the **held-out test
> set**. It says how much to trust the model. It is not a confidence in this
> prediction.

**Errors.**

| Situation | Status | `code` |
| --- | --- | --- |
| No such experiment | 404 | `experiment_not_found` |
| A malformed experiment id | 400 | `invalid_experiment_id` |
| The run has no stored model | 409 | `model_not_available` |
| A missing, unexpected or uncoercible feature; an empty or oversized batch; a record that is not an object | 422 | `invalid_prediction_input` |
| Malformed JSON, or a body that is not a `records` list | 422 | `invalid_request` |
| A stored artifact that cannot be read or verified | 500 | `model_artifact_unreadable` |
| The model raised while predicting | 500 | `prediction_failed` |

`model_not_available` is a **409, not a 404**, so "no such run" and "this run
has no model" stay distinguishable — a run from before persistence existed, or
one whose artifact was deleted, is reported rather than fabricated.

An `invalid_prediction_input` failure carries the reason in `details`: the
`index` of the offending record, and then `missing_features` /
`unexpected_features` with `expected_features` alongside them, or `feature` and
the `expected` kind when a value could not be read as the type that column was
trained as. A client can point at the box that is wrong.

```bash
curl -H 'Content-Type: application/json' \
     -d '{"records": [{"tenure_months": 30, "monthly_spend": 34.89,
                       "support_tickets": 2, "satisfaction_score": 6.7}]}' \
     http://localhost:8000/api/v1/experiments/exp_.../predict
```

---

## Knowledge

### `POST /api/v1/search`

Semantic search over the project's own documentation and its experiment
history. Retrieval only — no language model is involved, so this works with no
credential configured.

| Field | | Notes |
| --- | --- | --- |
| `query` | **required** | Non-empty; a server-configured maximum length applies |
| `top_k` | optional | Capped by the server's maximum |
| `similarity_threshold` | optional | Raise it to trade recall for precision |
| `filters` | optional | `source_types`, `task_type`, `dataset_fingerprint`, `target_column`, `selected_model`, `primary_metric`, `experiment_id` |

Filters are applied **before** ranking, so a filtered search returns the best
matches within the filter rather than the survivors of an unfiltered top-k.

**Returns** `results[]` — rank, score, the passage, its document and chunk ids,
source type, title, reference and a stable `citation_id` — plus
`candidate_count` and the similarity metric used.

**Errors.** `503 retrieval_index_not_built` when no index exists yet (build it
with `RagIndexer(...).index_documentation()`; the container entrypoint does
this at every start) · `400 invalid_request` for an unknown filter value.

### `POST /api/v1/ask`

Answer a question **from retrieved evidence**, with citations.

Same fields as `/search`, with `question` in place of `query`.

**Returns** `AskResponse`:

- `answer` and `status` — `answered`, `insufficient_evidence` or
  `grounding_failed`.
- `is_grounded`, `citations[]` (id, source type, title, reference, relevance,
  excerpt), `citation_ids`, `allowed_citations`, `rejected_citations`.
- `metadata` — provider, model, how many passages were retrieved and used,
  whether the context was truncated, approximate token counts and latency.

**The model is not the source of truth.** Every project-specific claim must
come from a retrieved passage. Citations are checked against the passages
actually supplied, and an answer citing a source it was not given is rejected —
that is `grounding_failed`, returned as **200** with the rejected ids listed.
`insufficient_evidence` means retrieval found nothing good enough; it is an
honest answer, not a failure.

**Errors.** `503 llm_not_configured` (no credential — the message says which
variable to set) · `503 retrieval_index_not_built` · `502 llm_provider_error` /
`llm_timeout` / `llm_rate_limited`.

### `GET /api/v1/knowledge/status`

What the knowledge endpoints can do right now: `search_available`,
`answering_available`, `index_built`, the similarity metric, the default and
maximum `top_k`, the maximum query length and the source types present.

Booleans and limits only — never a key, a path or a provider internal. The
dashboard header reads this to report features as unavailable rather than
letting a user hit a 503.

---

## Agent

### `POST /api/v1/agent/ask`

Answer a question by letting the system choose which of its own capabilities
the question needs — profiling, running an experiment, explaining a winner,
searching the history — and then answering from what those steps actually
returned.

| Field | | Notes |
| --- | --- | --- |
| `question` | **required** | 1–2000 characters |
| `max_tool_calls`, `max_iterations`, `max_context_chars` | optional | **May only lower** the server's limits; a larger value is rejected |

**Returns** `AgentAskResponse`:

- `status` — `completed`, `partial`, `insufficient_evidence` or
  `grounding_failed`. **All four are HTTP 200.**
- `final_answer`, `is_answer`, `citations[]`, `experiment_ids`, `warnings`.
- `tool_calls[]` and `observations[]` — the auditable trace: which tool ran,
  whether it succeeded, how long it took, an `input_summary` and the
  observation's result.
- `iterations`, `tool_call_count`, `tools_available`, `duration_ms`.

**What is never returned:** chain-of-thought, hidden reasoning, the system
prompt, provider internals, credentials, or raw dataset rows.

**What the agent cannot do:** execute Python, run a shell command, make an HTTP
request, or touch the filesystem. It can call exactly four registered tools;
anything else becomes a rejected observation it may correct from.

**Errors.** `400 invalid_agent_budget` (a budget above the server's ceiling) ·
`503 agent_unavailable` (no credential) · `502 agent_provider_error` /
`agent_planner_error` / `agent_run_failed`.

### `POST /api/v1/agent/ask-with-dataset`

The same bounded run, on a dataset uploaded with the question.

| Field | | Notes |
| --- | --- | --- |
| `file` | **required** | CSV, `.xlsx` or JSON |
| `question` | **required** | 1–2000 characters |
| `max_tool_calls`, `max_iterations`, `max_context_chars` | optional | As above |

The dataset is parsed in memory, lent to that one run under a name, and
released. The agent receives no `UploadFile`, no path and no file extension.
The response's `dataset` block reports the name, filename, source format,
fingerprint, shape, column names and `persisted: false`.

**This is one endpoint for all three formats.** There is no
`ask-with-excel` and no `ask-with-json`.

**Errors.** Every upload error above, plus every agent error above.

### `GET /api/v1/agent/status`

What the agent can do right now: `agent_available`, the registered `tools`,
whether dataset upload is supported, the supported formats, and the four
budget ceilings.

---

## Reading the numbers correctly

Three things this API is careful about, and a client should be too:

1. **Cross-validation is not the test score.** `selection.selection_score` is
   the mean over folds of the *training* rows.
   `evaluation.primary_metric_value` is one measurement on data no model saw.
   Report them separately.
2. **Not every metric improves upward.** Direction travels with the metric.
   Never sort a metric column without reading `higher_is_better`.
3. **Feature importance is not causation.** SHAP describes what the model does,
   not what drives the outcome in the world. The disclaimer is part of the
   contract, not decoration.

---

## Trying it

```bash
# Profile the demo dataset
curl -F "file=@examples/customer_churn.csv" -F "target_column=renewed" \
     http://localhost:8000/api/v1/datasets/profile

# Run an experiment on it
curl -F "file=@examples/customer_churn.csv" -F "target_column=renewed" \
     -F "models=logistic_regression" -F "models=random_forest_classifier" \
     http://localhost:8000/api/v1/experiments/run

# Search the project's own documentation
curl -H 'Content-Type: application/json' \
     -d '{"query": "cross-validation versus the final test evaluation"}' \
     http://localhost:8000/api/v1/search
```

On a deployment with `API_AUTH_ENABLED=true`, add the header to each of those:

```bash
curl -H "Authorization: Bearer $API_AUTH_KEY" \
     -F "file=@examples/customer_churn.csv" -F "target_column=renewed" \
     http://localhost:8000/api/v1/datasets/profile
```

`scripts/demo.sh` runs the whole sequence, and passes the header when
`API_AUTH_KEY` is set in the environment. See
[ARCHITECTURE.md](ARCHITECTURE.md) for how the pieces fit together.
