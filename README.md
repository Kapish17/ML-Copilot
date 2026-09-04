# ML Copilot

### AI Data Scientist — automated profiling, experimentation, explainability, RAG and agentic analysis

<!-- Replace OWNER/REPO with this repository's path once it has a remote.
     The workflow file is .github/workflows/ci.yml, so the badge path is
     .../actions/workflows/ci.yml/badge.svg — only the owner and repository
     name are unknown here. -->
[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Node](https://img.shields.io/badge/node-20%2B-green)

Upload a spreadsheet. Get back a profile of what is wrong with it, a
cross-validated comparison of six models, an unbiased score on data no model
saw, a SHAP explanation of the winner, and an answer to *"which model performs
best and why?"* that cites the run it came from.

**[Architecture](docs/ARCHITECTURE.md) · [API reference](docs/API.md) ·
[Production readiness](docs/PRODUCTION_READINESS.md) ·
[CI workflow](.github/workflows/ci.yml) · [Demo data](examples/README.md)**

```bash
docker compose up --build     # then open http://localhost:3000/dashboard
./scripts/demo.sh             # or watch the whole thing from a terminal
```

---

## What it is

A working AI data scientist, built as five Python packages and a TypeScript
dashboard. You give it a tabular dataset and a question; it decides which of
its own capabilities the question needs, runs them, and answers from what they
actually returned.

Everything below is implemented and covered by the test suite. Nothing on this
page is planned, aspirational, or a stub — what is *not* built is listed under
[Limitations](#limitations) and stated as plainly as what is.

## Why it is interesting

Most portfolio ML projects train a model and report a number. The interesting
problems are the ones that show up afterwards, and this project is organised
around four of them.

**The number is usually wrong.** Fit a scaler before the split and the test
score is inflated. Pick the best model *by* its test score and the test score
is no longer an estimate of anything. So the split happens before anything is
fitted, every transformer is fitted on training rows only, candidates are
cross-validated on the training rows alone, and the winner is measured
**once** on data it has never seen. Cross-validated and held-out scores are
separate fields in the record, separate columns in the API, and separate column
groups in the UI — because presenting them together invites the reader to
compare two numbers that answer different questions.

**A language model that answers from memory is a liability.** Every
project-specific claim has to come from a retrieved passage. Citations are
validated against the passages actually supplied, and an answer citing a source
it was not given is **rejected**, not quietly cleaned up. An answer that cannot
be grounded returns a status saying so — which is a result, not an error.

**An agent that can do anything can be talked into anything.** This one is
bounded by construction rather than by prompt: an explicit registry of four
tools, typed arguments validated before anything runs, and hard budgets a
request may lower and never raise. It cannot execute Python, run a shell
command, make an HTTP request or touch the filesystem, and it never receives a
file path or a file extension. Retrieved text and tool output are data, never
instructions.

**Explanations are the easiest thing to misread.** SHAP describes what a model
does, not what causes an outcome in the world. That disclaimer travels with the
numbers — in the stored record, in the API response, and rendered beside the
bars.

## Architecture

```
                    ┌──────────────────────┐
                    │   Next.js Dashboard  │
                    │  Upload / AI / ML UI │
                    └──────────┬───────────┘
                               │ HTTP
                               ▼
                    ┌──────────────────────┐
                    │      FastAPI API     │
                    └──────────┬───────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   Dataset/ML Pipeline       Agent                 RAG
          │                    │                    │
          ▼                    ▼                    ▼
   sklearn / SHAP       bounded tools       local retrieval
                               │
                               ▼
                            LLM layer
```

Five Python packages with **one-way dependencies enforced by tests that parse
the imports**, not by convention. `agent/` is the strictest: it imports no web
framework, no `pandas`, no `numpy`, no `scikit-learn`, no `shap`, no `openai`
and nothing from the backend, talking to every collaborator through a
structural `Protocol`.

**The frontend is a presentation layer.** ML computation, retrieval, agent
execution and every language-model call are server-side. There is no `sklearn`
or `openai` equivalent in the browser bundle and no credential can reach it.

Full detail, including the ingestion adapters, storage and deployment:
**[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Key capabilities

| | What it does |
| --- | --- |
| **Ingestion** | CSV, Excel (`.xlsx`) and JSON behind one adapter layer. Everything downstream sees a standardised DataFrame — one endpoint per capability, never one per format. Format detection is explicitly not the security boundary; the adapters and validators are. |
| **Identity from content** | A dataset is identified by a fingerprint of its normalised contents, never its filename. The same table as CSV, `.xlsx` and JSON produces one fingerprint and one shared history. |
| **Profiling** | Structure, per-column statistics, and data-quality findings: high missingness, constant columns, high cardinality, likely identifiers, duplicate rows, class imbalance. Plus an optional target analysis that suggests the task and says why. |
| **Preprocessing** | scikit-learn `Pipeline` + `ColumnTransformer`: imputation, one-hot encoding with a cardinality ceiling, scaling, datetime expansion — inferred from the profile or given explicitly. |
| **Leakage prevention** | Structural, not procedural. The train/test split happens before anything is fitted; every transformer is fitted on training rows only; the fitted preprocessing travels inside the model artefact so scoring cannot diverge from training. |
| **Model selection** | Six models, cross-validated on the **training rows only**, with fold-level scores, mean and spread. The winner is chosen from CV scores and never from the test set. |
| **Unbiased evaluation** | The winner is retrained on the full training set and measured **once** on the untouched test set, against a naive baseline, with metric direction carried alongside so nothing assumes higher is better. |
| **Explainability** | SHAP over the transformed features with readable names, global ranked importance and signed per-prediction contributions, and a permutation fallback for models SHAP cannot handle. A failed explanation is a status, not a 500. |
| **Experiment tracking** | Versioned JSON records: fingerprint, every preprocessing decision, candidate results, both scores, the baseline, the explanation, and the environment. Atomic writes. **No dataset rows are stored** — the record holds column names and statistics, never a cell. |
| **Model persistence** | A successful run's winning `Pipeline` — the preprocessing *as fitted*, plus the retrained estimator — is written to an application-owned artifact directory beside a manifest of the feature schema, checksummed and verified on load. Written only after evaluation succeeds; a failed write is a warning, not a failed experiment. |
| **Prediction** | `POST /api/v1/experiments/{id}/predict` runs new records through that exact stored pipeline. **Nothing is re-fitted**, so a prediction goes through the same transformation the held-out score was measured through. A request carries feature values and never a path. |
| **Retrieval** | Semantic search over the project's own documentation and its run history, with structure-aware chunking, pre-ranking metadata filters and stable citations. The default embedding provider is stateless — no download, no key, identical vectors everywhere. |
| **Grounded answers** | Evidence-first generation with validated citations, over any OpenAI-compatible endpoint — hosted, or a model on your laptop via `LLM_BASE_URL`. |
| **Bounded agent** | Four registered tools, typed arguments, hard budgets, four outcomes all returned as HTTP 200 with a status. No chain-of-thought is ever returned. |
| **Dashboard** | Upload, profile, ask, run, compare, explain, browse history, search knowledge. Runtime dependencies: Next.js, React, React DOM. No UI kit, no chart library, no state manager. |
| **Authentication** | Optional shared API key over `Authorization: Bearer`, compared in constant time, on the eleven endpoints that cost something. Off by default so the demo needs no secret; enabling it without a key fails at start-up rather than pretending to be protected. The key never reaches the browser, an image, a log or the schema. |

## The five-minute demo

Everything here needs **no API key** except step 4, which says so and carries on
without one.

```bash
docker compose up --build          # ~2 minutes on a cold cache
open http://localhost:3000/dashboard
```

**1 · Upload.** Drop `examples/customer_churn.csv` on the dashboard. 300 rows
of synthetic subscription data — [what is in it](examples/README.md).

**2 · Read the profile.** Eleven columns typed and summarised. Expect the
quality panel to flag `customer_id` as a likely identifier and
`satisfaction_score` as missing for 34 rows. *This is the point:* the data has
deliberate flaws, because a tool whose first job is to find them proves nothing
on a clean table.

**3 · Check the target.** Set the target to `renewed`. Expect
**classification**, with the reason given, and a mild imbalance reported —
218 of 300 renewed.

**4 · Ask the AI Data Scientist.** Type:

> **Which model performs best and why?**

Expect the answer to name a model, cite the run it came from, and show which
tools it ran to find out. Expect **no chain-of-thought** — the trace shows
*what* it did, not what it was thinking. With no credential configured, expect
instead a clear "language model not configured" message and an unchanged page:
every other step still works.

**5 · Run an experiment.** Leave the defaults. About fifteen seconds.

**6 · Compare the models.** The table has two column groups, and the separation
is the thing to look at. Expect `random_forest_classifier` to win on a
cross-validated F1 near **0.866 ± 0.039**, scored on training folds only, with
`uses_test_data: false`.

**7 · Read the held-out score.** Expect a test F1 near **0.866** on 60 rows no
model saw, against a majority-class baseline of **0.846** — an improvement of
about two points. That baseline is why the raw number is not the story.

**8 · Open the SHAP explanation.** Expect `tenure_months` (0.084),
`logins_last_30d` (0.061) and `support_tickets` (0.048) at the top — exactly
the three columns the generator built the signal from, so the explanation is
**checkable rather than merely plausible**. Expect the one-hot columns for
`region` and `signup_channel` in the bottom half at around 0.003–0.008: they
were generated with no relationship to the outcome at all. And expect the
causation disclaimer next to the bars.

**9 · Browse the history.** The run is there, findable by its content
fingerprint `60502bb371071023`. Now upload `examples/customer_churn.xlsx` —
the same table as a spreadsheet — and watch it land in the **same** history
under the **same** fingerprint.

**10 · Search the knowledge base.** Ask *"cross-validation versus the final
test evaluation"* on the Knowledge page. Expect cited passages from this
project's own documentation, each with a citation id — those ids are the only
sources an answer is allowed to cite. The default embedding provider is a
stateless hashing vectorizer, so it matches on terms rather than meaning and a
question in the documentation's own vocabulary retrieves noticeably better than
a short paraphrase. That is the trade for needing no download and no key;
`RAG_EMBEDDING_PROVIDER=sentence_transformer` is the alternative.

**11 · Predict with it.** Open the **Predict** tab on the run's page. The form
is built from the model's own declared schema, not from anything the page
already knew — type a row and get a class back with its probabilities. The
values run through the *same fitted preprocessing* the run produced; nothing is
re-fitted, which is what makes this prediction comparable to the held-out score
in step 7. Old runs recorded before this existed say so instead of showing a
form that could not work.

**From a terminal instead:** `./scripts/demo.sh` walks the same path with
`curl`, printing each result. It needs no key and no network. On Windows, run it
in Git Bash or WSL, or follow the equivalent commands in
[docs/API.md](docs/API.md).

## Tech stack

| Layer | Technology |
| --- | --- |
| API | Python 3.11, FastAPI, Uvicorn, Pydantic v2 |
| Data | pandas, openpyxl |
| ML | scikit-learn — `Pipeline`, `ColumnTransformer`, six estimators |
| Explainability | SHAP, with permutation-importance fallback |
| Retrieval | Local vector store behind a `VectorStore` interface; stateless hashed-n-gram embeddings by default, optional `all-MiniLM-L6-v2` |
| Language model | Provider abstraction over the OpenAI-compatible chat API — OpenAI, Azure, vLLM, Ollama, LM Studio, OpenRouter |
| Agent | Own package. No framework — no LangChain, LangGraph, AutoGen or CrewAI |
| Frontend | Next.js 15 (App Router), React 19, TypeScript, Tailwind |
| Storage | Local JSON records and a local index, on named volumes |
| Deployment | Docker Compose — two images, one command |
| CI | GitHub Actions — tests, gates, dependency audits, Docker stack smoke test |

## Quick start

**Requirements:** Python 3.11+ and Node.js 20+, or just Docker.

```bash
git clone <this repository>
cd ml-copilot
cp .env.example .env          # optional — every value has a working default
docker compose up --build
```

| | |
| --- | --- |
| Dashboard | <http://localhost:3000/dashboard> |
| API | <http://localhost:8000> |
| Interactive API docs | <http://localhost:8000/docs> |

**Without an API key**, profiling, experiments, cross-validation, SHAP, history
and retrieval search all work. Only answer generation and the agent need a
credential, and the dashboard reports those two as unavailable in its header
rather than failing. To enable them, put a key in `.env`:

```bash
LLM_API_KEY=your-key-here
```

`LLM_BASE_URL` points the same provider at any OpenAI-compatible endpoint, so a
local model works without an external credential.

## Docker

Two images, built from the committed Dockerfiles. The backend is a two-stage
build whose runtime carries no compiler and no pip cache; the frontend is three
stages whose runtime carries the Next.js standalone output and nothing else.
Both run as unprivileged users and define their own `HEALTHCHECK`, so
`docker compose up --wait` blocks until the application actually answers.

| Variable | Default | Notes |
| --- | --- | --- |
| `API_AUTH_ENABLED` | `false` | Require an API key on the protected endpoints. See [Authentication](#authentication) |
| `API_AUTH_KEY` | *(empty)* | The key itself. Backend only — never a build argument, never passed to the frontend, never in an image |
| `LLM_API_KEY` | *(empty)* | Unrelated to the above: this one lets the API *call* a language model. Never baked into an image, never passed to the frontend |
| `CORS_ALLOW_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Explicit list, never `*` |
| `NEXT_PUBLIC_API_BASE_URL` | `http://localhost:8000` | Inlined at **build** time — must be a URL the *browser* can resolve, never the Compose service name |
| `BIND_ADDRESS` | `127.0.0.1` | Host interface the ports are published on |
| `FRONTEND_PORT` / `BACKEND_PORT` | `3000` / `8000` | Update the URL and the CORS list to match |

**Both ports bind loopback by default.** Authentication is off unless you turn
it on, so a stock stack accepts file uploads from anyone who can reach the
port — and a published Docker port bypasses a host firewall rather than being
filtered by it. `BIND_ADDRESS=0.0.0.0` is the deliberate opt-in, and it is the
point at which [Authentication](#authentication) stops being optional.

| | Survives `down` | Survives `down -v` |
| --- | --- | --- |
| Experiment records | yes | no |
| Trained model artifacts | yes | no — and the runs then report `model_not_available` |
| Retrieval index | yes | rebuilt at next start |
| **Uploaded datasets** | **never stored at all** | — |

An uploaded dataset is parsed in memory for one request and released. There is
no upload directory, no temporary file, and deliberately no volume that could
become one. The model volume holds fitted `Pipeline` objects — learned
parameters, and column *names* in their manifests. It holds no dataset rows.
`MODEL_ARTIFACT_DIR` moves it; `MAX_PREDICTION_RECORDS` caps a prediction
batch.

## Local development

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r backend/requirements-dev.txt -r ml/requirements.txt \
            -r rag/requirements.txt -r llm/requirements.txt

python -c "from rag import RagIndexer, config_from_env; \
           print(RagIndexer(config_from_env()).index_documentation())"

uvicorn app.main:app --reload --app-dir backend
```

In a second terminal:

```bash
cd frontend && npm ci && npm run dev
```

Every log line carries the request it belongs to:

```
2026-09-03T11:20:14 INFO  app.api.middleware [3f9a1c7e2b8d4a15] POST /api/v1/experiments/run -> 200 in 14428.6ms
```

That id is also returned in the `X-Request-ID` response header, so a request
someone reports can be found in the log by its header. Send your own to have it
used instead. Set `LOG_LEVEL=DEBUG` for more; it raises the level only on this
project's own loggers, never on a third-party package.

## Authentication

**Off by default, and on with two environment variables.**

That default is deliberate. `docker compose up --build` has to bring up a
working system with no secret to configure, and the stack publishes both ports
on `127.0.0.1`, so what it starts is a local tool rather than an exposed
service. Turning authentication on is what you do before it stops being local.

```bash
# .env
API_AUTH_ENABLED=true
API_AUTH_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(24))")
```

Then every protected endpoint needs the key:

```bash
curl -H "Authorization: Bearer $API_AUTH_KEY" \
     -F "file=@examples/customer_churn.csv" \
     http://localhost:8000/api/v1/datasets/profile
```

Enabling it **without** a key is a start-up failure, not a warning. No key is
generated and there is no built-in default: a generated one would be a secret
nobody knows that changes every restart, and a shipped one would be a password
published in this repository. The remaining option — refusing to start — is the
only one that cannot leave an operator believing the service is protected when
it is not. A key under 32 characters is refused for the same reason.

### What is protected, and what is not

| | Endpoints | Why |
| --- | --- | --- |
| **Public** | `GET /` · `GET /health` · `/experiments/capabilities` · `/knowledge/status` · `/agent/status` | Liveness and capability booleans. A container healthcheck cannot carry a credential, and a client has to be able to ask *whether* it needs one before it has one. None of them reads stored state, costs CPU, or reports anything but limits and availability. |
| **Protected** | `/datasets/profile` · `/experiments/run` · `/experiments` · `/experiments/{id}` · `/experiments/compare` · `/search` · `/ask` · `/agent/ask` · `/agent/ask-with-dataset` | Everything that accepts an upload, trains a model, reads stored experiments, or reaches a language model. |

The split is asserted from the running application rather than kept in a list,
and a rule — *every non-GET route must be protected* — fails the suite if a new
expensive endpoint is ever added without a decision being made about it.

### Why the dashboard has no API key

**A browser application cannot hold a shared secret.** Anything shipped in a
Next.js bundle is readable by every visitor who opens developer tools, so a
`NEXT_PUBLIC_API_KEY` would not be a key — it would be a public string that
happens to open the API. There is none, there is no hard-coded credential in
the TypeScript, and Compose passes nothing of the sort to the frontend service.
A test walks the frontend source and fails if the name of the server-side
variable appears there at all.

So when the backend is protected, the dashboard is honest about it: the header
reads **"API key required — this dashboard cannot hold one"**, and any request
that is refused explains the same thing rather than suggesting a retry. It
reads that state from `GET /`, which reports `authentication_required` as a
boolean — a fact about the configuration that anyone could learn by making one
request and reading the 401.

CORS backs that up rather than working around it: `Authorization` is **not** in
the allowed request headers, so a browser on an allowed origin cannot send the
key cross-origin at all. Nothing legitimate is blocked by that — a
server-to-server caller is not subject to CORS, and a proxy that adds the
header server-side is not making a cross-origin browser request either — but
putting the key into JavaScript now fails visibly at the first request instead
of quietly working with a leaked credential.

To use a protected deployment from a browser, put something in front of the
dashboard that holds the credential server-side: a reverse proxy that injects
the header, or a small backend-for-frontend. **Neither is in this commit** —
adding a proxy purely to hide a key would replace an honest limitation with a
component nobody had asked for, and the security model would be no better for
it.

### What this is, and is not

This is a **single shared API key**, compared in constant time, with
`Authorization: Bearer <key>`. It is not identity management. There are no
users, no roles, no sessions, no expiry and no revocation short of changing the
key and restarting; everyone holding it is the same caller and the log cannot
tell them apart. No JWT, no OAuth, no SSO, no database of users and no
third-party authentication library — FastAPI's own security utilities and
`secrets` are the whole implementation.

It addresses one precise threat: an unauthenticated service that accepts file
uploads and runs model training synchronously lets anyone who can reach the
port spend the host's CPU. It does nothing about an authorised caller doing the
same thing, which is why the [existing hard
budgets](docs/PRODUCTION_READINESS.md) matter as much as the key does.

### It still needs TLS

```
Client ──HTTPS──▶ reverse proxy / load balancer ──HTTP──▶ FastAPI
```

A bearer token is a password sent on every request. Over plain HTTP anyone on
the path reads it once and has it forever, so **API-key authentication without
TLS is not sufficient for a remote deployment**. Terminating TLS is the
proxy's job, not this application's; no Nginx, Caddy, Traefik or cloud load
balancer is included here.

## CI and security

Four GitHub Actions jobs on every push and pull request to `main`, with a
`contents: read` token and **no secret of any kind** — so CI works unchanged on
a fork.

| Job | What it proves |
| --- | --- |
| **Backend tests** | Five pytest suites, plus a compile pass over every module |
| **Frontend tests** | `npm ci`, audit, lint, typecheck, Vitest, production build |
| **Dependency audit** | `pip-audit --strict` over the production **and** development closures |
| **Docker stack smoke test** | Builds both images, starts the stack, runs 30 checks against the running containers |

Dependabot watches every Python, npm and GitHub Actions manifest weekly, with
**no ignore rules and no automatic merging**. `npm audit --audit-level=high`
runs immediately after `npm ci`; nothing is suppressed with `|| true`, a lowered
threshold or `continue-on-error`, and the test suite asserts that.

Nine security assertions were each verified by making the corresponding mistake
and watching the suite fail. The last audit found four advisories and fixed
three of them by bumping `vitest`, `postcss` and `sharp` — deliberately *not*
by taking the major Next.js upgrade `npm audit fix --force` proposed.

The full checklist, including what is **not** production-grade:
**[docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md)**.

## Testing

```bash
pytest                       # all five suites from the repository root
pytest backend/tests         # or one at a time: backend ml rag llm agent
```

```bash
cd frontend
npm test                     # Vitest + Testing Library
npm run lint                 # ESLint
npm run typecheck            # tsc --noEmit
```

Every test builds its data in memory. Nothing reads an external dataset,
downloads a model or touches the network. Retrieval uses a deterministic fake
embedding provider and the language-model tests a deterministic fake provider,
so **no API key is needed to run anything**. Two optional tests skip themselves
unless explicitly enabled: the real sentence-transformer model, and the real
LLM provider.

`scripts/smoke-test.sh` runs 30 checks against a live stack — including the
three failures that are invisible from the outside: a browser bundle built with
the internal Compose hostname, a CORS mismatch, and an index that never got
built. The first two were verified to *fail* when the mistake is made.

## API documentation

Sixteen endpoints. The readable reference is **[docs/API.md](docs/API.md)**;
the running service serves the authoritative interactive schema at
**`/docs`** and the raw OpenAPI document at `/openapi.json`.

| | |
| --- | --- |
| `POST /api/v1/datasets/profile` | Profile an upload — structure, quality, target |
| `POST /api/v1/experiments/run` | Run and store a complete experiment |
| `GET /api/v1/experiments` | List stored runs, with filters and sorting |
| `GET /api/v1/experiments/{id}` | One run in full |
| `POST /api/v1/experiments/compare` | Rank several runs, respecting metric direction |
| `GET /api/v1/experiments/capabilities` | Models, metrics and limits this deployment allows |
| `GET /api/v1/experiments/{id}/model` | Whether a stored model exists, and the features it expects |
| `POST /api/v1/experiments/{id}/predict` | Predict with that model — feature values in, never a path |
| `POST /api/v1/search` | Semantic search with citations — no credential needed |
| `POST /api/v1/ask` | A grounded answer, or an honest status |
| `GET /api/v1/knowledge/status` | What the knowledge endpoints can do right now |
| `POST /api/v1/agent/ask` | The bounded agent |
| `POST /api/v1/agent/ask-with-dataset` | The bounded agent, on an uploaded dataset |
| `GET /api/v1/agent/status` | Tools, formats and budget ceilings |
| `GET /` · `GET /health` | Service identity and liveness |

Every failure answers in one envelope —
`{"error": {"code", "message", "details"}}` — with a stable `code`, a message
safe to show a user, and details that never carry a path, a credential or a
provider internal.

## Project structure

```
ml-copilot/
├── backend/      FastAPI service — api, core, schemas, services, tests
│   └── Dockerfile        Production image, built from the repository root
├── frontend/     Next.js dashboard — the presentation layer over the API
│   └── Dockerfile        Production image, three stages
├── ml/           Preprocessing, training, selection, explainability, tracking,
│                 model artifacts and prediction
├── rag/          Chunking, embeddings, vector store, retrieval, citations
├── llm/          Provider abstraction, prompts, grounding, citation validation
├── agent/        Bounded tool-calling agent — registry, planner, orchestrator
├── docs/         Architecture, API reference, production readiness
├── examples/     Synthetic demo data in all three formats, and its generator
├── scripts/      demo.sh · smoke-test.sh
├── data/         Your own local datasets — contents git-ignored, never committed
├── .github/      CI workflow and the Dependabot configuration
├── .env.example  Every setting, documented
└── docker-compose.yml
```

Each Python package has its own README with the depth this page deliberately
does not: **[ml](ml/README.md)** · **[rag](rag/README.md)** ·
**[llm](llm/README.md)** · **[agent](agent/README.md)** ·
**[backend](backend/README.md)** · **[frontend](frontend/README.md)**.

## Limitations

Stated plainly, because a portfolio project that hides its edges is not worth
reading.

- **Authentication is one shared key, not identity.** No users, no roles, no
  sessions, no expiry, no revocation short of restarting. It is off by default,
  so a stock `docker compose up` is still open to anyone who can reach the port
  — which is why Compose binds loopback. See [Authentication](#authentication).
- **No authorisation and no rate limiting.** Every holder of the key can do
  everything, as often as the budgets allow.
- **No TLS and no reverse proxy.** A bearer token over plain HTTP is readable
  by anyone on the path.
- **Training is synchronous.** A run holds its HTTP request open for its whole
  duration. No queue, no worker, no Celery, no Redis.
- **Model persistence is local and single-node.** A winning model is written to
  a directory on the server (a Docker volume in Compose), not to a registry. No
  MLflow, no S3, no versioning beyond one artifact per experiment, no rollback,
  no promotion, and no sharing between replicas. The artifact is a `joblib`
  file, so it is only loadable by a compatible scikit-learn — and only
  application-generated artifacts are ever loaded (see
  [PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md)). **Uploading a model
  file is not supported**, and no endpoint accepts one.
- **Prediction is synchronous and modest.** One request, up to 500 records by
  default, answered inline. No batch jobs, no streaming, no async scoring
  service.
- **Runs recorded before persistence existed have no model.** They report
  `model_not_available` rather than an artifact conjured after the fact.
- **No hyperparameter optimisation.** Six scikit-learn estimators at their
  defaults. No Optuna, no grid search, no XGBoost or LightGBM.
- **No database and no vector database.** Records and the index are local
  files. No PostgreSQL, no MLflow, no Qdrant.
- **No horizontal scaling, no cloud deployment, no multi-architecture images.**
  No Kubernetes and no Terraform.
- **No streaming, WebSockets or conversation memory.** Every question is
  independent.
- **Three ingestion formats.** CSV, `.xlsx` and JSON. No Parquet, SQL, Google
  Sheets, S3 or URL ingestion.
- **No agent framework.** LangChain, LangGraph, AutoGen and CrewAI are not
  used.

`docs/PRODUCTION_READINESS.md` says what each of these would take.

## Roadmap

1. ~~Project foundation — repository structure, FastAPI service, tests~~
2. ~~Dataset upload and profiling — validation, profile, quality, target~~
3. ~~Preprocessing and feature engineering — leakage-safe split~~
4. ~~Model training and evaluation — registry, metrics, baselines~~
5. ~~Cross-validation and model selection — one unbiased test evaluation~~
6. ~~Explainability with SHAP — global, local, fallback~~
7. ~~Experiment tracking — fingerprints, versioned records, comparison~~
8. ~~Experiment API — run, list, fetch and compare over HTTP~~
9. ~~Retrieval over documentation and run history — cited evidence~~
10. ~~Provider-agnostic LLM and grounded answers — citation validation~~
11. ~~Knowledge API — `/search` and `/ask`, with grounded statuses~~
12. ~~Bounded agent — tool registry, typed schemas, bounded execution~~
13. ~~Agent HTTP endpoint — budgets a request may lower~~
14. ~~Dataset-aware agent — request-scoped uploads, never persisted~~
15. ~~Multi-format ingestion — CSV, Excel and JSON behind one adapter layer~~
16. ~~Next.js dashboard — the whole product over the existing API~~
17. ~~Containerisation — production images and a one-command stack~~
18. ~~Continuous integration — suites, gates and a real Docker smoke test~~
19. ~~Dependency security — Dependabot, `pip-audit` and `npm audit` as gates~~
20. ~~Production readiness and demo experience — architecture and API docs, the
    demo dataset and script, request correlation and useful logs~~
21. ~~Lightweight API authentication — one shared key on the endpoints that
    cost something, off by default, kept out of the browser and out of every
    image, log and schema~~
22. **Model persistence and prediction** — the winning `Pipeline` written after
    a successful run, and a protected prediction endpoint that reuses the
    preprocessing *as fitted*, never re-fitting and never accepting a path
    *(current)*

**Next**, in the honest order: **TLS**, then background execution. TLS comes
first, and it is a precondition rather than a successor — a bearer token sent
over plain HTTP is captured once and reused forever, so the authentication that
now exists is only as good as the transport under it.
