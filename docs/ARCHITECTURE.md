# ML Copilot — Architecture

This describes the system as it is built, not as it might one day be. Anything
absent from this document is absent from the repository; the
[limitations](#what-this-architecture-does-not-do) at the end say so explicitly.

---

## The shape of it

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

Five Python packages and one TypeScript application. The Python packages depend
on each other in **one direction only**, and that direction is enforced by
tests that parse the import statements rather than by convention:

```
backend/app  ──▶  ml     ──▶  (pandas, scikit-learn, shap)
     │        ──▶  rag    ──▶  (numpy, scikit-learn)
     │        ──▶  llm    ──▶  (openai)
     └────────▶  agent   ──▶  (nothing)
```

`agent/` is the strictest of the five. It imports no web framework, no
`pandas`, no `numpy`, no `scikit-learn`, no `shap`, no `openai` and nothing
from `app`. It talks to every collaborator through a structural
`typing.Protocol`, so the whole bounded loop can be tested with fakes and no
model, no index and no credential.

---

## The architectural boundary

**The frontend is a presentation layer.** ML computation, retrieval, agent
execution and every interaction with a language model happen server-side.

The dashboard fetches JSON and renders it. It does not train, score, embed,
rank, retry a tool, decide which tool to run, or hold a credential. There is no
`sklearn`, `pandas`, `numpy`, `shap` or `openai` equivalent in the browser
bundle, and no API key can reach it: anything a Next.js build inlines is served
to every visitor, so the only build-time variable is `NEXT_PUBLIC_API_BASE_URL`,
which is a public URL by definition.

This boundary is what makes the backend usable without the dashboard. Every
capability in the product is an HTTP endpoint, documented in
[API.md](API.md) and in the generated OpenAPI schema at `/docs`.

---

## Frontend — Next.js dashboard

Next.js 15 (App Router), React 19, TypeScript, Tailwind. Four routes:

| Route | What it does |
| --- | --- |
| `/dashboard` | Upload a dataset, read its profile and quality findings, ask the AI Data Scientist, run an experiment |
| `/experiments` | Browse stored runs; select several and compare them |
| `/experiments/[id]` | One run in full: model comparison, metrics, SHAP, and predicting from its stored model |
| `/knowledge` | Search the project's own documentation and run history |

The API client is hand-written and fully typed — no `any` on a response — and
every backend error code maps to a human sentence, falling back to the
backend's own message when a code is unknown. Runtime dependencies are
Next.js, React and React DOM, and that is all: no UI kit, no chart library, no
state manager, no data-fetching library. Bars are `div`s with a width; the
confusion matrix is a `<table>`.

Details: [frontend/README.md](../frontend/README.md).

---

## API — FastAPI

`create_app()` is a factory. Nothing expensive happens in it: no index is read,
no embedding model is loaded, no language-model client is built. The
application therefore starts with no retrieval index present and no credential
configured, and the endpoints that need either say so when they are called.

Its jobs, and only these:

- **HTTP shape.** Typed request and response models, generated OpenAPI.
- **Translation.** `ml/`, `rag/`, `llm/` and `agent/` know nothing about HTTP;
  three translator modules map their exceptions onto status codes.
- **One error envelope.** Every failure, from every layer, is
  `{"error": {"code", "message", "details"}}`. A 5xx gets a generic message
  and its details dropped, with the real cause logged. Path-shaped values are
  stripped from details before they leave the process.
- **Bounds.** Upload size, row and column ceilings, agent budgets a request may
  lower and never raise.
- **Authentication.** Optional and off by default. When
  `API_AUTH_ENABLED=true`, a dependency on eleven routes checks
  `Authorization: Bearer <key>` against the configured key in constant time.
  The five liveness and capability routes stay open, so a container healthcheck
  never carries a credential. One shared key, not identity — see
  [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md).
- **CORS.** An explicit origin list from configuration, never a wildcard, with
  credentials off. Bearer authentication is explicit and unaffected by that:
  it travels in a header the caller sets, not in a cookie the browser attaches,
  which is why `allow_credentials` stays off.
- **Request correlation.** Every request gets an `X-Request-ID`, returned in
  the response and attached to every log line the request produces.

Details: [backend/README.md](../backend/README.md).

---

## Ingestion — one pipeline, three formats

`backend/app/services/datasets/ingestion/` is **the only place in the
application that knows a file extension exists.** A format is detected, an
adapter reads it, and everything downstream sees a standardised pandas
DataFrame.

```
UploadFile ──▶ detect_format ──▶ DatasetAdapter ──▶ DataFrame ──▶ everything else
                  CSV/XLSX/JSON      one per format
```

- **Detection** prefers the extension and consults the media type only when
  there is no extension. It is explicitly *not* the security boundary — the
  adapter and the validators are, so a `.csv` full of ZIP bytes is refused by
  the reader, not by its name.
- **Excel** reads the first worksheet through `openpyxl`. No formula is
  evaluated and no macro is executed; `.xls`, `.xlsb` and macro-enabled
  workbooks are not accepted formats and their readers are not installed.
- **JSON** accepts an array of objects, or one object whose record list is
  under a recognised key. Nested values are rejected rather than silently
  flattened into something the user did not write.

**Identity is the data, not the file.** A dataset is identified by a content
fingerprint computed from the normalised DataFrame, so the same table uploaded
as CSV, `.xlsx` and JSON produces the same fingerprint and the same experiment
history. The filename is never part of it.

**An uploaded dataset is never stored.** It is parsed in memory for one request
and released. There is no upload directory, no temporary file and — under
Docker — deliberately no volume that could turn a request-scoped loan into
permanent storage.

---

## ML pipeline

`ml/` is format-agnostic: it starts from a DataFrame and knows nothing about
CSV, Excel, JSON or HTTP.

```
profile ─▶ preprocessing config ─▶ leakage-safe split ─▶ cross-validated
selection ─▶ retrain winner ─▶ one test measurement ─▶ SHAP ─▶ stored record
                                                                     │
                                                     persist winner ─┘
                                                             │
                                                             ▼
                                            predict on new records later
```

- **Preprocessing** is a scikit-learn `Pipeline` + `ColumnTransformer`:
  imputation, one-hot encoding with a cardinality ceiling, scaling, datetime
  expansion. The configuration can be inferred from the profile or given
  explicitly.
- **Leakage prevention** is structural. The split happens before anything is
  fitted, every transformer is fitted on training rows only, and the fitted
  preprocessing travels inside the model artefact so scoring cannot diverge
  from training.
- **Selection** cross-validates every candidate on the **training rows only**
  and picks the winner from CV scores. Six models are available.
- **Evaluation** retrains the winner on the full training set and measures it
  **once** on the untouched test set. The two numbers are kept separate
  everywhere — in the record, in the API and in the UI — because a CV score and
  a held-out score answer different questions.
- **Metric direction** is carried with the metric. Nothing assumes higher is
  better; RMSE and MAE rank the other way, and the comparison endpoint reports
  the direction it used.
- **Baselines.** Every run is compared against a naive baseline, so "0.86
  accuracy" can be read against what predicting the majority class would score.

Details: [ml/README.md](../ml/README.md).

---

## Model artifacts and prediction

The winner of a successful run is the whole point of having run it, and it is
already a single object: `TrainedModel.pipeline` is a fitted
`Pipeline(preprocessing, estimator)` that accepts **raw feature rows**. So
persistence is that one object written out, and prediction is that one object
called.

```
run finishes  ─▶  joblib.dump(pipeline)  +  manifest (feature schema, classes,
                                              metrics, environment, sha256)
                          │
                          ▼
       POST /experiments/{id}/predict  ─▶  validate records against the
                                            manifest  ─▶  pipeline.predict
```

**Nothing is re-fitted and nothing is reassembled.** The preprocessing that
transforms a prediction request is the object fitted on that run's training
rows — the same one the held-out score was measured through. A test proves this
by making `fit` and `fit_transform` raise for the duration of a prediction.

`ml/artifacts/` holds three pieces: `schema.py` (the manifest and what a
feature is), `store.py` (a `ModelArtifactStore` `Protocol` and a local
filesystem implementation) and `prediction.py` (validating records against a
manifest, and predicting). The backend owns a store instance and passes it to
the experiment runner and the prediction service; a second implementation —
object storage, a registry — is a new class behind the same `Protocol`.

**Trust boundary.** A `joblib` file is a pickle, and loading one executes code,
so the control is *what may be loaded*. Four barriers, in order:

1. **No path ever comes from a request.** The API accepts feature values and an
   experiment id. There is no field, anywhere, that names a file.
2. **The id is validated as an id** before it is used to address anything.
3. **The resolved directory is re-checked** against the artifact root, so a
   traversal that survived step 2 still cannot escape.
4. **The filename is a constant.** It is never read from the manifest, so a
   tampered manifest cannot redirect the load.

Then the manifest is parsed and its schema version checked **before** anything
is unpickled, the model file's SHA-256 is verified against the manifest, a size
ceiling applies, and the loaded object must be a `Pipeline`. Anyone who can
write into the artifact directory can run code as the service — that directory
is trusted exactly as much as the application's own source, which is why it is
application-owned and why **no endpoint accepts a model file**. Third-party
model files are not supported.

Persistence happens **only after** evaluation succeeds, so an incomplete run
leaves nothing behind; a failed write is a warning on the run, not a failed
experiment. A run recorded before this existed answers `model_not_available`
rather than having an artifact conjured for it. And **no dataset row is written
at any point** — the manifest holds column names, kinds and dtypes.

---

## Explainability

SHAP over the **transformed** features, with the transformed feature names
carried through so `city=Berlin` is a readable name rather than `x37`. Global
ranked importance and signed per-prediction contributions. When SHAP cannot
handle a model, permutation importance is used instead and the record says
which method produced the numbers.

Explainability is allowed to fail without failing the run: a missing
explanation is a status on the record, not a 500.

**Feature importance describes model behaviour and association, not
causation.** That disclaimer travels with the numbers — it is in the record, in
the API response and rendered next to the bars in the dashboard, because an
importance chart without it is the single easiest thing in this system to
misread.

---

## Experiment tracking

Versioned JSON records under an `ExperimentStore` interface, with a local
atomic-write implementation. **MLflow is not implemented.**

One record holds the dataset fingerprint and shape, every preprocessing
decision, the candidate results, the CV score and its spread, the single test
measurement, the baseline comparison, the explanation and the environment
(Python version, platform, package versions, random state).

It does **not** hold dataset rows. It does not hold the fitted model either —
that lives in the artifact store described above, and the record carries only a
`model_artifact` note saying one was written and what features it expects. The
note is history; the store is the authority on whether a prediction can be made
today.

Identifiers are validated and resolved paths are re-checked against the store
root, so an experiment id can never address a file outside it.

---

## RAG — retrieval over documentation and history

Two corpora, one index:

- **Project documentation** — an explicit allowlist of Markdown files plus this
  `docs/` directory. Not a crawl: source code, datasets, model artefacts,
  virtual environments, `.git` and the experiment store are never candidates,
  and any filename that looks like a credential is refused even if it is
  configured.
- **Experiment records** — each stored run rendered as structured Markdown, so
  a question about feature importance retrieves the importance section rather
  than the whole run. Only stored facts are rendered; the renderer never
  invents prose.

Chunking follows document structure, not a character count. The default
embedding provider is a stateless `HashingVectorizer`: no vocabulary to fit, no
model to download, no API key, and identical vectors on every machine.
Retrieval is cosine similarity with pre-ranking metadata filters, and every
result carries a stable citation id.

The dependency runs one way — `ml/experiments → rag/ingestion → rag/retrieval`
— so experiments can be recorded with no index present, and the index can be
rebuilt from the store at any time. It is rebuilt incrementally at every
container start, so a wiped volume repairs itself.

Details: [rag/README.md](../rag/README.md).

---

## LLM layer

A provider abstraction over the OpenAI-compatible chat-completions shape. One
implementation therefore reaches OpenAI, Azure OpenAI, vLLM, Ollama, LM Studio
and OpenRouter by pointing `LLM_BASE_URL` at them — a model on a laptop works
the same as a hosted one.

Nothing is imported at package-import time and no client is built until a
generation is actually requested, so the layer imports and the whole test suite
runs with no key configured.

**The model is not the source of truth.** Every project-specific claim must
come from a retrieved passage. Citations are validated against the passages
actually supplied, and an answer citing a source it was not given is
**rejected**, not quietly cleaned up. Retrieved text is treated as untrusted
data, never as instructions.

An answer that cannot be grounded is a **result, not an error**: it returns 200
with a status saying so.

Details: [llm/README.md](../llm/README.md).

---

## The bounded agent

The agent decides which of the system's own capabilities a question needs. It
is bounded by construction rather than by prompt:

- **An explicit registry.** Four tools — `dataset_profile`, `run_experiment`,
  `explain_experiment`, `search_knowledge`. A tool not in the registry cannot
  be called, and a planner that names one gets a rejected observation it can
  correct from.
- **Typed argument schemas**, validated before anything runs. An unexpected
  argument is rejected, never ignored.
- **Hard budgets** on tool calls, iterations and context size. A request may
  *lower* them and can never raise them.
- **No arbitrary execution.** No Python, no shell, no HTTP, no filesystem
  access. The agent receives no `UploadFile`, no path and no file extension —
  a dataset reaches it as an already-parsed in-memory table under a name.
- **No chain-of-thought** in any response. The trace shows which tools ran and
  what they returned, which is the auditable part.

Four outcomes are all HTTP 200 with a status: `completed`, `partial`,
`insufficient_evidence`, `grounding_failed`. Only a provider failure is 502 or
503.

Details: [agent/README.md](../agent/README.md).

---

## Storage

| What | Where | Survives `down` | Survives `down -v` |
| --- | --- | --- | --- |
| Experiment records | `experiment-store` volume, JSON files | yes | no |
| Trained model artifacts | `model-artifacts` volume, one directory per experiment | yes | no |
| Retrieval index | `rag-index` volume, rebuilt at start | yes | no |
| Uploaded datasets | **nowhere — never written to disk** | — | — |

An artifact directory holds two files: the fitted pipeline and its manifest.
Learned parameters and column *names* — no dataset row, in either.

No database. No object store. No model registry.

---

## Deployment

Two images, one command.

```
docker compose up --build
```

- **Backend** — two-stage build; the builder resolves dependencies into a
  virtual environment and the runtime copies it, so no compiler and no pip
  cache ship. Runs as an unprivileged user that owns only the two data
  directories, so the process cannot modify its own code.
- **Frontend** — three stages; the runtime carries the Next.js standalone
  output and nothing else. No TypeScript, no ESLint, no Vitest.
- Both images define their own `HEALTHCHECK`, so `docker compose up --wait`
  blocks until the application actually answers rather than until a process
  exists.
- Both ports are published on `127.0.0.1` by default. This API has no
  authentication; `BIND_ADDRESS=0.0.0.0` is the deliberate opt-in.

`NEXT_PUBLIC_API_BASE_URL` is inlined at **build** time, because the browser is
what reads it. It must therefore be a URL the browser can resolve —
`http://localhost:8000`, never the Compose service name.

---

## CI and dependency security

Four GitHub Actions jobs on every push and pull request to `main`, with a
`contents: read` token and **no secret of any kind**, so CI works unchanged on
a fork:

| Job | What it proves |
| --- | --- |
| Backend tests | Five pytest suites, plus a compile pass over every module |
| Frontend tests | `npm ci`, audit, lint, typecheck, Vitest, production build |
| Dependency audit | `pip-audit --strict` over the production and development closures |
| Docker stack smoke test | Builds both images, starts the stack, runs 30 checks against it |

Dependabot watches every Python, npm and GitHub Actions manifest weekly, with
no ignore rules and no automatic merging.

---

## What this architecture does not do

Absent by decision, not by oversight:

- No identity, authorisation, rate limiting or multi-user separation. There is
  one optional shared API key (below) and it says *that* a caller is allowed,
  never *who* they are.
- No TLS and no reverse proxy — which the API key needs, since a bearer
  credential over plain HTTP is captured once and reused forever.
- No background execution — training is synchronous inside the request.
- No model registry, versioning, promotion or rollback. A run's winning
  pipeline is persisted to a local volume and can be predicted from; that is
  serving on one node, not a managed release.
- No database (PostgreSQL or otherwise) and no vector database (Qdrant or
  otherwise). Records and the index are local files.
- No hyperparameter optimisation, no gradient-boosting libraries, no MLflow.
- No agent framework — LangChain, LangGraph, AutoGen and CrewAI are not used.
- No streaming, no WebSockets, no conversation memory.
- No horizontal scaling, no cloud deployment, no multi-architecture images.
- No ingestion beyond CSV, `.xlsx` and JSON.

See [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) for what that means for
running this outside a local or demo deployment.
