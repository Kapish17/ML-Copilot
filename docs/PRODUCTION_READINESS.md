# ML Copilot — Production readiness

**Production-oriented architecture for a local or demo deployment.**

That sentence is the honest description of this project and it is chosen
carefully. The engineering practices below are the ones a production system
needs — bounded inputs, one error contract, non-root containers, dependency
audits as merge gates, leakage-safe evaluation, a real smoke test against
running containers, and an API key on everything that costs something. What is
missing is equally real: that key is not identity, there is no TLS, no
background execution and no horizontal scaling, so this is **not** something to
put on the public internet.

This document is the checklist. Anything unticked is stated as unticked.

---

## Security

| | Control | Where |
| --- | --- | --- |
| ⚠️ | **Lightweight API authentication.** One shared key, `Authorization: Bearer <key>`, compared in constant time over SHA-256 digests. Required on the nine endpoints that upload, train, read stored experiments or reach a language model; the five liveness and capability endpoints stay open. **Off by default** so the demo needs no secret, and enabling it without a key is a start-up failure rather than a service that reports itself protected and is not. Marked ⚠️ because it is a key, not identity — see the limitations below. | `backend/app/api/security.py` |
| ✅ | **The key stays server-side.** Never a `NEXT_PUBLIC_` variable, never a Docker build argument, never passed to the frontend service, never in an image layer, never in the OpenAPI schema, never echoed in a response and never written to a log. A rejected request logs `Authentication failed` and nothing else. A test walks the frontend source, the built bundle, both Dockerfiles, the resolved Compose configuration, `.env.example` and every Markdown file looking for it. | `backend/tests/test_authentication.py` |
| ✅ | **No browser-shipped API key.** A browser application cannot hold a shared secret — anything in the bundle is readable by every visitor — so the dashboard holds none and says so: its header reads *"API key required — this dashboard cannot hold one"* when the backend reports `authentication_required`. | `frontend/components/layout/SystemStatus.tsx` |
| ✅ | **Explicit CORS.** An origin list from configuration, never `*`. Credentials off. Empty list installs no cross-origin middleware at all. | `backend/app/main.py` |
| ✅ | **No secret reaches the frontend.** The only build-time variable is `NEXT_PUBLIC_API_BASE_URL`, a public URL. Anything a Next.js build inlines is served to every visitor, so nothing else may be one. | `frontend/Dockerfile` |
| ✅ | **No secret in source, images or logs.** The credential is read from the environment at the moment of use. Only the *variable name* and a boolean are ever exposed or logged. `.env.example` holds no real value. | `llm/config.py` |
| ✅ | **Non-root containers.** Backend runs as uid 10001 owning only its two data directories, so the process cannot modify its own code. Frontend runs as `node`. | both Dockerfiles |
| ✅ | **Uploads are never persisted.** Parsed in memory for one request and released. No upload directory, no temporary file, and deliberately no Docker volume that could become one. | `docker-compose.yml` |
| ✅ | **Prompt-injection defences.** Retrieved passages and tool observations are untrusted data, never instructions. Citations are validated against the passages actually supplied; an answer citing a source it was not given is rejected. | `llm/grounding.py`, `agent/grounding.py` |
| ✅ | **Bounded agent.** An explicit four-tool registry, typed argument schemas validated before anything runs, and hard budgets a request may only lower. No Python, shell, HTTP or filesystem access. No path or file extension ever reaches it. | `agent/registry.py` |
| ✅ | **Safe error mapping.** One envelope everywhere. 5xx gets a generic message with details dropped and the cause logged. Path-shaped values are stripped from details; no raw provider exception reaches a client. | `backend/app/api/error_handlers.py` |
| ✅ | **Path traversal closed.** Experiment identifiers are validated *and* the resolved path is re-checked against the store root. | `ml/experiments/local_store.py` |
| ✅ | **Dependency audits as merge gates.** `pip-audit --strict` over the production and development closures; `npm audit --audit-level=high` after `npm ci`. No `\|\| true`, no `continue-on-error`, no lowered threshold. Dependabot watches every manifest weekly with no ignore rules and no auto-merge. | `.github/` |
| ✅ | **Loopback by default.** Both published ports bind `127.0.0.1`; `BIND_ADDRESS=0.0.0.0` is a deliberate opt-in. A published Docker port bypasses a host firewall, so this default matters. | `docker-compose.yml` |
| ✅ | **CI needs no secret.** `contents: read`, nothing read from `secrets.`, so it works unchanged on a fork. | `.github/workflows/ci.yml` |
| ❌ | **Identity and authorisation.** The API key admits a caller; it does not say *who*. No users, no roles, no sessions, no expiry, and no revocation short of changing the key and restarting. Everyone holding it is the same caller and the log cannot tell them apart. | — |
| ❌ | **TLS.** The stack speaks plain HTTP, and **a bearer token is a password sent on every request** — over plain HTTP anyone on the path reads it once and has it forever. Terminate TLS in front: `Client ──HTTPS──▶ reverse proxy ──HTTP──▶ FastAPI`. No proxy is included here. | — |
| ⚠️ | **Resource protection is per request, not per caller.** Every expensive path is already bounded, and those bounds are why no new limit was added alongside authentication: upload size (`MAX_UPLOAD_MB`), parsed shape (`MAX_DATASET_ROWS`, `MAX_DATASET_COLUMNS`), what an experiment may run on (`MAX_EXPERIMENT_ROWS`, `MAX_CV_FOLDS`, `MAX_CANDIDATE_MODELS`), what SHAP may explain (`EXPLANATION_ROWS`), the agent's ceilings (`AGENT_MAX_TOOL_CALLS`, `AGENT_MAX_ITERATIONS`, `AGENT_MAX_CONTEXT_CHARS` — a request may lower these and never raise them), and retrieval's `RAG_MAX_TOP_K` and `RAG_MAX_QUERY_LENGTH`. **What none of them bounds is a rate**: nothing stops one holder of the key from sending the same bounded request a thousand times. | `backend/app/core/config.py` |
| ❌ | **Rate limiting.** Deliberately absent. Doing it properly across replicas needs shared state, and adding Redis to a single-container local tool would be more attack surface than it removes. | — |
| ❌ | **Secret management.** The credential comes from a `.env` file. No vault, no rotation, no per-tenant keys. | — |
| ❌ | **Static analysis and image scanning.** The audits read dependency manifests, not this project's own code or its built images. | — |

---

## Reliability

| | Control | Where |
| --- | --- | --- |
| ✅ | **Healthchecks in the images.** Both define their own `HEALTHCHECK` that asks the application for a real response, so `docker compose up --wait` blocks until it answers rather than until a process exists. | both Dockerfiles |
| ✅ | **CI on every push and pull request.** Four jobs: backend suites, frontend gates, dependency audit, and a Docker stack smoke test that builds the images, starts them and runs 30 checks against the running containers. | `.github/workflows/ci.yml` |
| ✅ | **Deterministic tests.** Every test builds its data in memory. Nothing reads an external dataset, downloads a model or touches the network. Retrieval uses a deterministic fake embedding provider and the language-model tests a deterministic fake provider — **no API key is needed to run anything.** | all five suites |
| ✅ | **One-command deployment.** `docker compose up --build`, with every value defaulted so the stack runs with no `.env` at all. | `docker-compose.yml` |
| ✅ | **Experiment persistence.** Atomic writes — serialise, write to a temporary file, `fsync`, `os.replace` — so an interrupted write leaves no half-record, not even an empty directory. Records live on a named volume. | `ml/experiments/local_store.py` |
| ✅ | **The index repairs itself.** Rebuilt incrementally at every container start: each document is hashed and the unchanged ones skipped, so it is a real build on a fresh volume and a sub-second no-op afterwards. Indexing failure is not fatal — the API starts and reports retrieval as unavailable. | `backend/docker-entrypoint.sh` |
| ✅ | **Graceful degradation.** With no credential, profiling, experiments, cross-validation, SHAP, history and retrieval all work; only answer generation and the agent report themselves unavailable, and the dashboard says so in its header instead of failing. | `/knowledge/status`, `/agent/status` |
| ✅ | **Request correlation.** Every response carries `X-Request-ID`, and every log line the request produces carries the same id. | `backend/app/api/middleware.py` |
| ⚠️ | **Bounded, but synchronous.** Upload size, row and column ceilings, fold and model counts and agent budgets are all enforced. But a training run occupies its request for its whole duration. | `backend/app/core/config.py` |
| ❌ | **Background execution.** No queue, no worker, no Celery, no Redis. A long run is a long request. | — |
| ❌ | **Horizontal scaling.** Records and the index are local files on volumes attached to one container. Two backend replicas would not share them. | — |
| ❌ | **Backups, retention, migration.** Nothing prunes old runs and nothing migrates a store between schema versions beyond refusing to read a version it does not know. | — |
| ❌ | **Metrics and alerting.** Logs only. No Prometheus, no OpenTelemetry, no dashboards, no alerts. | — |

---

## ML correctness

This is the section that matters most for a project that claims to be an
automated data scientist, because these are the mistakes that produce a number
that looks excellent and means nothing.

| | Control | Why it matters |
| --- | --- | --- |
| ✅ | **Leakage-safe split.** The split happens before anything is fitted; every transformer is fitted on training rows only; the fitted preprocessing travels inside the model artefact so scoring cannot diverge from training. | Fitting an imputer or a scaler on all rows leaks test statistics into training and inflates every score that follows. |
| ✅ | **Cross-validation on training rows only.** Every candidate is scored over folds of the training data, and the winner is chosen from those scores. | Choosing a model by its test score makes the test score a selection statistic, not an estimate of future performance. |
| ✅ | **One held-out evaluation.** The winner is retrained on the full training set and measured **once** on the untouched test set. `uses_test_data: false` on the selection block and `is_unbiased` on the evaluation block record this. | Measuring repeatedly and reporting the best is how an honest pipeline becomes a dishonest number. |
| ✅ | **CV and test kept separate everywhere.** Different fields in the record, different columns in the API, named column groups in the dashboard table. | They answer different questions. Presenting them in one column invites the reader to compare them. |
| ✅ | **Baseline comparison.** Every run is scored against a naive baseline. | "0.86 accuracy" is meaningless until you know the majority class is 0.85. |
| ✅ | **Metric direction carried with the metric.** Nothing assumes higher is better; RMSE and MAE rank the other way and the comparison response reports the direction it applied. | Sorting an error metric descending silently crowns the worst model. |
| ✅ | **SHAP causation disclaimer.** *Feature importance describes model behaviour and association, not causation.* Carried in the record, in the API response and rendered beside the bars. | An importance chart is the easiest artefact in this system to misread as a causal claim. |
| ✅ | **Explainability may fail without failing the run.** A missing explanation is a status on the record with a reason, not a 500, and a permutation fallback covers models SHAP cannot handle. | An unexplainable model is still a valid result. |
| ✅ | **Content fingerprints.** A dataset is identified by a hash of its normalised contents, never by filename, so the same table as CSV, `.xlsx` and JSON has one identity and one history. | Filename-based identity fragments the history of one dataset across three files. |
| ✅ | **Reproducibility recorded.** Random state, Python version, platform and package versions are stored with every run, alongside a configuration hash. | A score without its environment is not a result anyone can check. |
| ✅ | **Class imbalance surfaced.** Detected during profiling and reported as a quality finding; the split is stratified for classification. | Accuracy on an imbalanced target is the classic misleading metric. |
| ❌ | **Hyperparameter optimisation.** Models run at their defaults. No Optuna, no grid search, no nested cross-validation. | The reported scores are for untuned models and should be read that way. |
| ❌ | **Model persistence.** No fitted pipeline or explainer is written to disk, so there is no prediction endpoint and no model serving. | An experiment here is a measurement, not a deployable artefact. |
| ❌ | **Drift, monitoring, fairness analysis.** None. | — |

---

## Known limitations

Stated plainly, in one place:

- **Authentication is one shared key, and it is off by default.** A stock
  `docker compose up` is open to anyone who can reach the port — which is why
  the stack binds loopback. With `API_AUTH_ENABLED=true` the nine expensive
  endpoints require a bearer token, but that key is not identity: no users, no
  roles, no expiry, no revocation short of a restart, and no way to tell two
  holders apart in a log. It also cannot be rotated without downtime.
- **The dashboard cannot use a protected backend on its own.** A browser
  application cannot hold a shared secret, so with authentication on the
  dashboard reports "API key required" and does nothing else. Using it against
  a protected deployment needs something server-side in front that holds the
  credential — a reverse proxy that injects the header, or a
  backend-for-frontend. Neither is included.
- **No rate limiting.** The key stops a stranger from spending your CPU; it
  does nothing about the holder of the key doing so. The hard budgets are what
  bound that, and they bound one request at a time, not a rate.
- **No TLS or reverse proxy.** Plain HTTP — and a bearer credential over plain
  HTTP is readable by anyone on the path, so authentication and TLS are one
  decision, not two.
- **Synchronous training.** A run holds its HTTP request open for its whole
  duration; there is no queue and no worker.
- **No model serving.** Nothing is persisted that could answer a prediction
  request.
- **No horizontal scaling.** One backend container owns the local store and the
  local index.
- **Local storage.** Experiment records are JSON files on a volume. No
  database, no object store, no backups.
- **Local RAG index.** A local vector store on a volume, rebuilt at start. No
  Qdrant, no managed vector service.
- **No cloud deployment.** No Kubernetes, no Terraform, no managed service, no
  IaC of any kind.
- **No multi-architecture images.** The images are built for the machine that
  builds them.
- **Three ingestion formats.** CSV, `.xlsx` and JSON. No Parquet, no SQL, no
  Google Sheets, no S3, no URL ingestion.
- **Untuned models.** Six scikit-learn estimators at their defaults.

---

## If this were to go further

The honest order, smallest useful step first:

1. ~~**Authentication**~~ — done: one shared API key on the nine endpoints that
   cost something, off by default so the demo still needs no secret.
2. **TLS** — a reverse proxy terminating HTTPS in front of the API. This is now
   the largest remaining gap, and it is a *precondition* rather than a
   successor to the previous step: a bearer token sent over plain HTTP is
   captured once and reused forever, so the authentication that exists is only
   as good as the transport under it.
3. **Background execution** — move training off the request. This is the change
   that makes every other scaling question answerable.
4. **Model persistence** — store the fitted pipeline, and a prediction endpoint
   becomes possible.
5. **Shared storage** — a database for records and a managed vector store, at
   which point more than one replica becomes meaningful.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the current system is put
together, and [API.md](API.md) for what it exposes.
