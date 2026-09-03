# ML Copilot — Production readiness

**Production-oriented architecture for a local or demo deployment.**

That sentence is the honest description of this project and it is chosen
carefully. The engineering practices below are the ones a production system
needs — bounded inputs, one error contract, non-root containers, dependency
audits as merge gates, leakage-safe evaluation, a real smoke test against
running containers. What is missing is equally real: there is no
authentication, no TLS, no background execution and no horizontal scaling, so
this is **not** something to put on the public internet.

This document is the checklist. Anything unticked is stated as unticked.

---

## Security

| | Control | Where |
| --- | --- | --- |
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
| ❌ | **Authentication and authorisation.** None. Anyone who can reach the port can use every endpoint. | — |
| ❌ | **TLS.** The stack speaks plain HTTP. No reverse proxy is included. | — |
| ❌ | **Rate limiting.** Nothing bounds how often a caller may trigger a training run. | — |
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

- **No authentication.** Every endpoint is open to anyone who can reach the
  port. This is why the Compose stack binds loopback by default.
- **No TLS or reverse proxy.** Plain HTTP.
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

1. **Authentication and TLS** — an API key or OIDC in front, and a reverse
   proxy terminating TLS. Everything else is premature until a stranger cannot
   train a model on your CPU.
2. **Background execution** — move training off the request. This is the change
   that makes every other scaling question answerable.
3. **Model persistence** — store the fitted pipeline, and a prediction endpoint
   becomes possible.
4. **Shared storage** — a database for records and a managed vector store, at
   which point more than one replica becomes meaningful.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the current system is put
together, and [API.md](API.md) for what it exposes.
