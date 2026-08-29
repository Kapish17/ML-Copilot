# ML Copilot — Backend

FastAPI service that exposes the ML Copilot platform over HTTP: the
service-level endpoints, the dataset profiling API, and the experiment API that
runs the whole ML pipeline on an uploaded dataset and remembers what happened.

The service is an **adapter around the ML engine**, not a home for it. Every
statistic, split, score and explanation is computed in the top-level `ml/`
package; this service validates requests, orders the steps, and turns results
into JSON.

## Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── dependencies.py    Settings, services and the experiment store
│   │   ├── error_handlers.py  Exceptions → the shared error envelope
│   │   └── v1/
│   │       ├── router.py           Mounts the v1 routers under /api/v1
│   │       ├── datasets.py         POST /api/v1/datasets/profile
│   │       ├── experiments.py      The experiment endpoints
│   │       └── experiment_form.py  The multipart request contract
│   ├── core/
│   │   ├── config.py          Environment-driven Settings object
│   │   ├── errors.py          Typed domain errors with codes and statuses
│   │   └── ml_errors.py       ML-layer exceptions → codes and HTTP statuses
│   ├── models/                Persistence models (empty — no database yet)
│   ├── schemas/
│   │   ├── system.py          Schemas for `/` and `/health`
│   │   ├── errors.py          The error envelope
│   │   ├── dataset.py         Dataset profile response models
│   │   └── experiment.py      Experiment request and response models
│   ├── services/
│   │   ├── datasets/
│   │   │   ├── validation.py  Filename, extension and size checks
│   │   │   ├── loader.py      Decoding and parsing CSV into a DataFrame
│   │   │   ├── profiler.py    Dataset- and column-level statistics
│   │   │   ├── quality.py     Heuristic data-quality detectors
│   │   │   ├── target.py      Optional target-column analysis
│   │   │   ├── conversions.py NaN-safe value conversion helpers
│   │   │   └── service.py     Ingestion and profiling, used by both APIs
│   │   └── experiments/
│   │       ├── options.py     The validated description of one request
│   │       ├── runner.py      ExperimentRunner — the orchestration
│   │       └── history.py     Reading, filtering and comparing runs
│   └── main.py                Application factory and system routes
├── tests/
│   ├── factories.py           In-memory CSV builders used by the tests
│   ├── conftest.py            Client, settings, store and service fixtures
│   ├── test_main.py           System endpoints
│   ├── test_dataset_validation.py
│   ├── test_dataset_loader.py
│   ├── test_dataset_profiler.py
│   ├── test_dataset_quality.py
│   ├── test_dataset_target.py
│   ├── test_dataset_service.py
│   ├── test_api_datasets.py       Profiling contract and error handling
│   ├── test_api_experiments.py    Experiment endpoints, end to end
│   └── test_experiment_service.py Service layer and architecture rules
├── requirements.txt
└── README.md
```

### Design notes

- **Route handlers stay thin.** A handler takes the request, calls a service,
  validates the structured result against a response model and returns it.
  `api/v1/experiments.py` runs a full ML pipeline in four lines, because the
  orchestration is in `services/experiments/runner.py` and the machine
  learning is in `ml/`. A test asserts that no route module imports sklearn,
  SHAP, numpy or pandas.
- **One responsibility per module.** Each step of the dataset pipeline —
  validation, loading, profiling, quality, target — is importable and testable
  on its own, and the service only sequences them.
- **Errors are typed, not ad hoc.** Service code raises subclasses of
  `MLCopilotError`, each carrying a stable `code` and an HTTP status. A single
  set of handlers in `api/error_handlers.py` renders them, so every failure —
  domain, validation or unexpected — leaves the API in the same shape and
  stack traces never reach a client.
- **The ML layer keeps its independence.** `ml/` raises plain Python exceptions
  with no HTTP meaning. `core/ml_errors.py` is the one place that maps them to
  a code and a status, and it strips anything path-like from the details on the
  way out — so a "not found" never reveals where records are kept. A failure
  that maps to 5xx is logged with its real cause and answered generically.
- **Services do not import FastAPI.** `services/experiments/` is drivable from
  a script, a test or a future worker; the HTTP route is one caller among
  several. Tests assert both this and that `ml/` never imports FastAPI.
- **Settings flow through dependencies.** `create_app()` exposes `get_settings`
  as a FastAPI dependency, so tests can run against custom limits by calling
  `create_app(Settings(max_upload_bytes=...))` without touching the
  environment. There is no global mutable state.
- **Uploads are never kept.** They are treated as untrusted input: the filename
  is reduced to a bare name before use, the content is parsed in memory, and no
  request-supplied path reaches the filesystem. What an experiment stores is a
  record — the dataset's content fingerprint, shape and column types — never
  the dataset, the fitted pipeline or the SHAP explainer.
- **One ingestion path.** `DatasetProfilingService.load_upload` validates and
  parses; `profile_frame` profiles a DataFrame. The profiling endpoint and the
  experiment runner both use them, so file validation and CSV parsing exist
  once.
- **JSON stays valid.** pandas produces `NaN` and numpy scalars; every value
  passes through `conversions.py` so responses contain only JSON-legal values.
- The empty package (`models`) is a real package with a docstring describing its
  intended role. It is a placeholder for a later commit, not dead code.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service name, version, environment and docs URL |
| `GET` | `/health` | Liveness check |
| `POST` | `/api/v1/datasets/profile` | Profile an uploaded CSV |
| `POST` | `/api/v1/experiments/run` | Run a complete experiment and store it |
| `GET` | `/api/v1/experiments` | List stored experiments, filtered and sorted |
| `GET` | `/api/v1/experiments/capabilities` | Models, metrics and limits a request may use |
| `GET` | `/api/v1/experiments/{experiment_id}` | Fetch one stored experiment |
| `POST` | `/api/v1/experiments/compare` | Rank several stored experiments |

OpenAPI docs: <http://127.0.0.1:8000/docs> · schema: `/openapi.json`

**CSV is currently supported; the ML pipeline is intentionally
format-agnostic.** Excel, JSON, Parquet, SQL and API ingestion are not
implemented.

### Running an experiment

The dataset is a file and the configuration is form fields, so the request is
`multipart/form-data`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/experiments/run \
  -F "file=@customers.csv" \
  -F "target_column=renewed" \
  -F "models=logistic_regression" \
  -F "models=random_forest_classifier" \
  -F "folds=5" \
  -F "name=renewal baseline" \
  -F "tags=baseline"
```

Every field is optional. With none of them the service profiles the data,
prepares it, cross-validates every model that suits the detected task and keeps
the winner — and warns that it picked the target column by convention.

The flow, and where each step lives:

```
POST /api/v1/experiments/run          api/v1/experiments.py
        ↓
validate + parse in memory            services/datasets/  (CSV → DataFrame)
        ↓
profile                               services/datasets/profiler.py
        ↓
infer configuration + overrides       ml/features/inference.py
        ↓
leakage-safe train/test split         ml/pipelines/preparation.py
        ↓
validate candidates against the task  ml/models/spec.py
        ↓
cross-validate, select the winner     ml/models/selection.py
        ↓
retrain, evaluate once on the test set
        ↓
SHAP global explanation               ml/explainability/
        ↓
build + store the ExperimentRun       ml/experiments/
        ↓
JSON response                         schemas/experiment.py
```

Execution is **synchronous**: the response arrives when the run has finished.
No queue, worker or background execution is implemented, but `ExperimentRunner`
is a plain function of its arguments with no shared state, so moving it onto a
worker later is a change of caller rather than of runner.

There is **no prediction or model-serving endpoint**, deliberately: experiment
records do not contain the fitted model, so nothing exists to serve.

### The error contract

Every failure — dataset, preprocessing, model, explainability, experiment
storage or request validation — answers in one shape:

```json
{"error": {"code": "incompatible_model_task", "message": "...", "details": {}}}
```

| Status | When |
| --- | --- |
| `400` | Invalid configuration: unknown model, metric, strategy, fold count, sort key or experiment id |
| `404` | No experiment stored under that id |
| `409` | The request conflicts with the data: a classifier for a regression target, or a comparison of runs judged by different metrics |
| `413` | The upload or the dataset exceeds a configured limit |
| `415` | Unsupported file type — only `.csv` is implemented |
| `422` | The request or the data cannot be processed: missing target, no usable features, malformed CSV, too few rows |
| `500` | An unexpected internal failure — logged with its cause, answered generically |

Messages are written for a person reading a client. No stack trace, no internal
class name and no filesystem path appears in a response, on any path.

### Experiment history

Runs are stored as local JSON files (see `ml/README.md`). **MLflow is not
implemented, and neither is any database.** The listing endpoint reuses
Commit 7's `ExperimentQuery`:

```bash
curl "http://127.0.0.1:8000/api/v1/experiments?task_type=classification\
&sort_by=primary_metric&order=desc&limit=5"
```

Filters — `dataset_fingerprint`, `target_column`, `task_type`, `model_name`,
`strategy`, `primary_metric`, `tags` — are optional and combine with "and".
Sorting by `primary_metric` reads the metric's own declared direction, so
"best first" is the largest F1 but the smallest RMSE. Comparison refuses runs
that do not share a task and a metric rather than putting an RMSE and an F1 in
the same column.

Because a dataset is identified by a content fingerprint rather than a
filename, `?dataset_fingerprint=…` finds every run on the same data however
the file was named when it was uploaded.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | Environment name reported by the API |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `MAX_UPLOAD_MB` | `25` | Largest accepted upload |
| `MAX_DATASET_ROWS` | `1000000` | Largest accepted parsed dataset |
| `MAX_DATASET_COLUMNS` | `1000` | Widest accepted parsed dataset |
| `EXPERIMENT_STORE_DIR` | `ml/experiments/runs` | Where experiment records are written |
| `MAX_CV_FOLDS` | `10` | Largest accepted fold count |
| `MAX_CANDIDATE_MODELS` | `6` | Most models one experiment may try |
| `MAX_EXPERIMENT_ROWS` | `200000` | Largest dataset an experiment may run on |
| `EXPLANATION_ROWS` | `500` | Rows SHAP values are computed over |

Profiling thresholds, explanation limits, page sizes and the comparison limit
are fields on `Settings` in `app/core/config.py`, each with a named default
constant. Nothing in a route or service hard-codes a limit, and
`GET /api/v1/experiments/capabilities` reports the active ones.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

## Test

```bash
pytest              # from the repository root: backend and ML suites
pytest backend/tests   # this service only
```

Run `uvicorn` from this `backend/` directory — `app` is imported as a
top-level package. Test configuration lives in the repository root
`pytest.ini`, which puts both `backend/` and the repository root on the import
path.

## Relationship to the ML layer

The ML layer lives in the top-level `ml/` package and is a separate component
with its own dependencies. The dependency runs one way only:

```
API routes  →  application services  →  ml/  →  core abstractions
```

This service imports `ml/`. `ml/` does **not** import this service, does not
import FastAPI, and does not know HTTP exists — it consumes a dataset profile
*structurally*, through the protocols in `ml/features/inference.py`, so
profiling logic is never duplicated. Two tests enforce the rule: one parses
every module under `ml/` and fails if it imports `fastapi`, `starlette`, `app`
or `pydantic`, and one imports the whole ML layer in a fresh interpreter.

The same holds one level down: nothing under `app/services/` imports FastAPI,
which is what lets `ExperimentRunner` be driven by an HTTP route today and by a
worker or an agent tool later. `run_experiment(frame, ...)` in
`services/experiments/runner.py` is that programmatic entry point — data in, a
structured result out, no filesystem, pandas or sklearn detail in the answer.
**No agent, LLM or RAG integration is implemented.**

See `ml/README.md`.

## Dependencies

| Package | Why |
| --- | --- |
| `fastapi` | Web framework (brings Starlette and Pydantic) |
| `uvicorn` | ASGI server used to run the app locally |
| `python-multipart` | Required by FastAPI to parse multipart file uploads |
| `pandas` | CSV parsing and the statistics behind the profile |
| `pytest` | Test runner |
| `httpx2` | HTTP client required by Starlette's `TestClient` |

Running experiments additionally needs `ml/requirements.txt`
(scikit-learn and SHAP), since this service now calls the ML layer:

```bash
pip install -r backend/requirements.txt -r ml/requirements.txt
```

Dependencies are added only when the code that needs them lands. **Commit 8
added none.**
