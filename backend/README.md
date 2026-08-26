# ML Copilot — Backend

FastAPI service that exposes the ML Copilot platform over HTTP. It currently
provides the service-level endpoints and the dataset profiling API.

## Structure

```
backend/
├── app/
│   ├── api/
│   │   ├── dependencies.py    Settings and service providers
│   │   ├── error_handlers.py  Exceptions → the shared error envelope
│   │   └── v1/
│   │       ├── router.py      Mounts the v1 routers under /api/v1
│   │       └── datasets.py    POST /api/v1/datasets/profile
│   ├── core/
│   │   ├── config.py          Environment-driven Settings object
│   │   └── errors.py          Typed domain errors with codes and statuses
│   ├── models/                Persistence models (empty — no database yet)
│   ├── schemas/
│   │   ├── system.py          Schemas for `/` and `/health`
│   │   ├── errors.py          The error envelope
│   │   └── dataset.py         Dataset profile response models
│   ├── services/
│   │   └── datasets/
│   │       ├── validation.py  Filename, extension and size checks
│   │       ├── loader.py      Decoding and parsing CSV into a DataFrame
│   │       ├── profiler.py    Dataset- and column-level statistics
│   │       ├── quality.py     Heuristic data-quality detectors
│   │       ├── target.py      Optional target-column analysis
│   │       ├── conversions.py NaN-safe value conversion helpers
│   │       └── service.py     Orchestration used by the API layer
│   └── main.py                Application factory and system routes
├── tests/
│   ├── factories.py           In-memory CSV builders used by the tests
│   ├── conftest.py            Client, settings and service fixtures
│   ├── test_main.py           System endpoints
│   ├── test_dataset_validation.py
│   ├── test_dataset_loader.py
│   ├── test_dataset_profiler.py
│   ├── test_dataset_quality.py
│   ├── test_dataset_target.py
│   ├── test_dataset_service.py
│   └── test_api_datasets.py   Endpoint contract and error handling
├── requirements.txt
└── README.md
```

### Design notes

- **Route handlers stay thin.** `api/v1/datasets.py` accepts the upload and
  delegates to `DatasetProfilingService`; no validation, parsing or analysis
  happens in the HTTP layer.
- **One responsibility per module.** Each step of the dataset pipeline —
  validation, loading, profiling, quality, target — is importable and testable
  on its own, and the service only sequences them.
- **Errors are typed, not ad hoc.** Service code raises subclasses of
  `MLCopilotError`, each carrying a stable `code` and an HTTP status. A single
  set of handlers in `api/error_handlers.py` renders them, so every failure —
  domain, validation or unexpected — leaves the API in the same shape and
  stack traces never reach a client.
- **Settings flow through dependencies.** `create_app()` exposes `get_settings`
  as a FastAPI dependency, so tests can run against custom limits by calling
  `create_app(Settings(max_upload_bytes=...))` without touching the
  environment. There is no global mutable state.
- **Nothing is executed or stored.** Uploaded files are treated as untrusted
  input: the filename is reduced to a bare name before use, the content is
  parsed in memory, and no request-supplied path reaches the filesystem.
- **JSON stays valid.** pandas produces `NaN` and numpy scalars; every value
  passes through `conversions.py` so responses contain only JSON-legal values.
- The empty package (`models`) is a real package with a docstring describing its
  intended role. It is a placeholder for a later commit, not dead code.

## Endpoints

| Method | Path | Response |
| --- | --- | --- |
| `GET` | `/` | `{"name", "version", "environment", "docs_url"}` |
| `GET` | `/health` | `{"status": "ok", "version", "environment"}` |
| `POST` | `/api/v1/datasets/profile` | Dataset profile — see the root README |

OpenAPI docs: <http://127.0.0.1:8000/docs>

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | Environment name reported by the API |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `MAX_UPLOAD_MB` | `25` | Largest accepted upload |
| `MAX_DATASET_ROWS` | `1000000` | Largest accepted parsed dataset |
| `MAX_DATASET_COLUMNS` | `1000` | Widest accepted parsed dataset |

Profiling and heuristic thresholds are fields on `Settings` in
`app/core/config.py`, each with a named default constant. Nothing in the
service layer hard-codes a limit.

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
pytest
```

Run both commands from this `backend/` directory — `app` is imported as a
top-level package.

## Dependencies

| Package | Why |
| --- | --- |
| `fastapi` | Web framework (brings Starlette and Pydantic) |
| `uvicorn` | ASGI server used to run the app locally |
| `python-multipart` | Required by FastAPI to parse multipart file uploads |
| `pandas` | CSV parsing and the statistics behind the profile |
| `pytest` | Test runner |
| `httpx2` | HTTP client required by Starlette's `TestClient` |

Dependencies are added only when the code that needs them lands.
