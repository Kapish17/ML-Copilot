# ML Copilot — Backend

FastAPI service that will expose the ML Copilot platform over HTTP. In this
commit it provides only the two service-level endpoints needed to verify that
the application runs.

## Structure

```
backend/
├── app/
│   ├── api/          HTTP routers (empty — feature routers arrive later)
│   ├── core/         Configuration and shared infrastructure
│   │   └── config.py Environment-driven Settings object
│   ├── models/       Persistence models (empty — no database yet)
│   ├── schemas/      Pydantic request/response models
│   │   └── system.py Schemas for `/` and `/health`
│   ├── services/     Business logic (empty — no domain logic yet)
│   └── main.py       Application factory and route definitions
├── tests/
│   ├── conftest.py   Shared `TestClient` fixture
│   └── test_main.py  Tests for the available endpoints
├── requirements.txt
└── README.md
```

### Design notes

- `app/main.py` exposes `create_app()` plus a module-level `app` instance.
  The factory keeps the application configurable from tests without relying on
  import-time globals; `app` is what `uvicorn app.main:app` loads.
- `app/core/config.py` reads settings from environment variables using the
  standard library only. A heavier configuration layer will be introduced when
  external services actually need one.
- The empty packages (`api`, `models`, `services`) are real Python packages with
  docstrings describing their intended role. They are placeholders for the next
  commits, not dead code to delete.
- Response shapes are declared as Pydantic schemas rather than raw dicts, so the
  generated OpenAPI documentation stays accurate as the surface grows.

## Endpoints

| Method | Path | Response |
| --- | --- | --- |
| `GET` | `/` | `{"name", "version", "environment", "docs_url"}` |
| `GET` | `/health` | `{"status": "ok", "version", "environment"}` |

OpenAPI docs: <http://127.0.0.1:8000/docs>

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | Environment name reported by the API |
| `LOG_LEVEL` | `INFO` | Log verbosity (reserved for the logging setup) |

See `.env.example` at the repository root.

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
| `pytest` | Test runner |
| `httpx2` | HTTP client required by Starlette's `TestClient` |

Dependencies are added only when the code that needs them lands.
