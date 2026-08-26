# ML Copilot — AI Data Scientist

ML Copilot is a production-oriented AI system that will act as an assistant data
scientist: you give it a dataset and a question, it profiles the data, trains and
evaluates candidate models, explains what the models learned, and answers
follow-up questions in natural language — with every run tracked and every
answer grounded in retrievable context.

> **Status: early development.** The backend can currently ingest a CSV file and
> return a full dataset profile. Everything marked *planned* below is not
> implemented yet.

---

## What ML Copilot will do

- Ingest a tabular dataset and produce an automated profile of its structure and quality.
- Run a supervised learning workflow — preprocessing, model selection, training, evaluation.
- Explain results with feature attributions rather than a single opaque score.
- Answer questions about the data, the models, and the runs in natural language.
- Ground those answers in project documentation and prior run history through retrieval.
- Track every experiment so results are reproducible and comparable.

## Main capabilities

| Capability | Description | Status |
| --- | --- | --- |
| Dataset ingestion & profiling | CSV validation, structural profile, data-quality findings, target analysis | **implemented** |
| Automated ML pipeline | Preprocessing, feature handling, model training and evaluation | planned |
| Explainable AI | Global and per-prediction feature attributions (SHAP) | planned |
| Retrieval-augmented answers | Grounded responses over docs and past experiment results | planned |
| Agentic workflows | Multi-step, tool-using analysis planned and executed by an agent | planned |
| Experiment tracking | Reproducible run history, parameters, metrics and artifacts | planned |
| Web interface | Dataset upload, run monitoring, results and chat | planned |

## High-level architecture

```
             ┌───────────────────────┐
             │   Next.js frontend    │   (planned)
             └───────────┬───────────┘
                         │ HTTP
             ┌───────────▼───────────┐
             │    FastAPI backend    │   ← implemented: system + dataset routes
             │  api · services · db  │
             └─┬────────┬──────────┬─┘
               │        │          │
   ┌───────────▼──┐ ┌───▼───────┐ ┌▼──────────────┐
   │  ML layer    │ │ RAG layer │ │  Agent layer  │   (planned)
   │ pipelines,   │ │ ingestion,│ │ tools,        │
   │ evaluation,  │ │ retrieval,│ │ workflows,    │
   │ explainability│ │ prompts  │ │ state         │
   └───────┬──────┘ └─────┬─────┘ └───────┬───────┘
           │              │               │
   ┌───────▼──────┐ ┌─────▼─────┐ ┌───────▼───────┐
   │ Experiment   │ │  Qdrant   │ │  PostgreSQL   │   (planned)
   │  tracking    │ │  vectors  │ │   metadata    │
   └──────────────┘ └───────────┘ └───────────────┘
```

The backend is the single entry point. Domain logic lives in dedicated
top-level packages (`ml/`, `rag/`, `agents/`) so that each concern can be
developed and tested on its own and consumed by the API through a thin service
layer.

## Planned technology stack

| Layer | Technology | Status |
| --- | --- | --- |
| Backend API | Python, FastAPI, Uvicorn | **implemented** |
| Data handling | pandas | **implemented** |
| Machine learning | scikit-learn | planned |
| Explainability | SHAP | planned |
| Experiment tracking | MLflow | planned |
| LLM integration | Provider-agnostic LLM client | planned |
| Retrieval | Qdrant vector database | planned |
| Agents | Orchestrated multi-step workflows | planned |
| Database | PostgreSQL | planned |
| Frontend | Next.js, TypeScript | planned |
| Local orchestration | Docker Compose | skeleton only |

## Repository layout

```
ml-copilot/
├── backend/       FastAPI service (api, core, models, schemas, services, tests)
├── frontend/      Next.js application (placeholder)
├── ml/            Training pipelines, features, evaluation, experiments
├── rag/           Ingestion, retrieval and prompt assets
├── agents/        Agent tools, workflows and state
├── data/          Local datasets — raw and processed (git-ignored contents)
├── configs/       Configuration files
├── docs/          Project documentation
├── scripts/       Developer and operational scripts
├── .env.example   Template for local environment configuration
└── docker-compose.yml   Skeleton for the future local stack
```

Directories outside `backend/` are still placeholders; they hold the structure
the project will grow into.

---

## Dataset profiling

### Supported input

CSV only, comma-delimited, with a header row in the first line. UTF-8 (with or
without a byte order mark) is expected; Latin-1 is accepted as a fallback.
Uploads are processed **in memory and never stored**.

### What the profile contains

**Dataset level** — row and column counts, deep memory usage, duplicate row
count and percentage, total missing cells, and a breakdown of columns by
inferred type.

**Every column** — name, pandas dtype, inferred semantic type (`integer`,
`float`, `boolean`, `datetime`, `categorical`, `text`, `empty`), non-null count,
missing count and percentage, distinct count and percentage, and whether the
column is constant.

**Numeric columns** — mean, median, standard deviation, minimum, maximum, first
and third quartiles, plus zero and negative counts. Statistics that are
undefined (for example the standard deviation of a single row) are returned as
`null`, never as `NaN`.

**Categorical, boolean and text columns** — the most frequent values with counts
and percentages, and a flag showing whether the list was truncated. Numeric
statistics are not computed for these columns.

**Datetime-like columns** — the earliest and latest observed values. Text is
only treated as dates when nearly all sampled values parse; purely numeric text
is never reinterpreted as a timestamp.

### Data-quality findings

Each finding carries a code, a severity (`critical`, `warning`, `info`), a
plain-language message, the columns involved, and the numbers that triggered it.
Heuristic findings are named as such and are worded as observations, not
verdicts.

| Code | Severity | Meaning |
| --- | --- | --- |
| `empty_column` | critical | The column has no values at all |
| `high_missing_values` | warning | Missingness at or above the configured ratio |
| `missing_values` | info | Some values are missing, below that ratio |
| `duplicate_rows` | warning | Rows that exactly repeat an earlier row |
| `constant_column` | warning | A single value in every row |
| `mixed_type_column` | warning | Numbers mixed with text, so the column did not parse as numeric |
| `high_cardinality_column` | info | Many distinct values, and a high distinct share |
| `possible_id_column` | info | Near-unique, plus an id-like name or a consecutive integer run |
| `potentially_suspicious_column` | info | An outcome-like name beside the chosen target — possible leakage |

### Target column (optional)

When a target column is supplied it is validated against the dataset and
analysed: dtype, inferred type, missing count and percentage, and a suggested
task (`classification`, `regression` or `undetermined`) with the reason behind
the suggestion. Classification-like targets also get a value distribution and a
class-balance summary. **No model is trained.** Omitting the target still
returns the full profile.

### API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/v1/datasets/profile` | Profile an uploaded CSV file |

Request: `multipart/form-data` with

- `file` — the CSV file (required)
- `target_column` — name of the target column (optional)

#### Example request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/datasets/profile \
  -F "file=@customers.csv" \
  -F "target_column=churn"
```

For `customers.csv`:

```csv
customer_id,age,plan,churn
1,34,basic,no
2,,pro,yes
3,29,basic,no
```

#### Example response (abridged)

```json
{
  "filename": "customers.csv",
  "generated_at": "2026-08-26T13:47:38.500286Z",
  "dataset": {
    "row_count": 3,
    "column_count": 4,
    "memory_usage_bytes": 542,
    "duplicate_row_count": 0,
    "duplicate_row_percentage": 0.0,
    "missing_cell_count": 1,
    "missing_cell_percentage": 8.3333,
    "column_type_counts": { "integer": 1, "float": 1, "categorical": 2 }
  },
  "columns": [
    {
      "name": "age",
      "dtype": "float64",
      "inferred_type": "float",
      "non_null_count": 2,
      "missing_count": 1,
      "missing_percentage": 33.3333,
      "unique_count": 2,
      "unique_percentage": 66.6667,
      "is_constant": false,
      "numeric_stats": {
        "mean": 31.5, "median": 31.5, "std": 3.5355339059327378,
        "minimum": 29.0, "maximum": 34.0, "q1": 30.25, "q3": 32.75,
        "zero_count": 0, "negative_count": 0
      },
      "datetime_stats": null,
      "categorical_stats": null
    },
    {
      "name": "plan",
      "dtype": "str",
      "inferred_type": "categorical",
      "non_null_count": 3,
      "missing_count": 0,
      "missing_percentage": 0.0,
      "unique_count": 2,
      "unique_percentage": 66.6667,
      "is_constant": false,
      "numeric_stats": null,
      "datetime_stats": null,
      "categorical_stats": {
        "top_values": [
          { "value": "basic", "count": 2, "percentage": 66.6667 },
          { "value": "pro", "count": 1, "percentage": 33.3333 }
        ],
        "truncated": false
      }
    }
  ],
  "quality": {
    "issue_count": 2,
    "issues": [
      {
        "code": "missing_values",
        "severity": "info",
        "message": "Column 'age' has 1 missing value(s) (33.3%).",
        "columns": ["age"],
        "details": { "missing_count": 1, "missing_percentage": 33.3333 }
      },
      {
        "code": "possible_id_column",
        "severity": "info",
        "message": "Column 'customer_id' looks like it may be a row identifier (100.0% unique). Identifiers are usually excluded from training.",
        "columns": ["customer_id"],
        "details": {
          "unique_percentage": 100.0,
          "reasons": ["name_suggests_identifier", "consecutive_integer_sequence"]
        }
      }
    ]
  },
  "target": {
    "name": "churn",
    "dtype": "str",
    "inferred_type": "categorical",
    "missing_count": 0,
    "missing_percentage": 0.0,
    "task_suggestion": "classification",
    "task_reason": "The target is categorical with 2 distinct values.",
    "distribution": [
      { "value": "no", "count": 2, "percentage": 66.6667 },
      { "value": "yes", "count": 1, "percentage": 33.3333 }
    ],
    "class_balance": {
      "class_count": 2,
      "majority_class": "no", "majority_percentage": 66.6667,
      "minority_class": "yes", "minority_percentage": 33.3333,
      "is_imbalanced": false
    },
    "numeric_stats": null
  }
}
```

### Errors

Every failure returns the same envelope, so a frontend needs one handler:

```json
{
  "error": {
    "code": "target_column_not_found",
    "message": "Target column 'revenue' is not in the dataset.",
    "details": { "target_column": "revenue", "available_columns": ["age", "plan"] }
  }
}
```

| Status | Code | Cause |
| --- | --- | --- |
| 400 | `empty_file` | The upload contains no bytes |
| 413 | `file_too_large` | The upload exceeds `MAX_UPLOAD_MB` |
| 413 | `dataset_too_large` | The dataset exceeds the row or column limit |
| 415 | `unsupported_file_type` | The file is not a `.csv` |
| 422 | `malformed_csv` | Rows do not match the header |
| 422 | `missing_header` | No usable header row |
| 422 | `duplicate_columns` | The header repeats a column name |
| 422 | `empty_dataset` | A header with no data rows |
| 422 | `target_column_not_found` | The requested target is not a column |
| 422 | `invalid_request` | A required form field is missing or malformed |
| 500 | `internal_error` | Unexpected failure; details stay server-side |

Stack traces are never returned to API consumers.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | Environment name reported by the API |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `MAX_UPLOAD_MB` | `25` | Largest accepted upload |
| `MAX_DATASET_ROWS` | `1000000` | Largest accepted parsed dataset |
| `MAX_DATASET_COLUMNS` | `1000` | Widest accepted parsed dataset |

Profiling thresholds (top-value count, missingness, cardinality, identifier
uniqueness, class-count limits) live in `backend/app/core/config.py` as named
constants on the `Settings` object.

### Current limitations

- CSV only, comma-delimited. Excel, JSON, Parquet and other delimiters are not supported.
- The whole file is read into memory, so the upload limit is the practical size ceiling.
- Uploaded data is not persisted; each request is independent and nothing is stored between calls.
- A missing header can only be detected when the first row is blank or produces placeholder column names; a header row of plain numbers is indistinguishable from data.
- Malformed-row detection is asymmetric: a row with *more* fields than the header is rejected as `malformed_csv`, while a row with *fewer* fields is padded with missing values, following CSV convention. The padding then shows up in the profile as missing values.
- Datetime detection samples the first 200 non-missing values of a column.
- Data-quality findings are heuristics meant to prompt a human check, not conclusions.
- No authentication or rate limiting — do not expose this service publicly yet.

---

## Current implementation status

**Implemented**

- Repository structure for backend, ML, RAG, agent and frontend layers.
- FastAPI application with system endpoints and a versioned API.
- CSV upload validation, safe in-memory loading and structural profiling.
- Heuristic data-quality detection and optional target-column analysis.
- Consistent error envelope across every failure mode.
- Environment-driven configuration for limits and thresholds.
- Test suite covering the service layer and the API contract.

**Not implemented yet**

- ML training, evaluation and explainability
- RAG ingestion and retrieval
- LLM integration and agentic workflows
- Experiment tracking
- PostgreSQL, Qdrant and any database access
- Authentication
- Frontend application
- Containerisation and deployment

### Available endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service name, version, environment and docs URL |
| `GET` | `/health` | Liveness check — returns `{"status": "ok", ...}` |
| `POST` | `/api/v1/datasets/profile` | Profile an uploaded CSV file |

Interactive API documentation is served at `/docs`.

---

## Local setup

**Requirements:** Python 3.11 or newer.

```bash
git clone <repository-url>
cd ml-copilot
cp .env.example .env        # Windows: copy .env.example .env
```

Create a virtual environment and install the backend dependencies:

```bash
cd backend
python -m venv .venv

# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## Running the backend

From the `backend/` directory, with the virtual environment active:

```bash
uvicorn app.main:app --reload
```

The service starts on <http://127.0.0.1:8000>.

```bash
curl http://127.0.0.1:8000/
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/api/v1/datasets/profile -F "file=@customers.csv"
```

## Running the tests

From the `backend/` directory:

```bash
pytest
```

Add `-v` for per-test output.

## Roadmap

1. ~~**Project foundation** — repository structure, FastAPI service, tests~~
2. **Dataset upload and profiling** — validation, profile, data quality, target analysis *(current)*
3. ML training pipeline and evaluation
4. Explainability with SHAP
5. Experiment tracking
6. RAG layer over documentation and run history
7. LLM integration
8. Agentic workflows
9. Next.js frontend
10. Containerisation and deployment
