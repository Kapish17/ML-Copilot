# ML Copilot — AI Data Scientist

ML Copilot is a production-oriented AI system that will act as an assistant data
scientist: you give it a dataset and a question, it profiles the data, trains and
evaluates candidate models, explains what the models learned, and answers
follow-up questions in natural language — with every run tracked and every
answer grounded in retrievable context.

> **Status: early foundation.** This repository currently contains the project
> skeleton and a minimal FastAPI service. Everything described as *planned*
> below is not implemented yet.

---

## What ML Copilot will do

- Ingest a tabular dataset and produce an automated profile of its structure and quality.
- Run a supervised learning workflow — preprocessing, model selection, training, evaluation.
- Explain results with feature attributions rather than a single opaque score.
- Answer questions about the data, the models, and the runs in natural language.
- Ground those answers in project documentation and prior run history through retrieval.
- Track every experiment so results are reproducible and comparable.

## Main capabilities (target)

| Capability | Description |
| --- | --- |
| Automated ML pipeline | Preprocessing, feature handling, model training and evaluation |
| Explainable AI | Global and per-prediction feature attributions (SHAP) |
| Retrieval-augmented answers | Grounded responses over docs and past experiment results |
| Agentic workflows | Multi-step, tool-using analysis planned and executed by an agent |
| Experiment tracking | Reproducible run history, parameters, metrics and artifacts |
| Web interface | Dataset upload, run monitoring, results and chat |

## High-level architecture

```
             ┌───────────────────────┐
             │   Next.js frontend    │   (planned)
             └───────────┬───────────┘
                         │ HTTP
             ┌───────────▼───────────┐
             │    FastAPI backend    │   ← this commit (minimal service)
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
| Backend API | Python, FastAPI, Uvicorn | **implemented (minimal)** |
| Machine learning | scikit-learn, pandas | planned |
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

Directories outside `backend/` are placeholders in this commit; they hold the
structure the project will grow into.

## Current implementation status

**Implemented**

- Repository structure for backend, ML, RAG, agent and frontend layers.
- Minimal FastAPI application with two service endpoints.
- Environment-variable driven configuration (`APP_ENV`, `LOG_LEVEL`).
- Test suite covering the available endpoints.

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
```

## Running the tests

From the `backend/` directory:

```bash
pytest
```

Add `-v` for per-test output.

## Roadmap

1. **Project foundation** — repository structure, FastAPI service, tests *(current)*
2. Data ingestion and dataset profiling
3. ML training pipeline and evaluation
4. Explainability with SHAP
5. Experiment tracking
6. RAG layer over documentation and run history
7. LLM integration
8. Agentic workflows
9. Next.js frontend
10. Containerisation and deployment
