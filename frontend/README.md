# ML Copilot — Frontend

**Not implemented yet.** This directory is a placeholder; the frontend will be
built in a later commit.

## Planned

A Next.js (TypeScript) application that talks to the FastAPI backend and
provides:

- Dataset upload and profiling views
- Training run configuration and live status
- Model evaluation results and comparisons
- Explainability visualisations (feature attributions)
- A chat interface over the agent and RAG layers

## Current state

There is no `package.json`, no build step and nothing to run here. Until this
directory is implemented, the backend is used directly:

```bash
cd backend
uvicorn app.main:app --reload
# interactive API docs: http://127.0.0.1:8000/docs
```

The dataset profiling endpoint (`POST /api/v1/datasets/profile`) already returns
a stable JSON contract, including a single error envelope for every failure, so
the upload and profile views can be built against it as it stands. See the
[root README](../README.md) for the request and response shapes.
