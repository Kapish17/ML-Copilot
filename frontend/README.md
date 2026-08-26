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

See the [root README](../README.md) for the overall project status.
