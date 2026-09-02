#!/bin/sh
#
# Start the ML Copilot API inside a container.
#
# One thing happens before uvicorn: the retrieval index is brought up to date.
# The index is built from the project's own README files, so without it
# `POST /api/v1/search` answers 503 and the dashboard's Knowledge page is
# empty on a fresh volume — a poor first impression of a feature that works.
#
# `index_documentation()` is incremental: it hashes each document and skips
# the unchanged ones, so this is a real build on the first start and a
# sub-second no-op on every start after that. Running it unconditionally is
# therefore both cheaper and more robust than guarding it with a test for an
# existing index, and it self-heals if a volume is wiped.
#
# It is deliberately not fatal. Indexing needs no network, no API key and no
# model download, but if it ever fails the API should still start and report
# retrieval as unavailable through `GET /api/v1/knowledge/status`, which is
# exactly what the dashboard is built to display.
set -eu

if [ "${RAG_AUTO_INDEX:-1}" = "1" ]; then
    echo "entrypoint: updating the retrieval index at ${RAG_INDEX_DIR:-rag/index}"
    python -c "from rag import RagIndexer, config_from_env; print(RagIndexer(config_from_env()).index_documentation())" \
        || echo "entrypoint: indexing failed; the API will start and report search as unavailable"
fi

echo "entrypoint: starting uvicorn on ${API_HOST:-0.0.0.0}:${API_PORT:-8000}"
exec uvicorn app.main:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8000}" \
    --app-dir /app/backend \
    "$@"
