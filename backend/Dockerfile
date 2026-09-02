# ML Copilot — backend image.
#
# The build context is the **repository root**, not `backend/`. That is not an
# oversight: the API imports `ml`, `rag`, `llm` and `agent` as sibling
# packages, and the retrieval layer indexes the project's own README files by
# name. A context rooted at `backend/` could see none of them.
#
# Two stages. The builder resolves and installs every Python dependency into a
# virtual environment; the runtime copies that environment and the source, and
# carries no compiler, no pip cache and no build headers. A venv earns its
# place here precisely because it is one self-contained directory that a
# single COPY moves between stages.
#
# Nothing secret is baked in. Every credential and every path is read from the
# environment at start-up, so one image serves every configuration.

# ---------------------------------------------------------------------------
# Stage 1 — dependencies
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Compilers for any dependency without a manylinux wheel on this platform.
# They stay in this stage and never reach the runtime image.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Requirements first, so editing source does not re-resolve the dependency
# tree. All four layers, because the API imports every one of them. Each file
# is named `requirements.txt`, so each needs its own destination — copying
# them together would silently leave only the last.
COPY backend/requirements.txt /tmp/requirements/backend.txt
COPY ml/requirements.txt      /tmp/requirements/ml.txt
COPY rag/requirements.txt     /tmp/requirements/rag.txt
COPY llm/requirements.txt     /tmp/requirements/llm.txt

# `backend/requirements.txt` is runtime-only; the test dependencies live in
# `backend/requirements-dev.txt` and are deliberately not installed here.
RUN pip install --upgrade pip \
    && pip install \
        -r /tmp/requirements/backend.txt \
        -r /tmp/requirements/ml.txt \
        -r /tmp/requirements/rag.txt \
        -r /tmp/requirements/llm.txt

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# PYTHONPATH carries both roots: `/app` for the ml, rag, llm and agent
# packages, `/app/backend` for `app`. This mirrors `pythonpath = . backend`
# in pytest.ini, so the container imports exactly what the test suite does.
#
# The two data paths are absolute and point at directories the unprivileged
# user owns. The defaults in `.env.example` are repository-relative, which is
# right for a checkout and wrong for a container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:/app/backend" \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    EXPERIMENT_STORE_DIR=/data/experiments \
    RAG_INDEX_DIR=/data/rag-index

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# The source, and only the source. `.dockerignore` keeps the frontend, the
# caches, any local `.env` and every build artefact out of the context.
COPY README.md ./
COPY docs/ ./docs/
COPY backend/ ./backend/
COPY ml/ ./ml/
COPY rag/ ./rag/
COPY llm/ ./llm/
COPY agent/ ./agent/

# An unprivileged user owning only what it must write: the two data
# directories. The source tree stays root-owned, so the running process
# cannot modify its own code.
RUN useradd --create-home --uid 10001 mlcopilot \
    && chmod +x /app/backend/docker-entrypoint.sh \
    && mkdir -p /data/experiments /data/rag-index \
    && chown -R mlcopilot:mlcopilot /data

USER mlcopilot

EXPOSE 8000

# Verifies that the application answers, not merely that the process exists.
# Uses the interpreter already present rather than adding curl to the image.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "/app/backend/healthcheck.py"]

ENTRYPOINT ["/app/backend/docker-entrypoint.sh"]
