# Deploying ML Copilot to Render (Free tier)

This is a deployment guide, not a code change — it walks through the Render
dashboard by hand. It assumes the repository state as of this session:
Gemini added as an LLM provider, the experiment-history delete/clear
endpoints, and the `$PORT` fix in `backend/docker-entrypoint.sh` /
`backend/healthcheck.py` described in the accompanying report.

Everything below was checked against Render's own docs on 2026-09-20
(linked inline) rather than assumed. Where Render's docs didn't state a
number outright (exact free-instance RAM/CPU), that is called out rather
than invented.

## Before you start

Pick two service names now — Render URLs are predictable
(`https://<service-name>.onrender.com`), so choosing both names up front
lets you fill in `NEXT_PUBLIC_API_BASE_URL` and `CORS_ALLOW_ORIGINS`
correctly on each service's *first* deploy, with no CORS-fixing redeploy
afterwards. Example: `ml-copilot-api` and `ml-copilot-app`. If a name is
already taken, Render will tell you at creation time — adjust and recheck
the two URLs below before filling in env vars.

- Backend URL: `https://ml-copilot-api.onrender.com` (example)
- Frontend URL: `https://ml-copilot-app.onrender.com` (example)

## 1. Backend — Render Web Service (Docker)

Create a new **Web Service**, connect the `Kapish17/ML-Copilot` repo, and
choose **Docker** as the environment (per
[Render's Docker docs](https://render.com/docs/docker), you point at a
Dockerfile rather than a buildpack).

| Field | Value |
|---|---|
| Name | `ml-copilot-api` (or your choice) |
| Root Directory | *(leave blank — repo root)* |
| Dockerfile Path | `backend/Dockerfile` |
| Docker Build Context Directory | `.` (repo root — the backend Dockerfile does `COPY backend/ ml/ rag/ llm/ agent/` from the root; see the comment at the top of that file) |
| Instance Type | Free |
| Health Check Path | `/health` (already exists, already cheap — see `app/main.py`; it does not touch Gemini, RAG or the dataset pipeline, so it stays fast even mid-request) |

Render uses this health check to gate deploys and to auto-restart a wedged
instance — see
[Render's health-check docs](https://render.com/docs/health-checks).

### Backend environment variables

Render turns dashboard env vars into both build args (for `ARG`s the
Dockerfile declares) and runtime env vars automatically — see
[Environment Variables and Secrets](https://render.com/docs/configure-environment-variables).
The backend Dockerfile declares no build-time `ARG`, so every one of these
is a plain runtime variable.

**Render sets no `PORT` unless you do — its own default is 10000**
([Render env var docs](https://render.com/docs/environment-variables)).
The backend image's own default is 8000 (`ENV API_PORT=8000` in
`backend/Dockerfile`). Set `PORT` explicitly so the two agree:

```
PORT=8000
```

Core:

```
APP_ENV=production
LOG_LEVEL=INFO
API_AUTH_ENABLED=false
CORS_ALLOW_ORIGINS=https://ml-copilot-app.onrender.com
```

`API_AUTH_ENABLED` has to stay `false` for a public demo the frontend can
actually call: `app/main.py` deliberately excludes `Authorization` from the
CORS `allow_headers` list, so a browser on the allowed origin *cannot* send
the key even if you set one — that's a documented anti-pattern guard, not
a bug to work around. **Known consequence**: the live demo is unauthenticated,
so anyone with the URL can run experiments against your Gemini quota. There's
no built-in rate limiting; if that becomes a problem, Render's own request
limits or a WAF in front of the service are the next step, not a code change
here.

Storage paths — these directories exist inside the container's own
ephemeral disk on Free (no persistent disk is available on Free — see
[Render's free-tier docs](https://render.com/docs/free), "Free web services
cannot use persistent disks"). They're set to the same paths
`backend/Dockerfile` already creates and `chown`s, so nothing new is needed:

```
EXPERIMENT_STORE_DIR=/data/experiments
MODEL_ARTIFACT_DIR=/data/models
RAG_INDEX_DIR=/data/rag-index
```

Gemini:

```
LLM_PROVIDER=gemini
LLM_MODEL=gemini-flash-latest
LLM_API_KEY=<paste your real Gemini key directly into Render's dashboard — never into a file in this repo>
LLM_BASE_URL=
```

RAG — explicit even though these already match the code's own defaults, so
a future default change can't silently turn this into a 4 GB PyTorch
install on a 512 MB instance:

```
RAG_EMBEDDING_PROVIDER=hashing
RAG_EMBEDDING_DIMENSION=512
```

**Free-tier-tuned limits.** Every one of these is already fully
configurable (`backend/app/core/config.py`) and already has a working
default sized for a laptop, not a 512 MB container. Nothing in the code
changed for this section — only the values below, chosen for a portfolio
demo on small/medium datasets rather than the million-row ceiling the
defaults allow:

```
MAX_UPLOAD_MB=5
MAX_REQUEST_BODY_MB=5
MAX_DATASET_ROWS=50000
MAX_DATASET_COLUMNS=200
MAX_EXPERIMENT_ROWS=20000
MAX_CV_FOLDS=5
MAX_CANDIDATE_MODELS=4
MAX_ENCODED_FEATURES=1000
MAX_PREDICTION_RECORDS=200
EXPLANATION_ROWS=100
```

A request over any of these limits gets a structured 4xx error
(`413`/`422`, see `_RUN_ERRORS` in `backend/app/api/v1/experiments.py`) —
never a crashed container. This was true before this session and nothing
here changed it; the values above only make the ceiling appropriate for
Render Free instead of a workstation.

Agent (defaults are already conservative; listed for visibility, not
because they need to change):

```
AGENT_MAX_TOOL_CALLS=6
AGENT_MAX_ITERATIONS=8
```

## 2. Frontend — Render Web Service (Docker)

Second Web Service, same repo, Docker environment again.

| Field | Value |
|---|---|
| Name | `ml-copilot-app` (or your choice) |
| Root Directory | *(leave blank — repo root)* |
| Dockerfile Path | `frontend/Dockerfile` |
| Docker Build Context Directory | `frontend` (the frontend Dockerfile does `COPY package.json package-lock.json ./` and `COPY . .` relative to `frontend/`, exactly like `docker-compose.yml`'s `context: ./frontend`) |
| Instance Type | Free |
| Health Check Path | `/dashboard` (what the image's own `HEALTHCHECK` already probes — see `frontend/Dockerfile`) |

### Frontend environment variables

```
PORT=3000
NEXT_PUBLIC_API_BASE_URL=https://ml-copilot-api.onrender.com
```

`NEXT_PUBLIC_API_BASE_URL` **is** a Docker build arg
(`frontend/Dockerfile` declares `ARG NEXT_PUBLIC_API_BASE_URL`) and Render
passes dashboard env vars through as build args automatically, so setting
it here is enough — no separate "build args" UI to find. It has no secret
in it: it's the URL a browser calls, which every visitor sees anyway.

`PORT=3000` matches what `frontend/Dockerfile` already bakes in
(`ENV ... PORT=3000`, read by the Next.js standalone `server.js` and by the
image's own `HEALTHCHECK`) — setting it explicitly removes any ambiguity
against Render's own unset-`PORT` default of 10000.

## 3. Order of operations

1. Deploy the backend first. Wait for it to report healthy
   (`GET https://ml-copilot-api.onrender.com/health` → `{"status":"ok",...}`).
2. Deploy the frontend, pointed at the backend's real URL.
3. Open the frontend URL and confirm the dashboard talks to the backend —
   the header shows a live/unreachable indicator you already built; that's
   the fastest check.

If either service's actual assigned URL differs from what you guessed in
step 0 (name collision), update the *other* service's env var
(`CORS_ALLOW_ORIGINS` or `NEXT_PUBLIC_API_BASE_URL`) and redeploy that one
service — Docker build args mean a `NEXT_PUBLIC_API_BASE_URL` change always
needs a rebuild, same as it does locally with `docker compose up --build`.

## 4. What's ephemeral, and what that means for the demo

Free Render web services have no persistent disk
([Render free-tier docs](https://render.com/docs/free)). Every redeploy,
and every wake from the 15-minutes-idle spin-down, starts the container
filesystem from the image again. Concretely:

- **Experiment history and model artifacts** (`/data/experiments`,
  `/data/models`) are gone after a redeploy or a spin-down/spin-up cycle.
  This is the same "reload empties nothing, only an explicit delete does"
  guarantee as before — it's a *redeploy or cold start* that resets these,
  never a page reload. Framed as a demo: a visitor's session survives fine;
  the history won't survive you pushing a new deploy or the service
  sleeping overnight.
- **The RAG index** (`/data/rag-index`) rebuilds itself automatically and
  non-fatally on every container start (`backend/docker-entrypoint.sh`) —
  this was already true before Render entered the picture, and is exactly
  the behavior that makes an ephemeral disk fine for it.
- **Uploaded datasets** were never persisted anywhere, on Render or
  locally — they're parsed in memory for one request and released. No
  change in behavior at all.

None of this needed a code change: the app was already built to reinitialize
cleanly from empty storage (that's what makes the local `docker compose up`
demo work on a fresh checkout too). Render Free just means that "fresh
start" happens more often than a developer laptop would.

## 5. Cold starts

Free services spin down after 15 minutes with no inbound traffic and take
about a minute to spin back up on the next request
([Render free-tier docs](https://render.com/docs/free)). `/health` never
touches Gemini, RAG or the dataset pipeline (see `app/main.py`), so it
answers as soon as the process is up — it does not wait on anything slow.
The one-time RAG reindex on startup is incremental and skips unchanged
documents after the first run, so a warm image's cold start is dominated by
Render's own container boot, not by this application.

## 6. Live end-to-end checklist

Once both services are up, work through this with a small CSV (a few
hundred rows) — not the free tier's actual ceiling:

1. Open the frontend URL.
2. Upload a dataset.
3. Check dataset profiling renders.
4. Run an experiment.
5. View the experiment result (scores, SHAP).
6. Confirm it appears in Experiment history, and that **Delete** and
   **Clear all** (added this session) actually remove it.
7. Search Knowledge.
8. Ask the Knowledge Assistant a general question.
9. Ask an experiment-specific question (needs the run from step 4).
10. Confirm the answer came from Gemini (no `llm_unavailable`/`agent_provider_error`).
11. Try the Agent tab.
12. Reload the page — history must still show what step 4 created (proving
    reload ≠ delete).
13. Wait for the backend to spin down (15+ idle minutes, or check Render's
    dashboard for "sleeping"), then repeat step 4 — this is the honest cold
    start test.

Record each as pass/fail; anything that fails is worth pasting back with
the browser console and the backend's Render logs for that request's
`X-Request-ID`.

## 7. Security, one more time

- `LLM_API_KEY` is entered directly into Render's dashboard for the
  backend service only — it is never a build arg, never in `.env.example`,
  never in the frontend's env vars, never `NEXT_PUBLIC_*`.
- `CORS_ALLOW_ORIGINS` names the frontend's exact Render URL — no wildcard;
  `app/core/config.py` refuses one at startup if you try.
- Before pushing anything, re-run the checks in the accompanying report's
  security section (`git status`, `git diff`, a secret grep) yourself —
  this guide doesn't commit or push on your behalf.
