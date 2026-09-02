# ML Copilot — Frontend

The dashboard for ML Copilot: upload a dataset, read its profile, ask the AI
Data Scientist about it, run a cross-validated experiment, and read the
explanation — all against the FastAPI backend, over HTTP.

**This is a presentation layer and nothing else.** It computes no statistic,
fits no model, ranks no experiment and retrieves no passage. Every number on
every screen was produced by the backend and is rendered here; where a
component appears to make a judgement — which model won, whether a metric
improved, which run is best — it is displaying a decision the backend already
made and reported.

```
Frontend  ──HTTP──▶  FastAPI  ──▶  Services  ──▶  ML / RAG / LLM / Agent
```

---

## Quick start

### With Docker (the whole stack)

From the repository root:

```bash
cp .env.example .env          # optional: every value has a working default
docker compose up --build     # dashboard on :3000, API on :8000
```

That builds this image and the backend's, and starts both. See
**Run with Docker** in the [root README](../README.md) for logs, rebuilds,
persistence and troubleshooting.

### Without Docker

The backend must be running first; the frontend is useless without it.

```bash
# 1. Backend (from the repository root)
pip install -r backend/requirements-dev.txt -r ml/requirements.txt \
            -r rag/requirements.txt -r llm/requirements.txt
uvicorn app.main:app --app-dir backend --reload      # http://127.0.0.1:8000

# 2. Frontend (from this directory)
npm install
cp .env.example .env.local
npm run dev                                          # http://127.0.0.1:3000
```

Open <http://127.0.0.1:3000>. It redirects to the dashboard.

### Scripts

| Command | What it does |
| --- | --- |
| `npm run dev` | Development server with hot reload |
| `npm run build` | Production build |
| `npm run start` | Serve the production build |
| `npm run lint` | ESLint, over Next's recommended and TypeScript rules |
| `npm run typecheck` | `tsc --noEmit` |
| `npm test` | The component and page test suite (Vitest) |

---

## Configuration

One setting, and it is deliberately public:

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

The browser makes every request, so the backend's origin has to be readable by
the browser. That is also the reason **no secret may ever be configured here**:
anything named `NEXT_PUBLIC_` is inlined into the JavaScript bundle and served
to every visitor.

`LLM_API_KEY` — and every other backend secret — lives on the server, is used
only by the server, and never reaches this application. The frontend has no
credential of any kind, asks for none, and stores none. There is no code path
in this directory that could send one anywhere.

The variable is read in exactly one place, `lib/api/client.ts`, through
`apiBaseUrl()`. No host is hard-coded anywhere else; when the variable is
unset, a documented development default is used and a test pins that behaviour.

---

## Running in a container

`Dockerfile` is a three-stage build — install, build, run. The runtime stage
carries the compiled application and nothing else: no TypeScript, no ESLint,
no Vitest, no build tooling. `output: "standalone"` in `next.config.mjs` is
what makes that possible; it emits a self-contained server plus only the
`node_modules` the application actually reaches, which here is 79 MB against
665 MB for the full dependency tree.

The build runs this project's own gates — `npm run lint`, `npm run typecheck`,
`npm run build` — so a broken image fails at `docker compose build` rather than
at first page load.

### The one thing to understand before changing the Dockerfile

`NEXT_PUBLIC_API_BASE_URL` is inlined into the JavaScript bundle at **build**
time, because the browser is what reads it. Two consequences:

1. **It is a build argument, not a runtime variable.** Changing the backend
   URL means rebuilding: `docker compose up --build`.
2. **It must be a URL the browser can resolve.** `http://backend:8000` is the
   Compose service name; it resolves inside the container network and nowhere
   else, so a browser handed it would fail every request. Compose passes the
   published host URL instead — `http://localhost:8000` by default.

No backend secret is ever passed as a build argument. Anything named
`NEXT_PUBLIC_` is served to every visitor, so only genuinely public values may
be configured that way — and the only one this application has is the API URL.

`.dockerignore` keeps `node_modules`, `.next`, coverage output and any local
`.env` out of the build context; the image regenerates all of them and a host
copy of the first two would be wrong for the image's platform anyway.

The image runs as the base image's unprivileged `node` user and healthchecks
itself by requesting `/dashboard` — a wedged server fails the check, a merely
running process does not pass it.

---

## Architecture

```
frontend/
├── Dockerfile                   Three-stage production image
├── .dockerignore                What never enters the build context
├── app/                         Routes (Next.js App Router)
│   ├── layout.tsx               The shell every page sits in
│   ├── page.tsx                 /            → redirects to the dashboard
│   ├── dashboard/page.tsx       /dashboard   the main workflow
│   ├── experiments/page.tsx     /experiments history and comparison
│   ├── experiments/[id]/page.tsx  one stored run in full
│   └── knowledge/page.tsx       /knowledge   search and grounded answers
├── components/
│   ├── layout/                  App shell, navigation, system status
│   ├── common/                  Card, DataTable, Badge, Tabs, Button, …
│   ├── dataset/                 Upload, profile, quality findings, columns
│   ├── agent/                   Answer card, tool trace, citations
│   ├── experiments/             Model comparison, metrics, history, compare
│   ├── explainability/          Global importance, local explanation
│   └── knowledge/               Search results, grounded answer
├── lib/
│   ├── api/                     The typed client — one module per API area
│   │   ├── client.ts            Base URL, transport, error envelope parsing
│   │   ├── errors.ts            ApiError + code → friendly message
│   │   ├── types.ts             TypeScript mirrors of the backend contracts
│   │   ├── datasets.ts  experiments.ts  agent.ts  knowledge.ts
│   ├── format.ts                Numbers, dates, and metric direction
│   └── citations.ts             Citation ids → labels and in-app routes
└── tests/                       Vitest + Testing Library, API mocked at fetch
```

### The API client

Every request goes through `requestJson` in `lib/api/client.ts`, which resolves
the base URL from configuration, turns any failure into a typed `ApiError`
carrying the backend's own stable `code`, and refuses a body that is not JSON.
Components import from `lib/api/` and never call `fetch` themselves.

`lib/api/types.ts` mirrors the backend's response contracts by hand rather than
generating them, so that a field can say what it *means*: that
`selection_score` is cross-validated and `primary_metric_value` is not, that
`direction` decides whether a larger number is better, that `persisted` is
always false. **No API response is typed `any`.**

### Supported dataset formats

**CSV · Excel (`.xlsx`) · JSON** — the same three the backend reads, through
the same endpoints. There is no per-format route and no branch on format
anywhere in this code: the file is posted, the backend detects the format, and
the response reports it as `source_format` for display. Excel reads the first
worksheet; JSON must be an array of objects, one per row, or an object holding
one such array.

The upload control does check the extension before posting, but only as a
courtesy — the backend is the authority, since it looks at the bytes rather
than the name.

---

## The dashboard workflow

1. **Upload** a CSV, Excel or JSON file. Optionally name the target column.
2. **Profile** it — rows, columns, duplicates, missing cells, data-quality
   findings, per-column statistics, and the target's distribution and inferred
   task.
3. **Ask the AI Data Scientist** — *"Which model performs best and why?"* It
   chooses its own steps: profiling, a cross-validated experiment, a SHAP
   explanation, a documentation search. The answer arrives with its status, its
   citations, the tools that ran, and the dataset's fingerprint.
4. **Run an experiment** directly if you would rather drive it yourself.
5. **Read the result** — the model comparison, the metrics, the explanation.
6. **Open the experiment** from a citation, or from the history page.

Repeat with the same data as `.xlsx` and `.json`: the fingerprint is identical,
so all three runs appear as one dataset in the history. That is the clearest
demonstration that the format stopped mattering at ingestion.

### Two assistants, deliberately distinct

| | AI Data Scientist | Knowledge Assistant |
| --- | --- | --- |
| Where | Dashboard | `/knowledge` |
| Sees your data | Yes, for one request | No, never |
| Runs tools | Profiles, trains, explains, searches | None |
| Answers from | What its own steps returned | Retrieved passages |

---

## What the UI is careful about

**Cross-validation is not the test score.** Candidate models are scored over
folds of the *training* rows; only the winner is retrained and measured, once,
on rows no model has seen. The comparison table puts those in separate,
labelled column groups and says in words whether selection touched the test
set. Presenting them as one column is the easiest way to make a model look
better than it is.

**Metric direction is never assumed.** F1 rises when a model improves; RMSE
falls. Every metric is shown with the direction the backend reported, so a
table mixing R² and RMSE stays readable.

**Status leads, prose follows.** All four agent outcomes — `completed`,
`partial`, `insufficient_evidence`, `grounding_failed` — arrive as HTTP 200,
and only the first is an answer to act on. The verdict is rendered above the
answer text, not as a footnote.

**Citations link only where a page exists.** An experiment citation links to
that run's page in this app. A documentation citation is labelled with its
source file and is *not* linked, because this app does not serve that file and
a link that goes nowhere is indistinguishable from a fabricated citation.
Citations the backend rejected are shown as rejected.

**Attribution is not causation.** Every explanation carries the sentence
*"Feature importance describes model behaviour and association, not
causation."* — as a component, so it cannot be present in one place and
forgotten in another.

**Nothing hidden is reconstructed.** The backend returns no chain-of-thought,
no system prompt and no provider detail; this app renders none, and asks for
none. Tool *argument names* are shown; their values are not.

---

## Security

- **No secrets.** The frontend holds no credential and no API key. The only
  configuration is the public backend URL.
- **Uploads go to the configured backend and nowhere else.** There is no
  third-party upload target in this codebase.
- **Nothing about a dataset is stored in the browser.** No `localStorage`, no
  `sessionStorage`, no cookie, no IndexedDB. The `File` lives in React state
  for the life of the page and is read only by `fetch`. Tests assert that both
  storage areas are empty after a full workflow.
- **No dataset rows in a URL.** Nothing about the data is put in a query
  string, a path or a fragment.
- **No filesystem paths, tracebacks or provider exceptions are rendered.**
  Mapped error codes show their own sentence; an unmapped code falls back to
  the backend's message, which the backend guarantees is free of all three.

---

## Accessibility

- Semantic landmarks (`banner`, `navigation`, `main`, `contentinfo`) and a skip
  link as the first tab stop.
- One `h1` per page, and a heading outline that stays correct when a card is
  nested (`Card` takes a `headingLevel`).
- Every input has a real `<label>`; every button has an accessible name.
- Tables have captions, column headers and row headers — including the
  confusion matrix, which is a table of counts and is marked up as one.
- Tabs implement the tab pattern: roving focus, arrow keys, Home and End.
- **Status is never carried by colour alone.** Every badge pairs its colour
  with a word and a glyph; every bar prints its value as text beside it.
- Loading and error states are announced (`role="status"` with `aria-live`,
  `role="alert"`), so a long synchronous run is legible without sight.

The tests cover these structurally. They are not a substitute for an audit with
a real screen reader, and this project has not had one.

---

## Responsive design

Desktop is the primary target; the layout is usable on tablet and phone. The
dashboard's two columns collapse to one below `lg`, the header wraps, stat
tiles reflow from four columns to two, and **every wide table scrolls inside
its own container** so the page body never scrolls sideways. Verified at
390 px and 820 px with no horizontal overflow.

Animation is limited to the loading spinner.

---

## Testing

```bash
npm test
```

**Vitest + Testing Library, with the API mocked at `fetch`.** Mocking the
transport rather than the client modules keeps the URL building, the
error-envelope parsing and the error mapper under test on the real path a
request takes. No test needs a running backend, an index, or a language-model
credential.

The suite covers the API client and configuration, the error mapper, the upload
control for all three formats, the profile and quality views, classification
and regression metrics, the CV-versus-test distinction, SHAP and local
explanations, the experiment history, detail and comparison, retrieval, grounded
answers and all four agent outcomes, loading and empty states, malformed
responses, browser-storage emptiness, and the accessibility and responsive
contracts above.

Fixtures in `tests/fixtures.ts` were captured from live backend responses
rather than written from the schema, because the two differ exactly where it
matters — which fields are present, which are `null`, what a confusion matrix
or a baseline comparison really looks like.

---

## Screenshots

*Placeholder — no screenshots are checked in.* To capture your own, run the
backend and `npm run dev`, then use the workflow above. The views worth
capturing are the dashboard with a profile and a grounded answer, the
experiment result with its two column groups, and an experiment detail page.

---

## Dependencies

Runtime: **Next.js, React, React DOM.** That is all — no UI kit, no chart
library, no state manager, no data-fetching library. Bars are `div`s with a
width; the confusion matrix is a `<table>`.

Development: TypeScript, Tailwind CSS, ESLint (`eslint-config-next`), Vitest,
Testing Library, jsdom.

**Not used:** LangChain, LangGraph, MLflow, Qdrant, Redis, Celery, or any
second frontend framework. No `sklearn`, `pandas`, `numpy`, `shap` or `openai`
equivalent runs in the browser — that work belongs to the backend and stays
there.

---

## Requirements

- Node.js 20 or newer.
- A running ML Copilot backend, reachable at `NEXT_PUBLIC_API_BASE_URL`, with
  the dashboard's origin in its `CORS_ALLOW_ORIGINS`. The backend defaults
  allow `http://localhost:3000` and `http://127.0.0.1:3000`. Under Docker
  Compose both are wired for you.
- For grounded answers and the agent, the backend needs a language-model
  credential and a built retrieval index. Without either, the affected features
  report themselves unavailable in the header and everything else still works.
