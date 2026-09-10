# Release Checklist

What was actually verified before this release, and what could not be.

The distinction matters more than the list. A checklist that marks everything
green is worth nothing, because the reader cannot tell which items were run and
which were assumed. So every row below is in one of two sections: **Verified**
means a command was executed in this repository and its output was read;
**Environment-dependent / not executed** means the check could not run here and
says why.

The development environment used for this release has **no Docker daemon** and
**no language-model credential**. Everything that depends on either is in the
second section.

---

## Verified

Each of these was run from the repository root, and its result is what is
recorded.

| # | Check | Command | Result |
| --- | --- | --- | --- |
| 1 | Full Python test suite | `pytest` | **2,238 passed, 6 skipped** in 154s |
| 2 | Backend suite | `pytest backend/tests` | 892 tests collected |
| 3 | ML suite | `pytest ml/tests` | 647 tests collected |
| 4 | Agent suite | `pytest agent/tests` | 372 tests collected |
| 5 | RAG suite | `pytest rag/tests` | 189 tests collected |
| 6 | LLM suite | `pytest llm/tests` | 144 tests collected |
| 7 | Every module compiles | `python -m compileall -q backend ml rag llm agent` | clean |
| 8 | Frontend tests | `npx vitest run` | **201 passed**, 9 files |
| 9 | Frontend types | `npx tsc --noEmit` | clean |
| 10 | Frontend lint | `npm run lint` | clean |
| 11 | Frontend production build | `npm run build` | succeeds |
| 12 | Python dependency audit | `pip-audit --strict` over all five requirement files | see below |
| 13 | npm dependency audit | `npm audit --audit-level=high` | see below |
| 14 | Compose file is valid | `docker compose config -q` | exit 0 — Docker's own parser accepts it |
| 15 | Secret-pattern scan | pattern sweep over the tracked tree for `sk-`, `api[_-]?key\s*=`, bearer literals, private-key headers | no secret found; no value printed |
| 16 | `.env.example` holds no secret | `backend/tests/test_docker_config.py` | every credential-shaped variable is empty |
| 17 | Documented settings all exist | `backend/tests/test_documentation.py` | every variable `.env.example` assigns is read by code, Compose, a Dockerfile, a script or CI |
| 18 | Documentation carries no development narration | `backend/tests/test_documentation.py` | no "Commit N" narration in any user-facing document |
| 19 | Wording audit | `backend/tests/test_documentation.py` | no user-facing document overstates the project's operational maturity; the accurate wording is "production-oriented" and "production-readiness hardened" |
| 20 | Privacy | `backend/tests/test_privacy.py` | marker values planted in a dataset appear in no log, record, index entry or error |
| 21 | Leakage | `ml/tests/test_leakage.py` | transformers are fitted on training rows only; the test set is measured once |
| 22 | Prompt injection | `llm/tests/test_prompt_injection.py` | delimiters neutralised, evidence bounded, unretrieved citations rejected |
| 23 | Architecture rules | the import-parsing tests in each suite | one-way dependencies hold; `agent/` imports no web framework, pandas, numpy, scikit-learn, SHAP or openai |
| 24 | Live application pass | real stack on ports 8300/3000, real index, real experiments | profiling, experiment, cross-validation, SHAP, persistence, prediction, retrieval and one agent run all returned 200 with zero browser console errors |
| 25 | Screenshots | four captures in `docs/screenshots/` | reviewed image by image: no API key, no filesystem path, no personal data, only synthetic demo values |

### Dependency audit detail

`pip-audit --strict` is run over `backend/requirements.txt`,
`backend/requirements-dev.txt`, `ml/requirements.txt`, `rag/requirements.txt`
and `llm/requirements.txt` — the production **and** development closures, not
just the runtime one. `--strict` means an unresolvable dependency fails the run
rather than being skipped quietly.

`npm audit --audit-level=high` runs immediately after `npm ci` in CI. Nothing is
suppressed with `|| true`, a lowered threshold or `continue-on-error`, and
`backend/tests/test_ci_workflow.py` asserts that the workflow file has not grown
any of those escape hatches.

### The live application pass

The full stack was started in this environment — uvicorn on the real
application, the Next.js dashboard against it, the retrieval index built from
the project's own documentation — and driven through the whole product path with
a real browser. The four screenshots in `docs/screenshots/` come from that
session.

One qualification, stated plainly because the screenshot shows it: there is no
language-model credential in this environment, so the agent run used the
project's own `FakeLLMProvider` — a real `LLMProvider` implementation, part of
the test suite — scripted to return one valid workflow plan and one grounded
answer. Everything else on that path was real and unmodified: the plan was
validated against the real tool registry, the tools ran against the real
retrieval index and the real experiment store, and the citation was validated by
the real grounding code. What was substituted is the model, not the machinery.

---

## Environment-dependent / not executed

These are real parts of the release process. **None of them was run here**, and
this document does not claim otherwise.

| # | Check | Why it could not run | What would run it |
| --- | --- | --- | --- |
| 1 | Backend image builds | No Docker daemon in this environment | `docker build -f backend/Dockerfile .` |
| 2 | Frontend image builds | No Docker daemon | `docker build -f frontend/Dockerfile frontend` |
| 3 | Stack starts and becomes healthy | No Docker daemon | `docker compose up --build --wait` |
| 4 | Live stack smoke test (28 checks) | Needs a running stack | `./scripts/smoke-test.sh` |
| 5 | Compose validations that shell out to Docker | No Docker daemon | part of `pytest backend/tests/test_docker_config.py` — these are among the 6 skips |
| 6 | Container hardening review under real limits | No Docker daemon | starting the stack with `cap_drop: ALL`, a read-only root filesystem and a memory limit, and checking both services still boot |
| 7 | GitHub Actions run | This repository was not pushed from here, and no Git operation was performed | the CI badge at the top of the README goes green or red on the first push; **it has not been observed either way from here** |
| 8 | Real language-model provider | No credential in this environment | `LLM_API_KEY` set against any OpenAI-compatible endpoint, then `POST /api/v1/ask` and `POST /api/v1/agent/ask` |
| 9 | Optional sentence-transformer embeddings | Model download disabled in tests by design | `RAG_EMBEDDING_PROVIDER=sentence_transformer` with the optional test enabled — one of the 6 skips |
| 10 | Multi-architecture images | Not built, and not a goal | `docker buildx` — out of scope for this project |

---

## Before pushing the release

Not verification — the operator's remaining steps. Listed because a checklist
that stops at "the tests pass" leaves the last mile undocumented.

1. **Remove the two empty scaffolding directories** left from the original
   project skeleton, if they are still tracked: `agents/` (holding only
   `state/.gitkeep`, `tools/.gitkeep`, `workflows/.gitkeep`) and `configs/`
   (holding only `.gitkeep`). Neither is imported by anything; `agents/` is
   actively confusing next to the real `agent/` package.
2. **Confirm the LICENSE copyright holder.** `LICENSE` names *Kapish*, 2026.
3. **Push, then look at the CI badge.** It reflects the first workflow run on
   `main`. Until that run happens the badge shows no result, and nothing in this
   repository can make it green in advance.
4. **Before any deployment that is not localhost:** set `API_AUTH_ENABLED=true`
   with a real `API_AUTH_KEY`, put TLS termination in front of the API, and read
   [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md) — particularly the rows
   marked ⚠️, which are the ones a real deployment has to answer for.
