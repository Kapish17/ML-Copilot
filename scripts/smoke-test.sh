#!/usr/bin/env bash
#
# Smoke test for a running ML Copilot stack.
#
#   docker compose up -d --wait
#   ./scripts/smoke-test.sh
#
# It asks the running system questions a person would: does the dashboard
# load, does an upload get profiled, does an experiment produce a winner, is
# it in the history afterwards, does search return evidence. Every assertion
# is about a response, not about a file — this runs against whatever is
# listening on the two ports, containerised or not.
#
# Three checks are here specifically because they fail *quietly* in a
# container and nowhere else:
#
#   * the URL baked into the browser bundle. `http://backend:8000` resolves
#     inside the Compose network and nowhere else, so a stack built with it
#     looks healthy from the outside and is broken for every visitor.
#   * CORS. A mismatch between the published dashboard origin and the
#     backend's allowlist produces a page that loads and then fails every
#     request, with nothing in the server log.
#   * the retrieval index. The entrypoint builds it at start-up; if that step
#     failed the API still starts and only search is dead.
#
# No credential is needed and none is used. The deterministic paths — upload,
# profile, experiment, history, search — need no language model at all, and
# the two that do are checked for a well-formed documented response rather
# than for an answer they cannot produce without a key.
#
# Exits non-zero on the first failed check, so CI stops at the real cause.

set -euo pipefail

BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:3000}"
#: The hostname that must never reach a browser: a Compose service name.
INTERNAL_HOST="backend:8000"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-120}"

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

PASSED=0

pass() {
    PASSED=$((PASSED + 1))
    printf '  \033[32mPASS\033[0m  %s\n' "$1"
}

fail() {
    printf '  \033[31mFAIL\033[0m  %s\n' "$1" >&2
    exit 1
}

section() {
    printf '\n\033[1m%s\033[0m\n' "$1"
}

# Assert that a URL answers with an expected status.
expect_status() {
    local url="$1" expected="$2" label="$3" actual
    actual="$(curl -sS -o /dev/null -w '%{http_code}' -L --max-time 30 "$url" || echo 000)"
    [ "$actual" = "$expected" ] || fail "$label — expected HTTP $expected, got $actual"
    pass "$label ($actual)"
}

# Assert a property of a JSON body, described by a Python expression over
# `data`. Python rather than jq: it is on every runner, and an expression
# reads better than a filter for the things worth asserting here.
expect_json() {
    local file="$1" expression="$2" label="$3"
    if ! python3 -c "
import json, sys
data = json.load(open('$file'))
sys.exit(0 if ($expression) else 1)
" 2>/dev/null; then
        printf '  \033[31mFAIL\033[0m  %s\n' "$label — $expression was not true of:" >&2
        head -c 600 "$file" >&2
        printf '\n' >&2
        exit 1
    fi
    pass "$label"
}

# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------
# `docker compose up --wait` already waits on the healthchecks, but this
# script also runs against a stack started by hand, and waiting is cheap.

section "Waiting for the stack"
deadline=$(( $(date +%s) + READY_TIMEOUT_SECONDS ))
while :; do
    backend_up=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 "$BACKEND_URL/health" 2>/dev/null || echo 000)
    frontend_up=$(curl -sS -o /dev/null -w '%{http_code}' --max-time 5 "$FRONTEND_URL/dashboard" 2>/dev/null || echo 000)
    [ "$backend_up" = "200" ] && [ "$frontend_up" = "200" ] && break
    [ "$(date +%s)" -lt "$deadline" ] || fail "the stack did not become ready within ${READY_TIMEOUT_SECONDS}s (backend=$backend_up frontend=$frontend_up)"
    sleep 2
done
pass "both services are answering"

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

section "Backend"
expect_status "$BACKEND_URL/health" 200 "health endpoint"
expect_status "$BACKEND_URL/" 200 "service info"
expect_status "$BACKEND_URL/docs" 200 "API documentation"
expect_status "$BACKEND_URL/openapi.json" 200 "OpenAPI schema"

curl -sS "$BACKEND_URL/health" -o "$WORK_DIR/health.json"
expect_json "$WORK_DIR/health.json" "data['status'] == 'ok'" "health reports ok"

curl -sS "$BACKEND_URL/api/v1/experiments/capabilities" -o "$WORK_DIR/capabilities.json"
expect_json "$WORK_DIR/capabilities.json" \
    "len(data['models']) > 0 and data['supported_dataset_extensions'] == ['.csv', '.xlsx', '.json']" \
    "capabilities lists the models and the three formats"

curl -sS "$BACKEND_URL/api/v1/agent/status" -o "$WORK_DIR/agent-status.json"
expect_json "$WORK_DIR/agent-status.json" \
    "data['dataset_upload_supported'] is True and data['supported_dataset_formats'] == ['csv', 'xlsx', 'json']" \
    "agent status reports the upload formats"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
# The container entrypoint indexes the project's documentation before uvicorn
# starts. If that failed the API is still up and only this is broken, which is
# exactly the failure a healthcheck cannot see.

section "Retrieval"
curl -sS "$BACKEND_URL/api/v1/knowledge/status" -o "$WORK_DIR/knowledge.json"
expect_json "$WORK_DIR/knowledge.json" \
    "data['search_available'] is True and data['index_built'] is True" \
    "the index was built at start-up"

curl -sS -X POST "$BACKEND_URL/api/v1/search" \
    -H 'Content-Type: application/json' \
    -d '{"query": "cross-validation", "top_k": 3}' \
    -o "$WORK_DIR/search.json"
expect_json "$WORK_DIR/search.json" \
    "data['result_count'] > 0 and len(data['citations']) > 0" \
    "search returns passages with citations"

# ---------------------------------------------------------------------------
# The functional path: upload, profile, train, remember
# ---------------------------------------------------------------------------
# A tiny deterministic dataset — a real signal, small enough that a full
# cross-validated run with a SHAP explanation finishes in seconds.

section "Dataset and experiment"
python3 - "$WORK_DIR/smoke.csv" <<'PY'
"""Write a small, separable, entirely deterministic classification dataset."""
import sys

rows = ["income,tenure_months,segment,renewed"]
for index in range(60):
    high = index % 2 == 0
    income = 30_000 + (index % 20) * 500 + (12_000 if high else 0)
    tenure = 4 + (index % 12) + (18 if high else 0)
    segment = "business" if index % 3 == 0 else "retail"
    rows.append(f"{income},{tenure},{segment},{'yes' if high else 'no'}")
open(sys.argv[1], "w", encoding="utf-8").write("\n".join(rows) + "\n")
PY
pass "built a 60-row deterministic dataset"

curl -sS -X POST "$BACKEND_URL/api/v1/datasets/profile" \
    -F "file=@$WORK_DIR/smoke.csv" \
    -F "target_column=renewed" \
    -o "$WORK_DIR/profile.json"
expect_json "$WORK_DIR/profile.json" \
    "data['source_format'] == 'csv' and data['dataset']['row_count'] == 60 and data['target']['task_suggestion'] == 'classification'" \
    "profiling reads the upload and infers the task"

curl -sS -X POST "$BACKEND_URL/api/v1/experiments/run" \
    -F "file=@$WORK_DIR/smoke.csv" \
    -F "target_column=renewed" \
    -F "models=logistic_regression" \
    -F "folds=2" \
    -F "random_state=7" \
    -F "explain=true" \
    -o "$WORK_DIR/run.json"
expect_json "$WORK_DIR/run.json" \
    "data['selection']['selected_model'] == 'logistic_regression' and data['evaluation']['primary_metric_value'] is not None" \
    "an experiment ran and selected a model"
expect_json "$WORK_DIR/run.json" \
    "data['explainability']['status'] == 'available' and len(data['explainability']['feature_importances']) > 0" \
    "SHAP explained the winner"
expect_json "$WORK_DIR/run.json" \
    "data['execution']['stored'] is True and data['dataset']['source_format'] == 'csv'" \
    "the run was stored"

FINGERPRINT="$(python3 -c "import json;print(json.load(open('$WORK_DIR/run.json'))['dataset']['fingerprint'])")"
curl -sS "$BACKEND_URL/api/v1/experiments?dataset_fingerprint=$FINGERPRINT" -o "$WORK_DIR/history.json"
expect_json "$WORK_DIR/history.json" \
    "len(data['experiments']) > 0" \
    "the run is in the history, found by its content fingerprint"

EXPERIMENT_ID="$(python3 -c "import json;print(json.load(open('$WORK_DIR/run.json'))['experiment_id'])")"
expect_status "$BACKEND_URL/api/v1/experiments/$EXPERIMENT_ID" 200 "the stored record can be fetched"

# ---------------------------------------------------------------------------
# The language-model paths, without a language model
# ---------------------------------------------------------------------------
# Neither can produce an answer without a credential, and neither is supposed
# to fail obscurely without one. What is checked is that each returns the
# project's documented envelope — a structured refusal, never a traceback.

section "Agent and answering, with no credential configured"
agent_status=$(curl -sS -o "$WORK_DIR/agent.json" -w '%{http_code}' \
    -X POST "$BACKEND_URL/api/v1/agent/ask-with-dataset" \
    -F "file=@$WORK_DIR/smoke.csv" \
    -F "question=What is the target column?" || echo 000)
case "$agent_status" in
    200) expect_json "$WORK_DIR/agent.json" \
            "data['status'] in ('completed', 'partial', 'insufficient_evidence', 'grounding_failed')" \
            "the agent answered with a documented status" ;;
    502|503) expect_json "$WORK_DIR/agent.json" \
            "set(data) == {'error'} and 'code' in data['error']" \
            "the agent reported a structured failure ($agent_status)" ;;
    *) fail "the agent endpoint returned an unexpected HTTP $agent_status" ;;
esac

ask_status=$(curl -sS -o "$WORK_DIR/ask.json" -w '%{http_code}' \
    -X POST "$BACKEND_URL/api/v1/ask" \
    -H 'Content-Type: application/json' \
    -d '{"question": "What is cross-validation?"}' || echo 000)
case "$ask_status" in
    200) expect_json "$WORK_DIR/ask.json" \
            "data['status'] in ('grounded', 'insufficient_evidence', 'grounding_failed')" \
            "answering returned a documented status" ;;
    502|503) expect_json "$WORK_DIR/ask.json" \
            "set(data) == {'error'} and 'code' in data['error']" \
            "answering reported a structured failure ($ask_status)" ;;
    *) fail "the ask endpoint returned an unexpected HTTP $ask_status" ;;
esac

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

section "Frontend"
expect_status "$FRONTEND_URL/" 200 "root"
expect_status "$FRONTEND_URL/dashboard" 200 "dashboard"
expect_status "$FRONTEND_URL/experiments" 200 "experiments"
expect_status "$FRONTEND_URL/knowledge" 200 "knowledge"

# ---------------------------------------------------------------------------
# The browser's view of the backend
# ---------------------------------------------------------------------------
# The bundle is where the API URL actually lives, so the bundle is what gets
# inspected. Every script the dashboard loads is fetched and searched: the
# published URL must be in there, and the Compose service name must not.

section "Browser configuration"
curl -sS "$FRONTEND_URL/dashboard" -o "$WORK_DIR/dashboard.html"
python3 - "$WORK_DIR/dashboard.html" "$WORK_DIR/scripts.txt" <<'PY'
"""List the script URLs the page loads."""
import re
import sys

html = open(sys.argv[1], encoding="utf-8", errors="ignore").read()
sources = re.findall(r'<script[^>]+src="([^"]+)"', html)
open(sys.argv[2], "w", encoding="utf-8").write("\n".join(sources))
PY

: > "$WORK_DIR/bundle.txt"
cat "$WORK_DIR/dashboard.html" >> "$WORK_DIR/bundle.txt"
while read -r source; do
    [ -n "$source" ] || continue
    case "$source" in
        http*) url="$source" ;;
        *) url="$FRONTEND_URL$source" ;;
    esac
    curl -sS --max-time 30 "$url" >> "$WORK_DIR/bundle.txt" || true
done < "$WORK_DIR/scripts.txt"

grep -q "$INTERNAL_HOST" "$WORK_DIR/bundle.txt" \
    && fail "the browser bundle contains $INTERNAL_HOST, which only resolves inside the Compose network"
pass "the bundle names no internal Compose hostname"

grep -q "$BACKEND_URL" "$WORK_DIR/bundle.txt" \
    || fail "the browser bundle does not name $BACKEND_URL — the dashboard would call the wrong host"
pass "the bundle names the published backend URL ($BACKEND_URL)"

# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

section "CORS"
curl -sS -i -X OPTIONS "$BACKEND_URL/api/v1/datasets/profile" \
    -H "Origin: $FRONTEND_URL" \
    -H "Access-Control-Request-Method: POST" \
    -H "Access-Control-Request-Headers: content-type" \
    -o "$WORK_DIR/preflight.txt"
grep -qi "access-control-allow-origin: $FRONTEND_URL" "$WORK_DIR/preflight.txt" \
    || fail "the backend does not permit the dashboard's origin — every request from the page would be blocked"
pass "the dashboard's origin is permitted"

grep -qi "access-control-allow-origin: \*" "$WORK_DIR/preflight.txt" \
    && fail "the backend answered with a wildcard origin"
pass "the allowance is explicit, not a wildcard"

curl -sS -i "$BACKEND_URL/health" -H "Origin: http://not-the-dashboard.example" \
    -o "$WORK_DIR/foreign.txt"
grep -qi "access-control-allow-origin" "$WORK_DIR/foreign.txt" \
    && fail "an origin outside the allowlist was permitted"
pass "an origin outside the allowlist gets no allowance"

# ---------------------------------------------------------------------------
# Nothing leaked
# ---------------------------------------------------------------------------

section "Secrets"
: > "$WORK_DIR/responses.txt"
for path in / /health /openapi.json /api/v1/agent/status /api/v1/knowledge/status; do
    curl -sS "$BACKEND_URL$path" >> "$WORK_DIR/responses.txt" || true
done
cat "$WORK_DIR/bundle.txt" >> "$WORK_DIR/responses.txt"

# `sk-` at a word boundary: the endpoint path /agent/ask-with-dataset contains
# the letters and must not trip this.
grep -qE '(^|[^A-Za-z0-9])sk-[A-Za-z0-9]' "$WORK_DIR/responses.txt" \
    && fail "something credential-shaped appeared in a response or in the bundle"
pass "no credential-shaped string in any response or in the bundle"

grep -qiE 'LLM_API_KEY|api_key' "$WORK_DIR/bundle.txt" \
    && fail "the browser bundle mentions an API key"
pass "the browser bundle mentions no API key"

section "Result"
printf '  %d checks passed against %s and %s\n\n' "$PASSED" "$BACKEND_URL" "$FRONTEND_URL"
