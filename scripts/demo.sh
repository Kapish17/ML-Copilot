#!/usr/bin/env bash
#
# The five-minute demo, run from a terminal.
#
#   docker compose up -d --wait      # or run the backend however you like
#   ./scripts/demo.sh
#
# It walks the same path the README's demo walks in the dashboard — profile,
# ask, experiment, compare, explain, history, search, predict — and prints the
# answer at each step. The point is to make the product's claims *checkable* in one
# command: the leakage-safe split, the separation of cross-validation from the
# held-out measurement, the baseline comparison, and the fact that the same
# table in three formats has one identity.
#
# ---------------------------------------------------------------------------
# What it needs, and what it deliberately does not
# ---------------------------------------------------------------------------
# Needs: a running backend, `curl`, and `python3` for reading JSON — the same
# two tools `scripts/smoke-test.sh` uses, and both are already present anywhere
# this project runs.
#
# Does **not** need a network connection, a database, or any cloud service.
# Every step but one is entirely deterministic and local. The exception is
# step 3, which asks the AI Data Scientist — the one feature that needs a
# language model; without one configured it prints the documented "not
# configured" response and carries on, because a demo that dies on step three
# demonstrates nothing.
#
# Nor does it need an API key, on the default configuration where the backend
# has authentication switched off. Against a deployment that has it on:
#
#   API_AUTH_KEY=<the key> ./scripts/demo.sh
#
# The three status calls in step 0 deliberately go without the header even
# then, because they are public on every deployment and this demonstrates it.
#
# ---------------------------------------------------------------------------
# On Windows
# ---------------------------------------------------------------------------
# This is a bash script. Windows users have three good options: run it in Git
# Bash or WSL, or follow the equivalent `curl` commands in `docs/API.md`, which
# are the same requests written out one at a time.

set -euo pipefail

BACKEND_URL="${BACKEND_URL:-http://localhost:8000}"
DATASET="${DATASET:-examples/customer_churn.csv}"
TARGET="${TARGET:-renewed}"
QUESTION="${QUESTION:-Which model performs best and why?}"

#: Sent on every request when it is set, and omitted entirely when it is not.
#: A demo against a protected deployment is `API_AUTH_KEY=... ./scripts/demo.sh`
#: — the key comes from the environment, is never written to a file here, and
#: is never printed. `curl` is given the header through an array so it cannot
#: appear in a shell trace or a process list built from a single string.
API_AUTH_KEY="${API_AUTH_KEY:-}"
AUTH_HEADER=()
if [ -n "$API_AUTH_KEY" ]; then
    AUTH_HEADER=(-H "Authorization: Bearer $API_AUTH_KEY")
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

step() {
    printf '\n\033[1m\033[36m%s\033[0m\n' "$1"
}

note() {
    printf '  \033[2m%s\033[0m\n' "$1"
}

die() {
    printf '\n\033[31m%s\033[0m\n' "$1" >&2
    exit 1
}

# Read a JSON file with a Python expression over `data` and print the result.
# Python rather than jq: this project already requires it, and jq is one more
# thing to install before a demo can start.
read_json() {
    local file="$1" expression="$2"
    python3 -c "
import json
data = json.load(open('$file'))
print($expression)
"
}

cd "$(dirname "$0")/.."

[ -f "$DATASET" ] || die "No dataset at $DATASET. Run: python examples/generate_demo_dataset.py"

# ---------------------------------------------------------------------------
step "0. Is the stack up?"
# ---------------------------------------------------------------------------
curl -sS --max-time 10 "$BACKEND_URL/health" -o "$WORK_DIR/health.json" \
    || die "No backend at $BACKEND_URL. Start it with: docker compose up -d --wait"

note "$(read_json "$WORK_DIR/health.json" '"backend %s in %s" % (data["version"], data["environment"])')"

curl -sS --max-time 10 "$BACKEND_URL/api/v1/knowledge/status" -o "$WORK_DIR/knowledge.json"
curl -sS --max-time 10 "$BACKEND_URL/api/v1/agent/status" -o "$WORK_DIR/agent-status.json"

ANSWERING="$(read_json "$WORK_DIR/knowledge.json" 'str(data["answering_available"]).lower()')"
note "retrieval index built: $(read_json "$WORK_DIR/knowledge.json" 'str(data["index_built"]).lower()')"
note "language model configured: $ANSWERING"

if [ "$ANSWERING" != "true" ]; then
    note "-> step 3 will show the documented refusal instead of an answer."
    note "   Set LLM_API_KEY in .env and restart to see the agent work."
fi

# ---------------------------------------------------------------------------
step "1. Profile the dataset"
# ---------------------------------------------------------------------------
curl -sS --max-time 60 "${AUTH_HEADER[@]}" \
    -F "file=@$DATASET" -F "target_column=$TARGET" \
    "$BACKEND_URL/api/v1/datasets/profile" -o "$WORK_DIR/profile.json"

read_json "$WORK_DIR/profile.json" '
"  %d rows x %d columns, read as %s" % (
    data["dataset"]["row_count"], data["dataset"]["column_count"], data["source_format"])'

step "2. What the profiler found wrong with it"
# The demo data carries an identifier column, a column with real missingness
# and a mildly imbalanced target, because a table with no flaws demonstrates
# nothing about a tool whose job is to find them.
read_json "$WORK_DIR/profile.json" '
"\n".join("  - [%s] %s" % (i["severity"], i["message"]) for i in data["quality"]["issues"])
or "  (none)"'
read_json "$WORK_DIR/profile.json" '
"  target %r -> %s (%s)" % (
    data["target"]["name"], data["target"]["task_suggestion"], data["target"]["task_reason"])'

# ---------------------------------------------------------------------------
step "3. Ask the AI Data Scientist"
# ---------------------------------------------------------------------------
note "\"$QUESTION\""
python3 -c "
import json
print(json.dumps({'question': '''$QUESTION'''}))
" > "$WORK_DIR/agent-request.json"

curl -sS --max-time 300 "${AUTH_HEADER[@]}" -H 'Content-Type: application/json' \
    --data-binary "@$WORK_DIR/agent-request.json" \
    "$BACKEND_URL/api/v1/agent/ask" -o "$WORK_DIR/agent.json"

read_json "$WORK_DIR/agent.json" '
"  status: %s" % data.get("status", data.get("error", {}).get("code", "?"))'
read_json "$WORK_DIR/agent.json" '
"  " + (data.get("final_answer") or data.get("error", {}).get("message", ""))'
read_json "$WORK_DIR/agent.json" '
"  tools run: " + (", ".join(c["tool_name"] for c in data.get("tool_calls", [])) or "none")'
note "no chain-of-thought is returned — the trace is which tools ran, not why"

# ---------------------------------------------------------------------------
step "4. Run an experiment"
# ---------------------------------------------------------------------------
note "cross-validating every candidate on the training rows only..."
curl -sS --max-time 900 "${AUTH_HEADER[@]}" \
    -F "file=@$DATASET" -F "target_column=$TARGET" -F "explain=true" \
    -F "name=Demo run" -F "tags=demo" \
    "$BACKEND_URL/api/v1/experiments/run" -o "$WORK_DIR/run.json"

EXPERIMENT_ID="$(read_json "$WORK_DIR/run.json" 'data.get("experiment_id", "")')"
[ -n "$EXPERIMENT_ID" ] || die "The experiment did not run:
$(head -c 500 "$WORK_DIR/run.json")"

read_json "$WORK_DIR/run.json" '
"  stored as %s in %.1fs" % (data["experiment_id"], data["execution"]["duration_seconds"])'
read_json "$WORK_DIR/run.json" '
"  dataset fingerprint: %s (identity comes from the data, never the filename)"
% data["dataset"]["fingerprint"]'

step "5. Compare the candidates — cross-validated, on training rows only"
read_json "$WORK_DIR/run.json" '
"\n".join(
    "  %-32s %8s  +/- %-8s %s" % (
        c["model_name"],
        "%.4f" % c["score"] if c.get("score") is not None else "-",
        "%.4f" % c["score_std"] if c.get("score_std") is not None else "-",
        c["status"],
    )
    for c in data["selection"]["candidates"])'
read_json "$WORK_DIR/run.json" '
"  winner: %s  (chosen by %s, uses_test_data=%s)" % (
    data["selection"]["selected_model"],
    data["selection"]["primary_metric"],
    data["selection"]["uses_test_data"])'

step "6. Held-out test performance — measured once, on rows no model saw"
read_json "$WORK_DIR/run.json" '
"  %s on the test set: %s" % (
    data["evaluation"]["primary_metric"],
    "%.4f" % data["evaluation"]["primary_metric_value"]
    if data["evaluation"]["primary_metric_value"] is not None else "-")'
read_json "$WORK_DIR/run.json" '
(lambda b: "  baseline (%s): %.4f -> %+.4f (%s)" % (
    data["evaluation"].get("baseline_identifier") or "none",
    b["baseline_value"], b["absolute_improvement"],
    "beats it" if b["beats_baseline"] else "does NOT beat it")
 if b else "  baseline: none")(data["evaluation"].get("baseline_comparison"))'
note "this number and the cross-validated one above answer different questions"

step "7. What the model is doing — SHAP"
read_json "$WORK_DIR/run.json" '
"  method: %s (%s)" % (
    data["explainability"]["method"], data["explainability"]["status"])
if data.get("explainability") else "  (no explanation)"'
read_json "$WORK_DIR/run.json" '
"\n".join(
    "  %2d. %-32s %.4f" % (n, f["feature"], f["importance"])
    for n, f in enumerate((data.get("explainability") or {}).get("feature_importances", [])[:5], 1))
or "  (none)"'
note "importance describes model behaviour and association, NOT causation"

# ---------------------------------------------------------------------------
step "8. The run is in the history, found by its content fingerprint"
# ---------------------------------------------------------------------------
FINGERPRINT="$(read_json "$WORK_DIR/run.json" 'data["dataset"]["fingerprint"]')"
curl -sS --max-time 60 "${AUTH_HEADER[@]}" \
    "$BACKEND_URL/api/v1/experiments?dataset_fingerprint=$FINGERPRINT&limit=5" \
    -o "$WORK_DIR/history.json"

read_json "$WORK_DIR/history.json" '
"\n".join(
    "  %s  %-28s cv=%s test=%s" % (
        e["experiment_id"], e["selected_model"],
        "%.4f" % e["selection_score"] if e.get("selection_score") is not None else "-",
        "%.4f" % e["test_score"] if e.get("test_score") is not None else "-")
    for e in data["experiments"])'
note "upload the same table as .xlsx or .json and it lands in this same history"

# ---------------------------------------------------------------------------
step "9. Search the project's own documentation"
# ---------------------------------------------------------------------------
# The default embedding provider is a stateless `HashingVectorizer` — no
# download, no key, identical vectors on every machine. The trade is that it
# matches on terms rather than on meaning, so a question phrased in the
# vocabulary of the documentation retrieves noticeably better than a short
# paraphrase. That is a real property of the default and worth demonstrating
# honestly; `RAG_EMBEDDING_PROVIDER=sentence_transformer` is the alternative.
curl -sS --max-time 60 "${AUTH_HEADER[@]}" -H 'Content-Type: application/json' \
    -d '{"query": "cross-validation versus the final test evaluation", "top_k": 3}' \
    "$BACKEND_URL/api/v1/search" -o "$WORK_DIR/search.json"

read_json "$WORK_DIR/search.json" '
"\n".join(
    "  %d. [%s] %s (score %.3f)" % (
        r["rank"], r["citation_id"], r["source_title"], r["score"])
    for r in data.get("results", [])) or "  (no results)"'
note "every passage carries a citation id; answers may cite only these"

# ---------------------------------------------------------------------------
step "10. Predict with the model that run produced"
# ---------------------------------------------------------------------------
# The point of this step is what it does *not* do. It sends feature values and
# an experiment id — no path, no filename, no model reference. The backend
# loads the pipeline that run fitted and calls it; the preprocessing is the one
# fitted on the training rows above, and nothing here re-fits anything.
curl -sS --max-time 60 "${AUTH_HEADER[@]}" \
    "$BACKEND_URL/api/v1/experiments/$EXPERIMENT_ID/model" \
    -o "$WORK_DIR/model.json"

AVAILABLE="$(read_json "$WORK_DIR/model.json" 'str(data["available"]).lower()')"

if [ "$AVAILABLE" != "true" ]; then
    # Three states, not a boolean: `not_available` is a run from before model
    # persistence existed, `corrupted` is a stored artifact that does not check
    # out, and they have different fixes.
    note "$(read_json "$WORK_DIR/model.json" '"%s (%s): %s" % (data["status"], data.get("reason_code") or "?", data.get("reason") or "?")')"
elif [ "${DATASET##*.}" != "csv" ]; then
    note "skipped: this step reads its example row from a CSV, and DATASET is not one"
else
    read_json "$WORK_DIR/model.json" '
"  %s predicts %r from %d feature(s); scored %.4f %s on %d held-out rows" % (
    data["display_name"], data["target_column"], len(data["features"]),
    data["primary_metric_value"], data["primary_metric"], data["test_row_count"])'

    # One record, built from the first row of the demo file and narrowed to
    # exactly the features the model declares. A feature the model was not
    # trained on is refused rather than ignored, so narrowing is not optional.
    python3 -c "
import csv, json
with open('$DATASET', newline='') as handle:
    row = next(iter(csv.DictReader(handle)))
model = json.load(open('$WORK_DIR/model.json'))
record = {f['name']: (row.get(f['name']) or None) for f in model['features']}
print(json.dumps({'records': [record]}))
" > "$WORK_DIR/predict-request.json"

    curl -sS --max-time 60 "${AUTH_HEADER[@]}" -H 'Content-Type: application/json' \
        --data-binary "@$WORK_DIR/predict-request.json" \
        "$BACKEND_URL/api/v1/experiments/$EXPERIMENT_ID/predict" \
        -o "$WORK_DIR/predict.json"

    read_json "$WORK_DIR/predict.json" '
"  predicted %s = %r" % (
    data["model"]["target_column"], data["predictions"][0]["prediction"])
if "predictions" in data else "  " + data.get("error", {}).get("message", "?")'
    read_json "$WORK_DIR/predict.json" '
"  " + "  ".join(
    "%s=%.3f" % (label, value)
    for label, value in sorted(
        (data["predictions"][0].get("probabilities") or {}).items()))
if "predictions" in data else ""'
    note "the held-out score above measures the model, not this prediction"
fi

printf '\n\033[1m\033[32mDemo complete.\033[0m Experiment %s is at %s/api/v1/experiments/%s\n' \
    "$EXPERIMENT_ID" "$BACKEND_URL" "$EXPERIMENT_ID"
printf 'The same walkthrough in the dashboard: %s\n\n' "${FRONTEND_URL:-http://localhost:3000}/dashboard"
