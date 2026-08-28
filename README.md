# ML Copilot — AI Data Scientist

ML Copilot is a production-oriented AI system that will act as an assistant data
scientist: you give it a dataset and a question, it profiles the data, trains and
evaluates candidate models, explains what the models learned, and answers
follow-up questions in natural language — with every run tracked and every
answer grounded in retrievable context.

> **Status: early development.** The backend can ingest a CSV file and return a
> full dataset profile; the ML layer turns a profiled dataset into model-ready
> data, cross-validates a suite of models on the training rows, picks a winner
> without ever reading the test set, retrains it on the full training data and
> measures it once on the untouched test set. **There is no hyperparameter
> optimisation, no explainability, no experiment tracking, no LLM, no RAG and
> no agent.** Everything marked *planned* below is not implemented.

---

## What ML Copilot will do

- Ingest a tabular dataset and produce an automated profile of its structure and quality.
- Run a supervised learning workflow — preprocessing, model selection, training, evaluation.
- Explain results with feature attributions rather than a single opaque score.
- Answer questions about the data, the models, and the runs in natural language.
- Ground those answers in project documentation and prior run history through retrieval.
- Track every experiment so results are reproducible and comparable.

## Main capabilities

| Capability | Description | Status |
| --- | --- | --- |
| Dataset ingestion & profiling | CSV validation, structural profile, data-quality findings, target analysis | **implemented** |
| Preprocessing & feature engineering | Feature selection, imputation, encoding, scaling, datetime expansion, leakage-safe train/test split | **implemented** |
| Model training & evaluation | Six-model suite, task-appropriate metrics, baseline comparison | **implemented** |
| Cross-validated model selection | K-fold selection on training data, one unbiased test measurement | **implemented** |
| Hyperparameter optimisation | Automated search over model settings | planned |
| Explainable AI | Global and per-prediction feature attributions (SHAP) | planned |
| Retrieval-augmented answers | Grounded responses over docs and past experiment results | planned |
| Agentic workflows | Multi-step, tool-using analysis planned and executed by an agent | planned |
| Experiment tracking | Reproducible run history, parameters, metrics and artifacts | planned |
| Web interface | Dataset upload, run monitoring, results and chat | planned |

## High-level architecture

```
             ┌───────────────────────┐
             │   Next.js frontend    │   (planned)
             └───────────┬───────────┘
                         │ HTTP
             ┌───────────▼───────────┐
             │    FastAPI backend    │   ← implemented: system + dataset routes
             │  api · services · db  │
             └─┬────────┬──────────┬─┘
               │        │          │
   ┌───────────▼──┐ ┌───▼───────┐ ┌▼──────────────┐
   │  ML layer    │ │ RAG layer │ │  Agent layer  │
   │ preprocessing│ │ ingestion,│ │ tools,        │
   │ ✓ training ✓ │ │ retrieval,│ │ workflows,    │
   │ explainability│ │ prompts  │ │ state         │
   │ (planned)    │ │ (planned) │ │  (planned)    │
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
| Backend API | Python, FastAPI, Uvicorn | **implemented** |
| Data handling | pandas | **implemented** |
| Preprocessing | scikit-learn (`Pipeline`, `ColumnTransformer`) | **implemented** |
| Model training | scikit-learn estimators | **implemented** |
| Gradient boosting libraries | XGBoost, LightGBM | planned |
| Hyperparameter search | Optuna | planned |
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
├── ml/            Preprocessing and feature engineering; training comes later
├── rag/           Ingestion, retrieval and prompt assets
├── agents/        Agent tools, workflows and state
├── data/          Local datasets — raw and processed (git-ignored contents)
├── configs/       Configuration files
├── docs/          Project documentation
├── scripts/       Developer and operational scripts
├── .env.example   Template for local environment configuration
└── docker-compose.yml   Skeleton for the future local stack
```

`backend/` and `ml/` hold implemented code. The remaining directories are
placeholders holding the structure the project will grow into.

---

## Data ingestion formats

**CSV is the only implemented input format.** Excel, JSON, Parquet, SQL
databases and HTTP/API sources are planned.

The ML pipeline is intentionally **format-agnostic**: everything downstream of
ingestion operates on a standardised `pandas.DataFrame`, never on a file, a
path or a CSV-specific object.

```
ingestion adapter  ->  DataFrame  ->  profiling  ->  configuration  ->  preprocessing  ->  training
   (CSV today;                                                                            (planned)
    Excel, JSON,
    Parquet, SQL,
    API planned)
```

Adding a format therefore means writing one adapter that returns a DataFrame.
Profiling, preprocessing and the future training code need no changes and have
no knowledge of where the data came from.

---

## Dataset profiling

### Supported input

CSV only, comma-delimited, with a header row in the first line. UTF-8 (with or
without a byte order mark) is expected; Latin-1 is accepted as a fallback.
Uploads are processed **in memory and never stored**.

### What the profile contains

**Dataset level** — row and column counts, deep memory usage, duplicate row
count and percentage, total missing cells, and a breakdown of columns by
inferred type.

**Every column** — name, pandas dtype, inferred semantic type (`integer`,
`float`, `boolean`, `datetime`, `categorical`, `text`, `empty`), non-null count,
missing count and percentage, distinct count and percentage, and whether the
column is constant.

**Numeric columns** — mean, median, standard deviation, minimum, maximum, first
and third quartiles, plus zero and negative counts. Statistics that are
undefined (for example the standard deviation of a single row) are returned as
`null`, never as `NaN`.

**Categorical, boolean and text columns** — the most frequent values with counts
and percentages, and a flag showing whether the list was truncated. Numeric
statistics are not computed for these columns.

**Datetime-like columns** — the earliest and latest observed values. Text is
only treated as dates when nearly all sampled values parse; purely numeric text
is never reinterpreted as a timestamp.

### Data-quality findings

Each finding carries a code, a severity (`critical`, `warning`, `info`), a
plain-language message, the columns involved, and the numbers that triggered it.
Heuristic findings are named as such and are worded as observations, not
verdicts.

| Code | Severity | Meaning |
| --- | --- | --- |
| `empty_column` | critical | The column has no values at all |
| `high_missing_values` | warning | Missingness at or above the configured ratio |
| `missing_values` | info | Some values are missing, below that ratio |
| `duplicate_rows` | warning | Rows that exactly repeat an earlier row |
| `constant_column` | warning | A single value in every row |
| `mixed_type_column` | warning | Numbers mixed with text, so the column did not parse as numeric |
| `high_cardinality_column` | info | Many distinct values, and a high distinct share |
| `possible_id_column` | info | Near-unique, plus an id-like name or a consecutive integer run |
| `potentially_suspicious_column` | info | An outcome-like name beside the chosen target — possible leakage |

### Target column (optional)

When a target column is supplied it is validated against the dataset and
analysed: dtype, inferred type, missing count and percentage, and a suggested
task (`classification`, `regression` or `undetermined`) with the reason behind
the suggestion. Classification-like targets also get a value distribution and a
class-balance summary. **No model is trained.** Omitting the target still
returns the full profile.

### API

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/v1/datasets/profile` | Profile an uploaded CSV file |

Request: `multipart/form-data` with

- `file` — the CSV file (required)
- `target_column` — name of the target column (optional)

#### Example request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/datasets/profile \
  -F "file=@customers.csv" \
  -F "target_column=churn"
```

For `customers.csv`:

```csv
customer_id,age,plan,churn
1,34,basic,no
2,,pro,yes
3,29,basic,no
```

#### Example response (abridged)

```json
{
  "filename": "customers.csv",
  "generated_at": "2026-08-26T13:47:38.500286Z",
  "dataset": {
    "row_count": 3,
    "column_count": 4,
    "memory_usage_bytes": 542,
    "duplicate_row_count": 0,
    "duplicate_row_percentage": 0.0,
    "missing_cell_count": 1,
    "missing_cell_percentage": 8.3333,
    "column_type_counts": { "integer": 1, "float": 1, "categorical": 2 }
  },
  "columns": [
    {
      "name": "age",
      "dtype": "float64",
      "inferred_type": "float",
      "non_null_count": 2,
      "missing_count": 1,
      "missing_percentage": 33.3333,
      "unique_count": 2,
      "unique_percentage": 66.6667,
      "is_constant": false,
      "numeric_stats": {
        "mean": 31.5, "median": 31.5, "std": 3.5355339059327378,
        "minimum": 29.0, "maximum": 34.0, "q1": 30.25, "q3": 32.75,
        "zero_count": 0, "negative_count": 0
      },
      "datetime_stats": null,
      "categorical_stats": null
    },
    {
      "name": "plan",
      "dtype": "str",
      "inferred_type": "categorical",
      "non_null_count": 3,
      "missing_count": 0,
      "missing_percentage": 0.0,
      "unique_count": 2,
      "unique_percentage": 66.6667,
      "is_constant": false,
      "numeric_stats": null,
      "datetime_stats": null,
      "categorical_stats": {
        "top_values": [
          { "value": "basic", "count": 2, "percentage": 66.6667 },
          { "value": "pro", "count": 1, "percentage": 33.3333 }
        ],
        "truncated": false
      }
    }
  ],
  "quality": {
    "issue_count": 2,
    "issues": [
      {
        "code": "missing_values",
        "severity": "info",
        "message": "Column 'age' has 1 missing value(s) (33.3%).",
        "columns": ["age"],
        "details": { "missing_count": 1, "missing_percentage": 33.3333 }
      },
      {
        "code": "possible_id_column",
        "severity": "info",
        "message": "Column 'customer_id' looks like it may be a row identifier (100.0% unique). Identifiers are usually excluded from training.",
        "columns": ["customer_id"],
        "details": {
          "unique_percentage": 100.0,
          "reasons": ["name_suggests_identifier", "consecutive_integer_sequence"]
        }
      }
    ]
  },
  "target": {
    "name": "churn",
    "dtype": "str",
    "inferred_type": "categorical",
    "missing_count": 0,
    "missing_percentage": 0.0,
    "task_suggestion": "classification",
    "task_reason": "The target is categorical with 2 distinct values.",
    "distribution": [
      { "value": "no", "count": 2, "percentage": 66.6667 },
      { "value": "yes", "count": 1, "percentage": 33.3333 }
    ],
    "class_balance": {
      "class_count": 2,
      "majority_class": "no", "majority_percentage": 66.6667,
      "minority_class": "yes", "minority_percentage": 33.3333,
      "is_imbalanced": false
    },
    "numeric_stats": null
  }
}
```

### Errors

Every failure returns the same envelope, so a frontend needs one handler:

```json
{
  "error": {
    "code": "target_column_not_found",
    "message": "Target column 'revenue' is not in the dataset.",
    "details": { "target_column": "revenue", "available_columns": ["age", "plan"] }
  }
}
```

| Status | Code | Cause |
| --- | --- | --- |
| 400 | `empty_file` | The upload contains no bytes |
| 413 | `file_too_large` | The upload exceeds `MAX_UPLOAD_MB` |
| 413 | `dataset_too_large` | The dataset exceeds the row or column limit |
| 415 | `unsupported_file_type` | The file is not a `.csv` |
| 422 | `malformed_csv` | Rows do not match the header |
| 422 | `missing_header` | No usable header row |
| 422 | `duplicate_columns` | The header repeats a column name |
| 422 | `empty_dataset` | A header with no data rows |
| 422 | `target_column_not_found` | The requested target is not a column |
| 422 | `invalid_request` | A required form field is missing or malformed |
| 500 | `internal_error` | Unexpected failure; details stay server-side |

Stack traces are never returned to API consumers.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | Environment name reported by the API |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `MAX_UPLOAD_MB` | `25` | Largest accepted upload |
| `MAX_DATASET_ROWS` | `1000000` | Largest accepted parsed dataset |
| `MAX_DATASET_COLUMNS` | `1000` | Widest accepted parsed dataset |

Profiling thresholds (top-value count, missingness, cardinality, identifier
uniqueness, class-count limits) live in `backend/app/core/config.py` as named
constants on the `Settings` object.

### Current limitations

- CSV only, comma-delimited. Excel, JSON, Parquet and other delimiters are not supported.
- The whole file is read into memory, so the upload limit is the practical size ceiling.
- Uploaded data is not persisted; each request is independent and nothing is stored between calls.
- A missing header can only be detected when the first row is blank or produces placeholder column names; a header row of plain numbers is indistinguishable from data.
- Malformed-row detection is asymmetric: a row with *more* fields than the header is rejected as `malformed_csv`, while a row with *fewer* fields is padded with missing values, following CSV convention. The padding then shows up in the profile as missing values.
- Datetime detection samples the first 200 non-missing values of a column.
- Data-quality findings are heuristics meant to prompt a human check, not conclusions.
- No authentication or rate limiting — do not expose this service publicly yet.

---

## Preprocessing and feature engineering

The ML layer (`ml/`) turns a profiled dataset into model-ready training and
test data. It is a plain Python package with no web dependency: it takes a
DataFrame and a configuration and returns arrays, names and an account of every
decision. **It does not train models.**

`ml/README.md` documents it in full; the essentials follow.

### Configuration

A `PreprocessingConfig` says which column plays which role and how features are
treated. It can be **inferred from the Commit 2 profile** — profiling already
knows the semantic types, the constant columns, the free-text columns and the
possible identifiers, and that answer is reused rather than recomputed — and
**anything explicit overrides the inference**:

```python
from ml import infer_configuration, prepare_dataset

inferred = infer_configuration(profile, target_column="churn")
config = inferred.config.with_overrides(scaling_strategy="none", test_size=0.25)
prepared = prepare_dataset(frame, config, decisions=inferred.decisions)
```

### Feature groups

| Group | Source types | Treatment |
| --- | --- | --- |
| numeric | `integer`, `float` | impute (median by default) → scale (standard by default), plus a missing indicator |
| categorical | `categorical` | impute (most frequent by default) → one-hot encode, unknown categories tolerated |
| boolean | `boolean` | cast to 0/1 → impute; not scaled |
| datetime | `datetime` | expand to year, month, day, day of week → impute |

Columns that are not features are **never dropped silently**. Every column gets
a decision with a reason code — `profile_possible_id`, `free_text`,
`constant_column`, `high_cardinality`, `excluded_by_caller` and so on — and any
of them can be reinstated explicitly. Identifier-like and suspicious columns are
excluded from the feature set by default but remain in the dataset.

### Feature names

Names survive every transformation, which is what later makes feature
importance, SHAP and LLM explanations possible:

| Original | After preprocessing |
| --- | --- |
| `monthly_charges` | `monthly_charges` |
| `contract` | `contract_Month-to-month`, `contract_One-year`, `contract_Two-year` |
| `signup_date` | `signup_date_year`, `signup_date_month`, `signup_date_day`, `signup_date_day_of_week` |
| `age` (had gaps) | `age`, `missingindicator_age` |

### Feature engineering

Only safe, general-purpose transformations: datetime components, missing
indicators, and one-hot encoding. No polynomial expansion, no automatic
interactions, no domain-specific features, and nothing that would generate
hundreds of unreadable columns.

### Train/test split

`test_size` and `random_state` are configurable and the same seed always
reproduces the same split. Classification splits are stratified; regression
splits are not. Stratification is skipped — with the reason reported, never
silently — when the target has one class, when a class has a single row, or
when the test half is too small to hold every class. Cross-validation is not
implemented; the splitting functions are written so it can reuse them.

### Leakage prevention

Preprocessing is fitted on the training rows and on nothing else:

1. validate the configuration against the dataset,
2. separate the target from the features,
3. split into train and test,
4. `fit_transform` the preprocessor on the **training features only**,
5. `transform` the test features with the already-learned statistics.

So imputation values, scaler statistics, one-hot categories and missing
indicators all come from training data alone, and the target is never passed to
a feature transformer. This is checked directly in `ml/tests/test_leakage.py` —
including a test that corrupts the test rows' feature values and shows that
neither the fitted statistics nor the transformed training data change.

### Class imbalance

Measured, never corrected. The result reports class counts, percentages,
majority and minority classes and the imbalance ratio for the full dataset and
for each split half. No SMOTE, no oversampling, no undersampling.

### Result

`PreparedDataset` carries `X_train`, `X_test`, `y_train`, `y_test`, the fitted
preprocessor, feature names, feature groups, selected and excluded columns, the
per-column decisions, row counts and target distributions.
`prepared.summary()` returns a JSON-friendly view that deliberately omits the
fitted estimator and the data frames, so ML-internal objects never need to
appear in an API response.

Preprocessing is **not exposed over HTTP yet** — it is a library API.

---

## Model training and evaluation

On top of a `PreparedDataset`, the ML layer trains models, scores them against
a naive baseline and ranks them.

```python
from ml import select_and_evaluate_best_model, train_model

outcome = select_and_evaluate_best_model(prepared, folds=5)
outcome.selected_model_name   # chosen by cross-validation alone
outcome.final_test_score      # the single untouched-test measurement

trained = train_model(prepared, "random_forest_classifier")
trained.predict(raw_rows)     # raw columns in, predictions out
```

### Supported models

| Identifier | Model | Task |
| --- | --- | --- |
| `logistic_regression` | Logistic Regression | classification |
| `random_forest_classifier` | Random Forest Classifier | classification |
| `hist_gradient_boosting_classifier` | Histogram Gradient Boosting Classifier | classification |
| `linear_regression` | Linear Regression | regression |
| `random_forest_regressor` | Random Forest Regressor | regression |
| `hist_gradient_boosting_regressor` | Histogram Gradient Boosting Regressor | regression |

A small, deliberate suite. The registry is immutable and additive, so XGBoost
and LightGBM can be added later as definitions without touching the training
code.

### The trained artefact preserves preprocessing

Training fits a full pipeline, not a bare estimator:

```
raw feature rows -> preprocessing -> estimator -> prediction
```

The preprocessing step inside the model is fitted on the training rows alone,
so the finished model accepts the original columns — missing values and unseen
categories included — and applies exactly the transformations it learned. A
transformed matrix is never the only artefact.

### Metrics

Every metric declares whether higher or lower is better, and one that cannot be
computed is reported as unavailable with the reason rather than crashing.

**Classification** — accuracy, precision, recall, F1 and ROC-AUC, plus the
confusion matrix, class count and class distribution. Binary problems use
binary averaging against an explicitly reported positive class; three or more
classes use macro averaging. ROC-AUC is skipped when the model exposes no
probabilities or the test set holds a single class. Accuracy is never reported
alone.

**Regression** — MAE, MSE and RMSE (errors: **lower is better**) and R²
(explained variance: **higher is better**, 0 = no better than the mean). R² is
skipped when the target does not vary.

### Baseline comparison

"F1 = 0.82" means nothing on its own. Every model is measured against a
deliberately naive one on the same test rows: the **majority class** for
classification, the **training mean** for regression. Improvements are signed
so positive always means better than the baseline, whichever way the metric
runs.

```
baseline f1=0.710 accuracy=0.550

model                                      f1    acc  roc_auc  vs base
logistic_regression                     0.925  0.917    0.987   +0.216
random_forest_classifier                0.912  0.900    0.977   +0.202
hist_gradient_boosting_classifier       0.899  0.883    0.977   +0.189
```

---

## Cross-validated model selection

### Why not just use the test set?

A single train/test split gives one number per model, and that number carries
the luck of one particular division of the rows. Worse, a test set used to
*choose* a model has been spent: the winner's score is the best of several
draws, so reporting it as an estimate of future performance flatters it — the
more models compared, the more so.

Cross-validation moves the choice onto the training data. The training rows are
divided into *k* folds; each fold takes a turn as the validation set while the
others train the model. Every model is scored *k* times, and the comparison
rests on the average rather than a single draw.

> **Cross-validation selects the model; the held-out test set is reserved for
> the final evaluation.**

### The workflow

```
training data
      ↓
cross-validation  (k folds, test set never opened)
      ↓
compare models by mean fold score
      ↓
select the best
      ↓
retrain it on the COMPLETE training data
      ↓
ONE untouched test set
      ↓
final unbiased evaluation
```

```python
from ml import select_and_evaluate_best_model

outcome = select_and_evaluate_best_model(prepared, folds=5)
print(outcome.as_text())
```

```
Model                                   CV Mean F1  CV Std
----------------------------------------------------------
Logistic Regression                         0.9403  0.0160
Histogram Gradient Boosting Classifier      0.9167  0.0194
Random Forest Classifier                    0.9134  0.0187

Winner: Logistic Regression
Selected on 5-fold cross-validation of the training data; the test set was not used.

Final held-out test F1: 0.9254
Baseline (Majority class baseline): 0.7097  |  improvement: +0.2157
```

The cross-validated mean (0.9403) sits above the honest test score (0.9254).
That gap is exactly what a holdout-selected number hides.

### Splitters, mean and spread

Classification uses **`StratifiedKFold`** so every fold keeps the class
proportions — without it an imbalanced dataset can produce a fold with almost
none of the minority class, and a score that means nothing. Regression uses
plain **`KFold`**, since a continuous target has no classes to balance. Both
shuffle with the dataset's seed, so folds are random with respect to row order
and identical on every re-run.

Each fold produces the full metric set, aggregated into a **mean** and a
**standard deviation**. The spread is the honest half: a mean F1 of 0.84 ± 0.01
is a stable model, while the same mean ± 0.09 is a model whose score depends
heavily on which rows it saw.

The fold count is validated first — fewer than two folds, more folds than rows,
or a class with fewer members than folds all fail with a clear error rather
than producing misleading folds.

### Two strategies

| Strategy | Ranked by | Reads the test set? |
| --- | --- | --- |
| `holdout` (default for `compare_models`, unchanged) | score on the held-out test set | yes |
| `cross_validation` | mean of the training folds | **no** |

Under cross-validation the comparison object contains **no test-set numbers at
all** — no test metrics, no test baseline, no fitted model. They are absent
rather than merely unused, so selection cannot read them by accident. Ranking
reads the primary metric's declared direction: the maximum for F1 or R², the
minimum for RMSE. The naive baseline plays no part in choosing the winner; it
reappears at the end to frame the final test number.

Existing holdout callers are unaffected. Selecting through holdout reports
`final_evaluation_is_unbiased: false`, because the numbers that chose the model
are the numbers being reported.

### Not implemented

**No hyperparameter optimisation** — models run on their defaults plus whatever
a caller supplies. No Optuna, no nested or repeated cross-validation, no model
persistence, no SHAP, no MLflow, no XGBoost or LightGBM. Trained models live in
memory only.

Training and selection are **not exposed over HTTP yet** — they are a library
API, with `summary()` as the boundary that keeps sklearn objects out of any
future response.

---

## Current implementation status

**Implemented**

- Repository structure for backend, ML, RAG, agent and frontend layers.
- FastAPI application with system endpoints and a versioned API.
- CSV upload validation, safe in-memory loading and structural profiling.
- Heuristic data-quality detection and optional target-column analysis.
- Consistent error envelope across every failure mode.
- Environment-driven configuration for limits and thresholds.
- Format-agnostic preprocessing: feature selection, imputation, one-hot
  encoding, scaling and datetime expansion, driven by a configuration that can
  be inferred from the profile and overridden explicitly.
- Reproducible, leakage-safe train/test splitting with stratification for
  classification.
- A six-model training suite whose artefact is a full preprocessing pipeline,
  with task-appropriate metrics, naive baselines, fault-tolerant model
  comparison and direction-aware best-model selection.
- Stratified/K-fold cross-validation on the training data, with fold-level
  metrics, mean and standard deviation, CV-based model selection, and a single
  unbiased evaluation of the winner on the untouched test set.
- Test suites covering the backend service, the API contract and the ML layer.

**Not implemented yet**

- Hyperparameter optimisation (Optuna) and nested cross-validation
- Model persistence and explainability (SHAP)
- Ingestion formats other than CSV (Excel, JSON, Parquet, SQL, APIs)
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
| `POST` | `/api/v1/datasets/profile` | Profile an uploaded CSV file |

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

The ML layer is a separate component with its own dependencies. From the
repository root, install both for a full development environment:

```bash
pip install -r backend/requirements.txt -r ml/requirements.txt
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
curl -X POST http://127.0.0.1:8000/api/v1/datasets/profile -F "file=@customers.csv"
```

## Running the tests

From the repository root, which runs both the backend and the ML suites:

```bash
pytest
```

Or one suite at a time:

```bash
pytest backend/tests
pytest ml/tests
```

Add `-v` for per-test output. Every test builds its data in memory — none reads
an external dataset or touches the network.

## Roadmap

1. ~~**Project foundation** — repository structure, FastAPI service, tests~~
2. ~~**Dataset upload and profiling** — validation, profile, data quality, target analysis~~
3. ~~**Preprocessing and feature engineering** — configuration, pipeline, leakage-safe split~~
4. ~~**Model training and evaluation** — registry, metrics, baselines, comparison~~
5. **Cross-validation and model selection** — k-fold selection, one unbiased test evaluation *(current)*
6. Explainability with SHAP
7. Experiment tracking
8. RAG layer over documentation and run history
9. LLM integration
10. Agentic workflows
11. Next.js frontend
12. Containerisation and deployment
