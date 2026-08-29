# ML Copilot — AI Data Scientist

ML Copilot is a production-oriented AI system that will act as an assistant data
scientist: you give it a dataset and a question, it profiles the data, trains and
evaluates candidate models, explains what the models learned, and answers
follow-up questions in natural language — with every run tracked and every
answer grounded in retrievable context.

> **Status: early development.** The backend can ingest a CSV file and return a
> full dataset profile; the ML layer turns a profiled dataset into model-ready
> data, cross-validates a suite of models on the training rows, picks a winner
> without ever reading the test set, retrains it on the full training data,
> measures it once on the untouched test set, and explains what the chosen
> model is doing with SHAP, and records the whole run so it can be found and
> compared later — all of it reachable over HTTP. The retrieval layer then
> makes that documentation and history searchable, returning cited evidence.
> **There is no hyperparameter optimisation, no MLflow, no database, no model
> serving, no LLM and no agent** — retrieval returns evidence, and nothing
> generates an answer from it yet. Experiment history and the vector index are
> local files. Everything marked *planned* below is not implemented.

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
| Explainable AI | Global and per-prediction feature attributions (SHAP) | **implemented** |
| Retrieval over docs and experiments | Semantic search with metadata filtering and citations, over project documentation and run history | **implemented** |
| Retrieval-augmented answers | Grounded natural-language responses built from that evidence | planned (needs the LLM) |
| Agentic workflows | Multi-step, tool-using analysis planned and executed by an agent | planned |
| Experiment tracking | Reproducible run history — dataset fingerprint, configuration, metrics and explanations, stored locally as JSON | **implemented** (MLflow not implemented) |
| HTTP API | Dataset profiling, experiment execution, history and comparison over REST | **implemented** |
| Web interface | Dataset upload, run monitoring, results and chat | planned |

## High-level architecture

```
             ┌───────────────────────┐
             │   Next.js frontend    │   (planned)
             └───────────┬───────────┘
                         │ HTTP
             ┌───────────▼───────────┐
             │    FastAPI backend    │   ← system, dataset, experiment routes
             │  api · services · db  │
             └─┬────────┬──────────┬─┘
               │        │          │
   ┌───────────▼──┐ ┌───▼───────┐ ┌▼──────────────┐
   │  ML layer    │ │ RAG layer │ │  Agent layer  │
   │ preprocessing│ │ ingestion │ │ tools,        │
   │ ✓ training ✓ │ │ chunking  │ │ workflows,    │
   │ explainability│ │ retrieval│ │ state         │
   │      ✓       │ │     ✓     │ │  (planned)    │
   └───────┬──────┘ └─────┬─────┘ └───────┬───────┘
           │              │               │
   ┌───────▼──────┐ ┌─────▼─────┐ ┌───────▼───────┐
   │ Experiment   │ │  Vector   │ │  PostgreSQL   │
   │  tracking    │ │   store   │ │   metadata    │
   │ local JSON ✓ │ │ local ✓   │ │   (planned)   │
   │ MLflow: no   │ │ Qdrant: no│ │               │
   └───────┬──────┘ └─────▲─────┘ └───────────────┘
           │              │
           └──────────────┘
     experiments feed the index; the index never writes back
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
| Explainability | SHAP | **implemented** |
| Experiment tracking (storage) | Local JSON files behind an `ExperimentStore` interface | **implemented** |
| Experiment tracking (server) | MLflow | **not implemented** |
| LLM integration | Provider-agnostic LLM client | planned |
| Embeddings | Local, offline by default (hashed n-grams); optional `all-MiniLM-L6-v2` | **implemented** |
| Retrieval (index) | Local persistent vector store behind a `VectorStore` interface | **implemented** |
| Retrieval (server) | Qdrant vector database | **not implemented** |
| Agents | Orchestrated multi-step workflows | planned |
| Database | PostgreSQL | planned |
| Frontend | Next.js, TypeScript | planned |
| Local orchestration | Docker Compose | skeleton only |

## Repository layout

```
ml-copilot/
├── backend/       FastAPI service (api, core, models, schemas, services, tests)
├── frontend/      Next.js application (placeholder)
├── ml/            Preprocessing, training, selection, explainability, experiment tracking
├── rag/           Documentation and experiment retrieval (chunking, embeddings, vector store)
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
   (CSV today;
    Excel, JSON,
    Parquet, SQL,
    API planned)
```

Adding a format therefore means writing one adapter that returns a DataFrame.
Profiling, preprocessing, training and the experiment runner need no changes
and have no knowledge of where the data came from — the runner's entry point
takes a DataFrame, and the HTTP upload path exists only to produce one.

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
persistence, no MLflow, no XGBoost or LightGBM. Trained models live in memory
only.

---

## Explainability

Two questions, two answers.

### What SHAP does

A trained model turns a row of features into a number. SHAP answers: how much
did each feature contribute to *that* number, rather than to the model's usual
answer?

It works against a reference. The model has an average output over background
rows — the **base value** — and SHAP splits the gap between this row's output
and that average across the features. The split has a guarantee worth relying
on:

```
base value  +  every feature's contribution  =  the model's output
```

Nothing is left over, and nothing is invented. That property is what the tests
verify, rather than merely checking numbers came back.

### Global versus local

| | Question | Function |
| --- | --- | --- |
| **Global** | What drives this model in general? | `explain_global` |
| **Local** | Why did *this* row get *this* answer? | `explain_prediction` |

```python
from ml import explain_global, explain_prediction

overall = explain_global(outcome.final_model, prepared.X_train_raw)
why = explain_prediction(outcome.final_model, prepared.X_test_raw.iloc[[0]])
```

Global importance is the mean absolute SHAP value across many rows — how far
each feature moved the output on average, in either direction:

```
feature                  importance
--------------------------------------
tenure_months            0.2277
income                   0.1876
segment_business         0.1249
segment_public           0.0692
segment_retail           0.0135
```

A local explanation keeps the signs, because direction is the point:

```
prediction: no   probability: 0.6150   base value: 0.4448

feature                  value      contribution  direction
------------------------------------------------------------------
tenure_months            112.00     -0.3453       decreases_prediction
income                   41712.33   +0.2377       increases_prediction
segment_public           None       +0.1514       increases_prediction
segment_business         None       +0.0908       increases_prediction
segment_retail           None       +0.0356       increases_prediction
```

0.4448 + 0.1702 = 0.6150 — the base value plus the contributions is exactly the
probability the model produced.

### This is not causality

> **Explainability describes model behaviour and associations; it does not
> establish causal relationships.**

`increases_prediction` means the model's output was higher with this value than
at the reference. It does not mean changing that value in the real world would
change the outcome. A feature can rank highly because it stands in for
something the model never saw. Every result carries this caveat in its
`disclaimer` field so it cannot be lost downstream.

### Why transformed features, and why names matter

The trained artefact is `Pipeline(preprocessing, estimator)`, and SHAP needs the
numbers the *estimator* saw:

```
raw row -> already-fitted preprocessing -> transformed features -> estimator -> SHAP
```

Rows go through the preprocessing exactly as it was fitted during training —
never refitted — so explaining a model cannot change it. Because that step
carries Commit 3's feature names, explanations talk about
`contract_Month-to-month` and `missingindicator_age` rather than `x0` and `x7`.
That is the difference between a result someone can act on and a table of
numbers, and it is what will make this usable by a future LLM layer.

### Which explainer, and the fallback

| Model family | Explainer |
| --- | --- |
| Tree ensembles — random forest, gradient boosting, decision trees | `TreeExplainer` |
| Linear models — logistic, linear, ridge, lasso | `LinearExplainer` |
| Anything else | none — permutation importance instead |

No single explainer is forced onto every model. When none suits, **global**
importance falls back to permutation importance — shuffle a feature and see how
far the model's score falls. The method is always named in the result and the
reason SHAP was skipped is recorded, so a permutation number is never mistaken
for a SHAP one.

The fallback has two honest limits. It needs the labels, so without them the
result is `status: unavailable` rather than a guess. And it is **global only**:
a local explanation SHAP cannot produce comes back `unavailable` — with the
prediction and probabilities still reported, but no contributions invented from
the global method.

### Classification and sampling

Binary results name the predicted class, the explained class and the positive
class separately. Multiclass local explanations are per class; multiclass global
importances are averaged over classes, which the result states rather than
silently collapsing.

SHAP is not free, so rows are capped (`max_explanation_rows`, default 500) and
the excess sampled deterministically. Sampling is never silent: `sample_count`
reports the rows actually used and a warning records the reduction.

Training, selection and explanation are reached over HTTP through
`POST /api/v1/experiments/run` — see **The HTTP API** below. The `summary()`
boundary is what keeps sklearn objects out of those responses.

---

## Experiment tracking

Everything above is ephemeral: a pipeline run measures a winner, explains it,
and then the process exits and the result is gone. Experiment tracking gives
each complete run a small, readable, permanent record, so runs can be compared,
reproduced, and — in a later commit — read back by an assistant.

> **MLflow is NOT implemented yet.** Neither is PostgreSQL or any other
> database. Records are JSON files written into a local directory. This is a
> deliberate intermediate architecture, not a shortcut: local JSON needs no
> server, no migrations and no dependency, and stays readable in an editor
> while the pipeline is still changing shape. Storage sits behind an
> `ExperimentStore` interface, so MLflow or a database replaces it later
> without touching anything above.

### What is recorded

One `ExperimentRun` per complete pass through the pipeline:

- **identity** — schema version, experiment id, configuration hash, timestamp, name, description, tags
- **dataset** — content fingerprint, shape, columns and dtypes, target, task, source format, data-quality findings
- **preprocessing** — configuration, feature groups, per-column decisions, transformed feature names, split sizes, seed, stratification
- **selection** — strategy, folds, every candidate and its score, the winner, the selection score and spread, and whether test data was touched
- **evaluation** — the single unbiased test measurement, the baseline and the improvement over it
- **explainability** — method, explainer and ranked feature importances
- **environment** — Python version, platform, library versions, random seed

**Not recorded:** the dataset itself, the fitted pipeline, and the SHAP
explainer. Those are model artefacts, not history; the serialiser refuses them
rather than trusting callers to leave them out. A typical record is about 6 KiB.

### A dataset is identified by content

The fingerprint is a SHA-256 digest over the column names, dtypes and values —
never the filename or path. The same table exported twice under different names
fingerprints identically, so history survives a file being renamed, moved or
re-exported from another format. An edited value, an added row, a renamed
column or a different row order all make it a different dataset; the DataFrame
index does not.

### An experiment id is not just a timestamp

```
exp_84a8d53a1f5f_20260828T134042Z_b8c1
    └ configuration  └ UTC time     └ collision suffix
```

The first segment hashes the *inputs* — dataset fingerprint, preprocessing
settings, split, seed, candidate models, selection rule — and nothing the run
concluded. So an unchanged setup re-run tomorrow produces the same
configuration hash with a distinct id, which is what makes "have I run this
before?" answerable.

### Reading history back

Runs can be filtered by fingerprint, target, task, model, strategy, metric or
tag, and sorted by time, model or score. Sorting by score reads the metric's own
declared direction, so the largest F1 wins but the smallest RMSE does.
Comparison refuses to rank runs judged by different metrics or solving
different tasks rather than putting **RMSE and F1 in the same column**.

### Safety and integrity

- **Atomic writes.** A record is serialised before the filesystem is touched, then written to a temporary file, `fsync`ed and moved into place with an atomic rename. An interrupted write leaves the previous record intact, never a fragment.
- **Corruption is contained.** `get` raises on a broken record; `list` skips it with a warning so one bad file cannot hide a history; `verify()` reports exactly what is unreadable.
- **Path traversal is refused.** Experiment ids are restricted to letters, digits, underscores and hyphens, and the resolved path is confirmed to be inside the store directory — so `../../etc/passwd`, an absolute path or a backslash never becomes a write target.
- **No secrets are captured.** The environment section records interpreter, platform, library versions and the seed. No hostname, username, file path or environment variable is read, so no API key or token can reach a record.
- **Versioned records.** Every record carries `"schema_version": "1.0"`; an unknown or missing version is refused rather than half-read.

---

## The HTTP API

Everything above is reachable over HTTP. The API is an **adapter around the ML
engine**: it validates requests, orders the existing steps and turns results
into JSON. No statistic, split, score or explanation is computed in a route —
a test fails the build if a route module so much as imports sklearn, SHAP,
numpy or pandas.

```
                       FastAPI
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
    Dataset          Experiment        Experiment
    profiling          runner            history
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                      ML engine
                          │
        ┌─────────────────┼─────────────────┐
        ▼                 ▼                 ▼
   Cross-validated      SHAP           Experiment
     selection        explanation        storage
```

### Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service name, version, environment and docs URL |
| `GET` | `/health` | Liveness check |
| `POST` | `/api/v1/datasets/profile` | Profile an uploaded CSV |
| `POST` | `/api/v1/experiments/run` | Run a complete experiment and store it |
| `GET` | `/api/v1/experiments` | List stored experiments, filtered and sorted |
| `GET` | `/api/v1/experiments/capabilities` | Models, metrics and limits a request may use |
| `GET` | `/api/v1/experiments/{experiment_id}` | Fetch one stored experiment |
| `POST` | `/api/v1/experiments/compare` | Rank several stored experiments |

Interactive documentation is at `/docs`; the schema is at `/openapi.json`.

### Running an experiment

A dataset is a file and cannot travel inside a JSON body, so the request is
`multipart/form-data`: the file as an upload, the configuration as form fields.

```bash
curl -X POST http://127.0.0.1:8000/api/v1/experiments/run \
  -F "file=@customers.csv" \
  -F "target_column=renewed" \
  -F "models=logistic_regression" \
  -F "models=random_forest_classifier" \
  -F "folds=5" \
  -F "name=renewal baseline"
```

One request runs the whole pipeline:

```
upload → validate → DataFrame → profile → infer configuration
      → apply explicit overrides → leakage-safe train/test split
      → cross-validate every candidate on the training rows
      → select the winner → retrain on the full training data
      → ONE evaluation on the untouched test set
      → SHAP global explanation → ExperimentRun → stored
```

Every field is optional. With none of them the service profiles the data,
cross-validates every model that suits the detected task and keeps the winner
— and warns that it chose the target column by convention.

Execution is **synchronous**: the response arrives when the run has finished.
No queue, worker or background execution is implemented.

There is **no prediction or model-serving endpoint**. That is deliberate:
experiment records do not contain the fitted model, so nothing exists to serve.
Adding one would mean implementing model persistence, which is a separate
decision rather than a side effect of building an API.

### What comes back

The response is the stored experiment record plus how the run went: identity
and configuration hash, the dataset's content fingerprint and shape, the
preprocessing decisions, every candidate and its score, the winner, the single
untouched-test evaluation with its baseline and improvement, the ranked SHAP
importances, the reproducibility metadata, and any warnings.

It contains **only JSON-safe values**. A `Pipeline`, `ColumnTransformer`,
estimator, `numpy.ndarray`, `DataFrame` or SHAP explainer cannot appear: the
record is built by a serialiser that refuses them, and the response is then
validated against a Pydantic model that has no field able to hold one. Tests
assert both — that every leaf of a real response is a string, number, boolean
or null, and that no artefact's text appears anywhere in it.

### One error contract

Every failure — dataset, preprocessing, model, explainability, experiment
storage or request validation — answers in the same envelope:

```json
{"error": {"code": "incompatible_model_task", "message": "...", "details": {}}}
```

| Status | When |
| --- | --- |
| `400` | Invalid configuration: unknown model, metric, strategy, fold count, sort key or experiment id |
| `404` | No experiment stored under that id |
| `409` | The request conflicts with the data: a classifier for a regression target, or a comparison of runs judged by different metrics |
| `413` | The upload or the dataset exceeds a configured limit |
| `415` | Unsupported file type — only `.csv` is implemented |
| `422` | The request or the data cannot be processed: missing target, no usable features, malformed CSV, too few rows |
| `500` | An unexpected internal failure — logged with its cause, answered generically |

The ML layer raises plain Python exceptions with no HTTP meaning;
`backend/app/core/ml_errors.py` is the single place that maps them to a code
and a status, and it strips anything path-like from the details on the way out.
No stack trace, internal class name or filesystem path reaches a client on any
path — including a 404, which never reveals where records are kept.

### Experiment history over HTTP

```bash
curl "http://127.0.0.1:8000/api/v1/experiments?task_type=classification\
&sort_by=primary_metric&order=desc&limit=5"
```

Filters — dataset fingerprint, target, task, model, strategy, metric, tags —
are optional and combine with "and". Sorting by score reads the metric's own
declared direction. Comparison refuses runs that do not share a task and a
metric rather than ranking an RMSE against an F1. All of it is Commit 7's
`ExperimentQuery` and comparison logic; the API adds bounds, not a second query
language, and **no database is involved**.

Because a dataset is identified by its content, `?dataset_fingerprint=…` finds
every run on the same data however the file was named when it was uploaded.

### Uploads are not kept

The file is validated and parsed in memory and never written anywhere. The
client-supplied filename is reduced to a bare name before use, so no
request-supplied path reaches the filesystem, and experiment ids are restricted
to characters that cannot climb out of the store directory. What is stored is
the record — fingerprint, shape, decisions, scores, explanation — never the
dataset, the fitted pipeline or the explainer.

### Still CSV only

**CSV is currently supported; the ML pipeline is intentionally
format-agnostic.** The runner's real entry point takes a standardised
DataFrame — `run_experiment(frame, ...)` — and the file path exists only to
produce one:

```
Input adapter → DataFrame → Profiler → Preprocessor → ML
 (CSV today)
```

Adding Excel, JSON, Parquet, SQL or an HTTP source means writing one loader in
the dataset service. Nothing in the runner, the ML layer or the experiment
store changes, because none of them ever sees a file. Those formats are **not
implemented**.

---

## Retrieval over documentation and experiments

Everything above produces knowledge: documentation that says how the system
works, and experiment records that say what was actually run. The retrieval
layer makes both searchable, so a question can be answered from what this
project knows about itself rather than from what a model happens to remember.

> **LLM generation is not implemented yet.** This layer returns ranked
> passages with citations. It never writes an answer, draws a conclusion or
> interprets a result. What is built is the part that decides *what a future
> model gets to see*, and makes every sentence of a future answer traceable to
> a passage that exists.

**Also not implemented:** Qdrant, PostgreSQL, any vector database, LangChain,
LangGraph, agents, autonomous tool calling, and any hosted embedding API.

```
question → embed → filter by metadata → rank by cosine similarity
                                      → ranked passages, each with a citation
                                      → (later: LLM → grounded answer)
```

### What is indexed

**Project documentation** — the four READMEs, from an explicit allowlist plus
one optional `docs/` directory. Source code, datasets, model artefacts,
virtual environments, `.git` and the experiment store are not merely skipped;
they are never candidates. `.env` and anything whose name suggests a
credential is refused even when explicitly configured.

**Experiment records** — every stored `ExperimentRun`, rendered as structured
Markdown with a heading per section, so a question about feature importance
finds the importance section rather than the whole run:

```
## Model selection
Selection strategy: cross_validation
Selection score: 0.8623 ± 0.0465
Candidate results:
- logistic_regression: 0.8623 ± 0.0465 (succeeded)
- random_forest_classifier: 0.7998 ± 0.0474 (succeeded)

## Explainability
Top features by importance:
1. income: 0.9962
2. tenure_months: 0.8579
```

**Only stored facts are written.** The renderer never produces "the model
performed well" or "the forest was the better choice", because neither is in
the record — ungrounded prose in the index would be retrieved and cited as
fact later.

The dependency runs one way: `ml/experiments → rag/ingestion → rag/retrieval`.
Nothing in `ml/` knows the index exists, so experiments can be recorded with
no index present and the index can be rebuilt from the store at any time.

### Chunking, embeddings and the store

Chunking follows the document's own structure — sections, then paragraphs,
then a line boundary as a last resort — never a fixed character count. Fenced
code blocks are kept whole, tiny fragments are merged, and the heading path
travels with each chunk, so a passage read alone still says what it is about.

Embeddings are behind an interface. The **default provider runs offline**:
hashed word and character n-grams from scikit-learn, no download, no API key,
no network, and deterministic across machines. It matches on term overlap
rather than meaning — an honest limitation, and the reason retrieval quality
is measured rather than asserted. An optional
`sentence-transformers/all-MiniLM-L6-v2` provider is supported for real
semantic matching, lazily loaded and not installed by default (it costs about
4 GB of dependencies and a ~90 MB model download).

The store is a persistent local index — a float32 matrix, a JSONL record file
and a manifest, written atomically — that survives process restarts. Row *i*
of the matrix is the chunk on line *i* of the records, and a mismatch is
refused rather than guessed at.

### Searching

```python
from rag import RagConfig, RagIndexer, RetrievalService
from ml.experiments import LocalExperimentStore

indexer = RagIndexer(RagConfig())
indexer.index_documentation()
indexer.sync_experiments(LocalExperimentStore())

service = RetrievalService(RagConfig())
for result in service.search("How is data leakage prevented?", top_k=3):
    print(result.rank, round(result.score, 3), result.citation)
```

```
1 0.234 docs:readme#leakage-prevention
2 0.227 docs:ml-readme#leakage-prevention
3 0.220 docs:ml-readme#preprocessing
```

**The similarity metric is cosine.** Metadata filtering happens *before*
ranking, so asking for five classification experiments searches classification
experiments rather than ranking everything and discarding the rest:

```python
service.search_experiments(
    "which model scored best on the test set",
    task_type="classification",
    dataset_fingerprint="86494cff7a45cb7f",
)
```

### Citations

Every chunk carries a reference that resolves:

```
docs:ml-readme#leakage-prevention
experiment:exp_84a8d53a1f5f_20260828T134457Z_e420#final-evaluation
```

An experiment citation *is* the id the store and the HTTP API already use, so
`GET /api/v1/experiments/exp_84a8…` resolves it. When a future answer says
"according to experiment exp_84a8…", the claim can be checked.

### Measuring retrieval quality

A retriever that returns five confident passages about the wrong subject is
worse than one that returns nothing, because a model will use them. So quality
is measured with **Hit@K** (did any relevant document appear?) and **Recall@K**
(what fraction did?), over a small deterministic set of questions whose
answers really do live in the indexed documentation:

```
Retrieval evaluation over 5 queries at k=5
  Hit@5:    100.00%
  Recall@5: 90.00%
```

Answer quality is not evaluated, because there are no answers to evaluate.

### Indexing is incremental and reproducible

Document and chunk ids derive from content and position — no random UUID
appears anywhere — so re-indexing an unchanged source is a no-op, a changed
source has its old chunks deleted before the new ones are written, and two
machines indexing the same repository agree on every identifier. A manifest
records each source's hash and the embedding provider that built the index; a
provider change triggers a rebuild rather than a mix of incomparable vectors.

Retrieval is a library API in this commit — there is **no search endpoint**
yet. See `rag/README.md` for the full design.

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
- SHAP explainability: ranked global feature importance and signed
  per-prediction contributions over the transformed features, with a
  permutation-importance fallback for models SHAP cannot handle.
- Experiment tracking: content-based dataset fingerprints, configuration
  hashes, versioned JSON-safe run records, an `ExperimentStore` interface with
  a local atomic-write implementation, and filtering, sorting and
  direction-aware comparison over stored history.
- An HTTP API over the whole engine: run an experiment on an uploaded dataset,
  list and fetch stored experiments, and compare them — with one error
  envelope, JSON-safe responses and generated OpenAPI documentation.
- A retrieval layer over project documentation and experiment history:
  structure-aware chunking, a pluggable embedding interface with an offline
  default, a persistent local vector store, cosine search with metadata
  filtering, stable citations, and Recall@K/Hit@K evaluation.
- Test suites covering the backend service, the API contract, the ML layer and
  the retrieval layer.

**Not implemented yet**

- Hyperparameter optimisation (Optuna) and nested cross-validation
- Model persistence — no fitted pipeline or explainer is written to disk, so
  there is no prediction or model-serving endpoint
- **MLflow** — experiment tracking runs on local JSON files only
- Ingestion formats other than CSV (Excel, JSON, Parquet, SQL, APIs)
- **LLM generation** — retrieval returns evidence; nothing reads it yet
- Agentic workflows, autonomous tool calling, LangChain and LangGraph
- A search endpoint — retrieval is a library API in this commit
- PostgreSQL, Qdrant and any database access
- Background execution — no Celery, Redis, queue or worker; runs are synchronous
- Authentication and rate limiting
- Frontend application
- Containerisation and deployment

### Available endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Service name, version, environment and docs URL |
| `GET` | `/health` | Liveness check — returns `{"status": "ok", ...}` |
| `POST` | `/api/v1/datasets/profile` | Profile an uploaded CSV file |
| `POST` | `/api/v1/experiments/run` | Run a complete experiment and store it |
| `GET` | `/api/v1/experiments` | List stored experiments, filtered and sorted |
| `GET` | `/api/v1/experiments/capabilities` | Models, metrics and limits a request may use |
| `GET` | `/api/v1/experiments/{experiment_id}` | Fetch one stored experiment |
| `POST` | `/api/v1/experiments/compare` | Rank several stored experiments |

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

The ML layer is a separate component with its own dependencies (pandas,
scikit-learn and SHAP). From the repository root, install both for a full
development environment:

```bash
pip install -r backend/requirements.txt -r ml/requirements.txt -r rag/requirements.txt
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

From the repository root, which runs the backend, ML and retrieval suites:

```bash
pytest
```

Or one suite at a time:

```bash
pytest backend/tests
pytest ml/tests
pytest rag/tests
```

Add `-v` for per-test output. Every test builds its data in memory — none reads
an external dataset, downloads a model or touches the network. The retrieval
tests use a deterministic fake embedding provider; the one test that exercises
the real sentence-transformer model skips itself unless the package is
installed and the model is already cached.

## Roadmap

1. ~~**Project foundation** — repository structure, FastAPI service, tests~~
2. ~~**Dataset upload and profiling** — validation, profile, data quality, target analysis~~
3. ~~**Preprocessing and feature engineering** — configuration, pipeline, leakage-safe split~~
4. ~~**Model training and evaluation** — registry, metrics, baselines, comparison~~
5. ~~**Cross-validation and model selection** — k-fold selection, one unbiased test evaluation~~
6. ~~**Explainability with SHAP** — global importance, local contributions, fallback~~
7. ~~**Experiment tracking** — dataset fingerprints, versioned run records, local persistence, comparison~~ *(MLflow deferred)*
8. **Experiment API** — run, list, fetch and compare experiments over HTTP *(current)*
9. **Retrieval over documentation and run history** — chunking, embeddings, vector store, cited evidence *(current; LLM deferred)*
10. LLM integration — grounded answers built from that evidence
11. Agentic workflows
12. Next.js frontend
13. Containerisation and deployment
