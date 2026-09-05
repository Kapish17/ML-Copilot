# ML Copilot — ML Layer

Everything between a profiled dataset and an explained model: feature
configuration, leakage-safe preprocessing, model training, evaluation, baseline
comparison, cross-validated model selection, a single final measurement on
untouched data, SHAP explanations of what the chosen model is doing, and a
persistent record of every run.

> **Cross-validation selects the model; the held-out test set is reserved for
> the final evaluation.**
>
> **Explanations describe model behaviour and associations; they do not
> establish causal relationships.**

**Not implemented:** hyperparameter optimisation (no Optuna), **MLflow**, any
database (no PostgreSQL), model persistence, XGBoost or LightGBM. Experiment
tracking is implemented, but against a local JSON store written by this
package — trained models and SHAP explainers themselves live in memory for the
lifetime of the process and are never written to disk.

The stored records are read by the retrieval layer in `rag/`, which makes them
searchable, and the training and explanation functions here are what the
agent's `run_experiment` and `explain_experiment` tools call — and therefore
what `POST /api/v1/agent/ask` may reach. Both dependencies
run one way only: this package does not import `rag/`, `llm/` or `agent/`, and
does not know an index, a model or an agent exists. See `rag/README.md` and
`agent/README.md`.

## Format-agnostic by design

This package accepts a **standardised `pandas.DataFrame`** and nothing else. It
never sees a file, a path, an upload, or any CSV-specific object — passing
anything other than a DataFrame raises an error rather than being coerced.

```
ingestion adapter -> DataFrame -> profiling -> configuration -> preprocessing
   (CSV today;                                                        |
    Excel, JSON,                                                      v
    Parquet, SQL,                    model registry -> cross-validation
    API planned)                                       (training folds)
                                                                      |
                                                                      v
                                              model selection -> retrain winner
                                                                      |
                                                                      v
                                            ONE untouched test evaluation
                                                                      |
                                                                      v
                                              SHAP explanation (global + local)
```

CSV is the only ingestion format implemented today. Adding Excel, JSON,
Parquet, SQL or an HTTP source means writing an adapter that returns a
DataFrame; nothing in this package changes, and none of its tests need to know
where the data came from.

The same boundary applies in the other direction: the ML layer does not import
the backend. It consumes a dataset profile *structurally* — the protocols in
`features/inference.py` declare the handful of attributes it reads — so the
dependency runs one way only, and profiling logic is never duplicated here.

## Structure

```
ml/
├── features/
│   ├── types.py         Shared vocabulary (feature types, strategies, roles)
│   ├── config.py        PreprocessingConfig and its validation
│   ├── decisions.py     Per-column decisions, so selection is explainable
│   ├── inference.py     Deriving a configuration from a dataset profile
│   └── transformers.py  DatetimeComponentExtractor and small helpers
├── pipelines/
│   ├── preprocessing.py Builds the sklearn ColumnTransformer
│   ├── preparation.py   Validate → split → fit on train → transform
│   └── result.py        PreparedDataset, the structured outcome
├── evaluation/
│   ├── splitting.py         Train/test split and target-distribution reporting
│   ├── metrics.py           Task metrics, their directions, the primary metric
│   └── cross_validation.py  K-fold validation over the training rows
├── models/
│   ├── registry.py      Which estimators exist and how to build them
│   ├── spec.py          The validated request for one training run
│   ├── baselines.py     Naive reference models and the improvement over them
│   ├── training.py      Fitting Pipeline(preprocessing, estimator)
│   ├── comparison.py    Running several models and ranking them
│   ├── selection.py     Choosing a winner, then measuring it once
│   └── result.py        TrainedModel, the outcome of one run
├── explainability/
│   ├── types.py         Method, status and direction vocabulary
│   ├── config.py        Row limits and the seed for deterministic sampling
│   ├── results.py       GlobalExplanation and LocalExplanation
│   ├── strategy.py      Which SHAP explainer suits which estimator
│   ├── shap_backend.py  Running SHAP and normalising what it returns
│   ├── permutation.py   The global-only fallback
│   └── service.py       explain_global, explain_prediction
├── experiments/
│   ├── fingerprint.py   Content hash identifying a dataset, not a file
│   ├── identity.py      Configuration hashes, experiment ids, id validation
│   ├── serialization.py JSON safety, and what is refused outright
│   ├── run.py           ExperimentRun and its sections; the record schema
│   ├── store.py         ExperimentStore protocol, queries, sorting
│   ├── local_store.py   LocalExperimentStore — one JSON file per run
│   ├── builder.py       create_experiment_run: composing existing results
│   ├── comparison.py    Comparing runs on a metric they share
│   └── runs/            Stored history (git-ignored, created on first save)
├── artifacts/
│   ├── schema.py        The artifact manifest: feature schema, classes, metrics
│   ├── store.py         ModelArtifactStore protocol; LocalModelArtifactStore
│   ├── prediction.py    Validating records against a manifest, and predicting
│   └── models/          Persisted artifacts (git-ignored, created on first save)
├── tests/               Synthetic-data tests, including leakage proofs
└── requirements.txt
```

## Usage

```python
from ml import infer_configuration, prepare_dataset

# `profile` is the output of the dataset profiling service; `frame` is the
# same dataset as a DataFrame.
inferred = infer_configuration(profile, target_column="churn")

# Inference is a starting point — anything explicit overrides it.
config = inferred.config.with_overrides(
    scaling_strategy="none",
    excluded_columns=("promo_code",),
    test_size=0.25,
)

prepared = prepare_dataset(frame, config, decisions=inferred.decisions)

prepared.X_train        # transformed training features (DataFrame, named)
prepared.X_train_raw    # the source columns behind them, untransformed
prepared.y_train        # untouched target values
prepared.feature_names  # names after encoding and expansion
prepared.summary()      # JSON-friendly description of the whole run
```

Then choose a model and measure it:

```python
from ml import select_and_evaluate_best_model

outcome = select_and_evaluate_best_model(prepared, folds=5)

outcome.selected_model_name   # chosen by cross-validation alone
outcome.selection_score       # its mean over the training folds
outcome.final_test_score      # the single untouched-test measurement
outcome.summary()             # JSON-friendly, the two kept separate
print(outcome.as_text())
```

Or work with the pieces directly:

```python
from ml import compare_models, cross_validate_model, select_best_model, train_model

cv = cross_validate_model(prepared, "random_forest_classifier", folds=5)
cv.mean_primary_metric, cv.std_primary_metric

comparison = compare_models(prepared, strategy="cross_validation", folds=5)
best = select_best_model(comparison)

trained = train_model(prepared, "random_forest_classifier")
trained.predict(raw_rows)     # raw columns in, predictions out
```

Then ask what the model is doing:

```python
from ml import explain_global, explain_prediction

overall = explain_global(outcome.final_model, prepared.X_train_raw)
overall.top(5)                # the features that drive it in general

why = explain_prediction(outcome.final_model, prepared.X_test_raw.iloc[[0]])
why.prediction, why.probability, why.base_value
why.top(5)                    # what moved this one row
```

A configuration can also be written by hand, without a profile:

```python
from ml import PreprocessingConfig, prepare_dataset

config = PreprocessingConfig(
    target_column="price",
    numeric_columns=("size_sqm", "rooms"),
    categorical_columns=("district",),
    task_type="regression",
)
prepared = prepare_dataset(frame, config)
```

## Feature groups

Each column is routed to exactly one branch of the pipeline.

| Group | Source types | What happens |
| --- | --- | --- |
| numeric | `integer`, `float` | impute → scale, plus a missing indicator |
| categorical | `categorical` | impute → one-hot encode |
| boolean | `boolean` | cast to 0/1 → impute, no scaling |
| datetime | `datetime` | expand to calendar components → impute |

Columns that are not features are never dropped silently. Every column of the
dataset gets a `ColumnDecision` recording its role and the reason:

| Reason code | Meaning |
| --- | --- |
| `declared_target` | The column being predicted |
| `excluded_by_caller` | Excluded explicitly |
| `identifier_by_caller` | Marked as an identifier by the caller |
| `profile_possible_id` | Profiling flagged it as a possible identifier |
| `profile_suspicious` | Profiling flagged it as potentially suspicious |
| `constant_column` | One value in every row, so no signal |
| `no_values` | Entirely missing |
| `free_text` | Free text; a text encoder arrives later |
| `high_cardinality` | Too many distinct values to one-hot encode |
| `unsupported_type` | Type the pipeline cannot handle |

An excluded column can always be put back with `with_overrides` — the
configuration is a suggestion, not a verdict, and nothing is deleted from the
dataset.

## Configuration reference

| Field | Default | Purpose |
| --- | --- | --- |
| `target_column` | — | Column to predict; required |
| `numeric_columns` | inferred | Numeric feature group |
| `categorical_columns` | inferred | Categorical feature group |
| `boolean_columns` | inferred | Boolean feature group |
| `datetime_columns` | inferred | Datetime feature group |
| `identifier_columns` | inferred | Identifiers, kept out of the features |
| `excluded_columns` | inferred | Anything else deliberately excluded |
| `task_type` | from profile | `classification`, `regression` or `auto` |
| `scaling_strategy` | `standard` | `standard`, `minmax` or `none` |
| `numeric_imputation` | `median` | `median`, `mean` or `constant` |
| `categorical_imputation` | `most_frequent` | `most_frequent` or `constant` |
| `numeric_fill_value` | `0.0` | Used with constant numeric imputation |
| `categorical_fill_value` | `"Unknown"` | Used with constant categorical imputation |
| `add_missing_indicators` | `True` | Add a 0/1 feature marking imputed values |
| `datetime_components` | year, month, day, day_of_week | Components to extract |
| `max_categorical_cardinality` | `50` | Above this, a categorical is excluded |
| `max_classification_classes` | `20` | Distinct values still treated as classes |
| `test_size` | `0.2` | Fraction held out for testing |
| `random_state` | `42` | Seed, so runs are reproducible |

## Feature engineering

Deliberately conservative — only transformations that are safe on any tabular
dataset and stay readable:

- **Datetime components.** A timestamp is useless to most estimators; its year,
  month, day and day of week are ordinary numeric features.
- **Missing indicators.** A 0/1 feature marking which values were imputed, so
  "this was missing" stays available to the model. Indicators are created only
  for columns that had gaps in the *training* data, and they are not scaled.
- **One-hot encoding** with `handle_unknown="ignore"`, so a category never seen
  during training encodes as all zeros instead of raising at inference time.

No polynomial expansion, no automatic interactions, no domain-specific
features, and no resampling.

## Feature names

Names survive every transformation, which is what makes later explanations,
feature importance and LLM reasoning possible:

| Original | After preprocessing |
| --- | --- |
| `monthly_charges` | `monthly_charges` |
| `contract` | `contract_Month-to-month`, `contract_One-year`, `contract_Two-year` |
| `signup_date` | `signup_date_year`, `signup_date_month`, `signup_date_day`, `signup_date_day_of_week` |
| `age` (had gaps) | `age`, `missingindicator_age` |

## Train/test split

`evaluation/splitting.py` divides the data before anything is fitted.
`test_size` and `random_state` are configurable, and the same seed always
reproduces the same split.

Classification splits are stratified so class proportions survive on both
sides. Stratification is skipped — never silently — when it cannot apply, and
the reason is reported in `prepared.stratification_note`:

- the target is continuous (regression),
- the target has fewer than two classes,
- a class has only one row,
- the test half is too small to hold every class.

Cross-validation is not implemented. The splitting functions are small and
pure so a cross-validation strategy can reuse them later.

## Leakage prevention

Preprocessing is fitted on the training rows and on nothing else. The order of
operations in `prepare_dataset` is what enforces it:

1. validate the configuration against the dataset,
2. separate the target from the features,
3. split into train and test,
4. `fit_transform` the preprocessor on the **training features only**,
5. `transform` the test features with those already-learned statistics.

Consequences worth stating plainly:

- imputation statistics are the training medians/means/modes,
- scaler centring and scaling come from the training rows,
- one-hot categories are the categories present in training,
- missing indicators exist only for columns with gaps in training,
- the target is never passed to a feature transformer — the feature frame is
  built by *selecting* the configured feature columns, not by dropping the
  target, and `fit_transform` is called without `y`.

`tests/test_leakage.py` checks each of these directly. The decisive test
corrupts the feature values of the test rows and shows that neither the fitted
statistics nor the transformed training data move at all.

## Class imbalance

Measured, never corrected. `prepared.target_overall`, `target_train` and
`target_test` report class counts, percentages, majority and minority classes
and the imbalance ratio (or descriptive statistics for regression). No SMOTE,
no oversampling, no undersampling — the class distribution a caller sees is the
one the data has. Commit 5/6 can act on these numbers.

## Result object

`PreparedDataset` carries both the model-ready objects and the account of how
they were produced: `X_train`, `X_test`, `X_train_raw`, `X_test_raw`,
`y_train`, `y_test`, the fitted `preprocessor`, `feature_names`,
`feature_groups`, `selected_columns`, `excluded_columns`,
`identifier_columns`, `column_decisions`, row counts,
`rows_dropped_missing_target`, `stratified` and the target distributions.

The `_raw` frames are the untransformed source columns behind each split half.
They are what a full `Pipeline(preprocessing, estimator)` is fitted on, so a
trained model can accept raw feature rows rather than a transformed matrix.

`prepared.summary()` is the boundary for anything leaving the ML layer: it
returns only JSON-friendly values and deliberately omits the fitted estimator
and the data frames, so a fitted sklearn object never has to appear in an API
response.

---

## Model training

### Supported models

A small, deliberate suite rather than a long list. Every entry is a
`ModelDefinition` in `models/registry.py` carrying a stable identifier, a
display name, its task, its factory and its default parameters.

| Identifier | Model | Task | Notes |
| --- | --- | --- | --- |
| `logistic_regression` | Logistic Regression | classification | Linear baseline; readable coefficients |
| `random_forest_classifier` | Random Forest Classifier | classification | Robust with little tuning |
| `hist_gradient_boosting_classifier` | Histogram Gradient Boosting Classifier | classification | Usually strongest on tabular data |
| `linear_regression` | Linear Regression | regression | Ordinary least squares |
| `random_forest_regressor` | Random Forest Regressor | regression | Bagged trees |
| `hist_gradient_boosting_regressor` | Histogram Gradient Boosting Regressor | regression | Boosted trees |

The registry is immutable: `default_registry()` builds a fresh instance each
call and `registry.extend(definition)` returns a new registry. Adding XGBoost
or LightGBM later means appending a definition — `train_model` does not change.

### How training works

`train_model(prepared, spec)` validates the request against the registry and
the dataset's task, builds the estimator, wraps it behind the preprocessing,
fits, predicts and scores:

```
raw feature rows -> preprocessing -> estimator -> prediction
```

The trained artefact is a full `sklearn.pipeline.Pipeline` whose first step is
a fresh copy of the configured preprocessing, fitted here on the training rows
alone. A transformed matrix is never the only artefact, so the finished model
accepts the original columns — gaps and unseen categories included — and
applies exactly the transformations it was trained with.

`spec` may be a registry identifier for defaults, or a `ModelSpec`:

| Field | Purpose |
| --- | --- |
| `model_name` | Registry identifier |
| `task_type` | Optional; checked against the registry and the dataset |
| `hyperparameters` | Overrides on top of the model's defaults |
| `random_state` | Seed; defaults to the seed the dataset was split with |
| `primary_metric` | Metric this model is ranked by |

Invalid requests fail before any fitting, naming the valid options:
unknown model, incompatible task, unaccepted hyperparameter, unknown metric.
Hyperparameter *names* are checked against the estimator's own parameter list;
their *values* are left to scikit-learn, which validates them properly at fit
time and whose failure is re-raised as `ModelTrainingError`.

### Metrics

Every metric declares whether higher or lower is better, and a metric that
cannot be computed is reported as unavailable with the reason rather than
crashing or returning a meaningless number.

**Classification** — binary problems use `average="binary"` against a positive
class (the last label in sorted order, reported explicitly); three or more
classes use macro averaging, so no class dominates. The averaging used, the
class count, the class distribution and the confusion matrix all come back with
the scores, because a number like "F1 = 0.82" cannot be read without them.

| Metric | Direction | Notes |
| --- | --- | --- |
| `accuracy` | higher is better | Misleading on imbalanced data; never the only metric reported |
| `precision` | higher is better | |
| `recall` | higher is better | |
| `f1` | higher is better | Default primary metric |
| `roc_auc` | higher is better | Needs probabilities; one-vs-rest macro for multiclass |

ROC-AUC is skipped — with the reason — when the model exposes no
probabilities, when the test set holds a single class, or when it contains a
class the model never saw.

**Regression** — MAE, MSE and RMSE are errors, so **lower is better**; R² is a
share of explained variance, so **higher is better**, with 0 meaning "no better
than always predicting the mean".

| Metric | Direction | Notes |
| --- | --- | --- |
| `mae` | lower is better | Average absolute error, in the target's units |
| `mse` | lower is better | Punishes large mistakes |
| `rmse` | lower is better | Back in the target's units; default primary metric |
| `r2` | higher is better | Skipped when the target does not vary |

### Baseline comparison

A score means nothing without a reference. Every trained model is measured
against a deliberately naive one on the same test rows, through the same
pipeline and the same metric code:

- **classification** — always predict the majority training class
- **regression** — always predict the training mean

`absolute_improvement` and `relative_improvement` are signed so that
**positive always means better than the baseline**, whichever direction the
metric runs in. For RMSE, an improvement of 12.0 means 12.0 less error.

On a synthetic churn-style dataset (300 rows, majority class 55%):

```
baseline f1=0.710 accuracy=0.550

model                                      f1    acc  roc_auc  vs base
logistic_regression                     0.925  0.917    0.987   +0.216
random_forest_classifier                0.912  0.900    0.977   +0.202
hist_gradient_boosting_classifier       0.899  0.883    0.977   +0.189
```

The baseline's F1 of 0.710 is the point: without it, 0.925 has no scale.

---

## Cross-validation and model selection

### Why cross-validation

A single train/test split gives one number per model. That number carries the
luck of one particular division of the rows — swap a few rows between the
halves and the ranking can change. Worse, a test set used to *choose* a model
has been spent: the winner's score is then the best of several draws, which
flatters it. Reporting it as an estimate of future performance is optimistic,
and the more models compared, the more optimistic it gets.

Cross-validation fixes both problems by moving the choice onto the training
data. The training rows are divided into *k* folds; each fold takes a turn as
the validation set while the other *k−1* train the model. Every model is
scored *k* times, so the comparison rests on an average rather than a single
draw — and the test set is never opened.

> **Cross-validation selects the model; the held-out test set is reserved for
> the final evaluation.**

### The two splitters

| Task | Splitter | Why |
| --- | --- | --- |
| classification | `StratifiedKFold` | Every fold keeps the dataset's class proportions. Without it, an imbalanced dataset can produce a fold with almost none of the minority class, and a score that means nothing. |
| regression | `KFold` | A continuous target has no classes to balance. |

Both shuffle before splitting, using the dataset's seed, so the folds are
random with respect to row order but identical on every re-run.

The fold count is validated before anything runs. Fewer than two folds is not
cross-validation; more folds than rows is impossible; and for classification,
**a class with fewer members than folds** is refused with a clear error naming
the class and its size, rather than producing folds that quietly
misrepresent it.

### Mean and standard deviation

Each fold produces the full metric set for the task, and those are aggregated
per metric into a mean, a standard deviation, a minimum and a maximum.

The spread is the honest half of the result. A mean F1 of 0.84 with a spread of
0.01 is a stable model; the same mean with a spread of 0.09 is a model whose
score depends heavily on which rows it happened to see. The standard deviation
reported is the population spread over the folds that actually ran.

Classification runs also return a pooled confusion matrix — the fold matrices
summed. Because every training row is validated exactly once, that matrix
covers the whole training set.

### Ranking

`compare_models` takes a strategy:

| Strategy | Ranked by | Reads the test set? |
| --- | --- | --- |
| `holdout` (default, unchanged) | the model's score on the held-out test set | yes |
| `cross_validation` | the mean of the training folds | **no** |

Under cross-validation the comparison contains **no test-set numbers at all** —
no test metrics, no test baseline, no fitted model. They are absent rather than
merely unused, so selection cannot read them even by accident.

`select_best_model(comparison)` reads the primary metric's declared direction:
the **maximum** for a score metric such as F1 or R², the **minimum** for an
error metric such as RMSE. There is one source of truth for that direction —
the `MetricDefinition` from the metrics module — so no ranking code decides it
for itself. Models that failed, or that could not produce the primary metric,
are not candidates; if nothing is rankable it raises `NoSuccessfulModelError`
with the collected errors rather than returning `None`.

Each model runs inside its own error boundary. A model whose folds all fail is
recorded with its errors and ranked last; the others still run. A single fold
that fails does not abandon the run either — it is recorded, and the mean is
taken over the folds that succeeded.

The primary metric is configurable per run
(`compare_models(prepared, primary_metric="r2")`) or per model
(`ModelSpec(..., primary_metric="mae")`).

### Selecting, then measuring

`select_and_evaluate_best_model(prepared, folds=5)` runs the whole sequence:

1. cross-validate every candidate on the training rows,
2. rank by the mean of the folds and pick the winner,
3. retrain that winner on the **complete** training portion,
4. evaluate it **once** on the held-out test set, against the baseline.

Steps 1 and 2 never read the test set, so step 4 is the first and only time the
model meets that data — which is what makes the number an unbiased estimate.

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

The cross-validated mean (0.9403) sits slightly above the honest test score
(0.9254). That gap is exactly what a holdout-selected number hides.

`ModelSelectionResult.summary()` keeps the two apart by construction: a
`selection` section (`scored_on: "training_folds"`, `uses_test_data: false`)
and a `final_evaluation` section (`trained_on: "full_training_data"`,
`evaluated_on: "held_out_test_set"`, `is_unbiased: true`).

The holdout strategy still works and is still the default for
`compare_models`, so existing callers are unaffected. Selecting through it sets
`final_evaluation_is_unbiased` to `False`, because under holdout the numbers
that chose the model are the numbers being reported.

### Baselines under cross-validation

The naive baseline plays no part in choosing the winner — models are ranked
against each other on cross-validated scores. It reappears at the end, where
the selected model is compared against it on the untouched test set, so the
final figure still has a scale.

### Leakage prevention inside the folds

Every fold builds its own clone of the `Pipeline(preprocessing, estimator)` and
fits it on that fold's training rows alone:

```
training fold -> fit preprocessing -> fit model
validation fold -> transform -> predict -> score
```

The training data is never preprocessed as a whole before the folds. A
validation fold cannot contribute to the imputation values, scaler statistics
or encoder categories that are then applied to it.

`tests/test_cross_validation.py` proves this rather than asserting it: it
checks each fold's fitted imputer against the median of exactly the rows that
fold was given, and then corrupts the rows that fold 1 uses for validation —
fold 1's statistics are unchanged, while folds 2 and 3, which train on those
rows, both move.

`tests/test_selection.py` proves the other half: scrambling the test labels
leaves the cross-validated ranking, every fold mean and the winner exactly as
they were, while the final test score does change. Selection is blind to the
test set; the final measurement is the only thing that reads it.

### Result

`TrainedModel` carries the fitted `pipeline`, the `spec`, `metrics`,
`baseline`, `baseline_comparison`, `primary_metric`, `feature_names`,
`dataset` information and `training_seconds`. `trained.summary()` is the
boundary for anything leaving the ML layer — plain JSON-friendly values, with
the pipeline deliberately omitted, so a fitted estimator has no route into an
API response. `ModelComparison.as_table()` and `.summary()` do the same for a
comparison run.

### Programmatic API

Deterministic, structured functions, shaped so a later agent can call them as
tools. **No agent exists yet.**

| Function | Returns |
| --- | --- |
| `list_available_models(task_type=None)` | Serialisable records of every registered model |
| `get_model_spec(model_name, ...)` | A validated `ModelSpec` |
| `train_model(prepared, spec)` | A `TrainedModel` |
| `cross_validate_model(prepared, spec, folds=5)` | A `CrossValidationResult` |
| `compare_models(prepared, strategy=..., folds=...)` | A ranked `ModelComparison` |
| `select_best_model(comparison)` | The winning `ComparisonEntry` |
| `select_and_evaluate_best_model(prepared, ...)` | A `ModelSelectionResult` |

### Reproducibility

Estimators that accept a `random_state` are given one: the specification's if
set, otherwise the seed the dataset was split with, so a whole run is
reproducible from a single number. Estimators that take no seed — linear
regression — are never handed one. Repeated training on the same prepared
dataset gives identical predictions and identical metrics.

---

## Explainability

### What SHAP does

A trained model turns a row of features into a number. SHAP answers the
question "how much did each feature contribute to *that* number, rather than
to the model's usual answer?"

It works by comparing against a reference. The model has an average output over
some background set of rows — the **base value**. SHAP splits the difference
between this row's output and that base value across the features, and the
split has a useful guarantee: the contributions plus the base value add back up
to the prediction exactly. Nothing is left over, and nothing is invented.

```
base value  +  sum of every feature's contribution  =  the model's output
```

That property is what the tests check, rather than merely checking that numbers
came back.

### Global versus local

| | Question | Function | Answer |
| --- | --- | --- | --- |
| **Global** | What drives this model in general? | `explain_global` | One importance per feature, ranked |
| **Local** | Why did *this* row get *this* answer? | `explain_prediction` | One signed contribution per feature, ranked by size |

Global importance is the mean absolute SHAP value across many rows: how far
each feature moved the output on average, in either direction. Signs are
dropped deliberately — a feature that pushes hard both ways is influential, and
averaging signed values would hide it.

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

0.4448 + 0.1702 = 0.6150 — the base value plus the contributions is the
probability the model actually produced.

### Not causality

`increases_prediction` means the model's output was higher with this value than
it would have been at the reference. It does **not** mean that changing the
value in the real world would change the outcome. A feature can rank highly
because it stands in for something else the model never saw. Every result
carries this caveat in its `disclaimer` field, so it cannot be lost downstream.

### The pipeline boundary

A trained model is `Pipeline(preprocessing, estimator)`, and SHAP needs the
numbers the *estimator* saw — not the raw columns:

```
raw row -> already-fitted preprocessing -> transformed features -> estimator -> SHAP
```

So raw rows are pushed through the preprocessing step **exactly as it was
fitted during training**, and the transformed frame is handed to the explainer.
The original pipeline is left intact and still usable. Nothing is refitted,
which is why explanations cannot change a model — and why a set of deliberately
extreme rows can be explained without moving a single imputation value.

### Why feature names matter

Because the preprocessing step carries Commit 3's names, the explanation talks
about `contract_Month-to-month` and `missingindicator_age` rather than `x0` and
`x7`. That is the difference between a result a person can act on and a table
of numbers, and it is what makes the output usable by a future LLM layer.

### Which explainer, and when

There is no single SHAP explainer for every model, so the estimator's family
decides:

| Model family | Explainer | Needs background rows? |
| --- | --- | --- |
| Tree ensembles — random forest, gradient boosting, decision trees | `TreeExplainer` | no, it reads the trees |
| Linear models — logistic, linear, ridge, lasso | `LinearExplainer` | yes |
| Anything else | none | — |

Two structural checks catch models this package has not been told about: a
gradient-boosting library recognised by its module name, and any estimator
exposing `coef_`. Everything else is reported as unsupported with a reason
rather than being forced through the general-purpose kernel explainer, which is
slow enough on real data to be the wrong default.

### The fallback, and its limits

When no SHAP explainer suits a model, **global** importance can still be
measured by permutation importance: shuffle one feature's values and see how
far the model's score falls. The method is always named in the result, so a
permutation number is never mistaken for a SHAP one, and the reason SHAP was
skipped is recorded in the warnings.

Two honest limits come with it:

- **It needs the labels.** A score drop needs the right answers to measure
  against. Without `y_reference` the result is `status: unavailable` with that
  reason — nothing is invented. The model itself is untouched; permutation
  importance shuffles copies of the data.
- **It is global only.** It cannot say why one row got its prediction. A local
  explanation that SHAP cannot produce comes back as `status: unavailable`,
  with the prediction and probabilities still reported — those are facts about
  the model — but with no contributions filled in from the global method.

### Classification detail

For a **binary** problem the result names the predicted class, the explained
class and the positive class separately, so none has to be inferred. The
positive class follows the convention set in Commit 4 — the last of the
estimator's sorted classes. Some models produce one SHAP output per class;
others produce a single margin for the positive class, in which case explaining
the negative class is the exact negation of those values, and the result says
so in its warnings.

For a **multiclass** problem, local explanations are per class — the predicted
one by default, or any other via `target_class`. Global importances are
averaged over the classes, which the result states explicitly rather than
silently collapsing three classes into one number. Where a SHAP output shape
cannot be matched to the model's classes, the result is a structured
`unavailable` with the reason, never a guess.

### Sampling and limits

SHAP is not free, so the number of rows is capped and the excess sampled
deterministically:

| Setting | Default | Purpose |
| --- | --- | --- |
| `max_explanation_rows` | 500 | Rows SHAP values are computed over for global importance |
| `max_reference_rows` | 200 | Rows used as the background distribution |
| `permutation_repeats` | 10 | Shuffles per feature in the fallback |
| `random_state` | 42 | Seed for every sampling decision |

Sampling is never silent: `sample_count` reports how many rows were actually
used, and a warning records the reduction. The same seed gives the same rows in
the same order, so repeated explanations of the same model return identical
numbers.

### Programmatic API

Structured facts, not prose — the form the LLM layer receives and writes
sentences from. The retrieval layer in `rag/` indexes these findings so they
can be found and cited, and the agent's `explain_experiment` tool calls the
two functions below directly.

| Function | Returns |
| --- | --- |
| `explain_global(trained, X_reference, y_reference=None, ...)` | A `GlobalExplanation` |
| `explain_prediction(trained, row, background=None, ...)` | A `LocalExplanation` |
| `get_feature_importance(trained, X_reference, ...)` | Plain `{feature, importance, rank}` records |

`summary()` on either result is JSON-safe by construction: no explainer object,
no numpy array, no estimator, no DataFrame.

### Which rows to explain

The training features are the natural reference — they are what the model
learned from, so they describe the behaviour it actually acquired, and using
them keeps the held-out test set genuinely untouched. Explaining test rows is
also sound and answers a slightly different question; no target is involved and
nothing is fitted, so neither choice leaks. What the reference data does is
define the distribution each row is compared against, which is why it should be
representative.

---

## Experiment tracking

### Why it exists

Everything above this section is ephemeral. A pipeline run prepares data,
cross-validates half a dozen models, measures a winner once on untouched data
and explains it — and then the process exits and all of it is gone. The next
run cannot say whether it is better than the last one, whether it even used the
same data, or whether a change to the configuration helped.

A record fixes that. Each complete pass leaves behind a small, readable
document: which dataset, prepared how, which models were considered, which won
and by what rule, how it scored on data it had never seen, and what the
explanations said. That makes three things possible that were not before —
comparing runs, reproducing one, and later, letting an assistant read the
history and answer questions about it.

> **MLflow is NOT implemented yet.** Neither is PostgreSQL, nor any other
> database. Records are JSON files written by this package into a local
> directory. That is a deliberate intermediate step, described below.

### What a record contains

`ExperimentRun` is the whole record. It is a frozen dataclass of sections, each
of which is a summary of something the pipeline already produced:

| Section | Holds |
| --- | --- |
| identity | `schema_version`, `experiment_id`, `configuration_hash`, `created_at`, `name`, `description`, `tags` |
| `dataset` | fingerprint and its algorithm, row and column counts, column names and dtypes, target column, task type, source format, data-quality findings |
| `preprocessing` | the configuration, feature groups, selected and excluded columns, per-column decisions, transformed feature names, split sizes, seed, stratification |
| `selection` | strategy, folds, candidate models, every candidate's score, the winner, the selection score and its spread, what it was scored on, and whether test data was involved |
| `evaluation` | the final metrics on the untouched test set, the baseline and the improvement over it, classification detail, and whether the measurement is unbiased |
| `explainability` | method, explainer, ranked feature importances, sample and feature counts, and any warnings — `None` when nothing was explained |
| `environment` | Python version, platform, library versions, random seed |

`headline()` returns a one-line dict for listings; `to_dict()` returns the full
record; `from_dict()` reads one back with validation.

**What a record never contains.** Not the dataset — only its shape and
fingerprint. Not the fitted `Pipeline`, and not the SHAP explainer. Those are
model artefacts, not history: they are large, they are pickles, and they would
turn a readable document into an opaque blob. The serialiser refuses them
outright rather than trusting callers to remember (see below).

### Identifying a dataset by content

```python
from ml.experiments import fingerprint_dataset

fingerprint_dataset(frame).value   # '86494cff7a45cb7f'
```

A filename is not an identity. The same table gets exported twice under
different names, moved between folders, re-saved from Excel — and a run
recorded against `customers_final_v2.csv` tells you nothing about whether it
used the same rows as the one before it.

So the fingerprint is a SHA-256 digest over the *content*: the column names and
dtypes, then `pd.util.hash_pandas_object` over each column's values in order.
It is truncated to 16 hex characters, which is plenty to distinguish the
datasets one project accumulates.

What changes it and what does not is a design decision, not an accident:

- an edited cell, an added row, a renamed column — **different** dataset
- the row *order* — **different**, because it changes the train/test split, so
  it really is a different experiment
- the DataFrame index — **the same**, since an index is an artefact of loading
- the file path, name, or format it was read from — **the same**

No timestamp and no random value goes into it, which is what lets two runs of
the same data be recognised as such.

### Experiment identifiers

```
exp_84a8d53a1f5f_20260828T134042Z_b8c1
    └ configuration  └ UTC time     └ random suffix
```

The middle segment sorts readably and the last one keeps two runs of one
configuration in the same second from colliding — but the identifier is not
*only* a timestamp, and it is not a bare UUID. The first segment is a hash of
the configuration: dataset fingerprint, target, task, preprocessing settings,
selected and excluded columns, split size, seed, selection strategy, folds,
primary metric and the candidate models offered.

Only inputs are hashed. Nothing the run *concluded* — which model won, what it
scored — goes in, so re-running an unchanged setup produces the same
configuration hash. That is what makes "have I run this before?" a question the
store can answer, and what makes a reproducibility claim checkable rather than
merely asserted.

### JSON safety

`json.dumps(obj.__dict__)` fails on the first `numpy.float64`, and where it
does not fail it quietly writes something wrong. `serialization.py` converts
explicitly instead: numpy scalars and arrays, pandas Series, Index, Timestamp
and `NaT`, datetimes and dates to ISO 8601, enums to their values, dataclasses
and any object with `as_dict()`/`summary()` through it, and non-finite floats
(`NaN`, `inf`) to `null` — because JSON has no way to write them and
`allow_nan=False` on the writer catches any that escape.

It also refuses, loudly, four things:

- **sklearn or SHAP objects** — a model artefact is not experiment history
- **DataFrames** — records hold metadata, not data
- **sequences over 10 000 items** — a transformed feature matrix is not a field
- **cyclic or 32-deep structures** — recursion errors are not error messages

Each raises `SerializationError` naming what was refused.

### Storage behind an interface

`ExperimentStore` is a `Protocol` — `save`, `get`, `exists`, `list`, `delete` —
and every caller depends on it rather than on files. `LocalExperimentStore` is
the only implementation today:

```
ml/experiments/runs/
└── exp_84a8d53a1f5f_20260828T134042Z_b8c1/
    └── experiment.json
```

One directory per run, one readable JSON file inside it. Two failure modes get
explicit attention.

**A half-written file.** The record is serialised *before* the filesystem is
touched, so a record that cannot be written leaves nothing behind — not even an
empty directory. The write then goes to a temporary file in the same directory,
is flushed and `fsync`ed, and is moved into place with `os.replace`, which is
atomic. A reader sees the previous record or the complete new one, never a
fragment.

**A corrupted record.** `get` raises: a caller who names a run deserves to know
it is broken. `list` skips it with a logged warning, so one bad file cannot
hide an entire history, and `verify()` returns `(experiment_id, problem)` pairs
for everything unreadable.

**Path traversal.** An experiment id becomes a directory name, so it is the one
place an outside string touches the filesystem, and it is checked twice.
`validate_experiment_id` accepts only `[A-Za-z0-9][A-Za-z0-9_-]{0,127}` —
which excludes `/`, `\`, `..`, `.`, absolute paths, spaces and the empty string
— and the resolved path is then confirmed to lie inside the store root. Either
check would do; both are cheap.

### Querying history

```python
from ml.experiments import ExperimentQuery, ExperimentSortKey, LocalExperimentStore

store = LocalExperimentStore("ml/experiments/runs")
store.save(run)

store.list(ExperimentQuery(
    dataset_fingerprint="86494cff7a45cb7f",
    task_type="classification",
    tags=("baseline",),
    sort_by=ExperimentSortKey.PRIMARY_METRIC,
    descending=True,
    limit=5,
))
```

Filters — fingerprint, target column, task type, model, strategy, primary
metric, tags — are all optional and combine with "and". Sorting is by creation
time, primary metric or model name. `descending=True` means *best or newest
first*, and for the metric key "best" reads the metric's own declared
direction: the largest F1, but the smallest RMSE.

### Comparing runs

```python
from ml.experiments import compare_experiments

comparison = compare_experiments(store.list(ExperimentQuery(task_type="classification")))
print(comparison.as_text())
comparison.best().selected_model
```

Ranking only means something between runs judged the same way, so
`compare_experiments` refuses a set that mixes metrics or tasks with
`IncomparableExperimentsError`. **RMSE is never ranked against F1** — the
comparison establishes one shared metric first, reads its direction from the
same `MetricDefinition` the selection code uses, and orders accordingly.
Runs with no score sort last rather than winning by accident. The result
renders as a table of plain values, a JSON-safe `summary()`, or text.

### Building a record

`create_experiment_run` composes what earlier commits already produce — it
computes nothing on its own:

```python
from ml.experiments import LocalExperimentStore, create_experiment_run

run = create_experiment_run(
    frame,                       # the dataset, for its fingerprint only
    prepared,                    # PreparedDataset (Commit 3)
    outcome,                     # ModelSelectionResult (Commit 5)
    name="renewal baseline",
    explanation=explanation,     # GlobalExplanation (Commit 6), optional
    profile=profile,             # dataset profile (Commit 2), optional
    tags=("baseline",),
    source_format="csv",
)
LocalExperimentStore().save(run)
```

Feature importances are capped (50 by default) so a wide dataset cannot inflate
a record; a typical run serialises to roughly 6 KiB.

### Schema versioning

Every record carries `"schema_version": "1.0"`. Reading one written under an
unknown version raises `UnsupportedSchemaVersionError` rather than
half-interpreting it, and a record with no version is refused outright. Missing
required fields, wrong types and unparseable timestamps each raise
`InvalidExperimentRecordError` naming the field. When the schema changes, the
version goes up and the reader gains a migration path — that is the whole point
of writing it down now, while there is only one version.

### Reproducibility metadata

`environment` records the Python version, the platform string, the versions of
pandas, numpy, scikit-learn and shap, and the random seed. That plus the
configuration hash is what a reproduction attempt needs.

It records nothing identifying: no hostname, no username, no file paths, no
environment variables. **No API key, token or password can reach a record**,
because nothing reads the environment in the first place.

### Why local JSON, and what replaces it

This is not the final architecture and is not pretending to be. Local JSON was
chosen because it has no server to run, no schema migration to manage, no
dependency to install, and a record can be opened in an editor when something
looks wrong — which is the right trade while the pipeline itself is still
changing shape.

It will not stay. It has no concurrent-writer story, `list` reads every record
from disk, and there is no shared history between machines. The replacement —
MLflow, a PostgreSQL table, or both — implements `ExperimentStore` and nothing
above it changes. **Neither is implemented today.**

Stored runs are the raw material for the retrieval layer in `rag/`: each record
is a small, self-describing document with a stable identifier and readable
field names, which is what makes it retrievable. `rag/ingestion/experiments.py`
renders one into structured text and indexes it, so a question like "which
model won on this dataset" can be answered from history with a citation back
to the run.

The dependency runs one way — `ml/experiments → rag/ingestion → rag/retrieval`.
Nothing in this package imports `rag/`, so experiments can be recorded with no
index present, and the index can be rebuilt from the store at any time.
**No vector database or embedding is implemented here**, and none is needed:
the retrieval layer owns all of that.

## Model artifacts and prediction

`ml/artifacts/` persists the winner of a run and predicts with it later.

**There is one object to persist, and it already exists.**
`TrainedModel.pipeline` is a fitted `Pipeline(preprocessing, estimator)` that
accepts **raw feature rows** — the same object cross-validation scored and the
same one the held-out measurement was taken through. So persistence is that
object written out, and prediction is that object called. Nothing is
reassembled from the record, and **nothing is re-fitted**; a test proves the
second half by making `ColumnTransformer.fit`, `fit_transform` and
`Pipeline.fit` raise for the duration of a prediction.

```python
from ml.artifacts import LocalModelArtifactStore, build_frame, build_metadata, predict

store = LocalModelArtifactStore(Path("ml/experiments/models"))
store.save(
    experiment_id,
    selection.final_model.pipeline,
    build_metadata(experiment_id=experiment_id, prepared=prepared, selection=selection),
)

loaded = store.load(experiment_id)                       # -> LoadedModel
frame = build_frame([{"tenure_months": 30}], loaded.metadata)
result = predict(loaded, frame)                          # -> PredictionResult
```

An artifact is a directory named for the experiment holding exactly two files:
the pipeline, and a manifest. The manifest records the schema version, the
feature list (name, kind, dtype), the class labels, the target, the row counts,
the primary metric and its value, the environment, and the model file's
SHA-256. **No dataset row is written to either file.**

`ModelArtifactStore` is a `Protocol`; `LocalModelArtifactStore` is the
filesystem implementation. A model is written after `os.replace`, so an
interrupted save leaves no half-file, and the manifest is written second — an
artifact without its manifest is unreadable rather than half-trusted.

**Three states, decided once.** `store.status(experiment_id)` returns
`available`, `not_available` or `corrupted`, and it is the only code that
decides which — `exists()` is a reading of it, and both prediction endpoints
answer from it, so two callers cannot disagree about the same directory. The
check is cheap and shallow on purpose: the manifest is parsed and validated and
the model file is checked for presence and for the size the manifest recorded,
with nothing unpickled, so asking costs a JSON read and a `stat`. The deep
checks — the SHA-256 digest and that the object really is a `Pipeline` — need
the file itself and belong to `load`.

`not_available` and `corrupted` are kept apart because their fixes differ, and
`reason_code` carries which kind: `no_artifact`, `manifest_unreadable`,
`manifest_invalid`, `unsupported_schema_version`, `model_file_missing`,
`model_file_truncated`, `model_too_large`. The codes are stable strings; the
sentence a person reads is composed in the backend, which is the layer that
knows there is a person.

**Validation is strict in both directions.** Every trained-on feature must be
present in a record; one the model was *not* trained on is refused rather than
ignored, because the fitted `ColumnTransformer` uses `remainder="drop"` and
would silently discard a misspelt column, producing a confident prediction made
without that value. `None` and `""` are missing values — the imputation was
fitted for exactly that — and anything else is coerced to the column's trained
kind or rejected with the feature named.

**Loading executes code, so what may be loaded is the control.** A `joblib`
file is a pickle. No path ever comes from a caller's input: the store takes an
experiment id, validates it as an id, re-checks the resolved directory against
its root, and uses a constant filename that is never read from the manifest.
The manifest is parsed and its version checked *before* anything is unpickled;
then the digest is verified, a size ceiling applies, and the result must be a
`Pipeline`. Only artifacts this application wrote are ever loaded — **there is
no support for third-party or user-supplied model files**, and nothing in this
package accepts one.

## Errors

Plain Python exceptions with no HTTP meaning — the API layer translates them
when the ML layer is eventually exposed over HTTP.

| Error | Raised when |
| --- | --- |
| `MissingTargetError` | The target is blank or not in the dataset |
| `TargetLeakageError` | The target was also assigned to a feature group |
| `UnknownColumnError` | A configured column is not in the dataset |
| `DuplicateColumnAssignmentError` | A column is in two feature groups |
| `EmptyFeatureSetError` | No feature columns remain |
| `ConfigurationError` | An invalid strategy or split parameter |
| `InsufficientDataError` | Too few labelled rows to split, or an empty split half |
| `InvalidFoldCountError` | Fewer than two folds, more folds than rows, or a class smaller than the fold count |
| `UnknownModelError` | The model is not in the registry |
| `IncompatibleTaskError` | The model does not solve the dataset's task |
| `InvalidHyperparameterError` | A hyperparameter the estimator does not accept |
| `InvalidMetricError` | The metric does not exist for the task |
| `ModelTrainingError` | The estimator failed while fitting or predicting |
| `NoSuccessfulModelError` | No model in a comparison produced a rankable score |
| `InvalidTrainedModelError` | The object to explain is not a fitted trained model |
| `MissingFeatureColumnsError` | The data to explain lacks a fitted feature column |
| `EmptyExplanationDataError` | There are no rows to explain |
| `InvalidExplanationRowError` | A local explanation was given other than one row |
| `ExplainabilityError` | A linear model was given no background rows, or an unknown target class |
| `SerializationError` | A record holds something that cannot or must not be written as JSON |
| `InvalidExperimentIdError` | An experiment id is malformed, or would escape the store directory |
| `ExperimentNotFoundError` | Nothing is stored under that experiment id |
| `MalformedExperimentError` | A stored record is not valid JSON |
| `UnsupportedSchemaVersionError` | A record's schema version is missing or unreadable by this code |
| `InvalidExperimentRecordError` | A record is missing a field, or a field has the wrong type |
| `IncomparableExperimentsError` | Runs judged by different metrics or tasks were compared |
| `ModelArtifactNotFoundError` | No model artifact is stored for that experiment |
| `ModelArtifactUnreadableError` | A stored artifact is missing a file, fails its checksum, exceeds the size ceiling, declares an unknown schema version, or does not deserialise to a `Pipeline` |
| `ModelArtifactError` | A model could not be written |
| `PredictionInputError` | A record is missing a feature, carries one the model was not trained on, holds a value that cannot be read as that column's kind, or the batch is empty or too large |
| `PredictionError` | The fitted pipeline raised while predicting |

An estimator no explainer supports is **not** an error: it produces a
structured result with a reason.

## Setup and tests

```bash
pip install -r backend/requirements-dev.txt -r ml/requirements.txt \
            -r rag/requirements.txt -r llm/requirements.txt
pytest              # from the repository root: backend, ML and retrieval suites
pytest ml/tests     # ML layer only
```

Every test builds its data in memory with a fixed seed. Nothing reads an
external dataset or touches the network.

## Current limitations

**Modelling**

- **No hyperparameter optimisation.** Models run on registry defaults plus any
  hyperparameters a caller supplies. There is no search, no tuning, no Optuna.
- **Model persistence is one local artifact per experiment.** The winning
  `Pipeline` is written to an application-owned directory after a successful
  run, and `ml/artifacts/` loads it again to predict. There is no registry, no
  versioning beyond one artifact per experiment, no promotion or rollback, and
  no sharing between processes on different machines. `joblib` means the file
  is portable only to a compatible scikit-learn.

- **The SHAP explainer is not persisted.** Only the pipeline is, so a stored
  run can predict but cannot produce a *new* per-row explanation of itself.
  This is what the agent's `explain_experiment` tool reports honestly rather
  than working around: an experiment run in the current process can still be
  explained live, because its explainer has not been collected yet; an older
  one can only report the importances recorded when it ran, and a per-row
  explanation of it is answered `unavailable` with
  `reason: fitted_model_not_persisted`.

  Over HTTP that surfaces as a `partial` result from
  `POST /api/v1/agent/ask`: the experiment is reported in full and the missing
  explanation is stated as a warning, never filled in with an invented SHAP
  value. See `agent/README.md`.
- **Six models only.** No XGBoost or LightGBM yet; the registry is designed so
  adding them is one definition.
- Cross-validation is plain k-fold. There is no repeated k-fold, no
  group-aware or time-series splitting, and no nested cross-validation — the
  last of which would matter if hyperparameters were being tuned, which they
  are not.
- The cross-validated mean is itself an estimate with its own uncertainty. A
  gap between two models much smaller than their standard deviations is not
  evidence that one is better; the spread is reported so that can be judged.
- Under the `holdout` strategy, selection and final evaluation are the same
  measurement, so the reported score is optimistic. The result says so
  (`final_evaluation_is_unbiased` is `False`); prefer cross-validation.
- The binary positive class is the last label in sorted order. It is reported
  explicitly, but it is a convention rather than a choice the caller makes.
- `training_seconds` is wall-clock time on the machine that ran it — useful for
  relative comparison, not a benchmark.

**Explainability**

- Only tree and linear families have a SHAP explainer. Anything else gets
  permutation importance for global questions and nothing for local ones; the
  kernel explainer, which would cover everything, is too slow to enable by
  default.
- Contributions are on the model's own output scale — probabilities for a
  random forest, log-odds for a linear or boosted classifier — so the numbers
  from two model families are not directly comparable.
- Global importances describe correlation with the model's output, not
  causality, and a feature can rank highly by standing in for something else.
- One-hot columns are explained separately (`segment_retail`,
  `segment_business`), not rolled back up into a single `segment` importance.
- Multiclass global importances are averaged over classes; there is no
  per-class global ranking.
- SHAP interaction values, dependence plots and any visual output are not
  implemented — this layer returns structured numbers only.

**Experiment tracking**

- **MLflow is not implemented.** Neither is PostgreSQL or any other database.
  The only store writes JSON files locally.
- The local store has no locking, so two processes writing the *same*
  experiment id concurrently is undefined — though the atomic rename means
  neither leaves a truncated file.
- `list` reads every record from disk on each call. That is fine for the
  directory sizes a project accumulates by hand and would not be for thousands.
- History is per machine and per directory; nothing is shared or synchronised.
- A record cannot rebuild the model it describes, because no artefact is
  stored. Reproduction means re-running the recorded configuration.
- Comparison is single-metric by design and refuses mixed metrics rather than
  inventing a common scale.
- Deleting a run deletes its directory; there is no soft delete or history of
  deletions.

**Preprocessing**

- Text columns are excluded rather than encoded; a text representation comes later.
- Categorical columns above `max_categorical_cardinality` are excluded rather than grouped or target-encoded.
- One-hot encoding keeps every category (no `drop="first"`), which suits tree models and explanations but is collinear for plain linear models.
- Datetime components are treated as plain numbers; no cyclical encoding.
- Rows with a missing target are removed, since a supervised model cannot use them. The count is reported in the result.
- With `task_type="auto"` and no profile, whether a target is discrete is decided from its dtype and distinct count — supply the task explicitly when that matters.
- The whole dataset is held in memory; there is no out-of-core path.

## Future model expansion

Adding a model is adding a `ModelDefinition`:

```python
registry = default_registry().extend(
    ModelDefinition(
        identifier="xgboost_classifier",
        display_name="XGBoost Classifier",
        task_type=TaskType.CLASSIFICATION,
        factory=XGBClassifier,
        default_parameters={"n_estimators": 300},
        supports_random_state=True,
        supports_probabilities=True,
    )
)
compare_models(prepared, registry=registry)
```

`train_model`, the metrics, the baselines, the ranking and the result objects
all work unchanged, because none of them knows which estimators exist.
