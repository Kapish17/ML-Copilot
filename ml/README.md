# ML Copilot — ML Layer

Everything between a profiled dataset and an explained model: feature
configuration, leakage-safe preprocessing, model training, evaluation, baseline
comparison, cross-validated model selection, a single final measurement on
untouched data, and SHAP explanations of what the chosen model is doing.

> **Cross-validation selects the model; the held-out test set is reserved for
> the final evaluation.**
>
> **Explanations describe model behaviour and associations; they do not
> establish causal relationships.**

**Not implemented:** hyperparameter optimisation (no Optuna), experiment
tracking (no MLflow), model persistence, XGBoost or LightGBM, and any LLM, RAG
or agent integration. Trained models live in memory for the lifetime of the
process.

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

Structured facts, not prose — the form a future LLM layer would receive and
write sentences from. **No LLM, RAG or agent exists yet.**

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

An estimator no explainer supports is **not** an error: it produces a
structured result with a reason.

## Setup and tests

```bash
pip install -r backend/requirements.txt -r ml/requirements.txt
pytest              # from the repository root: runs the backend and ML suites
pytest ml/tests     # ML layer only
```

Every test builds its data in memory with a fixed seed. Nothing reads an
external dataset or touches the network.

## Current limitations

**Modelling**

- **No hyperparameter optimisation.** Models run on registry defaults plus any
  hyperparameters a caller supplies. There is no search, no tuning, no Optuna.
- **No model persistence.** Trained models live in memory for the lifetime of
  the process; nothing is written to disk or to a registry.
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
