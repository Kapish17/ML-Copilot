# ML Copilot — ML Layer

Everything between a profiled dataset and a chosen model: feature
configuration, leakage-safe preprocessing, model training, evaluation, baseline
comparison and model selection.

**Not implemented:** hyperparameter optimisation, explainability (SHAP),
experiment tracking (MLflow), model persistence, LLM/RAG/agent integration.
Trained models live in memory for the lifetime of the process.

## Format-agnostic by design

This package accepts a **standardised `pandas.DataFrame`** and nothing else. It
never sees a file, a path, an upload, or any CSV-specific object — passing
anything other than a DataFrame raises an error rather than being coerced.

```
ingestion adapter -> DataFrame -> profiling -> configuration -> preprocessing
   (CSV today;                                                        |
    Excel, JSON,                                                      v
    Parquet, SQL,                        model registry -> training -> evaluation
    API planned)                                                      |
                                                                      v
                                              comparison -> best model
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
│   ├── splitting.py     Train/test split and target-distribution reporting
│   └── metrics.py       Task metrics, their directions, the primary metric
├── models/
│   ├── registry.py      Which estimators exist and how to build them
│   ├── spec.py          The validated request for one training run
│   ├── baselines.py     Naive reference models and the improvement over them
│   ├── training.py      Fitting Pipeline(preprocessing, estimator)
│   ├── comparison.py    Running several models and ranking them
│   └── result.py        TrainedModel, the outcome of one run
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

Then train and choose a model:

```python
from ml import compare_models, select_best_model, train_model

trained = train_model(prepared, "random_forest_classifier")
trained.predict(raw_rows)          # raw columns in, predictions out
trained.summary()                  # JSON-friendly, no sklearn objects

comparison = compare_models(prepared)
best = select_best_model(comparison)
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

### Comparison and selection

`compare_models(prepared)` trains every registered model for the dataset's
task, sharing one baseline. Each model runs inside its own error boundary, so a
model that fails is recorded with its error and the rest still run — one broken
estimator never sinks the run.

`select_best_model(comparison)` returns the winner, reading the primary
metric's declared direction: the **maximum** for a score metric, the
**minimum** for an error metric. Models that failed, or that could not produce
the primary metric, are not candidates; if nothing is rankable it raises
`NoSuccessfulModelError` with the collected errors rather than returning
`None`.

The primary metric is configurable per run
(`compare_models(prepared, primary_metric="r2")`) or per model
(`ModelSpec(..., primary_metric="mae")`).

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
| `compare_models(prepared, ...)` | A ranked `ModelComparison` |
| `select_best_model(comparison)` | The winning `ComparisonEntry` |

### Reproducibility

Estimators that accept a `random_state` are given one: the specification's if
set, otherwise the seed the dataset was split with, so a whole run is
reproducible from a single number. Estimators that take no seed — linear
regression — are never handed one. Repeated training on the same prepared
dataset gives identical predictions and identical metrics.

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
| `UnknownModelError` | The model is not in the registry |
| `IncompatibleTaskError` | The model does not solve the dataset's task |
| `InvalidHyperparameterError` | A hyperparameter the estimator does not accept |
| `InvalidMetricError` | The metric does not exist for the task |
| `ModelTrainingError` | The estimator failed while fitting or predicting |
| `NoSuccessfulModelError` | No model in a comparison produced a rankable score |

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
- **No cross-validation.** Scores come from a single held-out test set, so they
  carry the variance of one split. The splitting functions are written so a
  cross-validation strategy can reuse them.
- **No model persistence.** Trained models live in memory for the lifetime of
  the process; nothing is written to disk or to a registry.
- **No explainability.** Feature names are preserved for it, but SHAP and
  feature importance are not implemented.
- **Six models only.** No XGBoost or LightGBM yet; the registry is designed so
  adding them is one definition.
- The test set is used for the reported metrics *and* for choosing the best
  model, which optimistically biases the winner's score. A separate validation
  split belongs with cross-validation.
- The binary positive class is the last label in sorted order. It is reported
  explicitly, but it is a convention rather than a choice the caller makes.
- `training_seconds` is wall-clock time on the machine that ran it — useful for
  relative comparison, not a benchmark.

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
