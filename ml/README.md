# ML Copilot — ML Layer

Preprocessing and feature engineering: everything between a profiled dataset
and model training. Model training itself is **not** implemented yet.

## Format-agnostic by design

This package accepts a **standardised `pandas.DataFrame`** and nothing else. It
never sees a file, a path, an upload, or any CSV-specific object — passing
anything other than a DataFrame raises an error rather than being coerced.

```
ingestion adapter  ->  DataFrame  ->  profiling  ->  configuration  ->  preprocessing  ->  training
   (CSV today;                                                                            (Commit 4)
    Excel, JSON,
    Parquet, SQL,
    API planned)
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
│   └── splitting.py     Train/test split and target-distribution reporting
├── models/              Empty — estimators arrive in a later commit
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
prepared.y_train        # untouched target values
prepared.feature_names  # names after encoding and expansion
prepared.summary()      # JSON-friendly description of the whole run
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
they were produced: `X_train`, `X_test`, `y_train`, `y_test`, the fitted
`preprocessor`, `feature_names`, `feature_groups`, `selected_columns`,
`excluded_columns`, `identifier_columns`, `column_decisions`, row counts,
`rows_dropped_missing_target`, `stratified` and the target distributions.

`prepared.summary()` is the boundary for anything leaving the ML layer: it
returns only JSON-friendly values and deliberately omits the fitted estimator
and the data frames, so a fitted sklearn object never has to appear in an API
response.

## Errors

Plain Python exceptions with no HTTP meaning — the API layer translates them
when preprocessing is eventually exposed over HTTP.

| Error | Raised when |
| --- | --- |
| `MissingTargetError` | The target is blank or not in the dataset |
| `TargetLeakageError` | The target was also assigned to a feature group |
| `UnknownColumnError` | A configured column is not in the dataset |
| `DuplicateColumnAssignmentError` | A column is in two feature groups |
| `EmptyFeatureSetError` | No feature columns remain |
| `ConfigurationError` | An invalid strategy or split parameter |
| `InsufficientDataError` | Too few labelled rows to split |

## Setup and tests

```bash
pip install -r backend/requirements.txt -r ml/requirements.txt
pytest              # from the repository root: runs the backend and ML suites
pytest ml/tests     # ML layer only
```

Every test builds its data in memory with a fixed seed. Nothing reads an
external dataset or touches the network.

## Current limitations

- Text columns are excluded rather than encoded; a text representation comes later.
- Categorical columns above `max_categorical_cardinality` are excluded rather than grouped or target-encoded.
- One-hot encoding keeps every category (no `drop="first"`), which suits tree models and explanations but is collinear for plain linear models.
- Datetime components are treated as plain numbers; no cyclical encoding.
- Rows with a missing target are removed, since a supervised model cannot use them. The count is reported in the result.
- With `task_type="auto"` and no profile, whether a target is discrete is decided from its dtype and distinct count — supply the task explicitly when that matters.
- The whole dataset is held in memory; there is no out-of-core path.
