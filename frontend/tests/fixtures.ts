/**
 * Fixture payloads, copied from what the real backend actually returns.
 *
 * Every object here was taken from a live response of the Commit 15 backend
 * rather than written from the schema, because the two differ in exactly the
 * places that matter: which fields are present, which are `null`, and what a
 * confusion matrix or a baseline comparison really looks like. A fixture
 * invented from a type would let a component pass a test and fail on real data.
 */

import type {
  AgentAnswer,
  AgentStatusResponse,
  AskResponse,
  DatasetProfile,
  ExperimentComparison,
  ExperimentListResponse,
  ExperimentRunResponse,
  KnowledgeStatus,
  SearchResponse,
} from "@/lib/api/types";

export const CLASSIFICATION_PROFILE: DatasetProfile = {
  filename: "customers.csv",
  source_format: "csv",
  generated_at: "2026-09-02T05:45:17.275714Z",
  dataset: {
    row_count: 180,
    column_count: 4,
    memory_usage_bytes: 25181,
    duplicate_row_count: 48,
    duplicate_row_percentage: 26.6667,
    missing_cell_count: 0,
    missing_cell_percentage: 0,
    column_type_counts: { integer: 2, categorical: 2 },
  },
  columns: [
    {
      name: "income",
      dtype: "int64",
      inferred_type: "integer",
      non_null_count: 180,
      missing_count: 0,
      missing_percentage: 0,
      unique_count: 40,
      unique_percentage: 22.2222,
      is_constant: false,
      numeric_stats: {
        mean: 43355.5555,
        median: 43600,
        std: 7418.0982,
        minimum: 30400,
        maximum: 57200,
        q1: 37400,
        q3: 48600,
        zero_count: 0,
        negative_count: 0,
      },
      datetime_stats: null,
      categorical_stats: null,
    },
    {
      name: "segment",
      dtype: "str",
      inferred_type: "categorical",
      non_null_count: 180,
      missing_count: 0,
      missing_percentage: 0,
      unique_count: 2,
      unique_percentage: 1.1111,
      is_constant: false,
      numeric_stats: null,
      datetime_stats: null,
      categorical_stats: {
        top_values: [
          { value: "retail", count: 120, percentage: 66.6667 },
          { value: "business", count: 60, percentage: 33.3333 },
        ],
        truncated: false,
      },
    },
    {
      name: "renewed",
      dtype: "str",
      inferred_type: "categorical",
      non_null_count: 180,
      missing_count: 0,
      missing_percentage: 0,
      unique_count: 2,
      unique_percentage: 1.1111,
      is_constant: false,
      numeric_stats: null,
      datetime_stats: null,
      categorical_stats: {
        top_values: [
          { value: "no", count: 91, percentage: 50.5556 },
          { value: "yes", count: 89, percentage: 49.4444 },
        ],
        truncated: false,
      },
    },
  ],
  quality: {
    issue_count: 2,
    issues: [
      {
        code: "duplicate_rows",
        severity: "warning",
        message: "48 of 180 rows are exact duplicates of an earlier row.",
        columns: [],
        details: { duplicate_row_count: 48, duplicate_row_percentage: 26.6667 },
      },
      {
        code: "high_missing",
        severity: "critical",
        message: "Column notes is 82% missing.",
        columns: ["notes"],
        details: { missing_percentage: 82 },
      },
    ],
  },
  target: {
    name: "renewed",
    dtype: "str",
    inferred_type: "categorical",
    missing_count: 0,
    missing_percentage: 0,
    task_suggestion: "classification",
    task_reason: "The target is categorical with 2 distinct values.",
    distribution: [
      { value: "no", count: 91, percentage: 50.5556 },
      { value: "yes", count: 89, percentage: 49.4444 },
    ],
    class_balance: {
      class_count: 2,
      majority_class: "no",
      majority_percentage: 50.5556,
      minority_class: "yes",
      minority_percentage: 49.4444,
      is_imbalanced: false,
    },
    numeric_stats: null,
  },
};

/** The same dataset uploaded as a workbook: identical below `source_format`. */
export const XLSX_PROFILE: DatasetProfile = {
  ...CLASSIFICATION_PROFILE,
  filename: "customers.xlsx",
  source_format: "xlsx",
};

export const JSON_PROFILE: DatasetProfile = {
  ...CLASSIFICATION_PROFILE,
  filename: "customers.json",
  source_format: "json",
};

export const CLASSIFICATION_RUN: ExperimentRunResponse = {
  schema_version: "1.0",
  experiment_id: "exp_e36e7bbf5267_20260902T054517Z_503e",
  configuration_hash: "b1946ac92492d234",
  created_at: "2026-09-02T05:45:17.275714Z",
  name: "customers.csv · renewed",
  description: null,
  tags: ["baseline"],
  dataset: {
    fingerprint: "9d610b7e1abef86c",
    fingerprint_algorithm: "sha256",
    row_count: 180,
    column_count: 4,
    target_column: "renewed",
    task_type: "classification",
    columns: ["income", "tenure_months", "segment", "renewed"],
    dtypes: {
      income: "int64",
      tenure_months: "int64",
      segment: "str",
      renewed: "str",
    },
    source_format: "csv",
    data_quality_issues: [
      {
        code: "duplicate_rows",
        severity: "warning",
        columns: [],
        message: "48 of 180 rows are exact duplicates of an earlier row.",
      },
    ],
  },
  preprocessing: {
    config: {},
    feature_groups: {},
    selected_columns: ["income", "tenure_months", "segment"],
    excluded_columns: [],
    identifier_columns: [],
    transformed_feature_names: ["income", "tenure_months", "segment_business"],
    column_decisions: [],
    train_row_count: 144,
    test_row_count: 36,
    test_size: 0.2,
    random_state: 7,
    stratified: true,
    stratification_note: null,
    rows_dropped_missing_target: 0,
  },
  selection: {
    strategy: "cross_validation",
    folds: 3,
    primary_metric: "f1",
    primary_metric_direction: "higher_is_better",
    candidate_models: ["logistic_regression", "random_forest_classifier"],
    candidates: [
      {
        model_name: "logistic_regression",
        display_name: "Logistic Regression",
        status: "succeeded",
        score: 0.8527877161598093,
        score_std: 0.016125350251944674,
        error: null,
      },
      {
        model_name: "random_forest_classifier",
        display_name: "Random Forest Classifier",
        status: "succeeded",
        score: 0.8114,
        score_std: 0.0242,
        error: null,
      },
      {
        model_name: "gradient_boosting_classifier",
        display_name: "Gradient Boosting Classifier",
        status: "failed",
        score: null,
        score_std: null,
        error: "The model could not be fitted on this data.",
      },
    ],
    selected_model: "logistic_regression",
    selection_score: 0.8527877161598093,
    selection_score_std: 0.016125350251944674,
    scored_on: "training folds",
    uses_test_data: false,
  },
  evaluation: {
    primary_metric: "f1",
    primary_metric_value: 0.9444444444444444,
    metrics: {
      accuracy: 0.9444444444444444,
      precision: 0.9444444444444444,
      recall: 0.9444444444444444,
      f1: 0.9444444444444444,
      roc_auc: 0.962962962962963,
    },
    unavailable_metrics: {},
    baseline_identifier: "majority_class_baseline",
    baseline_metrics: {
      accuracy: 0.5,
      precision: 0,
      recall: 0,
      f1: 0,
      roc_auc: 0.5,
    },
    baseline_comparison: {
      metric: "f1",
      direction: "higher_is_better",
      model_value: 0.9444444444444444,
      baseline_value: 0,
      absolute_improvement: 0.9444444444444444,
      relative_improvement: null,
      beats_baseline: true,
    },
    classification_details: {
      class_count: 2,
      class_labels: ["no", "yes"],
      class_distribution: { yes: 18, no: 18 },
      confusion_matrix: [
        [17, 1],
        [1, 17],
      ],
      averaging: "binary",
      positive_label: "yes",
    },
    test_row_count: 36,
    is_unbiased: true,
  },
  explainability: {
    status: "available",
    method: "shap",
    explainer: "LinearExplainer",
    aggregation: "mean absolute SHAP value per feature",
    explained_output: "the model's single output",
    feature_importances: [
      { feature: "income", importance: 0.8355665752047913, rank: 1 },
      { feature: "tenure_months", importance: 0.7731462919477246, rank: 2 },
      { feature: "segment_business", importance: 0.05004025174775922, rank: 3 },
    ],
    sample_count: 144,
    feature_count: 3,
    reason: null,
    warnings: [],
  },
  environment: {
    python_version: "3.11.9",
    platform: "Linux",
    packages: { "scikit-learn": "1.9.0" },
    random_state: 7,
  },
  warnings: [],
  execution: { duration_seconds: 1.42, stored: true, mode: "synchronous" },
};

/** A regression run: a different metric set, and a metric where lower wins. */
export const REGRESSION_RUN: ExperimentRunResponse = {
  ...CLASSIFICATION_RUN,
  experiment_id: "exp_regression_0001",
  name: "prices.xlsx · price",
  dataset: {
    ...CLASSIFICATION_RUN.dataset,
    target_column: "price",
    task_type: "regression",
    source_format: "xlsx",
  },
  selection: {
    ...CLASSIFICATION_RUN.selection,
    primary_metric: "rmse",
    primary_metric_direction: "lower_is_better",
    selected_model: "linear_regression",
    selection_score: 2011.5,
    selection_score_std: 88.2,
    candidates: [
      {
        model_name: "linear_regression",
        display_name: "Linear Regression",
        status: "succeeded",
        score: 2011.5,
        score_std: 88.2,
        error: null,
      },
    ],
  },
  evaluation: {
    primary_metric: "rmse",
    primary_metric_value: 1957.67,
    metrics: { mae: 1502.31, mse: 3832471.2, rmse: 1957.67, r2: 0.9312 },
    unavailable_metrics: { mape: "The target contains zeros." },
    baseline_identifier: "mean_baseline",
    baseline_metrics: { mae: 5210.4, mse: 41230000, rmse: 6421.06, r2: 0 },
    baseline_comparison: {
      metric: "rmse",
      direction: "lower_is_better",
      model_value: 1957.67,
      baseline_value: 6421.06,
      absolute_improvement: -4463.39,
      relative_improvement: null,
      beats_baseline: true,
    },
    classification_details: null,
    test_row_count: 24,
    is_unbiased: true,
  },
  explainability: {
    status: "unavailable",
    method: "none",
    explainer: null,
    aggregation: null,
    explained_output: null,
    feature_importances: [],
    reason: "fitted_model_not_persisted",
    warnings: [],
  },
};

export const EXPERIMENT_LIST: ExperimentListResponse = {
  count: 2,
  limit: 50,
  experiments: [
    {
      experiment_id: "exp_e36e7bbf5267_20260902T054517Z_503e",
      created_at: "2026-09-02T05:45:17.275714Z",
      name: "customers.csv · renewed",
      dataset_fingerprint: "9d610b7e1abef86c",
      task_type: "classification",
      target_column: "renewed",
      selected_model: "logistic_regression",
      strategy: "cross_validation",
      primary_metric: "f1",
      selection_score: 0.8527877161598093,
      test_score: 0.9444444444444444,
    },
    {
      experiment_id: "exp_second_run_0002",
      created_at: "2026-09-01T09:12:00.000000Z",
      name: "customers.json · renewed",
      dataset_fingerprint: "9d610b7e1abef86c",
      task_type: "classification",
      target_column: "renewed",
      selected_model: "random_forest_classifier",
      strategy: "cross_validation",
      primary_metric: "f1",
      selection_score: 0.8114,
      test_score: 0.9021,
    },
  ],
};

export const COMPARISON: ExperimentComparison = {
  task_type: "classification",
  primary_metric: "f1",
  direction: "higher_is_better",
  higher_is_better: true,
  run_count: 2,
  best_experiment_id: "exp_e36e7bbf5267_20260902T054517Z_503e",
  runs: [
    {
      experiment_id: "exp_e36e7bbf5267_20260902T054517Z_503e",
      created_at: "2026-09-02T05:45:17.275714Z",
      name: "customers.csv · renewed",
      model_name: "logistic_regression",
      strategy: "cross_validation",
      selection_score: 0.8527877161598093,
      selection_score_std: 0.0161,
      test_score: 0.9444444444444444,
      baseline_score: 0,
      improvement: 0.9444444444444444,
    },
    {
      experiment_id: "exp_second_run_0002",
      created_at: "2026-09-01T09:12:00.000000Z",
      name: "customers.json · renewed",
      model_name: "random_forest_classifier",
      strategy: "cross_validation",
      selection_score: 0.8114,
      selection_score_std: 0.0242,
      test_score: 0.9021,
      baseline_score: 0,
      improvement: 0.9021,
    },
  ],
  table: "…",
};

export const AGENT_COMPLETED: AgentAnswer = {
  question: "Which model performs best and why?",
  status: "completed",
  final_answer:
    "Logistic regression was selected. It scored 0.85 under 3-fold cross-validation on the training rows and 0.94 on the untouched test set [exp:exp_e36e7bbf5267_20260902T054517Z_503e].",
  is_answer: true,
  tool_calls: [
    {
      call_id: "call-01",
      tool_name: "dataset_profile",
      status: "ok",
      arguments: { dataset: "uploaded_dataset", target_column: "renewed" },
      duration_ms: 41,
    },
    {
      call_id: "call-02",
      tool_name: "run_experiment",
      status: "ok",
      arguments: { dataset: "uploaded_dataset", target_column: "renewed" },
      duration_ms: 1382,
    },
    {
      call_id: "call-03",
      tool_name: "explain_experiment",
      status: "ok",
      arguments: { experiment_id: "exp_e36e7bbf5267_20260902T054517Z_503e" },
      duration_ms: 220,
    },
  ],
  observations: [
    {
      call_id: "call-03",
      tool_name: "explain_experiment",
      status: "ok",
      output: {
        status: "ok",
        scope: "global",
        source: "recomputed",
        method: "shap",
        explainer: "LinearExplainer",
        aggregation: "mean absolute SHAP value per feature",
        feature_importances: [
          { feature: "income", importance: 0.8355, rank: 1 },
          { feature: "tenure_months", importance: 0.7731, rank: 2 },
        ],
      },
      error: null,
      error_code: null,
      duration_ms: 220,
      citations: [],
    },
  ],
  citations: [
    {
      citation_id: "exp:exp_e36e7bbf5267_20260902T054517Z_503e",
      source_type: "experiment",
      source_title: "customers.csv · renewed",
      source_reference: "experiment record",
      score: 0.91,
    },
    {
      citation_id: "docs:ml-readme#baselines-under-cross-validation",
      source_type: "project_documentation",
      source_title: "ML Copilot — ML Layer",
      source_reference: "ml/README.md",
      score: 0.38,
    },
  ],
  citation_ids: [
    "exp:exp_e36e7bbf5267_20260902T054517Z_503e",
    "docs:ml-readme#baselines-under-cross-validation",
  ],
  rejected_citations: [],
  allowed_citations: [],
  experiment_ids: ["exp_e36e7bbf5267_20260902T054517Z_503e"],
  warnings: [],
  iterations: 4,
  tool_call_count: 3,
  tools_available: [
    "dataset_profile",
    "run_experiment",
    "explain_experiment",
    "search_knowledge",
  ],
  dataset: {
    name: "uploaded_dataset",
    filename: "customers.csv",
    source_format: "csv",
    fingerprint: "9d610b7e1abef86c",
    row_count: 180,
    column_count: 4,
    columns: ["income", "tenure_months", "segment", "renewed"],
    persisted: false,
  },
  error_code: null,
  duration_ms: 1720,
};

/** A local explanation, which only ever arrives inside an observation. */
export const AGENT_LOCAL_EXPLANATION: AgentAnswer = {
  ...AGENT_COMPLETED,
  question: "Why did the model make that prediction?",
  observations: [
    {
      call_id: "call-04",
      tool_name: "explain_experiment",
      status: "ok",
      output: {
        status: "ok",
        scope: "prediction",
        source: "recomputed",
        method: "shap",
        row_index: 7,
        prediction: "yes",
        predicted_class: "yes",
        base_value: 0.02,
        feature_contributions: [
          { feature: "income", contribution: 0.31, direction: "increases prediction", rank: 1 },
          { feature: "tenure_months", contribution: -0.12, direction: "decreases prediction", rank: 2 },
        ],
      },
      error: null,
      error_code: null,
      duration_ms: 180,
      citations: [],
    },
  ],
};

export const AGENT_PARTIAL: AgentAnswer = {
  ...AGENT_COMPLETED,
  status: "partial",
  final_answer: "The older run could not be explained, so this is incomplete.",
  tool_calls: [
    {
      call_id: "call-01",
      tool_name: "run_experiment",
      status: "ok",
      arguments: { dataset: "uploaded_dataset" },
      duration_ms: 900,
    },
    {
      call_id: "call-02",
      tool_name: "explain_experiment",
      status: "unavailable",
      arguments: { experiment_id: "exp_from_last_week" },
      duration_ms: 12,
    },
  ],
  observations: [],
  warnings: ["The fitted model for that experiment is not persisted."],
  experiment_ids: [],
};

export const AGENT_INSUFFICIENT: AgentAnswer = {
  ...AGENT_COMPLETED,
  status: "insufficient_evidence",
  final_answer: "INSUFFICIENT_EVIDENCE",
  is_answer: false,
  citations: [],
  citation_ids: [],
  observations: [],
  experiment_ids: [],
};

export const AGENT_GROUNDING_FAILED: AgentAnswer = {
  ...AGENT_COMPLETED,
  status: "grounding_failed",
  final_answer: "It works by magic [docs:secret-internal#nope].",
  is_answer: false,
  citations: [],
  citation_ids: [],
  rejected_citations: ["docs:secret-internal#nope"],
  observations: [],
  experiment_ids: [],
};

/** A run whose planner tried a tool that is not registered. */
export const AGENT_REJECTED_TOOL: AgentAnswer = {
  ...AGENT_COMPLETED,
  status: "partial",
  tool_calls: [
    {
      call_id: "call-01",
      tool_name: "shell",
      status: "rejected",
      arguments: {},
      duration_ms: 1,
    },
  ],
  observations: [],
};

export const AGENT_STATUS: AgentStatusResponse = {
  agent_available: true,
  tools: ["search_knowledge", "explain_experiment"],
  dataset_upload_supported: true,
  supported_dataset_formats: ["csv", "xlsx", "json"],
  max_tool_calls: 6,
  max_iterations: 8,
  max_context_chars: 20000,
  max_answer_length: 4000,
};

export const KNOWLEDGE_STATUS: KnowledgeStatus = {
  search_available: true,
  answering_available: true,
  index_built: true,
  similarity_metric: "cosine",
  default_top_k: 5,
  max_top_k: 50,
  max_query_length: 2000,
  source_types: ["project_documentation", "experiment", "ml_reference"],
};

export const SEARCH_RESPONSE: SearchResponse = {
  query: "cross-validation",
  results: [
    {
      rank: 1,
      score: 0.3817122280597687,
      content:
        "The naive baseline plays no part in choosing the winner — models are ranked against each other on cross-validated scores.",
      document_id: "project_documentation:ml-readme.md",
      chunk_id: "project_documentation:ml-readme.md#0034-429721d7",
      source_type: "project_documentation",
      source_title: "ML Copilot — ML Layer",
      source_reference: "ml/README.md",
      citation_id: "docs:ml-readme#baselines-under-cross-validation",
      metadata: { heading: "Baselines under cross-validation" },
    },
  ],
  result_count: 1,
  top_k: 5,
  similarity_threshold: 0.15,
  similarity_metric: "cosine",
  candidate_count: 12,
  citations: ["docs:ml-readme#baselines-under-cross-validation"],
};

export const ASK_GROUNDED: AskResponse = {
  question: "What is cross-validation?",
  answer:
    "Candidates are scored over folds of the training rows only [docs:ml-readme#baselines-under-cross-validation].",
  status: "grounded",
  is_grounded: true,
  citations: [
    {
      citation_id: "docs:ml-readme#baselines-under-cross-validation",
      source_type: "project_documentation",
      source_title: "ML Copilot — ML Layer",
      source_reference: "ml/README.md",
      relevance_score: 0.38,
      excerpt: "The naive baseline plays no part in choosing the winner…",
    },
  ],
  citation_ids: ["docs:ml-readme#baselines-under-cross-validation"],
  rejected_citations: [],
  allowed_citations: ["docs:ml-readme#baselines-under-cross-validation"],
  warnings: [],
  error_code: null,
  metadata: {
    provider: "fake",
    model: "fake-model",
    retrieved_count: 5,
    context_count: 3,
    context_truncated: false,
    context_characters: 2400,
    approximate_context_tokens: 600,
    below_threshold_count: 2,
    latency_seconds: 0.4,
    prompt_tokens: null,
    completion_tokens: null,
    finish_reason: "stop",
  },
};

/** A tiny CSV as a `File`, for upload tests. */
export function csvFile(name = "customers.csv"): File {
  return new File(["income,renewed\n50000,yes\n"], name, { type: "text/csv" });
}

/** A `File` named like a workbook. Bytes do not matter — the backend decides. */
export function xlsxFile(name = "customers.xlsx"): File {
  return new File([new Uint8Array([0x50, 0x4b, 0x03, 0x04])], name, {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
}

/** A JSON `File` shaped as the backend's array-of-objects form. */
export function jsonFile(name = "customers.json"): File {
  return new File(['[{"income":50000,"renewed":"yes"}]'], name, {
    type: "application/json",
  });
}
