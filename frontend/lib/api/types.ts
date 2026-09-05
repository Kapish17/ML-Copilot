/**
 * TypeScript mirrors of the backend's response contracts.
 *
 * These are written by hand against the generated OpenAPI schema rather than
 * generated from it, for one reason: a hand-written type can say what a field
 * *means*, and several fields here mean something a schema cannot express —
 * that `selection_score` is cross-validated and `primary_metric_value` is not,
 * that `direction` decides whether a larger number is better, that
 * `persisted` is always false. Those are the distinctions the dashboard exists
 * to make visible, so they are documented where they are declared.
 *
 * Nothing here is `any`. Where the backend genuinely returns an open map —
 * a metric name to its value, a tool's validated arguments — the type is a
 * `Record` of `JsonValue`, which is honest about the shape without discarding
 * type safety.
 */

/** Any value that can appear in a JSON document. */
export type JsonValue =
  | string
  | number
  | boolean
  | null
  | JsonValue[]
  | { [key: string]: JsonValue };

/** An open map of JSON values, as several endpoints return. */
export type JsonObject = { [key: string]: JsonValue };

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/** The body of every failed request, whatever failed. */
export interface ErrorResponse {
  error: {
    code: string;
    message: string;
    details?: JsonObject;
  };
}

// ---------------------------------------------------------------------------
// Datasets — POST /api/v1/datasets/profile
// ---------------------------------------------------------------------------

/** The physical formats the backend reads. */
export type SourceFormat = "csv" | "xlsx" | "json";

/** How strongly a data-quality finding should be acted on. */
export type IssueSeverity = "info" | "warning" | "critical";

/** The semantic type inferred for a column from its parsed values. */
export type InferredType =
  | "integer"
  | "float"
  | "boolean"
  | "datetime"
  | "categorical"
  | "text"
  | "empty"
  | "unknown";

/** What the backend thinks a target column is for. */
export type TaskSuggestion = "classification" | "regression" | "undetermined";

export interface ValueCount {
  value: string;
  count: number;
  percentage: number;
}

export interface NumericStats {
  mean: number | null;
  median: number | null;
  std: number | null;
  minimum: number | null;
  maximum: number | null;
  q1: number | null;
  q3: number | null;
  zero_count: number;
  negative_count: number;
}

export interface DatetimeStats {
  earliest: string | null;
  latest: string | null;
  parsed_count: number;
  unparsed_count: number;
}

export interface CategoricalStats {
  top_values: ValueCount[];
  truncated: boolean;
}

export interface ColumnProfile {
  name: string;
  dtype: string;
  inferred_type: InferredType;
  non_null_count: number;
  missing_count: number;
  missing_percentage: number;
  unique_count: number;
  unique_percentage: number;
  is_constant: boolean;
  numeric_stats: NumericStats | null;
  datetime_stats: DatetimeStats | null;
  categorical_stats: CategoricalStats | null;
}

export interface DatasetSummary {
  row_count: number;
  column_count: number;
  memory_usage_bytes: number;
  duplicate_row_count: number;
  duplicate_row_percentage: number;
  missing_cell_count: number;
  missing_cell_percentage: number;
  column_type_counts: Record<string, number>;
}

export interface QualityIssue {
  code: string;
  severity: IssueSeverity;
  message: string;
  columns: string[];
  details?: JsonObject;
}

export interface QualityReport {
  issue_count: number;
  issues: QualityIssue[];
}

export interface ClassBalance {
  class_count: number;
  majority_class: string | null;
  majority_percentage: number | null;
  minority_class: string | null;
  minority_percentage: number | null;
  is_imbalanced: boolean;
}

export interface TargetProfile {
  name: string;
  dtype: string;
  inferred_type: InferredType;
  missing_count: number;
  missing_percentage: number;
  task_suggestion: TaskSuggestion;
  task_reason: string;
  distribution: ValueCount[] | null;
  class_balance: ClassBalance | null;
  numeric_stats: NumericStats | null;
}

export interface DatasetProfile {
  filename: string;
  /** How the upload was read. Reported as context; profiling is identical. */
  source_format: SourceFormat | null;
  generated_at: string;
  dataset: DatasetSummary;
  columns: ColumnProfile[];
  quality: QualityReport;
  target: TargetProfile | null;
}

// ---------------------------------------------------------------------------
// Experiments
// ---------------------------------------------------------------------------

/** Whether a larger value of a metric is better. */
export type MetricDirection = "higher_is_better" | "lower_is_better";

export interface ExperimentDataset {
  fingerprint: string;
  fingerprint_algorithm: string;
  row_count: number;
  column_count: number;
  target_column: string;
  task_type: string;
  columns: string[];
  dtypes: Record<string, string>;
  source_format: SourceFormat | string | null;
  data_quality_issues?: QualityIssue[];
}

/** One model that was considered, and how it scored under the strategy. */
export interface CandidateModel {
  model_name: string;
  display_name: string;
  status: string;
  score: number | null;
  score_std: number | null;
  error: string | null;
}

export interface ExperimentSelection {
  strategy: string;
  folds: number | null;
  primary_metric: string;
  primary_metric_direction: MetricDirection | string;
  candidate_models: string[];
  candidates: CandidateModel[];
  selected_model: string;
  /**
   * The winner's score under the selection strategy. Under
   * `cross_validation` this is the mean over folds of the *training* rows —
   * it is not, and must never be presented as, the final measurement.
   */
  selection_score: number | null;
  selection_score_std: number | null;
  scored_on: string;
  /** False under cross-validation: the test set was not touched to choose. */
  uses_test_data: boolean;
  /**
   * One sentence on why the winning model won, composed by the backend from
   * this run's own recorded numbers. Displayed as sent: it is not written by
   * a language model, and the frontend must not compose its own version.
   * Optional, because records written before it exist.
   */
  rationale?: string;
}

export interface BaselineComparison {
  metric: string;
  direction: MetricDirection | string;
  model_value: number | null;
  baseline_value: number | null;
  absolute_improvement: number | null;
  relative_improvement: number | null;
  beats_baseline: boolean;
}

export interface ClassificationDetails {
  class_count: number;
  class_labels: string[];
  class_distribution: Record<string, number>;
  confusion_matrix: number[][];
  averaging: string;
  positive_label: string | null;
}

export interface ExperimentEvaluation {
  primary_metric: string;
  /** The winner's score on the untouched test set. Measured once. */
  primary_metric_value: number | null;
  metrics: Record<string, number | null>;
  unavailable_metrics?: Record<string, string>;
  baseline_identifier: string | null;
  baseline_metrics?: Record<string, number | null>;
  baseline_comparison?: BaselineComparison | null;
  classification_details?: ClassificationDetails | null;
  test_row_count: number;
  is_unbiased: boolean;
  /**
   * Situations in this run worth a second look. Signals, not verdicts, and
   * not failures — a run with diagnostics completed exactly as asked.
   * Optional, because records written before diagnostics existed have none.
   */
  diagnostics?: ExperimentDiagnostic[];
  warning_count?: number;
}

/** One signal about a finished run. `code` is stable; `message` is for people. */
export interface ExperimentDiagnostic {
  code: string;
  severity: "warning" | "info" | string;
  message: string;
  details?: JsonObject;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
  rank: number;
}

export interface ExperimentExplainability {
  status: string;
  method: string;
  explainer: string | null;
  aggregation: string | null;
  explained_output: string | null;
  feature_importances?: FeatureImportance[];
  sample_count?: number;
  feature_count?: number;
  reason: string | null;
  warnings?: string[];
}

export interface ExperimentPreprocessing {
  config: JsonObject;
  feature_groups: JsonObject;
  selected_columns: string[];
  excluded_columns: string[];
  identifier_columns: string[];
  transformed_feature_names: string[];
  column_decisions: JsonObject[];
  train_row_count: number;
  test_row_count: number;
  test_size: number;
  random_state: number;
  stratified: boolean;
  stratification_note: string | null;
  rows_dropped_missing_target: number;
}

export interface ExperimentEnvironment {
  python_version: string;
  platform: string;
  packages: Record<string, string>;
  random_state: number | null;
}

export interface ExperimentExecution {
  duration_seconds: number;
  stored: boolean;
  mode?: string;
}

/** One stored experiment, as `GET /api/v1/experiments/{id}` returns it. */
export interface ExperimentRecord {
  schema_version: string;
  experiment_id: string;
  configuration_hash: string;
  created_at: string;
  name: string;
  description: string | null;
  tags?: string[];
  dataset: ExperimentDataset;
  preprocessing: ExperimentPreprocessing;
  selection: ExperimentSelection;
  evaluation: ExperimentEvaluation;
  explainability: ExperimentExplainability | null;
  model_artifact: ExperimentModelArtifact | null;
  environment: ExperimentEnvironment;
}

/**
 * That this run's winning model was persisted, and what it expects.
 *
 * A note about what happened when the run finished, not a promise about now —
 * an artifact can be deleted. `GET /api/v1/experiments/{id}/model` answers
 * whether a prediction can actually be made today, and is what the prediction
 * panel reads.
 *
 * `null` on every run recorded before model persistence existed.
 */
export interface ExperimentModelArtifact {
  stored: boolean;
  model_name: string;
  task_type: string;
  target_column: string;
  feature_names: string[];
  feature_count: number;
  class_labels: string[];
  artifact_schema_version: string;
  created_at: string | null;
}

/** One column a stored model expects, and how it is treated. */
export interface PredictedFeature {
  name: string;
  /** `numeric`, `categorical`, `boolean` or `datetime`. */
  kind: string;
  dtype: string;
}

/** The lifecycle state of an experiment's stored model. */
export type ModelStatus = "available" | "not_available" | "corrupted";

/**
 * Whether an experiment can be predicted from, and with what.
 *
 * Answered from the artifact store rather than from the record, so a model
 * deleted after the run that made it reports `not_available` rather than a
 * form that cannot work. All three states arrive as a 200 — whether a run can
 * be predicted from is an answer, not a failure — so this interface is what
 * the panel branches on, and an `ApiError` means something else went wrong.
 */
export interface ModelAvailability {
  experiment_id: string;
  /**
   * `available` — there is a usable model. `not_available` — this run has no
   * artifact, which is normal for one recorded before model persistence
   * existed. `corrupted` — an artifact is stored and does not check out.
   */
  status: ModelStatus;
  /** `status === "available"`, for a caller that only needs to branch. */
  available: boolean;
  /** A stable code for *why* there is no usable model, or `null`. */
  reason_code: string | null;
  /** The same thing as a sentence, ending in what to do about it. */
  reason: string | null;
  max_records: number;
  model_name: string | null;
  display_name: string | null;
  task_type: string | null;
  target_column: string | null;
  classes: JsonValue[];
  features: PredictedFeature[];
  created_at: string | null;
  train_row_count: number | null;
  /** How many held-out rows `primary_metric_value` was measured on. */
  test_row_count: number | null;
  primary_metric: string | null;
  primary_metric_value: number | null;
  /** Whether a prediction from this model can carry class probabilities. */
  supports_probabilities: boolean;
  artifact_schema_version: string | null;
}

/** Which model produced a prediction, and what it was trained on. */
export interface PredictedModel {
  experiment_id: string;
  created_at: string;
  model_name: string;
  display_name: string;
  task_type: string;
  target_column: string;
  classes: JsonValue[];
  features: PredictedFeature[];
  train_row_count: number;
  /** How many held-out rows the metric below was measured on. */
  test_row_count: number | null;
  primary_metric: string;
  /**
   * The model's score on the **held-out test set** — a measurement of the
   * model over many rows, not a confidence in this prediction. The two are
   * easy to conflate and the UI keeps them apart.
   */
  primary_metric_value: number | null;
  /** Whether this model reports a probability per class. */
  supports_probabilities: boolean;
  artifact_schema_version: string | null;
}

/** One record's result. */
export interface PredictionItem {
  /** Position of the record in the submitted batch. */
  index: number;
  prediction: JsonValue;
  /** Per-class probability, or `null` for regression. */
  probabilities: Record<string, number> | null;
}

/** `POST /api/v1/experiments/{id}/predict`. */
export interface PredictionResponse {
  predictions: PredictionItem[];
  prediction_count: number;
  model: PredictedModel;
}

/** A completed run, as `POST /api/v1/experiments/run` returns it. */
export interface ExperimentRunResponse extends ExperimentRecord {
  warnings?: string[];
  execution: ExperimentExecution;
}

export interface ExperimentHeadline {
  experiment_id: string;
  created_at: string;
  name: string;
  dataset_fingerprint: string;
  task_type: string;
  target_column: string;
  selected_model: string;
  strategy: string;
  primary_metric: string;
  selection_score: number | null;
  selection_score_std?: number | null;
  test_score: number | null;
  train_row_count?: number | null;
  test_row_count?: number | null;
  feature_count?: number | null;
  warning_count?: number;
  is_unbiased?: boolean;
}

export interface ExperimentListResponse {
  count: number;
  limit: number;
  experiments: ExperimentHeadline[];
}

export interface ComparisonRow {
  experiment_id: string;
  created_at: string;
  name: string;
  model_name: string;
  strategy: string;
  task_type?: string;
  primary_metric?: string;
  /** The score that chose the model. Never a test result. */
  selection_score: number | null;
  /** Spread across folds — how much they disagreed, not a confidence interval. */
  selection_score_std: number | null;
  /** The held-out measurement, taken once after selection. */
  test_score: number | null;
  baseline_score: number | null;
  improvement: number | null;
  train_row_count?: number | null;
  test_row_count?: number | null;
  /** Features after encoding, as the model saw them. */
  feature_count?: number | null;
  /** Diagnostics on this run worth more than a glance. */
  warning_count?: number;
  is_unbiased?: boolean;
  rationale?: string;
}

export interface ExperimentComparison {
  task_type: string;
  primary_metric: string;
  primary_metric_label?: string;
  direction: MetricDirection | string;
  higher_is_better: boolean;
  run_count: number;
  best_experiment_id: string | null;
  runs: ComparisonRow[];
  table: string;
}

export interface ModelInfo {
  identifier: string;
  display_name: string;
  task_type: string;
  supports_probabilities: boolean | null;
  supports_random_state: boolean | null;
  default_parameters?: JsonObject;
  description: string | null;
}

export interface ExperimentCapabilities {
  models: ModelInfo[];
  metrics: Record<string, string[]>;
  strategies: string[];
  sort_keys: string[];
  limits: JsonObject;
  supported_dataset_extensions: string[];
}

/** The form fields `POST /api/v1/experiments/run` accepts. */
export interface ExperimentOptions {
  target_column?: string;
  models?: string[];
  primary_metric?: string;
  strategy?: string;
  folds?: number;
  test_size?: number;
  random_state?: number;
  explain?: boolean;
  name?: string;
  description?: string;
  tags?: string[];
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/**
 * The four outcomes of a run, all of which arrive as HTTP 200.
 *
 * Telling them apart is the thing a client most needs to get right: only
 * `completed` may be shown to a person as an answer.
 */
export type AgentStatus =
  | "completed"
  | "partial"
  | "insufficient_evidence"
  | "grounding_failed";

export interface AgentToolCall {
  call_id: string;
  tool_name: string;
  /** `ok`, `unavailable`, `rejected` or `failed`. */
  status: string;
  arguments?: JsonObject;
  duration_ms: number | null;
}

export interface AgentObservation {
  call_id: string;
  tool_name: string;
  status: string;
  input_summary?: JsonObject;
  output?: JsonObject;
  error: string | null;
  error_code: string | null;
  duration_ms: number | null;
  citations?: string[];
}

export interface AgentCitation {
  citation_id: string;
  source_type?: string;
  source_title?: string;
  source_reference?: string;
  score?: number | null;
}

/** Facts about an upload. Never any of its rows. */
export interface AgentDatasetInfo {
  name: string;
  filename: string;
  source_format: SourceFormat | string;
  fingerprint: string;
  row_count: number;
  column_count: number;
  columns: string[];
  /** Always false — the dataset lived in memory for one request. */
  persisted?: boolean;
}

/** One planned step, and what became of it. */
export interface AgentWorkflowStep {
  step: string;
  tool: string;
  /** A short label for the step — what it was for, never why it was chosen. */
  purpose: string;
  /** `ok`, `unavailable`, `rejected`, `failed`, or `skipped` — never called. */
  status: string;
  depends_on: string[];
  /** Why it did not run or did not work. An authored sentence. */
  reason: string | null;
}

/**
 * The plan a run executed, beside what happened to each step.
 *
 * `null` on a run answered one decision at a time, which is what every run
 * looked like before planning existed — so anything reading this must handle
 * its absence rather than assume it.
 *
 * There is no field for a step's arguments, deliberately: they are the one
 * place a planner could put text of its own choosing into something a person
 * reads, and what a call actually received is already in the tool trace.
 */
export interface AgentWorkflow {
  goal: string;
  objective: string;
  steps: AgentWorkflowStep[];
  /** The plan as a person reads it, one numbered line per step. */
  summary: string[];
  planned_step_count: number;
  completed_step_count: number;
  is_complete: boolean;
}

/** The shape of a run, without reading its observations. */
export interface AgentExecutionSummary {
  planned: boolean;
  steps_planned: number;
  steps_completed: number;
  workflow_complete: boolean;
  tools_used: string[];
  tool_call_count: number;
  /** True whenever the answer covers less than the question asked for. */
  partial: boolean;
  stopped_by: string | null;
}

export interface AgentAnswer {
  question: string;
  status: AgentStatus | string;
  final_answer: string;
  is_answer: boolean;
  tool_calls: AgentToolCall[];
  observations: AgentObservation[];
  citations: AgentCitation[];
  citation_ids: string[];
  rejected_citations: string[];
  allowed_citations: string[];
  experiment_ids: string[];
  warnings: string[];
  /** The plan this run executed, when it had one. */
  workflow?: AgentWorkflow | null;
  execution_summary?: AgentExecutionSummary | null;
  iterations: number;
  tool_call_count: number;
  tools_available: string[];
  dataset: AgentDatasetInfo | null;
  error_code: string | null;
  duration_ms: number | null;
}

export interface AgentStatusResponse {
  agent_available: boolean;
  tools: string[];
  dataset_upload_supported: boolean;
  supported_dataset_formats: string[];
  max_tool_calls: number;
  max_iterations: number;
  max_context_chars: number;
  max_answer_length: number;
}

/** Budgets a request may lower and never raise. */
export interface AgentBudgets {
  max_tool_calls?: number;
  max_iterations?: number;
  max_context_chars?: number;
}

// ---------------------------------------------------------------------------
// Knowledge — search and grounded answers
// ---------------------------------------------------------------------------

export interface SearchResult {
  rank: number;
  score: number;
  content: string;
  document_id: string;
  chunk_id: string;
  source_type: string;
  source_title: string;
  source_reference: string;
  citation_id: string;
  metadata?: JsonObject;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  result_count: number;
  top_k: number;
  similarity_threshold: number;
  similarity_metric?: string;
  candidate_count: number;
  citations: string[];
}

export interface AnswerCitation {
  citation_id: string;
  source_type: string;
  source_title: string;
  source_reference: string;
  relevance_score: number;
  excerpt?: string;
}

export interface AnswerMetadata {
  provider: string;
  model: string;
  retrieved_count: number;
  context_count: number;
  context_truncated: boolean;
  context_characters: number;
  approximate_context_tokens: number;
  below_threshold_count: number;
  latency_seconds: number | null;
  prompt_tokens: number | null;
  completion_tokens: number | null;
  finish_reason: string | null;
}

export interface AskResponse {
  question: string;
  answer: string;
  /** `grounded`, `insufficient_evidence` or `grounding_failed`. */
  status: string;
  is_grounded: boolean;
  citations: AnswerCitation[];
  citation_ids: string[];
  rejected_citations: string[];
  allowed_citations: string[];
  warnings: string[];
  error_code: string | null;
  metadata: AnswerMetadata;
}

/** `GET /` — who is running, and whether the protected endpoints are locked. */
export interface ServiceInfo {
  name: string;
  version: string;
  environment: string;
  docs_url: string;
  /**
   * Whether the protected endpoints require `Authorization: Bearer <key>`.
   *
   * A fact about the deployment's configuration, never about the key. The
   * dashboard reads it to say so in the header — it cannot act on it, because
   * a browser application cannot hold a shared backend secret.
   */
  authentication_required: boolean;
}

/** `GET /health` — liveness, and what both container healthchecks call. */
export interface HealthStatus {
  status: string;
  version: string;
  environment: string;
}

export interface KnowledgeStatus {
  search_available: boolean;
  answering_available: boolean;
  index_built: boolean;
  similarity_metric: string;
  default_top_k: number;
  max_top_k: number;
  max_query_length: number;
  source_types: string[];
}
