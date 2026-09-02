"use client";

import { useCallback, useMemo, useState } from "react";
import { Button } from "@/components/common/Button";
import { Card } from "@/components/common/Card";
import { EmptyState } from "@/components/common/EmptyState";
import { ErrorBanner } from "@/components/common/ErrorBanner";
import { Loading } from "@/components/common/Spinner";
import { Tabs } from "@/components/common/Tabs";
import { AgentAsk } from "@/components/agent/AgentAsk";
import { DatasetUpload } from "@/components/dataset/DatasetUpload";
import { DatasetProfileView } from "@/components/dataset/DatasetProfileView";
import { ExperimentSummary } from "@/components/experiments/ExperimentSummary";
import { ModelComparisonTable } from "@/components/experiments/ModelComparisonTable";
import { MetricsPanel } from "@/components/experiments/MetricsPanel";
import { RunExperimentForm } from "@/components/experiments/RunExperimentForm";
import { GlobalImportance } from "@/components/explainability/GlobalImportance";
import { askAgentWithDataset } from "@/lib/api/agent";
import { profileDataset } from "@/lib/api/datasets";
import { runExperiment } from "@/lib/api/experiments";
import type {
  AgentAnswer,
  DatasetProfile,
  ExperimentOptions,
  ExperimentRunResponse,
} from "@/lib/api/types";

/**
 * The main workflow: upload → analyse → experiment → explain.
 *
 * One page holds the file and three independent results, because the file is
 * what ties them together — the same upload feeds the profile, the run and
 * the agent, and re-picking it for each would be the most tedious part of
 * using this. Everything below that is a component; this page owns the file,
 * the three request states, and nothing else.
 *
 * The `File` lives in React state for the lifetime of the page and is never
 * written to `localStorage`, `sessionStorage` or a URL. Its contents are read
 * only by `fetch`, when posting it to the configured backend.
 */

/** The stages a person waits through, in the words of what is happening. */
const STAGE_LABELS = {
  profiling: "Profiling dataset…",
  experiment: "Running experiment… training and cross-validating models",
  agent: "Thinking… the agent may profile, train and explain before answering",
} as const;

export default function DashboardPage() {
  const [file, setFile] = useState<File | null>(null);
  // Shared by profiling and by the run: saying what you predict, once.
  const [target, setTarget] = useState("");

  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [profiling, setProfiling] = useState(false);
  const [profileError, setProfileError] = useState<unknown>(null);

  const [run, setRun] = useState<ExperimentRunResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<unknown>(null);

  const [answer, setAnswer] = useState<AgentAnswer | null>(null);
  const [asking, setAsking] = useState(false);
  const [agentError, setAgentError] = useState<unknown>(null);

  const busy = profiling || running || asking;
  const columns = useMemo(
    () => profile?.columns.map((column) => column.name) ?? [],
    [profile],
  );

  const onSelectFile = useCallback((next: File | null) => {
    // A new file invalidates everything derived from the old one. Leaving a
    // stale profile beside a new upload is how someone reads the wrong result.
    setFile(next);
    setProfile(null);
    setProfileError(null);
    setRun(null);
    setRunError(null);
    setAnswer(null);
    setAgentError(null);
    setTarget("");
  }, []);

  async function onProfile() {
    if (!file) return;
    setProfiling(true);
    setProfileError(null);
    try {
      setProfile(await profileDataset(file, target || undefined));
    } catch (error) {
      setProfile(null);
      setProfileError(error);
    } finally {
      setProfiling(false);
    }
  }

  async function onRunExperiment(options: ExperimentOptions) {
    if (!file) return;
    setRunning(true);
    setRunError(null);
    try {
      setRun(await runExperiment(file, options));
    } catch (error) {
      setRun(null);
      setRunError(error);
    } finally {
      setRunning(false);
    }
  }

  async function onAsk(question: string) {
    if (!file) return;
    setAsking(true);
    setAgentError(null);
    try {
      setAnswer(await askAgentWithDataset(file, question));
    } catch (error) {
      setAnswer(null);
      setAgentError(error);
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-ink-900">
          AI Data Scientist
        </h1>
        <p className="mt-1 text-sm text-ink-600">
          Upload → Analyse → Experiment → Explain. Every number on this page is
          computed by the backend; nothing is modelled in the browser.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
        <div className="space-y-4">
          <Card title="1 · Upload a dataset">
            <DatasetUpload file={file} onSelect={onSelectFile} disabled={busy} />

            <div className="mt-4 flex flex-wrap gap-2">
              <Button onClick={onProfile} disabled={!file || busy}>
                {profiling ? "Profiling…" : "Profile dataset"}
              </Button>
            </div>

            {profiling && <Loading className="mt-3" label={STAGE_LABELS.profiling} />}
            {!profiling && profileError != null && (
              <div className="mt-3">
                <ErrorBanner
                  error={profileError}
                  title="Profiling failed"
                  onRetry={onProfile}
                />
              </div>
            )}
          </Card>

          <Card
            title="2 · Run an experiment"
            description="Profile, prepare, cross-validate every candidate, retrain the winner and measure it once on rows it has never seen."
          >
            {file ? (
              <>
                <RunExperimentForm
                  columns={columns}
                  disabled={busy}
                  busyLabel="Running…"
                  target={target}
                  onTargetChange={setTarget}
                  onRun={onRunExperiment}
                />
                {running && (
                  <Loading className="mt-3" label={STAGE_LABELS.experiment} />
                )}
                {!running && runError != null && (
                  <div className="mt-3">
                    <ErrorBanner error={runError} title="The experiment failed" />
                  </div>
                )}
              </>
            ) : (
              <EmptyState
                title="Upload a dataset first"
                hint="CSV, Excel (.xlsx) or JSON."
              />
            )}
          </Card>
        </div>

        <div className="space-y-6">
          <section aria-labelledby="dataset-heading">
            <h2 id="dataset-heading" className="sr-only">
              Dataset profile
            </h2>
            {profile ? (
              <DatasetProfileView profile={profile} />
            ) : (
              <Card title="Dataset">
                <EmptyState
                  title={file ? "Not profiled yet" : "No dataset yet"}
                  hint={
                    file
                      ? "Select Profile dataset to see rows, columns, quality findings and the target. Naming a target first analyses it too."
                      : "Upload a CSV, Excel or JSON file to begin. The same file feeds the profile, the experiment and the AI Data Scientist."
                  }
                />
              </Card>
            )}
          </section>

          <section aria-labelledby="agent-heading">
            <h2
              id="agent-heading"
              className="mb-3 text-lg font-semibold tracking-tight text-ink-900"
            >
              3 · Ask the AI Data Scientist
            </h2>
            <AgentAsk
              onAsk={onAsk}
              busy={asking}
              busyLabel={STAGE_LABELS.agent}
              answer={answer}
              error={agentError}
              disabled={!file}
              disabledReason="Upload a dataset to ask about it."
            />
          </section>

          {run && (
            <section aria-labelledby="run-heading" className="space-y-4">
              <h2
                id="run-heading"
                className="text-lg font-semibold tracking-tight text-ink-900"
              >
                4 · Experiment result
              </h2>

              <ExperimentSummary record={run} execution={run.execution} />

              {run.warnings && run.warnings.length > 0 && (
                <Card title="Warnings" headingLevel={3}>
                  <ul className="list-disc space-y-1 pl-5 text-sm text-amber-900">
                    {run.warnings.map((warning) => (
                      <li key={warning}>{warning}</li>
                    ))}
                  </ul>
                </Card>
              )}

              <Card title="Models and metrics" headingLevel={3}>
                <Tabs
                  ariaLabel="Experiment result sections"
                  tabs={[
                    {
                      id: "models",
                      label: "Model comparison",
                      badge: run.selection.candidates.length,
                      content: (
                        <ModelComparisonTable
                          selection={run.selection}
                          evaluation={run.evaluation}
                        />
                      ),
                    },
                    {
                      id: "metrics",
                      label: "Metrics",
                      content: <MetricsPanel evaluation={run.evaluation} />,
                    },
                    {
                      id: "explain",
                      label: "Explainability",
                      content: (
                        <GlobalImportance explainability={run.explainability} />
                      ),
                    },
                  ]}
                />
              </Card>
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
