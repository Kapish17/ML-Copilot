"use client";

import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type {
  AgentAnswer,
  DatasetProfile,
  ExperimentRunResponse,
} from "@/lib/api/types";

/**
 * The Dashboard's upload and its three derived results, lifted out of the
 * page component and into a provider that sits above every route.
 *
 * `DashboardPage` fully unmounts whenever the user follows a link to another
 * route — the Knowledge Assistant, an experiment's own page — and remounts
 * fresh when they come back. That is ordinary Next.js App Router behaviour
 * for a routed page component, but it is also what made an uploaded file
 * "disappear" from the Dashboard after a visit to Knowledge: the file itself
 * was never lost, the `useState` holding it was torn down with the
 * component. `AppShell`, by contrast, wraps every route and is never
 * unmounted by navigating between them, so state kept here — in a provider
 * rendered inside `AppShell` — survives exactly the trip that was breaking
 * it. This is in-memory only, same as before: nothing here is written to
 * `localStorage`, `sessionStorage` or a URL, and a full page reload still
 * starts clean.
 */
export interface DashboardState {
  file: File | null;
  target: string;
  setTarget: (target: string) => void;
  /**
   * Replaces the uploaded file and clears everything derived from the
   * previous one. Leaving a stale profile or result beside a new upload is
   * how someone ends up reading the wrong answer.
   */
  selectFile: (file: File | null) => void;

  profile: DatasetProfile | null;
  setProfile: (profile: DatasetProfile | null) => void;
  profiling: boolean;
  setProfiling: (value: boolean) => void;
  profileError: unknown;
  setProfileError: (error: unknown) => void;

  run: ExperimentRunResponse | null;
  setRun: (run: ExperimentRunResponse | null) => void;
  running: boolean;
  setRunning: (value: boolean) => void;
  runError: unknown;
  setRunError: (error: unknown) => void;

  answer: AgentAnswer | null;
  setAnswer: (answer: AgentAnswer | null) => void;
  asking: boolean;
  setAsking: (value: boolean) => void;
  agentError: unknown;
  setAgentError: (error: unknown) => void;
}

const DashboardStateContext = createContext<DashboardState | null>(null);

export function DashboardStateProvider({ children }: { children: ReactNode }) {
  const [file, setFile] = useState<File | null>(null);
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

  const selectFile = useCallback((next: File | null) => {
    setFile(next);
    setProfile(null);
    setProfileError(null);
    setRun(null);
    setRunError(null);
    setAnswer(null);
    setAgentError(null);
    setTarget("");
  }, []);

  const value = useMemo<DashboardState>(
    () => ({
      file,
      target,
      setTarget,
      selectFile,
      profile,
      setProfile,
      profiling,
      setProfiling,
      profileError,
      setProfileError,
      run,
      setRun,
      running,
      setRunning,
      runError,
      setRunError,
      answer,
      setAnswer,
      asking,
      setAsking,
      agentError,
      setAgentError,
    }),
    [
      file,
      target,
      selectFile,
      profile,
      profiling,
      profileError,
      run,
      running,
      runError,
      answer,
      asking,
      agentError,
    ],
  );

  return (
    <DashboardStateContext.Provider value={value}>
      {children}
    </DashboardStateContext.Provider>
  );
}

/** The Dashboard's state, persisted above the routed page. Must be used under `DashboardStateProvider`. */
export function useDashboardState(): DashboardState {
  const context = useContext(DashboardStateContext);
  if (!context) {
    throw new Error(
      "useDashboardState must be used within a DashboardStateProvider",
    );
  }
  return context;
}
