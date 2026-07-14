import React, { createContext, useContext, useMemo, useReducer, useRef } from "react";
import { streamRun } from "../api/client";
import type { RunFinished, SpanEvent, StreamEvent } from "../api/types";

export type RunStatus = "idle" | "running" | "done" | "error";

interface RunState {
  status: RunStatus;
  mode: "live" | "replay" | null;
  runId: string | null;
  taskId: string | null;
  question: string;
  graded: boolean;
  spans: SpanEvent[]; // ordered as they arrive == construction order
  finished: RunFinished | null;
  answer: string | null;
  outcome: string | null;
  error: string | null;
  selectedNode: string | null;
  // Step-through scrubber: how many spans to reveal (Infinity => all).
  revealCount: number;
  follow: boolean; // auto-advance scrubber as events arrive
}

const initial: RunState = {
  status: "idle",
  mode: null,
  runId: null,
  taskId: null,
  question: "",
  graded: false,
  spans: [],
  finished: null,
  answer: null,
  outcome: null,
  error: null,
  selectedNode: null,
  revealCount: 0,
  follow: true,
};

type Action =
  | { type: "start"; mode: "live" | "replay" }
  | { type: "event"; ev: StreamEvent }
  | { type: "select"; id: string | null }
  | { type: "reveal"; n: number }
  | { type: "follow"; on: boolean }
  | { type: "abort" }
  | { type: "reset" };

function reducer(s: RunState, a: Action): RunState {
  switch (a.type) {
    case "start":
      return { ...initial, status: "running", mode: a.mode, follow: true };
    case "event": {
      const ev = a.ev;
      if (ev.kind === "run_started") {
        return {
          ...s,
          runId: ev.run_id,
          taskId: ev.task_id,
          question: ev.question,
          graded: ev.graded,
          status: "running",
        };
      }
      if (ev.kind === "span") {
        const spans = [...s.spans, ev];
        return { ...s, spans, revealCount: s.follow ? spans.length : s.revealCount };
      }
      if (ev.kind === "run_finished") {
        return {
          ...s,
          finished: ev,
          answer: ev.answer,
          outcome: ev.outcome,
          status: "done",
          revealCount: s.follow ? s.spans.length : s.revealCount,
        };
      }
      if (ev.kind === "error") {
        return { ...s, status: "error", error: ev.error };
      }
      return s;
    }
    case "select":
      return { ...s, selectedNode: a.id };
    case "reveal":
      return { ...s, follow: false, revealCount: Math.max(0, a.n) };
    case "follow":
      return { ...s, follow: a.on, revealCount: a.on ? s.spans.length : s.revealCount };
    case "abort":
      return { ...s, status: s.status === "running" ? "idle" : s.status };
    case "reset":
      return initial;
    default:
      return s;
  }
}

interface RunContextValue {
  state: RunState;
  run: (body: any) => void;
  replay: (body: any) => void;
  abort: () => void;
  select: (id: string | null) => void;
  reveal: (n: number) => void;
  setFollow: (on: boolean) => void;
}

const RunContext = createContext<RunContextValue | null>(null);

export function RunProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initial);
  const abortRef = useRef<(() => void) | null>(null);

  const startStream = (path: "/api/runs" | "/api/replay", body: any, mode: "live" | "replay") => {
    abortRef.current?.();
    dispatch({ type: "start", mode });
    abortRef.current = streamRun(
      path,
      body,
      (ev) => dispatch({ type: "event", ev }),
      () => {},
      (msg) => dispatch({ type: "event", ev: { kind: "error", run_id: "", error: msg } }),
    );
  };

  const value: RunContextValue = useMemo(
    () => ({
      state,
      run: (body) => startStream("/api/runs", body, "live"),
      replay: (body) => startStream("/api/replay", body, "replay"),
      abort: () => {
        abortRef.current?.();
        dispatch({ type: "abort" });
      },
      select: (id) => dispatch({ type: "select", id }),
      reveal: (n) => dispatch({ type: "reveal", n }),
      setFollow: (on) => dispatch({ type: "follow", on }),
    }),
    [state],
  );

  return <RunContext.Provider value={value}>{children}</RunContext.Provider>;
}

export function useRun() {
  const ctx = useContext(RunContext);
  if (!ctx) throw new Error("useRun must be used inside RunProvider");
  return ctx;
}
