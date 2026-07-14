import type { Endpoint } from "../App";
import type { HistoryEntry } from "../api/types";
import AcgGraph from "../components/AcgGraph";
import { Card } from "../components/Card";
import ExecutionTimeline from "../components/ExecutionTimeline";
import ExperimentPanel from "../components/ExperimentPanel";
import NodeInspector from "../components/NodeInspector";
import ReasoningPanel from "../components/ReasoningPanel";
import ReplayGallery from "../components/ReplayGallery";
import ReportPanel from "../components/ReportPanel";
import RetrievalPanel from "../components/RetrievalPanel";
import { useRun } from "../state/store";

function RunBanner() {
  const { state } = useRun();
  if (!state.question && state.status === "idle") return null;
  const outcome = state.outcome || (state.status === "running" ? "running" : "");
  return (
    <div className="card">
      <div className="card-body">
        <div className="spread" style={{ marginBottom: 8 }}>
          <div className="tiny dim">
            {state.mode === "replay" ? "REPLAY" : "LIVE RUN"}
            {state.taskId ? ` · ${state.taskId}` : ""}
            {state.runId ? ` · ${state.runId}` : ""}
          </div>
          {outcome && (
            <span className={`badge ${state.status === "running" ? "" : outcome}`}>
              {state.status === "running" ? <><span className="spinner" /> running</> : outcome}
            </span>
          )}
        </div>
        <div style={{ fontWeight: 600, marginBottom: 6 }}>{state.question || "—"}</div>
        {state.status === "error" ? (
          <div className="badge incorrect" style={{ whiteSpace: "normal" }}>error: {state.error}</div>
        ) : state.answer ? (
          <div className="doc-hit" style={{ background: "var(--bg-elev2)" }}>
            <span className="dim small">final answer:</span> {state.answer}
          </div>
        ) : state.status === "running" ? (
          <div className="progressbar"><span style={{ width: `${Math.min(95, state.spans.length * 8)}%` }} /></div>
        ) : null}
      </div>
    </div>
  );
}

export default function Dashboard({
  serverUp,
  prefill,
  onPrefillConsumed,
  endpoint,
  setEndpoint,
}: {
  serverUp: boolean;
  prefill: HistoryEntry | null;
  onPrefillConsumed: () => void;
  endpoint: Endpoint | null;
  setEndpoint: (e: Endpoint) => void;
}) {
  return (
    <div className="dash-grid">
      <div className="col-stack">
        <ExperimentPanel
          serverUp={serverUp}
          prefill={prefill}
          onPrefillConsumed={onPrefillConsumed}
          endpoint={endpoint}
          setEndpoint={setEndpoint}
        />
        <ReplayGallery />
      </div>

      <div className="col-stack">
        <RunBanner />
        <Card title="Agentic Computation Graph" right={<span className="tiny faint">scroll to zoom · drag to pan · click a node</span>}>
          <AcgGraph tall />
        </Card>

        <div className="grid-2">
          <Card title="Execution timeline" collapsible>
            <ExecutionTimeline />
          </Card>
          <Card title="Node inspector" collapsible>
            <NodeInspector />
          </Card>
        </div>

        <div className="grid-2">
          <Card title="Retrieval" collapsible>
            <RetrievalPanel />
          </Card>
          <Card title="Model reasoning" collapsible>
            <ReasoningPanel />
          </Card>
        </div>

        <Card title="Execution report">
          <ReportPanel />
        </Card>
      </div>
    </div>
  );
}
