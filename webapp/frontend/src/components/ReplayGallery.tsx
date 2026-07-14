import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { TraceInfo } from "../api/types";
import { fmtBytes } from "../lib/format";
import { useRun } from "../state/store";
import { Card } from "./Card";

export default function ReplayGallery() {
  const { replay, state } = useRun();
  const [traces, setTraces] = useState<TraceInfo[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [runs, setRuns] = useState<Record<string, any[]>>({});
  const [speed, setSpeed] = useState(1);

  useEffect(() => {
    api.traces().then((t) => setTraces(t.traces.filter((x) => x.num_runs > 0)));
  }, []);

  const toggle = async (file: string) => {
    if (expanded === file) return setExpanded(null);
    setExpanded(file);
    if (!runs[file]) {
      const r = await api.traceRuns(file);
      setRuns((m) => ({ ...m, [file]: r.runs }));
    }
  };

  const running = state.status === "running";

  return (
    <Card
      title="Replay archived traces"
      collapsible
      right={
        <label className="switch tiny">
          speed
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} style={{ width: 70 }}>
            <option value={0}>instant</option>
            <option value={1}>1×</option>
            <option value={2}>2×</option>
            <option value={4}>4×</option>
          </select>
        </label>
      }
    >
      <div className="tiny faint" style={{ marginBottom: 10 }}>
        {traces.length} trace files in <span className="mono">traces/</span>. Streams a recorded run through the same
        visualization — works with the model server down.
      </div>
      <div style={{ maxHeight: 360, overflow: "auto" }}>
        {traces.map((t) => (
          <div key={t.file} className="list-item" style={{ marginBottom: 7, padding: "9px 11px" }}>
            <div className="spread" style={{ cursor: "pointer" }} onClick={() => toggle(t.file)}>
              <div>
                <span className="mono small">{t.name}</span>
                <div className="tiny faint">
                  {t.num_runs} runs · {t.num_spans} spans · {fmtBytes(t.size_bytes)}
                  {t.tasks.length ? ` · ${t.tasks.slice(0, 6).join(", ")}` : ""}
                </div>
              </div>
              <button
                className="btn sm"
                disabled={running}
                onClick={(e) => {
                  e.stopPropagation();
                  replay({ file: t.file, speed });
                }}
              >
                ▶ replay first
              </button>
            </div>
            {expanded === t.file && runs[t.file] && (
              <div style={{ marginTop: 8, borderTop: "1px solid var(--border)", paddingTop: 8 }}>
                <table className="tbl">
                  <thead>
                    <tr><th>run</th><th>task</th><th>outcome</th><th>nodes</th><th>tokens</th><th></th></tr>
                  </thead>
                  <tbody>
                    {runs[t.file].slice(0, 40).map((r) => (
                      <tr key={r.trace_id}>
                        <td className="mono tiny">{r.run_id || r.trace_id.slice(0, 8)}</td>
                        <td>{r.task_id}</td>
                        <td><span className={`badge ${r.outcome}`}>{r.outcome}</span></td>
                        <td className="mono">{r.node_count}</td>
                        <td className="mono">{r.total_tokens}</td>
                        <td>
                          <button className="btn sm ghost" disabled={running}
                            onClick={() => replay({ file: t.file, trace_id: r.trace_id, speed })}>
                            replay
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        ))}
      </div>
    </Card>
  );
}
