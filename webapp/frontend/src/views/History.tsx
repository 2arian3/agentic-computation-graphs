import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { HistoryEntry } from "../api/types";
import { Card } from "../components/Card";
import { fmtInt, fmtS, fmtTime } from "../lib/format";

export default function History({ onRerun }: { onRerun: (e: HistoryEntry) => void }) {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [detail, setDetail] = useState<any | null>(null);

  const load = () => api.history().then((r) => setEntries(r.entries));
  useEffect(() => { load(); }, []);

  const del = async (id: string) => {
    if (!confirm("Delete this history entry?")) return;
    await api.historyDelete(id);
    load();
    if (detail?.__id === id) setDetail(null);
  };
  const open = async (e: HistoryEntry) => {
    const d = await api.historyGet(e.id);
    setDetail({ ...d, __id: e.id });
  };

  return (
    <div>
      <div className="spread" style={{ marginBottom: 14 }}>
        <div>
          <h2 style={{ margin: 0 }}>Experiment history</h2>
          <div className="tiny faint">{entries.length} runs · stored under <span className="mono">webapp/data/history/</span></div>
        </div>
        <button className="btn" onClick={load}>⟳ Refresh</button>
      </div>

      <Card title="Runs">
        {!entries.length ? (
          <div className="empty">No runs yet. Runs (and saved replays) appear here automatically.</div>
        ) : (
          <div style={{ overflow: "auto" }}>
            <table className="tbl">
              <thead>
                <tr>
                  <th>when</th><th>mode</th><th>task</th><th>model</th><th>outcome</th>
                  <th>nodes</th><th>tokens</th><th>wall</th><th></th>
                </tr>
              </thead>
              <tbody>
                {entries.map((e) => (
                  <tr key={e.id}>
                    <td className="tiny">{fmtTime(e.timestamp)}</td>
                    <td><span className="tag">{e.mode}</span></td>
                    <td>{e.task_id}</td>
                    <td className="mono tiny">{e.model}</td>
                    <td><span className={`badge ${e.outcome}`}>{e.outcome}</span></td>
                    <td className="mono">{fmtInt(e.node_count)}</td>
                    <td className="mono">{fmtInt(e.total_tokens)}</td>
                    <td className="mono">{fmtS(e.wall_clock_s)}</td>
                    <td>
                      <div className="btn-row">
                        <button className="btn sm ghost" onClick={() => open(e)}>view</button>
                        <button className="btn sm" onClick={() => onRerun(e)}>rerun</button>
                        <button className="btn sm danger ghost" onClick={() => del(e.id)}>✕</button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {detail && (
        <div className="modal-back" onClick={() => setDetail(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3 style={{ margin: 0 }}>{detail.task_id} · {detail.run_id}</h3>
              <div className="spacer" style={{ flex: 1 }} />
              <button className="btn sm ghost" onClick={() => setDetail(null)}>close</button>
            </div>
            <div className="modal-body">
              <div className="spread" style={{ marginBottom: 8 }}>
                <span className={`badge ${detail.outcome}`}>{detail.outcome}</span>
                <span className="tiny faint">{detail.mode}</span>
              </div>
              <div style={{ fontWeight: 600, marginBottom: 6 }}>{detail.question}</div>
              {detail.answer && <div className="doc-hit" style={{ marginBottom: 12 }}><span className="dim small">answer:</span> {detail.answer}</div>}
              <div className="tiny dim" style={{ marginBottom: 4 }}>CONFIG</div>
              <pre className="code" style={{ maxHeight: 180 }}>{JSON.stringify(detail.config, null, 2)}</pre>
              <div className="tiny dim" style={{ margin: "10px 0 4px" }}>REPORT</div>
              <pre className="code" style={{ maxHeight: 220 }}>{JSON.stringify(detail.report, null, 2)}</pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
