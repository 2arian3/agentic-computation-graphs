import { useMemo } from "react";
import { useRun } from "../state/store";

export default function RetrievalPanel() {
  const { state, select } = useRun();

  const searches = useMemo(
    () =>
      state.spans
        .filter((s) => s.node_type === "tool_call" && s.tool_name === "search")
        .map((s) => ({
          id: s.node_id,
          query: (s.tool_args || {}).query,
          results: (s.tool_result?.results as any[]) || [],
          nested: s.is_nested,
        })),
    [state.spans],
  );
  const reads = useMemo(
    () =>
      state.spans
        .filter((s) => s.node_type === "tool_call" && s.tool_name === "read_document")
        .map((s) => ({ id: s.node_id, docId: (s.tool_args || {}).doc_id, title: s.tool_result?.title })),
    [state.spans],
  );

  if (!searches.length && !reads.length) {
    return <div className="empty">Retrieval (search + read) will appear here as the agent queries the store.</div>;
  }

  return (
    <div>
      {searches.map((s, i) => (
        <div key={s.id} style={{ marginBottom: 14 }}>
          <div className="spread" style={{ marginBottom: 6 }}>
            <div className="small">
              <span className="dim">query #{i + 1}: </span>
              <span className="mono">{String(s.query || "")}</span>{" "}
              {s.nested && <span className="nested-pill">sub-agent</span>}
            </div>
            <button className="btn sm ghost" onClick={() => select(s.id)}>show node</button>
          </div>
          {s.results.map((r) => (
            <div key={r.doc_id} className="doc-hit">
              <div className="spread">
                <div><span className="mono small">{r.doc_id}</span> · <b className="small">{r.title}</b></div>
                <span className="score-chip">score {r.score}</span>
              </div>
              <div className="tiny faint" style={{ marginTop: 4 }}>{r.snippet}</div>
            </div>
          ))}
        </div>
      ))}
      {reads.length > 0 && (
        <div>
          <div className="tiny dim" style={{ margin: "6px 0" }}>DOCUMENTS READ ({reads.length})</div>
          {reads.map((r, i) => (
            <span key={r.id + i} className="tag" style={{ marginRight: 6, cursor: "pointer" }} onClick={() => select(r.id)}>
              {r.docId}{r.title ? ` · ${r.title}` : ""}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
