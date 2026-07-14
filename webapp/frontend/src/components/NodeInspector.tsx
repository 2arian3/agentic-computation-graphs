import { useMemo } from "react";
import { fmtNs, prettyJSON } from "../lib/format";
import { useRun } from "../state/store";

export default function NodeInspector() {
  const { state, select } = useRun();
  const id = state.selectedNode;

  const span = useMemo(
    () => [...state.spans].reverse().find((s) => s.node_id === id) || null,
    [state.spans, id],
  );
  const gnode = useMemo(
    () => state.finished?.graph.nodes.find((n) => n.id === id) || null,
    [state.finished, id],
  );

  if (!id) return <div className="empty">Click a node in the graph (or a step) to inspect it.</div>;

  const type = span?.node_type || gnode?.type || "?";
  return (
    <div>
      <div className="spread" style={{ marginBottom: 10 }}>
        <span className={`badge ${type === "llm_call" ? "llm" : type === "tool_call" ? "tool" : ""}`}>{type}</span>
        <button className="btn sm ghost" onClick={() => select(null)}>clear</button>
      </div>
      <div className="kv">
        <span className="k">node id</span><span className="v">{id}</span>
        {gnode?.level != null && (<><span className="k">graph level</span><span className="v">{gnode.level}</span></>)}
        {span?.step != null && (<><span className="k">step</span><span className="v">{span.step}</span></>)}
        {(span?.depends_on?.length ?? 0) > 0 && (<><span className="k">depends on</span><span className="v">{span!.depends_on.join(", ")}</span></>)}
        {span?.duration_ns != null && (<><span className="k">duration</span><span className="v">{fmtNs(span.duration_ns)}</span></>)}
        {type === "llm_call" && (
          <>
            <span className="k">model</span><span className="v">{span?.model}</span>
            <span className="k">temp / top_p</span><span className="v">{span?.temperature} / {span?.top_p}</span>
            <span className="k">seed</span><span className="v">{String(span?.seed ?? "—")}</span>
            <span className="k">tokens</span><span className="v">{span?.input_tokens} in / {span?.output_tokens} out</span>
            <span className="k">finish</span><span className="v">{span?.finish_reasons?.join(", ")}</span>
          </>
        )}
        {type === "tool_call" && (
          <>
            <span className="k">tool</span><span className="v">{span?.tool_name}</span>
            {span?.is_nested && (<><span className="k">nested</span><span className="v">yes (sub-agent subtree)</span></>)}
          </>
        )}
        {(gnode?.repeat_labels?.length ?? 0) > 0 && (
          <><span className="k">re-reasoning</span><span className="v">{gnode!.repeat_labels.join("; ")}</span></>
        )}
      </div>

      {type === "tool_call" && span?.tool_args && Object.keys(span.tool_args).length > 0 && (
        <>
          <div className="tiny dim" style={{ margin: "12px 0 4px" }}>INPUT</div>
          <pre className="code">{prettyJSON(span.tool_args)}</pre>
        </>
      )}
      {type === "tool_call" && span?.tool_result != null && (
        <>
          <div className="tiny dim" style={{ margin: "12px 0 4px" }}>OUTPUT</div>
          <pre className="code">{prettyJSON(span.tool_result)}</pre>
        </>
      )}
      {type === "llm_call" && span?.completion && (
        <>
          <div className="tiny dim" style={{ margin: "12px 0 4px" }}>COMPLETION</div>
          <pre className="code">{prettyJSON(span.completion)}</pre>
        </>
      )}
      {type === "llm_call" && span?.prompt && (
        <details style={{ marginTop: 10 }}>
          <summary className="small dim" style={{ cursor: "pointer" }}>prompt sent to model</summary>
          <pre className="code" style={{ marginTop: 6 }}>{prettyJSON(span.prompt)}</pre>
        </details>
      )}
    </div>
  );
}
