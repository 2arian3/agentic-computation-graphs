import { useState } from "react";
import type { SpanEvent } from "../api/types";
import { fmtNs, prettyJSON, truncate } from "../lib/format";
import { useRun } from "../state/store";

function StepBody({ ev }: { ev: SpanEvent }) {
  const [showPrompt, setShowPrompt] = useState(false);
  if (ev.node_type === "llm_call") {
    const tcs = ev.completion?.tool_calls || [];
    return (
      <div>
        {ev.completion?.content ? (
          <div className="small" style={{ marginBottom: 8 }}>
            <span className="dim">reasoning: </span>{truncate(ev.completion.content, 500)}
          </div>
        ) : null}
        {tcs.length ? (
          <div className="small">
            <span className="dim">decided to call:</span>
            <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>
              {tcs.map((tc) => (
                <li key={tc.id} className="mono tiny">
                  {tc.name}(<span className="dim">{truncate(tc.arguments, 90)}</span>)
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <div className="small dim">no tool call → produced the final answer.</div>
        )}
        <button className="btn sm ghost" style={{ marginTop: 8 }} onClick={() => setShowPrompt((s) => !s)}>
          {showPrompt ? "hide" : "show"} prompt sent
        </button>
        {showPrompt && <pre className="code" style={{ marginTop: 8 }}>{prettyJSON(ev.prompt)}</pre>}
      </div>
    );
  }
  if (ev.node_type === "tool_call") {
    const args = ev.tool_args || {};
    if (args.thought) {
      // reasoning-elicitation mode surfaces the model's rationale on the tool args
    }
    return (
      <div>
        {args.thought && (
          <div className="small" style={{ marginBottom: 6 }}>
            <span className="nested-pill">thought</span> {truncate(String(args.thought), 300)}
          </div>
        )}
        <div className="kv small" style={{ marginBottom: 8 }}>
          <span className="k">input</span>
          <span className="v">{prettyJSON({ ...args, thought: undefined })}</span>
        </div>
        <details>
          <summary className="small dim" style={{ cursor: "pointer" }}>output</summary>
          <pre className="code" style={{ marginTop: 6 }}>{prettyJSON(ev.tool_result)}</pre>
        </details>
      </div>
    );
  }
  return <div className="small dim">Question: {ev.question}</div>;
}

export default function ExecutionTimeline() {
  const { state, select } = useRun();
  const [open, setOpen] = useState<Set<string>>(new Set());
  const spans = state.spans;
  const reveal = state.revealCount;

  if (!spans.length) {
    return (
      <div className="empty">
        {state.status === "running" ? (
          <><span className="spinner" /> waiting for the first step…</>
        ) : (
          "Run an experiment (or replay a trace) to see each step stream in."
        )}
      </div>
    );
  }

  const toggle = (id: string) =>
    setOpen((s) => {
      const n = new Set(s);
      n.has(id) ? n.delete(id) : n.add(id);
      return n;
    });

  return (
    <div className="timeline">
      {spans.map((ev, i) => {
        const id = ev.node_id + ":" + i;
        const dimmed = i >= reveal;
        const isOpen = open.has(id);
        const selected = state.selectedNode === ev.node_id;
        const ico = ev.node_type === "llm_call" ? "LLM" : ev.node_type === "tool_call" ? "T" : "▶";
        const title =
          ev.node_type === "llm_call"
            ? `LLM call #${ev.step}`
            : ev.node_type === "tool_call"
            ? ev.tool_name
            : "Agent run started";
        const sub =
          ev.node_type === "llm_call"
            ? `${ev.tool_call_count ?? 0} tool call(s) chosen`
            : ev.node_type === "tool_call"
            ? Object.entries(ev.tool_args || {})
                .filter(([k]) => k !== "thought")
                .map(([k, v]) => `${k}=${truncate(String(v), 30)}`)
                .join("  ")
            : ev.question;
        return (
          <div key={id} className={`step ${dimmed ? "dimmed" : ""}`} style={selected ? { borderColor: "var(--accent)" } : undefined}>
            <div className="step-head" onClick={() => { toggle(id); select(ev.node_id); }}>
              <span className={`step-ico ${ev.node_type}`}>{ico}</span>
              <div style={{ minWidth: 0 }}>
                <div className="step-title">
                  {title}{" "}
                  {ev.is_nested && <span className="nested-pill">sub-agent</span>}
                </div>
                <div className="step-sub">{truncate(String(sub || ""), 80)}</div>
              </div>
              <div className="step-meta">
                {ev.node_type === "llm_call" && (
                  <span className="tag">{ev.input_tokens}→{ev.output_tokens} tok</span>
                )}
                <span className="tag">{fmtNs(ev.duration_ns)}</span>
                <span className="chev" style={{ transform: isOpen ? "rotate(0)" : "rotate(-90deg)" }}>▾</span>
              </div>
            </div>
            {isOpen && (
              <div className="step-body">
                <StepBody ev={ev} />
              </div>
            )}
          </div>
        );
      })}
      {state.status === "running" && (
        <div className="step-sub dim" style={{ padding: "4px 6px" }}>
          <span className="spinner" /> executing…
        </div>
      )}
    </div>
  );
}
