import { useMemo } from "react";
import { useRun } from "../state/store";

// Surfaces the model's reasoning wherever the trace exposes it:
//  - LLM completion content (free-text the model wrote before/with a tool call),
//  - the `thought` argument added to tool calls when elicit_reasoning is on.
export default function ReasoningPanel() {
  const { state, select } = useRun();

  const items = useMemo(() => {
    const out: { id: string; kind: string; step: number | null; text: string; nested: boolean }[] = [];
    for (const s of state.spans) {
      if (s.node_type === "llm_call" && s.completion?.content?.trim()) {
        out.push({ id: s.node_id, kind: "LLM", step: s.step, text: s.completion.content.trim(), nested: s.is_nested });
      }
      if (s.node_type === "tool_call" && (s.tool_args || {}).thought) {
        out.push({ id: s.node_id, kind: `→ ${s.tool_name}`, step: s.step, text: String(s.tool_args!.thought), nested: s.is_nested });
      }
    }
    return out;
  }, [state.spans]);

  if (!items.length) {
    return (
      <div className="empty">
        No verbalized reasoning in this run. Enable <span className="mono">elicit_reasoning</span> to require a
        “thought” on every tool call, or the model may reason silently.
      </div>
    );
  }

  return (
    <div>
      {items.map((it, i) => (
        <div key={it.id + i} className="doc-hit" style={{ cursor: "pointer" }} onClick={() => select(it.id)}>
          <div className="tiny dim" style={{ marginBottom: 3 }}>
            step {it.step} · <span className="mono">{it.kind}</span> {it.nested && <span className="nested-pill">sub-agent</span>}
          </div>
          <div className="small">{it.text}</div>
        </div>
      ))}
    </div>
  );
}
