import { useMemo } from "react";
import type { Report } from "../api/types";
import { fmtInt, fmtS } from "../lib/format";
import { useRun } from "../state/store";

function liveReport(spans: any[]): Partial<Report> {
  let inTok = 0, outTok = 0, llm = 0, tool = 0, searches = 0, reads = 0, subs = 0;
  for (const s of spans) {
    if (s.node_type === "llm_call") { llm++; inTok += s.input_tokens || 0; outTok += s.output_tokens || 0; }
    if (s.node_type === "tool_call") {
      tool++;
      if (s.tool_name === "search") searches++;
      if (s.tool_name === "read_document") reads++;
      if (s.tool_name === "sub_agent") subs++;
    }
  }
  return {
    num_llm_calls: llm, num_tool_calls: tool, num_searches: searches, num_reads: reads,
    num_sub_agents: subs, reasoning_iterations: llm, input_tokens: inTok, output_tokens: outTok,
    total_tokens: inTok + outTok,
  };
}

function Metric({ v, label }: { v: any; label: string }) {
  return (
    <div className="metric">
      <div className="m-val">{v}</div>
      <div className="m-lbl">{label}</div>
    </div>
  );
}

export default function ReportPanel() {
  const { state } = useRun();
  const r = state.finished?.report;
  const live = useMemo(() => liveReport(state.spans), [state.spans]);
  const rep: Partial<Report> = r || live;

  if (!state.spans.length && !r) {
    return <div className="empty">Execution statistics will populate here.</div>;
  }

  const st = r?.stage_times;
  const total = st ? Math.max(st.llm_s + st.tool_s + st.overhead_s, 0.0001) : 0;
  const tb = rep.tool_breakdown || {};

  return (
    <div>
      <div className="metric-grid">
        <Metric v={fmtInt(rep.node_count)} label="ACG nodes" />
        <Metric v={fmtInt(rep.num_llm_calls)} label="LLM calls" />
        <Metric v={fmtInt(rep.num_tool_calls)} label="tool calls" />
        <Metric v={fmtInt(rep.depth)} label="depth" />
        <Metric v={fmtInt(rep.width)} label="width (emitted)" />
        <Metric v={fmtInt(rep.width_executed)} label="width (executed)" />
        <Metric v={fmtInt(rep.total_tokens)} label="total tokens" />
        <Metric v={r ? fmtS(r.wall_clock_s) : "…"} label="wall clock" />
      </div>

      <div className="divider" />
      <div className="tiny dim" style={{ marginBottom: 6 }}>PIPELINE ACTIVITY</div>
      <div className="metric-grid">
        <Metric v={fmtInt(rep.num_searches)} label="retrievals" />
        <Metric v={fmtInt(rep.num_reads)} label="docs read" />
        <Metric v={fmtInt(rep.num_sub_agents)} label="sub-agents" />
        <Metric v={fmtInt(rep.reasoning_iterations)} label="reasoning iters" />
      </div>

      {st && (
        <>
          <div className="divider" />
          <div className="spread" style={{ marginBottom: 6 }}>
            <span className="tiny dim">STAGE TIME</span>
            <span className="tiny faint">LLM {fmtS(st.llm_s)} · tool {fmtS(st.tool_s)} · overhead {fmtS(st.overhead_s)}</span>
          </div>
          <div className="bar-track">
            <span style={{ width: `${(st.llm_s / total) * 100}%`, background: "var(--llm)" }} title={`LLM ${fmtS(st.llm_s)}`} />
            <span style={{ width: `${(st.tool_s / total) * 100}%`, background: "var(--tool)" }} title={`tool ${fmtS(st.tool_s)}`} />
            <span style={{ width: `${(st.overhead_s / total) * 100}%`, background: "var(--agent)" }} title={`overhead ${fmtS(st.overhead_s)}`} />
          </div>
        </>
      )}

      <div className="divider" />
      <div className="tiny dim" style={{ marginBottom: 6 }}>TOKENS & COST</div>
      <div className="kv small">
        <span className="k">input tokens</span><span className="v">{fmtInt(rep.input_tokens)}</span>
        <span className="k">output tokens</span><span className="v">{fmtInt(rep.output_tokens)}</span>
        <span className="k">est. cost</span>
        <span className="v">
          {r ? `$${(r.cost?.usd ?? 0).toFixed(6)}` : "—"} <span className="faint tiny">{r?.cost?.note}</span>
        </span>
        <span className="k">memory</span><span className="v faint">not exposed by the instrument</span>
      </div>

      {Object.keys(tb).length > 0 && (
        <>
          <div className="divider" />
          <div className="tiny dim" style={{ marginBottom: 6 }}>TOOL BREAKDOWN</div>
          <div>
            {Object.entries(tb).map(([k, v]) => (
              <span key={k} className="tag" style={{ marginRight: 6 }}>{k}: {v as any}</span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
