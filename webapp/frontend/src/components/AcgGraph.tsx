import { useEffect, useMemo, useRef, useState } from "react";
import { fromGraphJSON, fromSpans, NODE_H, NODE_W, VizGraph } from "../lib/graph";
import { useRun } from "../state/store";

const COLORS: Record<string, string> = {
  agent_run: "#8a94a6",
  llm_call: "#4c78a8",
  tool_call: "#f58518",
};

export default function AcgGraph({ tall }: { tall?: boolean }) {
  const { state, select, reveal, setFollow } = useRun();
  const wrapRef = useRef<HTMLDivElement>(null);
  const [t, setT] = useState({ x: 20, y: 10, k: 1 });
  const dragRef = useRef<{ x: number; y: number; ox: number; oy: number } | null>(null);
  const [fitToken, setFitToken] = useState(0);

  const viz: VizGraph = useMemo(() => {
    if (state.finished) return fromGraphJSON(state.finished.graph);
    return fromSpans(state.spans);
  }, [state.finished, state.spans]);

  const totalSpans = state.spans.length;
  const revealedIds = useMemo(() => {
    if (state.revealCount >= totalSpans) return null; // null == all revealed
    return new Set(state.spans.slice(0, state.revealCount).map((s) => s.node_id));
  }, [state.spans, state.revealCount, totalSpans]);

  const isRevealed = (id: string) => revealedIds == null || revealedIds.has(id);

  // Fit-to-view.
  const fit = () => {
    const el = wrapRef.current;
    if (!el || viz.nodes.length === 0) return;
    const pad = 40;
    const kw = (el.clientWidth - pad) / Math.max(viz.width, 1);
    const kh = (el.clientHeight - pad) / Math.max(viz.height, 1);
    const k = Math.min(1.1, Math.max(0.25, Math.min(kw, kh)));
    setT({
      x: (el.clientWidth - viz.width * k) / 2,
      y: (el.clientHeight - viz.height * k) / 2,
      k,
    });
  };
  useEffect(fit, [fitToken]);
  // auto-fit when the graph first appears / finishes
  useEffect(() => {
    if (viz.nodes.length) setFitToken((x) => x + 1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.finished ? "final" : "live", viz.nodes.length <= 1]);

  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const el = wrapRef.current!;
    const rect = el.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    setT((prev) => {
      const k = Math.min(2.5, Math.max(0.15, prev.k * factor));
      return { k, x: mx - (mx - prev.x) * (k / prev.k), y: my - (my - prev.y) * (k / prev.k) };
    });
  };
  const onDown = (e: React.MouseEvent) => {
    dragRef.current = { x: e.clientX, y: e.clientY, ox: t.x, oy: t.y };
  };
  const onMove = (e: React.MouseEvent) => {
    if (!dragRef.current) return;
    setT((p) => ({ ...p, x: dragRef.current!.ox + (e.clientX - dragRef.current!.x), y: dragRef.current!.oy + (e.clientY - dragRef.current!.y) }));
  };
  const onUp = () => (dragRef.current = null);

  const sel = state.selectedNode;
  const nodeById = useMemo(() => Object.fromEntries(viz.nodes.map((n) => [n.id, n])), [viz]);

  const edgePath = (s: string, d: string) => {
    const a = nodeById[s], b = nodeById[d];
    if (!a || !b) return "";
    const x0 = a.x + NODE_W / 2, y0 = a.y;
    const x1 = b.x - NODE_W / 2, y1 = b.y;
    const mx = (x0 + x1) / 2;
    return `M ${x0} ${y0} C ${mx} ${y0}, ${mx} ${y1}, ${x1} ${y1}`;
  };

  return (
    <div>
      <div className={`graph-wrap ${tall ? "tall" : ""}`} ref={wrapRef} onWheel={onWheel}
        onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp}
        style={{ cursor: dragRef.current ? "grabbing" : "grab" }}>
        <div className="graph-controls">
          <button className="icon-btn" onClick={() => setT((p) => ({ ...p, k: Math.min(2.5, p.k * 1.2) }))}>+</button>
          <button className="icon-btn" onClick={() => setT((p) => ({ ...p, k: Math.max(0.15, p.k / 1.2) }))}>−</button>
          <button className="icon-btn" onClick={() => setFitToken((x) => x + 1)}>Fit</button>
        </div>

        {viz.nodes.length === 0 ? (
          <div className="empty" style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center" }}>
            The ACG will build here as the run executes.
          </div>
        ) : (
          <svg width="100%" height="100%">
            <defs>
              <marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L7,3 L0,6 Z" fill="var(--border-strong)" />
              </marker>
              <marker id="arrow-rep" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth">
                <path d="M0,0 L7,3 L0,6 Z" fill="var(--accent-2)" />
              </marker>
            </defs>
            <g transform={`translate(${t.x},${t.y}) scale(${t.k})`}>
              {viz.edges.map((e, i) => {
                const shown = isRevealed(e.source) && isRevealed(e.target);
                return (
                  <path key={i} d={edgePath(e.source, e.target)} fill="none"
                    stroke={e.repeat ? "var(--accent-2)" : "var(--border-strong)"}
                    strokeWidth={e.repeat ? 1.6 : 1.6}
                    strokeDasharray={e.repeat ? "5 4" : undefined}
                    markerEnd={e.repeat ? "url(#arrow-rep)" : "url(#arrow)"}
                    opacity={shown ? (e.repeat ? 0.9 : 0.75) : 0.08} />
                );
              })}
              {viz.nodes.map((n) => {
                const revealed = isRevealed(n.id);
                const selected = sel === n.id;
                return (
                  <g key={n.id} className={`gnode ${selected ? "selected" : ""} ${revealed ? "" : "faded"}`}
                    transform={`translate(${n.x - NODE_W / 2},${n.y - NODE_H / 2})`}
                    onClick={(e) => { e.stopPropagation(); select(selected ? null : n.id); }}>
                    <rect width={NODE_W} height={NODE_H} rx={9}
                      fill={COLORS[n.type] || "#666"}
                      stroke={n.isNested ? "var(--accent-2)" : "rgba(0,0,0,.35)"}
                      strokeWidth={n.isNested ? 2 : 1}
                      strokeDasharray={n.isNested ? "4 3" : undefined} />
                    <text className="gnode-label" x={NODE_W / 2} y={n.sublabel ? 23 : 30} textAnchor="middle">
                      {n.label}
                    </text>
                    {n.sublabel && (
                      <text className="gnode-sub" x={NODE_W / 2} y={38} textAnchor="middle">{n.sublabel}</text>
                    )}
                    {n.isRepeat && (
                      <circle cx={NODE_W - 10} cy={10} r={5} fill="var(--accent-2)">
                        <title>re-reasoning loop: {n.repeatLabels.join("; ")}</title>
                      </circle>
                    )}
                  </g>
                );
              })}
            </g>
          </svg>
        )}

        <div className="graph-legend">
          <span className="legend-item"><span className="legend-swatch" style={{ background: COLORS.llm_call }} /> LLM call</span>
          <span className="legend-item"><span className="legend-swatch" style={{ background: COLORS.tool_call }} /> tool call</span>
          <span className="legend-item"><span className="legend-swatch" style={{ background: COLORS.agent_run }} /> agent run</span>
          <span className="legend-item"><span className="legend-swatch" style={{ background: "var(--accent-2)", borderRadius: 8 }} /> sub-agent / repeat</span>
        </div>
      </div>

      {/* Step-through scrubber */}
      {totalSpans > 1 && (
        <div className="scrubber">
          <button className="btn sm" onClick={() => { setFollow(false); reveal(Math.max(1, state.revealCount - 1)); }}>◀</button>
          <input type="range" min={1} max={totalSpans} value={Math.min(state.revealCount, totalSpans)}
            onChange={(e) => reveal(Number(e.target.value))} />
          <button className="btn sm" onClick={() => { setFollow(false); reveal(Math.min(totalSpans, state.revealCount + 1)); }}>▶</button>
          <span className="mono tiny" style={{ minWidth: 96, textAlign: "right" }}>
            step {Math.min(state.revealCount, totalSpans)}/{totalSpans}
          </span>
          <button className="btn sm ghost" onClick={() => setFollow(true)} title="Reveal all / follow live">all</button>
        </div>
      )}
    </div>
  );
}
