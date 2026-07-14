// Build a layout-ready graph from either live spans or the final reconstructed graph.
import type { GraphJSON, GraphNode, SpanEvent } from "../api/types";

export interface VizNode {
  id: string;
  type: string; // agent_run | llm_call | tool_call
  label: string;
  sublabel: string;
  level: number;
  order: number; // reveal order (index the node appeared)
  tokensIn: number;
  tokensOut: number;
  durationNs: number;
  toolName?: string | null;
  isNested: boolean;
  isRepeat: boolean;
  repeatLabels: string[];
  step: number | null;
  x: number;
  y: number;
}

export interface VizEdge {
  source: string;
  target: string;
  repeat?: boolean;
}

export interface VizGraph {
  nodes: VizNode[];
  edges: VizEdge[];
  width: number;
  height: number;
  maxLevel: number;
}

export const COL_GAP = 210;
export const ROW_GAP = 92;
export const NODE_W = 158;
export const NODE_H = 56;

function spanLabel(ev: SpanEvent): { label: string; sublabel: string } {
  if (ev.node_type === "agent_run") return { label: "START", sublabel: "agent run" };
  if (ev.node_type === "llm_call")
    return { label: `LLM #${ev.step}`, sublabel: `${ev.input_tokens}→${ev.output_tokens} tok` };
  const name = ev.tool_name || "tool";
  const args = ev.tool_args || {};
  if (name === "read_document" && args.doc_id) return { label: `read ${args.doc_id}`, sublabel: "tool call" };
  if (name === "search" && args.query) return { label: `search`, sublabel: String(args.query).slice(0, 22) };
  if (name === "sub_agent") return { label: "sub_agent", sublabel: String(args.question || "").slice(0, 22) };
  if (name === "finish") return { label: "finish", sublabel: "tool call" };
  return { label: name.replace(/_/g, " "), sublabel: "tool call" };
}

// Longest-path level from root using depends_on edges (mirrors backend _levels_from_root).
function computeLevels(ids: string[], deps: Record<string, string[]>): Record<string, number> {
  const level: Record<string, number> = {};
  ids.forEach((id) => (level[id] = 0));
  // iterate to fixpoint (DAG, small)
  let changed = true;
  let guard = 0;
  while (changed && guard++ < ids.length + 2) {
    changed = false;
    for (const id of ids) {
      for (const d of deps[id] || []) {
        if (level[d] != null && level[d] + 1 > level[id]) {
          level[id] = level[d] + 1;
          changed = true;
        }
      }
    }
  }
  return level;
}

function place(nodes: VizNode[]): VizGraph {
  const byLevel: Record<number, VizNode[]> = {};
  let maxLevel = 0;
  for (const n of nodes) {
    (byLevel[n.level] ||= []).push(n);
    maxLevel = Math.max(maxLevel, n.level);
  }
  let maxRows = 1;
  Object.keys(byLevel).forEach((lv) => {
    const col = byLevel[+lv].sort((a, b) => (a.step ?? -1) - (b.step ?? -1) || a.id.localeCompare(b.id));
    maxRows = Math.max(maxRows, col.length);
    col.forEach((n, i) => {
      n.x = n.level * COL_GAP + NODE_W / 2 + 20;
      n.y = (i - (col.length - 1) / 2) * ROW_GAP;
    });
  });
  // shift y so min is a margin
  const minY = Math.min(...nodes.map((n) => n.y), 0);
  nodes.forEach((n) => (n.y += -minY + 40));
  return {
    nodes,
    edges: [],
    width: (maxLevel + 1) * COL_GAP + NODE_W + 40,
    height: maxRows * ROW_GAP + 80,
    maxLevel,
  };
}

export function fromSpans(spans: SpanEvent[]): VizGraph {
  const seen = new Map<string, SpanEvent>();
  spans.forEach((s) => {
    if (s.node_id) seen.set(s.node_id, s);
  });
  const ids = [...seen.keys()];
  const deps: Record<string, string[]> = {};
  ids.forEach((id) => (deps[id] = (seen.get(id)!.depends_on || []).filter((d) => seen.has(d))));
  const levels = computeLevels(ids, deps);

  const nodes: VizNode[] = ids.map((id, idx) => {
    const ev = seen.get(id)!;
    const { label, sublabel } = spanLabel(ev);
    return {
      id,
      type: ev.node_type,
      label,
      sublabel,
      level: levels[id] || 0,
      order: idx,
      tokensIn: ev.input_tokens,
      tokensOut: ev.output_tokens,
      durationNs: ev.duration_ns || 0,
      toolName: ev.tool_name,
      isNested: ev.is_nested,
      isRepeat: false,
      repeatLabels: [],
      step: ev.step,
      x: 0,
      y: 0,
    };
  });
  const g = place(nodes);
  const nodeSet = new Set(ids);
  const edges: VizEdge[] = [];
  ids.forEach((id) => (deps[id] || []).forEach((d) => nodeSet.has(d) && edges.push({ source: d, target: id })));
  g.edges = edges;
  return g;
}

export function fromGraphJSON(gj: GraphJSON): VizGraph {
  const nodes: VizNode[] = gj.nodes.map((n: GraphNode, idx) => ({
    id: n.id,
    type: n.type,
    label: n.label,
    sublabel:
      n.type === "llm_call"
        ? `${n.input_tokens}→${n.output_tokens} tok`
        : n.type === "tool_call"
        ? "tool call"
        : "agent run",
    level: n.level,
    order: idx,
    tokensIn: n.input_tokens,
    tokensOut: n.output_tokens,
    durationNs: n.duration_ns,
    toolName: n.tool_name,
    isNested: n.is_nested,
    isRepeat: n.is_repeat,
    repeatLabels: n.repeat_labels || [],
    step: n.step,
    x: 0,
    y: 0,
  }));
  const g = place(nodes);
  const repeatPairs = new Set(gj.behavioral_repeats.map((r) => `${r.from_node}->${r.to_node}`));
  g.edges = gj.edges.map((e) => ({ ...e }));
  gj.behavioral_repeats.forEach((r) => g.edges.push({ source: r.from_node, target: r.to_node, repeat: true }));
  void repeatPairs;
  return g;
}
