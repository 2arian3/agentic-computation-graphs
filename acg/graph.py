"""Offline reconstruction of the Agentic Computation Graph (ACG) from traces.

This module reads the JSONL span store produced by acg.tracing and, for each run
(one OTel trace), rebuilds the ACG as a directed acyclic graph and computes the
size/structure metrics the proposal calls for:

  * node count, split by type (LLM calls vs tool calls, and per tool)
  * edge count and the dependency structure
  * depth  -- the longest dependency chain (latency-related)
  * width  -- structural (emitted) branching: the most nodes sharing one dependency
              level, i.e. how many tool calls the model issued together. This is
              *potential* parallelism, not necessarily parallelism that happened.
  * width_executed -- realized parallelism: the most tool spans that actually overlap
              in wall-clock time. Requires a concurrent executor (acg/agent.py); on
              serial traces it is 1, which is the honest reading.
  * total tokens -- input + output summed across all LLM calls (cost-related)
  * wall-clock latency -- per node and for the whole run
  * task outcome

Edges are the explicit data dependencies recorded on each span (`acg.depends_on`):
"one step's output feeds another step's input". The OTel parent/child links are kept
as a cross-check. Reconstruction is pure and offline, so analysis never affects runs.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import networkx as nx

from . import tracing as T


# --------------------------------------------------------------------------- #
# Loading + grouping
# --------------------------------------------------------------------------- #
def load_spans(trace_file: str | Path) -> list[dict]:
    spans = []
    for line in Path(trace_file).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            spans.append(json.loads(line))
    return spans


def group_by_trace(spans: list[dict]) -> dict[str, list[dict]]:
    by_trace: dict[str, list[dict]] = defaultdict(list)
    for s in spans:
        by_trace[s["trace_id"]].append(s)
    return by_trace


def _attr(span: dict, key: str, default=None):
    return (span.get("attributes") or {}).get(key, default)


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def build_graph(spans: list[dict]) -> nx.DiGraph:
    """Build the ACG DAG for the spans of a single run/trace.

    Nodes are keyed by their stable acg.node_id. Edges go dep -> node, meaning the
    output of `dep` feeds the input of `node`.
    """
    g = nx.DiGraph()
    id_to_span = {}

    for s in spans:
        node_id = _attr(s, T.ACG_NODE_ID)
        if node_id is None:
            continue
        node_type = _attr(s, T.ACG_NODE_TYPE, "unknown")
        id_to_span[node_id] = s
        g.add_node(
            node_id,
            type=node_type,
            name=s.get("name"),
            step=_attr(s, T.ACG_STEP),
            tool_name=_attr(s, T.GEN_AI_TOOL_NAME),
            tool_args=_parse_tool_args(_attr(s, T.ACG_TOOL_ARGS)),
            input_tokens=_attr(s, T.GEN_AI_USAGE_INPUT_TOKENS, 0) or 0,
            output_tokens=_attr(s, T.GEN_AI_USAGE_OUTPUT_TOKENS, 0) or 0,
            duration_ns=s.get("duration_ns") or 0,
            start_time_ns=s.get("start_time_ns"),
            end_time_ns=s.get("end_time_ns"),
            outcome=_attr(s, T.ACG_OUTCOME),
            question=_attr(s, "acg.question"),
            answer=_attr(s, "acg.answer"),
            task_id=_attr(s, T.ACG_TASK_ID),
        )

    # Edges from explicit data dependencies.
    for s in spans:
        node_id = _attr(s, T.ACG_NODE_ID)
        if node_id is None:
            continue
        raw = _attr(s, T.ACG_DEPENDS_ON)
        deps = json.loads(raw) if isinstance(raw, str) else (raw or [])
        for dep in deps:
            if dep in g:                    # ignore dangling refs defensively
                g.add_edge(dep, node_id)

    return g


@dataclass
class BehavioralRepeat:
    """Behavioral re-reasoning loop: a later LLM step repeats an earlier LLM's tool decision."""
    kind: str              # "exact" | "same_tool"
    from_node: str         # later LLM (decides to repeat after more context)
    to_node: str           # earlier LLM (first made that tool choice)
    tool_name: str
    detail: str            # human-readable, e.g. "re-call read D03"


def _parse_tool_args(raw) -> dict:
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def _tool_args_signature(tool_name: str, args: dict) -> str | None:
    """Stable key for detecting exact repeats (same tool + same target)."""
    if tool_name == "read_document":
        doc_id = str(args.get("doc_id", "")).strip()
        return doc_id.lower() if doc_id else None
    if tool_name == "search":
        query = str(args.get("query", "")).strip().lower()
        return query if query else None
    return None


def _llm_parent(g: nx.DiGraph, node_id: str) -> str | None:
    """The LLM call whose tool output this node consumed (immediate predecessor)."""
    for pred in g.predecessors(node_id):
        if g.nodes[pred].get("type") == T.NODE_TYPE_LLM:
            return pred
    return None


def _add_llm_repeat(
    repeats: list[BehavioralRepeat],
    seen: dict[tuple[str, str], str],
    g: nx.DiGraph,
    *,
    first_tool: str,
    repeat_tool: str,
    kind: str,
    tool: str,
    detail: str,
) -> None:
    early_llm = _llm_parent(g, first_tool)
    late_llm = _llm_parent(g, repeat_tool)
    if not early_llm or not late_llm or early_llm == late_llm:
        return
    key = (late_llm, early_llm)
    if key in seen and seen[key] == "exact" and kind == "same_tool":
        return
    if key in seen and seen[key] == kind:
        return
    seen[key] = kind
    repeats.append(BehavioralRepeat(kind, late_llm, early_llm, tool, detail))
    g.nodes[late_llm]["is_repeat"] = True
    g.nodes[late_llm]["repeat_kind"] = kind
    g.nodes[late_llm].setdefault("repeat_labels", []).append(detail)


def detect_behavioral_repeats(g: nx.DiGraph) -> list[BehavioralRepeat]:
    """Find when a later LLM step re-decides a tool the agent already used.

    The loop is LLM → tool → … → LLM (re-reasons) → same tool again.
    We draw it as a semantic arc from the later LLM back to the earlier LLM.
    """
    for n, d in g.nodes(data=True):
        if d.get("type") in (T.NODE_TYPE_LLM, T.NODE_TYPE_TOOL):
            for key in ("is_repeat", "repeat_kind", "repeat_labels"):
                d.pop(key, None)

    repeats: list[BehavioralRepeat] = []
    seen_pairs: dict[tuple[str, str], str] = {}
    seen_exact: dict[tuple[str, str], str] = {}   # (tool, sig) -> first tool node
    seen_tool: dict[str, str] = {}                # tool_name -> first tool node

    for n in nx.topological_sort(g):
        d = g.nodes[n]
        if d.get("type") != T.NODE_TYPE_TOOL:
            continue
        tool = d.get("tool_name") or "unknown"
        if tool == "finish":
            continue
        sig = _tool_args_signature(tool, d.get("tool_args") or {})
        if sig:
            key = (tool, sig)
            if key in seen_exact:
                detail = f"re-call {tool.replace('_', ' ')} ({sig!r})"
                _add_llm_repeat(
                    repeats, seen_pairs, g,
                    first_tool=seen_exact[key], repeat_tool=n,
                    kind="exact", tool=tool, detail=detail,
                )
            else:
                seen_exact[key] = n
        if tool in seen_tool:
            _add_llm_repeat(
                repeats, seen_pairs, g,
                first_tool=seen_tool[tool], repeat_tool=n,
                kind="same_tool", tool=tool,
                detail=f"re-decide {tool.replace('_', ' ')}",
            )
        else:
            seen_tool[tool] = n

    return repeats


def _run_metadata(g: nx.DiGraph) -> dict:
    root = find_root(g)
    if root is None:
        return {}
    d = g.nodes[root]
    return {
        "question": d.get("question") or "",
        "answer": d.get("answer") or "",
        "outcome": d.get("outcome") or "unknown",
        "task_id": d.get("task_id") or "",
    }


def _wrap_text(text: str, width: int = 92) -> str:
    words = (text or "").split()
    if not words:
        return ""
    lines, cur = [], words[0]
    for w in words[1:]:
        if len(cur) + 1 + len(w) <= width:
            cur += " " + w
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return "\n".join(lines)


def find_root(g: nx.DiGraph) -> str | None:
    for n, d in g.nodes(data=True):
        if d.get("type") == T.NODE_TYPE_AGENT_RUN:
            return n
    # fallback: a node with no predecessors
    for n in g.nodes():
        if g.in_degree(n) == 0:
            return n
    return None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass
class ACGMetrics:
    run_id: str
    task_id: str
    trace_id: str
    outcome: str
    # size
    node_count: int               # LLM + tool nodes (excludes the synthetic root)
    num_llm_calls: int
    num_tool_calls: int
    tool_breakdown: dict          # {tool_name: count}
    edge_count: int               # dependency edges (includes root->first)
    # structure
    depth: int                    # longest dependency chain, in edges
    width: int                    # structural/emitted: max nodes sharing one dependency level
    width_executed: int           # realized: max tool spans overlapping in wall-clock time
    # cost
    input_tokens: int
    output_tokens: int
    total_tokens: int
    # latency
    wall_clock_s: float           # whole-run latency (root span duration)
    llm_time_s: float
    tool_time_s: float

    def to_row(self) -> dict:
        d = asdict(self)
        d.pop("tool_breakdown")
        for k, v in self.tool_breakdown.items():
            d[f"tool_{k}"] = v
        return d


def _max_temporal_overlap(intervals: list[tuple]) -> int:
    """Max number of [start_ns, end_ns] intervals open at the same instant (sweep line).

    This turns tool-span wall-clock times into *executed* parallelism: how many tool
    calls were genuinely running at once. Intervals that merely touch (one ends exactly
    as the next starts) do NOT count as overlap -- ends are processed before starts at an
    equal timestamp. Returns 0 when there are no valid intervals.
    """
    events: list[tuple[int, int]] = []
    for start, end in intervals:
        if start is None or end is None or end < start:
            continue
        events.append((start, 1))     # +1 at start
        events.append((end, -1))      # -1 at end
    events.sort(key=lambda e: (e[0], e[1]))   # at a tie, -1 (end) before +1 (start)
    active = peak = 0
    for _, delta in events:
        active += delta
        peak = max(peak, active)
    return peak


def _levels_from_root(g: nx.DiGraph, root: str) -> dict[str, int]:
    """Longest-path distance (in edges) from root to each node, via topo-order DP."""
    level = {n: 0 for n in g.nodes()}
    for n in nx.topological_sort(g):
        for succ in g.successors(n):
            if level[n] + 1 > level[succ]:
                level[succ] = level[n] + 1
    return level


def compute_metrics(g: nx.DiGraph, *, run_id="", task_id="", trace_id="") -> ACGMetrics:
    root = find_root(g)
    real = [n for n, d in g.nodes(data=True) if d.get("type") in (T.NODE_TYPE_LLM, T.NODE_TYPE_TOOL)]
    llm_nodes = [n for n in real if g.nodes[n]["type"] == T.NODE_TYPE_LLM]
    tool_nodes = [n for n in real if g.nodes[n]["type"] == T.NODE_TYPE_TOOL]

    tool_breakdown = Counter(g.nodes[n].get("tool_name") or "unknown" for n in tool_nodes)

    # depth / width over the full DAG (root included) — root anchors the chain.
    if g.number_of_nodes() and nx.is_directed_acyclic_graph(g):
        depth = nx.dag_longest_path_length(g)        # in edges
        if root is not None:
            levels = _levels_from_root(g, root)
            width = max(Counter(levels.values()).values()) if levels else 0
        else:
            width = 0
    else:
        depth = width = 0

    # Executed (realized) parallelism: how many tool spans overlapped in wall-clock time.
    # On serial traces this is 1 (or 0 with no tools); it exceeds 1 only when the executor
    # actually ran tool calls concurrently (see cfg.max_tool_workers).
    width_executed = _max_temporal_overlap([
        (g.nodes[n].get("start_time_ns"), g.nodes[n].get("end_time_ns"))
        for n in tool_nodes
    ])

    in_tok = sum(g.nodes[n]["input_tokens"] for n in llm_nodes)
    out_tok = sum(g.nodes[n]["output_tokens"] for n in llm_nodes)
    llm_time = sum(g.nodes[n]["duration_ns"] for n in llm_nodes) / 1e9
    tool_time = sum(g.nodes[n]["duration_ns"] for n in tool_nodes) / 1e9
    wall = (g.nodes[root]["duration_ns"] / 1e9) if root is not None else 0.0
    outcome = g.nodes[root].get("outcome") if root is not None else None

    return ACGMetrics(
        run_id=run_id, task_id=task_id, trace_id=trace_id, outcome=outcome or "unknown",
        node_count=len(real), num_llm_calls=len(llm_nodes), num_tool_calls=len(tool_nodes),
        tool_breakdown=dict(tool_breakdown), edge_count=g.number_of_edges(),
        depth=depth, width=width, width_executed=width_executed,
        input_tokens=in_tok, output_tokens=out_tok, total_tokens=in_tok + out_tok,
        wall_clock_s=round(wall, 4), llm_time_s=round(llm_time, 4), tool_time_s=round(tool_time, 4),
    )


@dataclass
class ReconstructedRun:
    trace_id: str
    run_id: str
    task_id: str
    graph: nx.DiGraph
    metrics: ACGMetrics


def reconstruct_runs(trace_file: str | Path) -> list[ReconstructedRun]:
    """Reconstruct every run found in a trace file."""
    runs = []
    for trace_id, spans in group_by_trace(load_spans(trace_file)).items():
        g = build_graph(spans)
        if g.number_of_nodes() == 0:
            continue
        root = find_root(g)
        run_id = (g.nodes[root].get("name") if root else None) or ""
        # pull run_id/task_id from any span attribute
        run_id = next((_attr(s, T.ACG_RUN_ID) for s in spans if _attr(s, T.ACG_RUN_ID)), "")
        task_id = next((_attr(s, T.ACG_TASK_ID) for s in spans if _attr(s, T.ACG_TASK_ID)), "")
        m = compute_metrics(g, run_id=run_id, task_id=task_id, trace_id=trace_id)
        runs.append(ReconstructedRun(trace_id, run_id, task_id, g, m))
    return runs


# --------------------------------------------------------------------------- #
# Rendering — "draw its graph from the captured trace" (Month-1 milestone)
# --------------------------------------------------------------------------- #
_COLOR_MAP = {
    T.NODE_TYPE_AGENT_RUN: ("#e0e0e0", "#333333"),
    T.NODE_TYPE_LLM: ("#4c78a8", "#ffffff"),
    T.NODE_TYPE_TOOL: ("#f58518", "#ffffff"),
}


def _node_label(g: nx.DiGraph, n: str) -> str:
    d = g.nodes[n]
    t = d.get("type")
    if t == T.NODE_TYPE_LLM:
        return f"LLM#{d.get('step')} ({d['input_tokens']}->{d['output_tokens']} tok)"
    if t == T.NODE_TYPE_TOOL:
        return f"tool:{d.get('tool_name')}"
    if t == T.NODE_TYPE_AGENT_RUN:
        return "START"
    return n


def _node_short_label(g: nx.DiGraph, n: str) -> tuple[str, str]:
    """Return (primary, secondary) labels for PNG boxes."""
    d = g.nodes[n]
    t = d.get("type")
    if t == T.NODE_TYPE_LLM:
        step = d.get("step")
        return (f"LLM call #{step}", f"{d['input_tokens']} in / {d['output_tokens']} out tok")
    if t == T.NODE_TYPE_TOOL:
        name = d.get("tool_name") or "tool"
        args = d.get("tool_args") or {}
        if name == "read_document" and args.get("doc_id"):
            primary = f"read {args['doc_id']}"
        elif name == "search" and args.get("query"):
            primary = f"search: {args['query']}"
        elif name == "finish" and args.get("answer"):
            primary = f"finish: {args['answer']}"
        else:
            primary = name.replace("_", " ")
        return (primary, "tool call")
    if t == T.NODE_TYPE_AGENT_RUN:
        return ("START", "agent run")
    return (n, "")


def _box_size_for_labels(primary: str, secondary: str) -> tuple[float, float]:
    """Data-coordinate box (width, height) sized to fit label text."""
    char_w_primary, char_w_secondary = 0.105, 0.088
    pad_x, pad_y = 0.65, 0.45
    min_w, max_w = 2.1, 8.5
    w = max(len(primary) * char_w_primary, len(secondary or "") * char_w_secondary) + pad_x
    w = max(min_w, min(max_w, w))
    h = 1.18 if secondary else 0.9
    return w, h


def _layout_positions(
    g: nx.DiGraph,
    node_sizes: dict[str, tuple[float, float]],
    *,
    x_gap: float = 0.75,
    y_gap: float = 2.5,
) -> dict[str, tuple[float, float]]:
    """Left-to-right layout; column spacing follows the widest box at each level."""
    root = find_root(g)
    if root is None:
        return {n: (0.0, 0.0) for n in g.nodes()}

    levels = _levels_from_root(g, root)
    by_level: dict[int, list[str]] = defaultdict(list)
    for n, lv in levels.items():
        by_level[lv].append(n)

    pos: dict[str, tuple[float, float]] = {}
    x_cursor = 0.0
    for lv in sorted(by_level):
        nodes = sorted(
            by_level[lv],
            key=lambda x: (
                g.nodes[x].get("step") if g.nodes[x].get("step") is not None else -1,
                g.nodes[x].get("tool_name") or "",
                x,
            ),
        )
        col_w = max(node_sizes[n][0] for n in nodes)
        x_center = x_cursor + col_w / 2
        n_at_level = len(nodes)
        for i, n in enumerate(nodes):
            y = (i - (n_at_level - 1) / 2.0) * y_gap
            pos[n] = (x_center, y)
        x_cursor += col_w + x_gap
    return pos


def draw_ascii(g: nx.DiGraph) -> str:
    """A simple, dependency-ordered text rendering of the ACG."""
    meta = _run_metadata(g)
    lines = []
    if meta.get("question"):
        lines.append(f"Q: {meta['question']}")
    if meta.get("answer"):
        lines.append(f"A: {meta['answer']}  ({meta.get('outcome', '?')})")
    if lines:
        lines.append("")

    root = find_root(g)
    if root is None:
        return "(empty graph)"
    levels = _levels_from_root(g, root)
    by_level: dict[int, list[str]] = defaultdict(list)
    for n, lv in levels.items():
        by_level[lv].append(n)
    for lv in sorted(by_level):
        for n in sorted(by_level[lv], key=lambda x: str(g.nodes[x].get("step"))):
            indent = "  " * lv
            succ = ", ".join(_node_label(g, s) for s in g.successors(n))
            arrow = f"  ->  {succ}" if succ else ""
            lines.append(f"{indent}{_node_label(g, n)}{arrow}")
    return "\n".join(lines)


def draw_png(g: nx.DiGraph, out_path: str | Path, title: str = "") -> str:
    """Render the ACG to a PNG: task header + DAG with variable-width node boxes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    if g.number_of_nodes() == 0:
        raise ValueError("cannot draw an empty graph")

    meta = _run_metadata(g)
    node_sizes = {
        n: _box_size_for_labels(*_node_short_label(g, n))
        for n in g.nodes()
    }
    pos = _layout_positions(g, node_sizes)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]

    root = find_root(g) or next(iter(g))
    levels = _levels_from_root(g, root)
    n_levels = max(levels.values()) + 1
    max_parallel = max(
        len([n for n, lv in levels.items() if lv == level])
        for level in range(n_levels)
    )
    total_w = max(p[0] + node_sizes[n][0] / 2 for n, p in pos.items()) - min(
        p[0] - node_sizes[n][0] / 2 for n, p in pos.items()
    )
    fig_w = max(12.0, total_w * 0.55 + 3.0)
    fig_h = max(6.0, 1.3 * max_parallel + 3.5)

    fig = plt.figure(figsize=(fig_w, fig_h))
    header_h = 0.26 if (meta.get("question") or meta.get("answer")) else 0.08
    gs = fig.add_gridspec(2, 1, height_ratios=[header_h, 1.0 - header_h], hspace=0.08)
    ax_head = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1])

    max_box_w = max(w for w, _ in node_sizes.values())
    max_box_h = max(h for _, h in node_sizes.values())
    pad_x = 1.5 + 0.05 * n_levels
    pad_y = 1.0 + 0.15 * max_parallel
    ax.set_xlim(min(xs) - max_box_w / 2 - pad_x, max(xs) + max_box_w / 2 + pad_x)
    ax.set_ylim(min(ys) - max_box_h / 2 - pad_y, max(ys) + max_box_h / 2 + pad_y)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    ax_head.axis("off")
    head_lines = []
    if title:
        head_lines.append(title)
    elif meta.get("task_id"):
        head_lines.append(f"Task {meta['task_id']} · {meta.get('outcome', 'unknown')}")
    if meta.get("question"):
        head_lines.append(f"Question: {_wrap_text(meta['question'], width=110)}")
    if meta.get("answer"):
        ans = meta["answer"]
        if len(ans) > 280:
            ans = ans[:277] + "…"
        head_lines.append(f"Answer: {ans}")
    elif meta.get("outcome") == "no_answer":
        head_lines.append("Answer: (none — model did not finish)")
    ax_head.text(
        0.01, 0.98, "\n".join(head_lines),
        transform=ax_head.transAxes, va="top", ha="left",
        fontsize=10, linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7f7f7", edgecolor="#cccccc"),
    )

    for n, (x, y) in pos.items():
        d = g.nodes[n]
        box_w, box_h = node_sizes[n]
        fill, text_color = _COLOR_MAP.get(d.get("type"), ("#999999", "#ffffff"))
        primary, secondary = _node_short_label(g, n)
        rect = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            linewidth=1.2,
            edgecolor="#444444",
            facecolor=fill,
            zorder=2,
            clip_on=False,
        )
        ax.add_patch(rect)
        fs_primary = 8 if len(primary) > 36 else 9
        ax.text(x, y + (0.14 if secondary else 0), primary, ha="center", va="center",
                fontsize=fs_primary, fontweight="bold", color=text_color, zorder=3, clip_on=False)
        if secondary:
            ax.text(x, y - 0.24, secondary, ha="center", va="center", fontsize=7,
                    color=text_color, zorder=3, clip_on=False)

    for u, v in g.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w0, h0 = node_sizes[u]
        w1, h1 = node_sizes[v]
        arrow = FancyArrowPatch(
            (x0 + w0 / 2, y0),
            (x1 - w1 / 2, y1),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="#555555",
            connectionstyle="arc3,rad=0.0",
            shrinkA=0,
            shrinkB=0,
            zorder=1,
            clip_on=False,
        )
        ax.add_patch(arrow)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.97, bottom=0.05, hspace=0.10)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, facecolor="white", pad_inches=0.5)
    plt.close(fig)
    return str(out_path)
