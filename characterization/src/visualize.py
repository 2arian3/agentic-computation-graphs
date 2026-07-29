"""Render a canonical agentic computation graph (ACG) as a figure.

Produces one image per session with three panels:

  1. **Stats header** -- everything the trace actually knows about the session:
     provider, models, rounds, token totals, prefix-cache hit ratio, tool mix,
     errors, wall-clock duration, and the structural metrics.
  2. **DAG view** -- the computation graph itself. Columns are topological
     levels (so siblings that could run concurrently share a column), colour is
     node type, and edge style encodes the dependency kind.
  3. **Timeline view** -- the same nodes on a wall-clock axis. Only drawn when
     the source ships timestamps. Overlapping tool bars here are *measured*
     concurrency, not inferred.

Everything rendered is read from data/graphs/<dataset>/graphs.jsonl. Nothing is
recomputed from the raw trace and nothing absent is invented -- a null field is
drawn as "n/a".

Usage:
    python -m src.visualize tracelab --index 0
    python -m src.visualize tracelab --graph-id tracelab:claude:375aff66-...
    python -m src.visualize tracelab --pick most-parallel --rounds 0:12
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.common import GRAPHS, REPO  # noqa: E402

# Node type -> (fill, label prefix). Colour-blind-safe, works on white.
NODE_STYLE = {
    "llm": ("#4C78A8", "L"),
    "tool": ("#F58518", "T"),
    "user": ("#54A24B", "U"),
    "retrieval": ("#B279A2", "R"),
    "memory": ("#9D755D", "M"),
    "agent": ("#E45756", "A"),
    "verifier": ("#72B7B2", "V"),
}
# Edge type -> (colour, linestyle, width)
EDGE_STYLE = {
    "data": ("#333333", "-", 1.4),
    "control": ("#888888", "--", 1.0),
    "order": ("#BBBBBB", ":", 0.9),
    "agent_msg": ("#54A24B", "-", 1.2),
}


def _ts(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _fmt(n: Any, suffix: str = "") -> str:
    if n is None:
        return "n/a"
    if isinstance(n, float):
        return f"{n:,.0f}{suffix}"
    return f"{n:,}{suffix}"


# --------------------------------------------------------------------- loading


def iter_graphs(dataset: str) -> Iterator[dict]:
    path = GRAPHS / dataset / "graphs.jsonl"
    if not path.exists():
        raise SystemExit(f"no graphs at {path} -- run the extractor first")
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_graph(dataset: str, index: int | None, graph_id: str | None,
               pick: str | None) -> dict:
    if graph_id:
        for g in iter_graphs(dataset):
            if g["graph_id"] == graph_id:
                return g
        raise SystemExit(f"graph_id {graph_id!r} not found")

    if pick:
        best, best_key = None, None
        for g in iter_graphs(dataset):
            nodes = g["nodes"]
            if pick == "most-parallel":
                key = (_measured_width(nodes), -len(nodes))
            elif pick == "median":
                key = -abs(len(nodes) - 46)
            elif pick == "largest":
                key = len(nodes)
            else:
                raise SystemExit(f"unknown --pick {pick!r}")
            if best_key is None or key > best_key:
                best, best_key = g, key
        assert best is not None
        return best

    for i, g in enumerate(iter_graphs(dataset)):
        if i == (index or 0):
            return g
    raise SystemExit(f"index {index} out of range")


def _measured_width(nodes: list[dict]) -> int:
    iv = [(a, b) for n in nodes
          if (a := _ts(n.get("start_ts"))) is not None
          and (b := _ts(n.get("end_ts"))) is not None and b >= a]
    if not iv:
        return 0
    ev = sorted([(a, 1) for a, _ in iv] + [(b, -1) for _, b in iv])
    cur = best = 0
    for _, d in ev:
        cur += d
        best = max(best, cur)
    return best


# ------------------------------------------------------------- "useful data"


def graph_summary(g: dict) -> dict[str, Any]:
    """Everything the trace actually knows about this session."""
    nodes, edges = g["nodes"], g["edges"]
    llm = [n for n in nodes if n["node_type"] == "llm"]
    tool = [n for n in nodes if n["node_type"] == "tool"]

    def _sum(ns: list[dict], f: str) -> int | None:
        vals = [n[f] for n in ns if n.get(f) is not None]
        return sum(vals) if vals else None

    starts = [t for n in nodes if (t := _ts(n.get("start_ts"))) is not None]
    ends = [t for n in nodes if (t := _ts(n.get("end_ts"))) is not None]

    in_tok = _sum(llm, "input_tokens")
    cache = _sum(llm, "cache_hit_tokens")
    tool_lat = [n["tool_latency_ms"] for n in tool if n.get("tool_latency_ms") is not None]
    gen_lat = [n["wall_latency_ms"] for n in llm if n.get("wall_latency_ms") is not None]

    levels = _levels(nodes, edges)
    depth = max(levels.values()) if levels else 0

    return {
        "graph_id": g["graph_id"],
        "dataset": g["dataset"],
        "source_domain": g["source_domain"],
        "provenance": g.get("provenance", {}),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "node_types": dict(Counter(n["node_type"] for n in nodes)),
        "edge_types": dict(Counter(e["edge_type"] for e in edges)),
        "depth": depth,
        "max_fan_out": max(Counter(e["src"] for e in edges).values(), default=0),
        "measured_parallel_width": _measured_width(nodes),
        "models": dict(Counter(n["model"] for n in llm if n.get("model"))),
        "input_tokens": in_tok,
        "output_tokens": _sum(llm, "output_tokens"),
        "prefill_tokens": _sum(llm, "prefill_tokens"),
        "cache_hit_tokens": cache,
        "cache_hit_ratio": (cache / in_tok) if in_tok else None,
        "tool_histogram": dict(Counter(n["tool_name"] for n in tool if n.get("tool_name")).most_common()),
        "tool_status": dict(Counter(n["tool_status"] for n in tool if n.get("tool_status"))),
        "tool_latency_ms_total": sum(tool_lat) if tool_lat else None,
        "gen_latency_ms_total": sum(gen_lat) if gen_lat else None,
        "wall_seconds": (max(ends) - min(starts)) if starts and ends else None,
    }


def _levels(nodes: list[dict], edges: list[dict]) -> dict[str, int]:
    """Longest-path level per node; siblings that may run together share one."""
    ids = [n["node_id"] for n in nodes]
    preds: defaultdict[str, list[str]] = defaultdict(list)
    succ: defaultdict[str, list[str]] = defaultdict(list)
    indeg = dict.fromkeys(ids, 0)
    for e in edges:
        if e["src"] in indeg and e["dst"] in indeg:
            preds[e["dst"]].append(e["src"])
            succ[e["src"]].append(e["dst"])
            indeg[e["dst"]] += 1
    level = dict.fromkeys(ids, 0)
    queue = [i for i in ids if indeg[i] == 0]
    seen = 0
    while queue:
        cur = queue.pop(0)
        seen += 1
        for nxt in succ[cur]:
            level[nxt] = max(level[nxt], level[cur] + 1)
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if seen < len(ids):  # cycle: fall back to insertion order
        for i, nid in enumerate(ids):
            level.setdefault(nid, i)
    return level


# ------------------------------------------------------------------- rendering


def _slice_rounds(g: dict, rounds: str | None) -> dict:
    """Keep only nodes whose turn_id falls in [a,b), and edges between them."""
    if not rounds:
        return g
    a, _, b = rounds.partition(":")
    lo = int(a) if a else 0
    hi = int(b) if b else 10**9
    keep = {n["node_id"] for n in g["nodes"]
            if n.get("turn_id") is not None and lo <= n["turn_id"] < hi}
    return {**g,
            "nodes": [n for n in g["nodes"] if n["node_id"] in keep],
            "edges": [e for e in g["edges"] if e["src"] in keep and e["dst"] in keep]}


def _auto_wrap(n_levels: int, limit: int = 15) -> int:
    """Levels per row. A 36-level chain drawn on one row is 3,000 px of tiny
    text; wrapping it into rows of ~15 keeps the aspect ratio readable."""
    if n_levels <= limit:
        return n_levels or 1
    import math
    rows = math.ceil(n_levels / limit)
    return math.ceil(n_levels / rows)


def _draw_dag(ax, nodes: list[dict], edges: list[dict], wrap: int | None = None) -> None:
    level = dict(_levels(nodes, edges))

    # Layout-only: a user node has no predecessor, so the longest-path level
    # pins every one of them to column 0 and their edges arc across the whole
    # figure. Seat each just left of the round it feeds. This does not touch
    # `_levels`, so the reported depth metric is unaffected.
    succs: defaultdict[str, list[str]] = defaultdict(list)
    for e in edges:
        succs[e["src"]].append(e["dst"])
    for n in nodes:
        if n["node_type"] == "user":
            kids = [level[s] for s in succs.get(n["node_id"], []) if s in level]
            if kids:
                level[n["node_id"]] = max(min(kids) - 1, 0)

    by_level: defaultdict[int, list[dict]] = defaultdict(list)
    order = {"user": 0, "llm": 1, "agent": 2, "tool": 3}
    for n in nodes:
        by_level[level[n["node_id"]]].append(n)

    n_levels = max(by_level) + 1 if by_level else 1
    w = wrap or _auto_wrap(n_levels)

    # Vertical budget per wrapped row = the widest column it contains.
    row_height: dict[int, float] = {}
    for lv, ns in by_level.items():
        r = lv // w
        row_height[r] = max(row_height.get(r, 1.0), float(len(ns)))
    row_base: dict[int, float] = {}
    acc = 0.0
    for r in sorted(row_height):
        acc -= row_height[r] / 2 + 0.9
        row_base[r] = acc
        acc -= row_height[r] / 2

    # Serpentine (boustrophedon): odd rows run right-to-left, so the wrap hop
    # from the end of one row to the start of the next is a short step down
    # instead of a long diagonal back across the whole figure.
    pos: dict[str, tuple[float, float]] = {}
    for lv, ns in by_level.items():
        ns.sort(key=lambda n: (order.get(n["node_type"], 9), n["node_id"]))
        span = (len(ns) - 1) / 2
        r = lv // w
        c = lv % w if r % 2 == 0 else (w - 1 - lv % w)
        for i, n in enumerate(ns):
            pos[n["node_id"]] = (c, row_base[r] + (i - span))

    for e in edges:
        if e["src"] not in pos or e["dst"] not in pos:
            continue
        col, ls, lw = EDGE_STYLE.get(e["edge_type"], ("#999", "-", 1.0))
        x0, y0 = pos[e["src"]]
        x1, y1 = pos[e["dst"]]
        # An edge that crosses a wrap boundary would otherwise cut straight back
        # across the figure -- bow it out so it reads as "continued below".
        wraps = (level[e["src"]] // w) != (level[e["dst"]] // w)
        # arc3's rad is relative to the endpoint distance, and a wrap edge spans
        # the whole row, so a fixed rad balloons into a huge sweep. Scale it
        # down by the span to keep the curve tight.
        rad = -min(0.9 / max(abs(x1 - x0), 1.0), 0.14) if wraps else 0.06
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=col, linestyle=ls,
                                    linewidth=lw, shrinkA=11, shrinkB=13,
                                    alpha=0.55 if wraps else 1.0,
                                    connectionstyle=f"arc3,rad={rad}"), zorder=1)

    for n in nodes:
        x, y = pos[n["node_id"]]
        fill, pre = NODE_STYLE.get(n["node_type"], ("#999", "?"))
        ax.scatter([x], [y], s=340, c=fill, edgecolors="white", linewidths=1.4,
                   zorder=3, marker="o")
        if n["node_type"] == "llm":
            lab = f"{pre}{n.get('turn_id', '')}"
        elif n["node_type"] in ("tool", "agent"):
            # agent nodes keep their tool name (spawn_agent, Agent, ...) so the
            # specific delegation op stays visible, not just the category
            lab = (n.get("tool_name") or n["node_type"])[:12]
        else:
            lab = pre
        ax.text(x, y - 0.22, lab, ha="center", va="top", fontsize=6.5,
                color="#222", zorder=4, clip_on=False)
        # token annotation for llm nodes, when the source has tokens
        if n["node_type"] == "llm" and n.get("input_tokens") is not None:
            ax.text(x, y + 0.20,
                    f"{n['input_tokens'] // 1000}k/{n.get('output_tokens') or 0}",
                    ha="center", va="bottom", fontsize=5.5, color="#4C78A8",
                    zorder=4, clip_on=False)
        if n["node_type"] == "tool" and n.get("tool_status") == "error":
            ax.scatter([x], [y], s=560, facecolors="none", edgecolors="#E45756",
                       linewidths=1.6, zorder=2)

    # A single-lane graph has zero y-extent, so autoscale would collapse the
    # axis and throw the labels outside it -- pin the limits explicitly.
    ys = [p[1] for p in pos.values()]
    ax.set_ylim(min(ys) - 0.85, max(ys) + 0.85)
    xs = [p[0] for p in pos.values()]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6)
    ax.set_yticks([])
    lab = "topological level  (siblings in one column may run concurrently)"
    if n_levels > w:
        lab += f"  —  wrapped every {w} levels; rows alternate direction (see ▸/◂)"
        ax.set_xticks([])
        # Mark each row's reading direction, since odd rows run right-to-left.
        for r in sorted(row_base):
            rightward = r % 2 == 0
            ax.text(-0.55 if rightward else w - 0.45, row_base[r],
                    "▸" if rightward else "◂",
                    ha="center", va="center", fontsize=13, color="#BBB",
                    zorder=0, clip_on=False)
    ax.set_xlabel(lab, fontsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#EEE", linewidth=0.6)
    ax.set_axisbelow(True)


def _draw_timeline(ax, nodes: list[dict]) -> bool:
    """Wall-clock Gantt, one row per round. False if the source is untimed.

    A row per round is what makes concurrency legible: tools issued by the same
    LLM round sit on the same row, so overlap is visible as bars that literally
    overlap. Packing bars into greedy lanes instead collapses a whole session
    onto one row and hides exactly the thing worth seeing.

    The LLM bar is the *generation* window (round start -> its first tool call),
    not the round span. The round span for Claude includes human think time,
    which would swamp the axis and misrepresent compute.
    """
    timed = [n for n in nodes if _ts(n.get("start_ts")) is not None]
    if not timed:
        return False
    t0 = min(_ts(n["start_ts"]) for n in timed)

    rounds = sorted({n["turn_id"] for n in nodes if n.get("turn_id") is not None})
    row_of = {r: i for i, r in enumerate(rounds)}
    if not rounds:
        return False

    tools_by_round: defaultdict[Any, list[dict]] = defaultdict(list)
    for n in nodes:
        if n["node_type"] == "tool":
            tools_by_round[n.get("turn_id")].append(n)

    # Tool durations span orders of magnitude in one session (exec_command
    # seconds vs write_stdin tens of ms), so short bars land below one pixel and
    # a row looks empty when it is not. Clamp to a visible floor and say so on
    # the axis -- presence is real, the drawn width of a clamped bar is not.
    total_span = max(
        (_ts(n["end_ts"]) for n in nodes if _ts(n.get("end_ts")) is not None),
        default=t0,
    ) - t0
    min_w = max(total_span * 0.004, 1e-3)
    clamped = 0

    span_end = 0.0
    for n in nodes:
        a = _ts(n.get("start_ts"))
        b = _ts(n.get("end_ts"))
        if a is None or n.get("turn_id") not in row_of:
            continue
        row = row_of[n["turn_id"]]
        fill, _ = NODE_STYLE.get(n["node_type"], ("#999", "?"))

        if n["node_type"] == "tool" and b is not None:
            x0, w = a - t0, b - a
            if w < min_w:
                clamped += 1
            ax.barh(row, max(w, min_w), left=x0, height=0.62, color=fill,
                    alpha=0.95, edgecolor="white", linewidth=0.5, zorder=3)
            span_end = max(span_end, x0 + max(w, min_w))
        elif n["node_type"] == "llm":
            # Two bars, both evidenced, neither inferred:
            #  - pale: the round span, min..max of the round's event timestamps
            #  - solid: the generation window from wall_latency_ms (TraceLab's
            #    documented proxy), ending at this round's first tool call.
            # The proxy is null whenever no input event precedes the first tool
            # call -- common for Codex, whose tool_results are timestamped after
            # its own tool_calls -- so the pale bar is what stays dense.
            if b is not None:
                ax.barh(row, max(b - a, 1e-3), left=a - t0, height=0.62,
                        color=fill, alpha=0.28, edgecolor="white",
                        linewidth=0.5, zorder=1)
                span_end = max(span_end, b - t0)
            kids = [_ts(t["start_ts"]) for t in tools_by_round.get(n["turn_id"], [])
                    if _ts(t.get("start_ts")) is not None]
            lat = n.get("wall_latency_ms")
            if kids and lat is not None:
                gen_end = min(kids)
                x0, w = (gen_end - lat / 1000.0) - t0, lat / 1000.0
                ax.barh(row, max(w, 1e-3), left=x0, height=0.40, color=fill,
                        alpha=0.95, edgecolor="white", linewidth=0.4, zorder=2)
                span_end = max(span_end, x0 + w)
        elif n["node_type"] == "user":
            ax.plot([a - t0], [row], marker="D", markersize=4.5,
                    color=NODE_STYLE["user"][0], zorder=4)
            span_end = max(span_end, a - t0)

    for n in nodes:
        if n["node_type"] != "tool" or n.get("turn_id") not in row_of:
            continue
        a, b = _ts(n.get("start_ts")), _ts(n.get("end_ts"))
        if a is None or b is None:
            continue
        w = b - a
        if span_end and w > span_end * 0.06:
            ax.text(a - t0 + w / 2, row_of[n["turn_id"]], (n.get("tool_name") or "")[:14],
                    ha="center", va="center", fontsize=5.5, color="white", zorder=5)

    note = f"; {clamped} bar(s) shorter than {min_w:.2g}s drawn at minimum width" if clamped else ""
    ax.set_xlabel(
        f"seconds from session start  (bars overlapping on a row = measured concurrency{note})",
        fontsize=8)
    ax.set_ylabel("round", fontsize=8)
    step = max(1, len(rounds) // 25)
    ax.set_yticks([row_of[r] for r in rounds[::step]])
    ax.set_yticklabels([str(r) for r in rounds[::step]], fontsize=6)
    ax.set_ylim(len(rounds) - 0.4, -0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#EEE", linewidth=0.6)
    ax.set_axisbelow(True)
    return True


def _stats_text(s: dict) -> str:
    p = s["provenance"]
    models = ", ".join(f"{k} x{v}" for k, v in list(s["models"].items())[:3]) or "n/a"
    tools = ", ".join(f"{k} x{v}" for k, v in list(s["tool_histogram"].items())[:6]) or "none"
    ratio = f"{s['cache_hit_ratio']:.1%}" if s["cache_hit_ratio"] is not None else "n/a"
    nt = s["node_types"]
    return (
        f"provider {p.get('provider') or 'n/a'}   project {p.get('project') or 'n/a'}   "
        f"rounds {p.get('n_rounds', 'n/a')}   models: {models}\n"
        f"nodes {s['n_nodes']:,} (llm {nt.get('llm', 0)}, tool {nt.get('tool', 0)}, "
        f"user {nt.get('user', 0)})   edges {s['n_edges']:,} "
        f"({', '.join(f'{k} {v}' for k, v in s['edge_types'].items())})\n"
        f"depth {s['depth']}   max fan-out {s['max_fan_out']}   "
        f"measured parallel width {s['measured_parallel_width']}   "
        f"wall {_fmt(s['wall_seconds'], ' s')}\n"
        f"tokens in {_fmt(s['input_tokens'])}  out {_fmt(s['output_tokens'])}  "
        f"prefill {_fmt(s['prefill_tokens'])}  cache-hit {_fmt(s['cache_hit_tokens'])} "
        f"({ratio} of input)\n"
        f"tool time {_fmt(s['tool_latency_ms_total'], ' ms')}   "
        f"gen time {_fmt(s['gen_latency_ms_total'], ' ms')}   "
        f"status {s['tool_status'] or 'n/a'}\n"
        f"tools: {tools}"
    )


def render(g: dict, out: Path, rounds: str | None = None,
           max_nodes: int = 500, wrap: int | None = None,
           timeline: bool = True) -> dict:
    summary = graph_summary(g)  # full-session stats, before any windowing
    view = _slice_rounds(g, rounds)
    nodes, edges = view["nodes"], view["edges"]

    truncated = None
    if len(nodes) > max_nodes:
        keep = {n["node_id"] for n in nodes[:max_nodes]}
        truncated = (len(nodes), max_nodes)
        nodes = [n for n in nodes if n["node_id"] in keep]
        edges = [e for e in edges if e["src"] in keep and e["dst"] in keep]

    lv = _levels(nodes, edges)
    n_levels = (max(lv.values()) + 1) if lv else 1
    n_lanes = max(Counter(lv.values()).values(), default=1)
    n_rounds = len({n.get("turn_id") for n in nodes if n.get("turn_id") is not None}) or 1

    w = wrap or _auto_wrap(n_levels)
    n_rows = -(-n_levels // w)  # ceil
    # Height must cover every wrapped row's widest column, not just the widest
    # column overall, or tall rows collide.
    lanes_per_row: Counter[int] = Counter()
    per_level = Counter(lv.values())
    for level_i, count in per_level.items():
        lanes_per_row[level_i // w] = max(lanes_per_row[level_i // w], count)
    total_lanes = sum(lanes_per_row.values()) + 1.4 * n_rows

    width = float(min(max(11.0, w * 0.72 + 3.0), 60.0))
    dag_h = float(min(max(2.6, total_lanes * 0.42), 40.0))
    tl_h = float(min(max(2.2, n_rounds * 0.20), 14.0)) if timeline else 0.0
    header_h = 1.9

    fig = plt.figure(figsize=(width, header_h + dag_h + tl_h), dpi=150)
    ratios = [header_h, dag_h] + ([tl_h] if timeline else [])
    gs = fig.add_gridspec(len(ratios), 1, height_ratios=ratios, hspace=0.30)

    ax0 = fig.add_subplot(gs[0])
    ax0.axis("off")
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.text(0, 1.02, g["graph_id"], fontsize=12, fontweight="bold",
             va="top", family="monospace")
    ax0.text(0, 0.80, _stats_text(summary), fontsize=8.0, va="top",
             family="monospace", linespacing=1.45)

    ax1 = fig.add_subplot(gs[1])
    _draw_dag(ax1, nodes, edges, wrap=w)
    title = "Agentic computation graph"
    if rounds:
        title += f"  (rounds {rounds})"
    if truncated:
        title += f"  [showing {truncated[1]} of {truncated[0]} nodes]"
    ax1.set_title(title, fontsize=11, loc="left", pad=8)

    if timeline:
        ax2 = fig.add_subplot(gs[2])
        if _draw_timeline(ax2, nodes):
            ax2.set_title("Execution timeline (real timestamps)", fontsize=11,
                          loc="left", pad=6)
        else:
            ax2.axis("off")
            ax2.text(0.5, 0.5, "no timestamps in this source -- timeline not available",
                     ha="center", va="center", fontsize=10, color="#888")

    handles = [mpatches.Patch(color=c, label=t)
               for t, (c, _) in NODE_STYLE.items()
               if any(n["node_type"] == t for n in nodes)]
    handles += [Line2D([0], [0], color=c, linestyle=ls, lw=lw, label=f"{t} edge")
                for t, (c, ls, lw) in EDGE_STYLE.items()
                if any(e["edge_type"] == t for e in edges)]
    # Anchored to the figure so it can never collide with an axis label.
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.012),
               ncol=len(handles), fontsize=8, frameon=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Render an ACG as a figure")
    ap.add_argument("dataset")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--graph-id", default=None)
    ap.add_argument("--pick", choices=["most-parallel", "median", "largest"], default=None)
    ap.add_argument("--rounds", default=None, help="window as A:B over turn_id")
    ap.add_argument("--max-nodes", type=int, default=500)
    ap.add_argument("--wrap", type=int, default=None,
                    help="levels per row before wrapping (default: auto)")
    ap.add_argument("--no-timeline", action="store_true",
                    help="omit the timeline panel")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    g = load_graph(args.dataset, args.index, args.graph_id, args.pick)
    stem = g["graph_id"].replace(":", "_").replace("/", "_")[:80]
    out = args.out or (REPO / "reports" / "figures" / args.dataset / f"{stem}.png")
    summary = render(g, out, args.rounds, args.max_nodes, wrap=args.wrap,
                     timeline=not args.no_timeline)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
