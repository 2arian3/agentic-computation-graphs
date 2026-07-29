"""Render a showcase PDF: one trace per dataset, graph plus per-step content.

Each dataset contributes a graph page (stats header, the DAG, and a wall-clock
timeline where the source is timed) followed by walkthrough pages that print,
for every step, the evidence the source actually carries: the model's reasoning,
the tool arguments, the observation that came back, and the cost fields. A field
the source does not ship is printed as "absent in this source" rather than left
blank -- the cost/semantics split that divides this corpus is the finding, so it
has to be legible rather than inferred from a gap on the page.

Traces are drawn at random but seeded, and only from a legibility-constrained
pool: a 201-node OpenHands chain carrying 4 KB tool outputs does not fit on a
page. The constraint used for each dataset is printed on that dataset's page, so
nothing about the sampling is hidden.

Text is laid out with matplotlib rather than a PDF toolkit because matplotlib is
already the only rendering dependency, and `pdf.fonttype 42` keeps the output
selectable and searchable.

Usage:
    .venv/bin/python scripts/render_showcase_pdf.py
    .venv/bin/python scripts/render_showcase_pdf.py --seed 7 --steps 8
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.common import GRAPHS, REPO  # noqa: E402
from src.visualize import (  # noqa: E402
    EDGE_STYLE,
    NODE_STYLE,
    _auto_wrap,
    _draw_dag,
    _draw_timeline,
    _levels,
    _slice_rounds,
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42,  # embed TrueType: text stays selectable and searchable
})

# ------------------------------------------------------------------ page layout

PAGE = (11.69, 8.27)  # A4 landscape -- projector-shaped, and wide enough that a
                      # monospace line holds ~180 chars of tool output
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 0.45, 0.45, 0.42, 0.34
MONO_ADV = 0.6023  # DejaVu Sans Mono advance width, in em


@dataclass
class Pick:
    """How one dataset's example trace is chosen, and how it is introduced."""

    dataset: str
    title: str
    blurb: str
    min_nodes: int
    max_nodes: int
    # The graph JSONL runs to 22 GB, so candidates are restricted to the first
    # `scan_limit` records; anything further in costs a multi-GB read to reach.
    scan_limit: int
    require_agents: bool = False
    dag_max_nodes: int = 40
    notes: list[str] = field(default_factory=list)


PICKS = [
    Pick(
        dataset="tracelab",
        title="TraceLab — production Claude Code / Codex sessions",
        blurb="Real developer sessions on their own machines. The only source with "
              "tokens, latency, timestamps and prefix-cache accounting — and the only "
              "one with named sub-agents. Sanitized, so it carries no text at all.",
        min_nodes=8, max_nodes=40, scan_limit=4265, require_agents=True,
        notes=["Pool restricted to sessions containing at least one sub-agent node, so the "
               "`agent` node type is visible; only 300 of 4,265 sessions have any."],
    ),
    Pick(
        dataset="swe_rebench_openhands",
        title="SWE-rebench / OpenHands — real GitHub issues",
        blurb="OpenHands agent (Qwen3-Coder-480B) on 1,823 real repositories. Structured "
              "tool calls plus the exact tool schemas the model was offered.",
        min_nodes=20, max_nodes=80, scan_limit=1500, dag_max_nodes=34,
    ),
    Pick(
        dataset="swe_agent_traj",
        title="SWE-agent — same domain, different harness",
        blurb="SWE-agent scaffold (Llama-70b) on real repositories. Paired with OpenHands "
              "on purpose: same task type, different harness, so shape differences are "
              "attributable to the scaffold. Reasoning coverage is 100%.",
        min_nodes=16, max_nodes=34, scan_limit=4000,
    ),
    Pick(
        dataset="osworld_gelato",
        title="OSWorld / Gelato — GUI computer-use on real desktop apps",
        blurb="Agent driving real Chrome, LibreOffice, GIMP and VS Code in an Ubuntu VM "
              "via pyautogui. The only non-coding domain in the corpus.",
        min_nodes=16, max_nodes=34, scan_limit=2111,
    ),
]

ABSENT = "absent in this source"


# ------------------------------------------------------------------- text utils


def _clean(s: str) -> str:
    """Tabs expanded, CR dropped, unprintables blanked. `$` is escaped later."""
    s = s.replace("\t", "    ").replace("\r", "")
    return "".join(c if (c.isprintable() or c == "\n") else " " for c in s)


def _esc(line: str) -> str:
    """matplotlib parses `$...$` as mathtext, so a literal dollar must escape."""
    return line.replace("$", r"\$")


def _wrap(text: str, width: int, hard: bool) -> list[str]:
    """Wrap to `width` columns. `hard` splits mid-token, for code and JSON where
    a word boundary is meaningless; prose wraps on whitespace instead."""
    out: list[str] = []
    for raw in _clean(text).split("\n"):
        if not raw.strip():
            out.append("")
        elif hard:
            while len(raw) > width:
                out.append(raw[:width])
                raw = raw[width:]
            out.append(raw)
        else:
            out.extend(textwrap.wrap(raw, width) or [""])
    return out


def _fmt(v: Any, suffix: str = "") -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.0f}{suffix}"
    return f"{v:,}{suffix}"


class Deck:
    """Flowing monospace text across PDF pages.

    `y` is inches from the top of the page. Every writer advances it and asks
    `_room` whether a continuation page is needed, so a caller never has to know
    where a page break falls.
    """

    def __init__(self, pdf: PdfPages) -> None:
        self.pdf = pdf
        self.fig: plt.Figure | None = None
        self.y = MARGIN_T
        self.n_pages = 0
        self._title = ""
        self._sub = ""

    # -- page lifecycle ------------------------------------------------------
    def new_page(self, title: str = "", sub: str = "", *, remember: bool = True) -> plt.Figure:
        self.end_page()
        self.fig = plt.figure(figsize=PAGE)
        self.n_pages += 1
        self.y = MARGIN_T
        if remember:
            self._title, self._sub = title, sub
        if title:
            self.line(title, size=12.5, mono=False, weight="bold", color="#111")
            if sub:
                self.line(sub, size=7.6, mono=False, color="#666")
            self.rule()
        return self.fig

    def end_page(self) -> None:
        if self.fig is None:
            return
        self.fig.text(1 - MARGIN_R / PAGE[0], MARGIN_B / 2 / PAGE[1], str(self.n_pages),
                      ha="right", va="center", fontsize=7, color="#999")
        self.pdf.savefig(self.fig)
        plt.close(self.fig)
        self.fig = None

    # -- geometry ------------------------------------------------------------
    @staticmethod
    def wrap_width(size: float, indent: float = 0.0) -> int:
        usable = PAGE[0] - MARGIN_L - MARGIN_R - indent
        return max(20, int(usable / (MONO_ADV * size / 72)))

    def _room(self, inches: float) -> bool:
        return self.y + inches <= PAGE[1] - MARGIN_B

    def ensure(self, inches: float) -> None:
        if not self._room(inches):
            self.new_page(f"{self._title}  (cont.)", self._sub, remember=False)

    def lines_left(self, size: float) -> int:
        """Lines of `size` still fitting on this page. Used to trim the last block
        of a page rather than spill two orphan lines onto the next one."""
        return int((PAGE[1] - MARGIN_B - self.y) / (size * 1.42 / 72))

    # -- writers -------------------------------------------------------------
    def line(self, text: str = "", *, size: float = 7.0, color: str = "#222",
             mono: bool = True, weight: str = "normal", indent: float = 0.0) -> None:
        h = size * 1.42 / 72
        self.ensure(h)
        if text:
            assert self.fig is not None
            self.fig.text((MARGIN_L + indent) / PAGE[0], 1 - self.y / PAGE[1], _esc(text),
                          ha="left", va="top", fontsize=size, color=color, weight=weight,
                          family="monospace" if mono else "sans-serif")
        self.y += h

    def para(self, text: str, *, size: float = 7.0, color: str = "#222", indent: float = 0.0,
             hard: bool = False, max_lines: int | None = None, mono: bool = True) -> None:
        width = self.wrap_width(size, indent) if mono else int(
            (PAGE[0] - MARGIN_L - MARGIN_R - indent) / (0.50 * size / 72))
        lines = _wrap(text, width, hard)
        clipped = 0
        if max_lines is not None and len(lines) > max_lines:
            clipped = sum(len(x) for x in lines[max_lines:])
            lines = lines[:max_lines]
        for ln in lines:
            self.line(ln, size=size, color=color, indent=indent, mono=mono)
        if clipped:
            self.line(f"… [+{clipped:,} more characters in the trace]",
                      size=size - 0.4, color="#B07000", indent=indent)

    def field(self, label: str, text: str | None, *, size: float = 6.5, max_lines: int = 6,
              hard: bool = False, label_color: str = "#4C78A8", absent: str = ABSENT) -> None:
        """`label: value`, wrapped and indented under the label.

        A block is kept whole: if it does not fit in what is left of the page but
        would fit on a fresh one, the page breaks first. Two lines of tool output
        stranded on a page of their own is worse than an early break.
        """
        pad = 11
        if not text:
            self.line(f"{label:<{pad}} — {absent}", size=size, color="#999")
            return
        lines = _wrap(text, self.wrap_width(size, 0.34), hard)
        clipped = sum(len(x) for x in lines[max_lines:])
        lines = lines[:max_lines]

        lh = size * 1.42 / 72
        need = (len(lines) + 1 + bool(clipped)) * lh
        if not self._room(need):
            if len(lines) + 2 <= int((PAGE[1] - MARGIN_B - MARGIN_T - 0.45) / lh):
                self.ensure(need)
            else:  # taller than any page: keep what fits here and say what was cut
                room = max(1, self.lines_left(size) - 2)
                clipped += sum(len(x) for x in lines[room:])
                lines = lines[:room]
        self.line(f"{label:<{pad}}", size=size, color=label_color, weight="bold")
        for ln in lines:
            self.line(ln, size=size, color="#333", indent=0.34)
        if clipped:
            self.line(f"… [+{clipped:,} more characters in the trace]",
                      size=size - 0.4, color="#B07000", indent=0.34)

    def rule(self, color: str = "#DDD", gap: float = 0.055) -> None:
        self.y += gap
        self.ensure(0.02)
        assert self.fig is not None
        self.fig.add_artist(Line2D([MARGIN_L / PAGE[0], 1 - MARGIN_R / PAGE[0]],
                                   [1 - self.y / PAGE[1]] * 2, color=color, lw=0.8))
        self.y += gap

    def band(self, text: str, color: str, height: float = 0.185) -> None:
        """Section header on a tinted band -- the step separators in a walkthrough."""
        self.ensure(height + 0.06)
        assert self.fig is not None
        x0, x1 = MARGIN_L / PAGE[0], 1 - MARGIN_R / PAGE[0]
        y1 = 1 - self.y / PAGE[1]
        self.fig.add_artist(mpatches.Rectangle((x0, y1 - height / PAGE[1]), x1 - x0,
                                               height / PAGE[1], facecolor=color,
                                               edgecolor="none", alpha=0.16))
        self.fig.text(x0 + 0.004, y1 - height / 2 / PAGE[1], _esc(text), ha="left",
                      va="center", fontsize=7.6, color="#111", weight="bold",
                      family="monospace")
        self.y += height + 0.05

    def space(self, inches: float = 0.06) -> None:
        self.y += inches


# ------------------------------------------------------------------- selection


def choose(pick: Pick, seed: int) -> tuple[int, dict[str, Any], str]:
    """Return (record index, per-graph metrics row, human description of the pool)."""
    mp = GRAPHS / pick.dataset / "per_graph_metrics.jsonl"
    if not mp.exists():
        raise SystemExit(f"missing {mp} — run src.characterize {pick.dataset} first")

    pool: list[tuple[int, dict[str, Any]]] = []
    with open(mp) as f:
        for i, line in enumerate(f):
            if i >= pick.scan_limit:
                break
            if not line.strip():
                continue
            r = json.loads(line)
            if not pick.min_nodes <= r["n_nodes"] <= pick.max_nodes:
                continue
            if pick.require_agents and not r["types"].get("agent"):
                continue
            pool.append((i, r))
    if not pool:
        raise SystemExit(f"no candidate graphs for {pick.dataset}")

    idx, row = random.Random(f"{seed}:{pick.dataset}").choice(pool)
    how = (f"drawn at random (seed {seed}) from {len(pool):,} candidates: "
           f"{pick.min_nodes}–{pick.max_nodes} nodes"
           + (", ≥1 sub-agent node" if pick.require_agents else "")
           + f", among the first {pick.scan_limit:,} records")
    return idx, row, how


def load_by_index(dataset: str, index: int) -> dict[str, Any]:
    """Read the index-th graph, parsing only that line -- the file is up to 22 GB."""
    with open(GRAPHS / dataset / "graphs.jsonl", "rb") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    raise SystemExit(f"index {index} beyond end of {dataset}/graphs.jsonl")


# --------------------------------------------------------------- graph helpers


def node_labels(nodes: list[dict]) -> dict[str, str]:
    """Compact per-node handles (U0, L3, T3.0, A37.1) used in the edge listings.

    Node ids carry a session uuid and are far too long to print in an edge list;
    these labels are unique within one graph and match the DAG's LLM labels.
    """
    seen: Counter[tuple[str, Any]] = Counter()
    out: dict[str, str] = {}
    for n in nodes:
        t, turn = n["node_type"], n.get("turn_id")
        pre = {"llm": "L", "tool": "T", "agent": "A", "user": "U"}.get(t, t[:1].upper())
        k = (pre, turn)
        seen[k] += 1
        stem = f"{pre}{turn if turn is not None else '?'}"
        out[n["node_id"]] = stem if pre in ("L", "U") else f"{stem}.{seen[k] - 1}"
    return out


def edge_index(edges: list[dict]) -> tuple[dict, dict]:
    ins: defaultdict[str, list[dict]] = defaultdict(list)
    outs: defaultdict[str, list[dict]] = defaultdict(list)
    for e in edges:
        ins[e["dst"]].append(e)
        outs[e["src"]].append(e)
    return ins, outs


def coverage(nodes: list[dict], fields: Iterable[str]) -> float:
    if not nodes:
        return 0.0
    return sum(1 for n in nodes if any(n.get(f) is not None for f in fields)) / len(nodes)


def cost_bits(n: dict) -> str:
    bits = []
    if n.get("input_tokens") is not None:
        s = f"in {n['input_tokens']:,} tok"
        if n.get("cache_hit_tokens") is not None and n["input_tokens"]:
            s += f" (prefix-cache {n['cache_hit_tokens']:,} = {n['cache_hit_tokens'] / n['input_tokens']:.0%})"
        bits.append(s)
    if n.get("output_tokens") is not None:
        bits.append(f"out {n['output_tokens']:,} tok")
    if n.get("prefill_tokens") is not None:
        bits.append(f"prefill {n['prefill_tokens']:,}")
    for f, lab in (("wall_latency_ms", "wall"), ("tool_latency_ms", "tool")):
        if n.get(f) is not None:
            bits.append(f"{lab} {n[f]:,.0f} ms")
    if n.get("start_ts"):
        bits.append(f"ts {n['start_ts'][11:23]}" + (f"→{n['end_ts'][11:23]}" if n.get("end_ts") else ""))
    ex = n.get("extra") or {}
    sizes = [f"{lab} {ex[k]:,} chars" for k, lab in
             (("content_chars", "content"), ("reasoning_chars", "reasoning"),
              ("input_chars", "args"), ("arg_chars", "args"),
              ("command_chars", "command"), ("result_chars", "result"))
             if isinstance(ex.get(k), int)]
    if sizes:
        bits.append("sizes: " + ", ".join(sizes))
    return " · ".join(bits) if bits else f"no tokens, latency or timestamps — {ABSENT}"


# ------------------------------------------------------------------- the pages


def cover_page(deck: Deck, metrics: dict[str, dict], picked: dict[str, str]) -> None:
    deck.new_page("Agentic Computation Graph Corpus",
                  f"Worked examples: one trace per extracted dataset · generated {date.today().isoformat()}")
    deck.para(
        "Every agent session in this corpus is normalised into one graph: nodes are runtime "
        "operations (LLM call, tool call, sub-agent boundary, user turn) and edges are data or "
        "control dependencies. This deck takes one trace from each extracted dataset and shows "
        "the whole thing — the graph, then every step's reasoning, tool arguments and result — "
        "so that what each source does and does not record is visible on the page rather than "
        "asserted in a table.",
        size=8.2, mono=False)
    deck.space(0.10)
    deck.para("Design rule the corpus depends on: absent means null, never a guess. A field the "
              "source never shipped is printed below as \"absent in this source\".",
              size=8.2, mono=False, color="#444")
    deck.rule()

    hdr = (f"{'dataset':<24}{'domain':<9}{'traces':>9}{'nodes':>13}{'med n':>7}"
           f"{'med dep':>8}{'fan-out':>9}{'reasoning':>11}{'tokens':>8}")
    deck.line(hdr, size=7.2, weight="bold")
    deck.line("-" * len(hdr), size=7.2, color="#BBB")
    for p in PICKS:
        m, d = metrics[p.dataset], metrics[p.dataset]["dist"]
        sem = m.get("coverage_semantic", {})
        fan = f"{int(d['max_fan_out']['median'])}/{int(d['max_fan_out']['max'])}"
        deck.line(
            f"{p.dataset:<24}{m['source_domain']:<9}{m['n_graphs']:>9,}{m['total_nodes']:>13,}"
            f"{int(d['n_nodes']['median']):>7,}{int(d['depth']['median']):>8,}{fan:>9}"
            f"{sem.get('reasoning_of_llm_nodes', 0):>11.0%}"
            f"{m['coverage_node_weighted']['tokens']:>8.0%}",
            size=7.2)
    deck.line("-" * len(hdr), size=7.2, color="#BBB")
    deck.line("fan-out is median/max.  reasoning = share of llm nodes with a stated rationale.  "
              "tokens = share of all nodes with token accounting.", size=6.4, color="#777")
    deck.space(0.10)
    deck.rule()

    deck.line("The four sources, and the trace shown for each", size=8.6, mono=False,
              weight="bold")
    deck.space(0.04)
    for p in PICKS:
        deck.line(f"{p.dataset}", size=7.6, weight="bold", color=NODE_STYLE["llm"][0])
        deck.para(p.blurb, size=7.2, indent=0.18, mono=False, color="#333")
        deck.line(f"example: {picked[p.dataset]}", size=6.6, indent=0.18, color="#777")
        deck.space(0.05)
    deck.space(0.05)
    deck.rule()
    deck.line("Reading the graph pages", size=8.6, mono=False, weight="bold")
    deck.space(0.04)
    for txt in (
        "nodes    L = LLM call · T = tool call · A = sub-agent · U = user turn",
        "edges    data (solid) = an observation feeds the next LLM call · control (dashed) = the "
        "LLM decided to call this tool",
        "         agent_msg (green) = a human turn enters the round · order (dotted) = ordering "
        "only, no data passed",
        "columns  topological level: nodes sharing a column have no dependency between them and "
        "could run concurrently",
    ):
        deck.para(txt, size=7.0, indent=0.10, max_lines=2)


def graph_page(deck: Deck, g: dict, pick: Pick, row: dict, how: str, index: int) -> None:
    nodes, edges = g["nodes"], g["edges"]
    types = Counter(n["node_type"] for n in nodes)
    etypes = Counter(e["edge_type"] for e in edges)
    prov = g.get("provenance", {})
    llm = [n for n in nodes if n["node_type"] == "llm"]
    action = [n for n in nodes if n["node_type"] in ("tool", "agent", "retrieval")]

    deck.new_page(pick.title, f"{g['graph_id']}   ·   record #{index} in {pick.dataset}/graphs.jsonl")
    deck.para(pick.blurb, size=7.8, mono=False, color="#333")
    deck.space(0.05)

    # Provenance keys differ per dataset by design; print whichever exist.
    ident = [(k, prov[k]) for k in
             ("repo", "instance_id", "application", "task_id", "project", "provider",
              "scaffold", "model_name", "n_rounds", "steps", "resolved", "exit_status",
              "result_score", "gen_tests_correct")
             if prov.get(k) is not None]
    deck.line("  ".join(f"{k}={v}" for k, v in ident), size=7.0, color="#333")
    models = Counter(n["model"] for n in llm if n.get("model"))
    deck.line("models: " + (", ".join(f"{k} ×{v}" for k, v in models.most_common(3)) or "—"),
              size=7.0, color="#333")
    deck.line(
        f"nodes {len(nodes):,} ("
        + ", ".join(f"{t} {types[t]}" for t in ("llm", "tool", "agent", "user") if types[t])
        + f")   edges {len(edges):,} ("
        + ", ".join(f"{t} {c}" for t, c in etypes.most_common()) + ")",
        size=7.0)
    deck.line(
        f"depth {row['depth']}   max fan-out {row['max_fan_out']}   loop iterations "
        f"{row['loop_iterations']}   measured parallel width "
        f"{_fmt(row['measured_parallel_width'])}   DAG {'yes' if row['is_dag'] else 'no'}   "
        f"branches {row['n_branches']} (abandoned {row['abandoned_nodes']})",
        size=7.0)
    deck.line(
        f"this trace — cost: tokens {row['cov_tokens']:.0%}, latency {row['cov_latency']:.0%}, "
        f"timestamps {row['cov_timestamps']:.0%}, KV {row['cov_kv']:.0%}"
        f"   semantic: reasoning {coverage(llm, ('reasoning_text',)):.0%} of llm, "
        f"tool i/o {coverage(action, ('tool_input', 'tool_output')):.0%} of actions",
        size=7.0, color="#1A6B4A")
    deck.para(f"selection: {how}", size=6.5, color="#777", max_lines=2)
    deck.rule()

    # DAG (+ timeline when the source is timed), sharing the rest of the page.
    view = g
    shown, total = len(nodes), len(nodes)
    if total > pick.dag_max_nodes:
        turns = sorted({n["turn_id"] for n in nodes if n.get("turn_id") is not None})
        for k in range(1, len(turns) + 1):
            cand = _slice_rounds(g, f"0:{turns[k - 1] + 1}")
            if len(cand["nodes"]) > pick.dag_max_nodes:
                break
            view, shown = cand, len(cand["nodes"])
    vnodes, vedges = view["nodes"], view["edges"]

    # Height the DAG actually needs: `_draw_dag` wraps long chains into rows, and
    # a two-row GUI trace stretched over half a page reads as a rendering fault.
    lv = _levels(vnodes, vedges)
    per_level = Counter(lv.values())
    n_levels = max(per_level) + 1 if per_level else 1
    wrap = _auto_wrap(n_levels)
    lanes: defaultdict[int, int] = defaultdict(int)
    for level_i, count in per_level.items():
        lanes[level_i // wrap] = max(lanes[level_i // wrap], count)
    n_rows = len(lanes) or 1
    dag_need = sum(lanes.values()) * 0.42 + 1.1 * n_rows

    # Titles and axis labels live outside the axes rectangle, so each panel gets
    # its own budget -- otherwise the DAG's x-label lands on the timeline title.
    DAG_TITLE, DAG_XLAB, TL_TITLE, TL_XLAB, LEGEND = 0.30, 0.34, 0.28, 0.38, 0.30

    timed = any(n.get("start_ts") for n in vnodes)
    n_rounds = len({n.get("turn_id") for n in vnodes if n.get("turn_id") is not None}) or 1
    assert deck.fig is not None
    top = deck.y
    avail = PAGE[1] - MARGIN_B - top
    tl_block = min(avail * 0.45, 0.17 * n_rounds + TL_TITLE + TL_XLAB + 0.45) if timed else 0.0
    dag_block = max(1.7, min(avail - tl_block - LEGEND, dag_need + DAG_TITLE + DAG_XLAB))

    def rect(top_in: float, height_in: float) -> list[float]:
        return [MARGIN_L / PAGE[0], (PAGE[1] - top_in - height_in) / PAGE[1],
                1 - (MARGIN_L + MARGIN_R) / PAGE[0], height_in / PAGE[1]]

    ax = deck.fig.add_axes(rect(top + DAG_TITLE, dag_block - DAG_TITLE - DAG_XLAB))
    _draw_dag(ax, vnodes, vedges)
    title = "Computation graph"
    if shown < total:
        title += f"  —  first {shown} of {total} nodes (whole-trace figures above)"
    ax.set_title(title, fontsize=9, loc="left", pad=6)
    bottom = top + dag_block

    if timed:
        ax2 = deck.fig.add_axes(rect(bottom + TL_TITLE, tl_block - TL_TITLE - TL_XLAB))
        if _draw_timeline(ax2, vnodes):
            ax2.set_title("Wall-clock execution timeline (real timestamps; bars overlapping on a "
                          "row are measured concurrency)", fontsize=8.5, loc="left", pad=4)
            bottom += tl_block
        else:
            ax2.remove()

    handles = [mpatches.Patch(color=c, label=t) for t, (c, _) in NODE_STYLE.items()
               if any(n["node_type"] == t for n in vnodes)]
    handles += [Line2D([0], [0], color=c, linestyle=ls, lw=lw, label=f"{t} edge")
                for t, (c, ls, lw) in EDGE_STYLE.items()
                if any(e["edge_type"] == t for e in vedges)]
    deck.fig.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, 1 - (bottom + 0.06) / PAGE[1]),
                    ncol=len(handles), fontsize=7.5, frameon=False)
    deck.y = PAGE[1]  # the axes own the rest of the page


def prompt_page(deck: Deck, g: dict, pick: Pick) -> bool:
    """The inputs the run started from: system prompt, tool schemas, task text.

    Returns False without drawing anything when the source ships none of the
    three -- TraceLab's case. A page of three "absent" lines reads as a
    rendering failure; the absence is stated on the walkthrough page instead.
    """
    prov = g.get("provenance", {})
    prompts_path = GRAPHS / pick.dataset / "prompts.json"
    prompts = json.loads(prompts_path.read_text()) if prompts_path.exists() else {}

    sys_txt = prompts.get(prov.get("system_prompt_sha256") or "")
    tool_txt = prompts.get(prov.get("tool_schemas_sha256") or "")
    users = [n for n in g["nodes"] if n["node_type"] == "user"]
    task = users[0].get("reasoning_text") if users else None
    if not (sys_txt or tool_txt or task):
        return False

    deck.new_page(f"{pick.title}  —  inputs to the run",
                  "what the model was given before step 1")

    deck.band("SYSTEM PROMPT", NODE_STYLE["llm"][0])
    if sys_txt:
        deck.line(f"sha256 {prov['system_prompt_sha256'][:16]}…   {len(sys_txt):,} chars   "
                  f"identical across trajectories, so it is stored once in "
                  f"data/graphs/{pick.dataset}/prompts.json", size=6.4, color="#777")
        deck.field("excerpt", sys_txt, max_lines=11)
    else:
        deck.field("prompt", None, absent=f"{ABSENT} — this release ships no system prompt")
    deck.space(0.06)

    deck.band("TOOL SCHEMAS OFFERED", NODE_STYLE["tool"][0])
    if tool_txt:
        try:
            names = [s.get("function", {}).get("name", "?") for s in json.loads(tool_txt)]
        except (json.JSONDecodeError, AttributeError):
            names = []
        deck.line(f"sha256 {prov['tool_schemas_sha256'][:16]}…   {len(tool_txt):,} chars   "
                  f"{len(names)} tools: {', '.join(names)}", size=6.6, color="#333")
        deck.field("excerpt", tool_txt, max_lines=7, hard=True)
    else:
        deck.field("schemas", None,
                   absent=f"{ABSENT} — the offered action space is not recoverable")
    deck.space(0.06)

    deck.band("TASK / INPUT PROMPT", NODE_STYLE["user"][0])
    if not users:
        note = "this trace has no user node"
        if pick.dataset == "osworld_gelato":
            note += (f" — the release ships only the OSWorld task uuid "
                     f"({prov.get('task_id', '?')}), not the instruction text, so the task "
                     f"statement is not recoverable without joining upstream")
        deck.field("task", None, absent=f"{ABSENT}: {note}")
    else:
        deck.line(f"{len(users)} user node(s); first shown. {cost_bits(users[0])}",
                  size=6.4, color="#777")
        deck.field("text", task, max_lines=max(4, min(20, deck.lines_left(6.5) - 2)),
                   absent=f"{ABSENT} — TraceLab's sanitizer replaced the message with a "
                          f"character count before release")
    return True


def walkthrough_pages(deck: Deck, g: dict, pick: Pick, n_steps: int,
                      inputs_shown: bool, marks: list[tuple[int, float]] | None = None) -> None:
    """Walk the first `n_steps` steps. `marks` collects (steps drawn, y reached at
    the end of that step) so `fit_steps` can pick a count that fills its pages."""
    nodes = g["nodes"]
    labels = node_labels(nodes)
    ins, outs = edge_index(g["edges"])
    order = {nid: i for i, nid in enumerate(n["node_id"] for n in nodes)}

    by_turn: defaultdict[Any, list[dict]] = defaultdict(list)
    for n in nodes:
        by_turn[n.get("turn_id")].append(n)
    turns = sorted(by_turn, key=lambda t: (t is None, t))
    steps = [t for t in turns if any(x["node_type"] != "user" for x in by_turn[t])]

    # Sub-agent delegation is the one structure only this corpus's production
    # source has, and it can sit well past the first few steps -- pull the first
    # such step into the window rather than let the example miss it.
    shown = steps[:n_steps]
    late_agent = next((t for t in steps
                       if t not in shown
                       and any(x["node_type"] == "agent" for x in by_turn[t])), None)
    if late_agent is not None:
        shown = shown + [late_agent]

    sub = (f"showing {len(shown)} of {len(steps)} steps · every field the trace holds for each "
           f"node, verbatim and truncated only where marked")
    if not inputs_shown:
        sub += " · this source ships no system prompt and no task text"
    deck.new_page(f"{pick.title}  —  step-by-step content", sub)

    rank = {"user": 0, "llm": 1, "agent": 2, "tool": 3}
    prev_i, drawn = 0, 0
    for t in shown:
        si = steps.index(t) + 1
        group = sorted(by_turn[t], key=lambda n: (rank.get(n["node_type"], 9), order[n["node_id"]]))
        deck.ensure(1.15)
        if si > prev_i + 1:
            deck.line(f"⋯ steps {prev_i + 1}–{si - 1} skipped; jumping to the first step that "
                      f"delegates to a sub-agent", size=6.8, color="#B07000")
            deck.space(0.04)
        prev_i = si
        deck.band(f"STEP {si}   (turn_id {t})   "
                  + " → ".join(labels[n["node_id"]] for n in group), "#4C78A8")

        for n in group:
            lab, nt = labels[n["node_id"]], n["node_type"]
            colour = NODE_STYLE.get(nt, ("#999", "?"))[0]
            head = f"{lab:<7}{nt.upper()}"
            if nt == "agent":
                head += f"  ⟨SUB-AGENT⟩  {n.get('tool_name') or ''}"
            elif nt in ("tool", "retrieval"):
                head += f"  {n.get('tool_name') or '?'}"
                if n.get("tool_status"):
                    head += f"  [{n['tool_status']}]"
            elif nt == "llm":
                head += f"  {n.get('model') or '—'}"
            deck.ensure(0.42)
            deck.line(head, size=7.2, color=colour, weight="bold")

            wired = [f"{labels.get(e['src'], '?')} —{e['edge_type']}→ {lab}" for e in ins[n["node_id"]]]
            wired += [f"{lab} —{e['edge_type']}→ {labels.get(e['dst'], '?')}" for e in outs[n["node_id"]]]
            deck.line("edges      " + ("; ".join(wired) if wired else "none"),
                      size=6.3, color="#777", indent=0.06)
            deck.line("cost       " + cost_bits(n), size=6.3, color="#777", indent=0.06)

            if nt in ("llm", "user"):
                deck.field("reasoning", n.get("reasoning_text"), max_lines=7,
                           absent=f"{ABSENT} (text stripped before release; only counts survive)"
                           if pick.dataset == "tracelab" else
                           f"{ABSENT} for this node — the model acted without narrating")
            else:
                deck.field("input", n.get("tool_input"), max_lines=6, hard=True,
                           absent=f"{ABSENT} (arguments stripped; {(n.get('extra') or {}).get('input_chars', '?')} chars recorded)"
                           if pick.dataset == "tracelab" else ABSENT)
                cmd = (n.get("extra") or {}).get("command")
                if cmd:
                    deck.field("executed", cmd, max_lines=3, hard=True, label_color="#8A5A00")
                deck.field("output", n.get("tool_output"), max_lines=7, hard=True,
                           absent=f"{ABSENT} (result stripped; {(n.get('extra') or {}).get('result_chars', '?')} chars recorded)"
                           if pick.dataset == "tracelab" else ABSENT)
            deck.space(0.045)
        deck.space(0.05)
        if t is not late_agent:  # the sub-agent step is always kept, never counted
            drawn += 1
            if marks is not None:
                marks.append((drawn, deck.y))

    if len(shown) < len(steps):
        deck.ensure(0.3)
        deck.line(f"⋯ trace continues: {len(steps) - len(shown)} further steps not shown "
                  f"(whole-trace totals are on the graph page).", size=6.8, color="#B07000")


def closing_page(deck: Deck, metrics: dict[str, dict]) -> None:
    deck.new_page("What the four examples show together",
                  "each claim below is visible on the preceding pages")
    tl, oh = metrics["tracelab"], metrics["swe_rebench_openhands"]

    points = [
        ("Cost and semantics are disjoint in public data.",
         f"TraceLab pages carry tokens, prefix-cache splits, latency and timestamps on every "
         f"step and not one word of text. The other three carry reasoning, arguments and "
         f"observations verbatim and zero cost fields "
         f"({tl['coverage_node_weighted']['tokens']:.0%} vs "
         f"{oh['coverage_node_weighted']['tokens']:.0%} token coverage). Relating why a node ran "
         f"to what it cost needs two datasets, or an instrumented run of our own."),
        ("The harness decides graph shape, not the model.",
         f"Across {sum(m['total_nodes'] for m in metrics.values()):,} nodes, the three benchmark "
         f"sources never exceed fan-out {max(metrics[p.dataset]['dist']['max_fan_out']['max'] for p in PICKS[1:])} "
         f"— every step is one tool, so the graph is the chain. TraceLab's production sessions "
         f"reach fan-out {tl['dist']['max_fan_out']['max']} with a median of "
         f"{tl['dist']['max_fan_out']['median']}, and 88.2% of its multi-tool rounds overlap in "
         f"measured wall-clock time. A corpus built only from benchmark rollouts would conclude, "
         f"wrongly, that agent computation graphs are inherently serial."),
        ("Trace size is a configuration setting.",
         f"OpenHands stops at 201 nodes (100 iterations) and OSWorld at 100 nodes (50 steps); "
         f"SWE-agent has no cap and decays smoothly to "
         f"{metrics['swe_agent_traj']['dist']['n_nodes']['max']:,}. TraceLab is unbounded: "
         f"p99 {tl['dist']['n_nodes']['p99']:,}, max {tl['dist']['n_nodes']['max']:,}."),
        ("Only production traffic has a conversation in it.",
         "The benchmark traces each begin with exactly one injected task statement and then run "
         "uninterrupted until a cap or a `submit`. TraceLab sessions have a human interrupting, "
         "redirecting and resuming — mean 11.3 user turns, and 4.2% of sessions span more than a "
         "day. That is what produces prompt-cache reuse across hours, which a benchmark rollout "
         "never exercises."),
        ("Sub-agents appear in production and nowhere else.",
         f"{tl['node_type_histogram']['agent']:,} `agent` nodes across 300 of 4,265 TraceLab "
         f"sessions (Claude `Agent`, Codex `spawn_agent`/`wait_agent`); all three semantic "
         f"sources are single-agent. No dataset in hand has reasoning AND sub-agents together, "
         f"which is the clearest gap to target next."),
    ]
    for i, (head, body) in enumerate(points, 1):
        deck.ensure(0.9)
        deck.line(f"{i}.  {head}", size=9.0, mono=False, weight="bold", color="#111")
        deck.para(body, size=8.0, mono=False, indent=0.28, color="#333")
        deck.space(0.10)

    deck.rule()
    deck.para("Reproduce this deck: .venv/bin/python scripts/render_showcase_pdf.py --seed "
              "<n>. Traces are sampled from a seeded, legibility-constrained pool; the "
              "constraint is printed on each dataset's graph page.", size=7.0, color="#777")


# -------------------------------------------------------------------- assembly


def fit_steps(g: dict, pick: Pick, n_steps: int, inputs_shown: bool) -> int:
    """Largest step count (≤ n_steps) whose final page is at least 45% full.

    Step content varies by three orders of magnitude — a `hotkey` call with a
    114-character result against an 8 KB `cat` — so a fixed count strands two
    lines of tool output on a page of its own often enough to matter in a deck.
    Rendering is cheap, so the walkthrough is laid out once into a throwaway PDF
    just to see where the page breaks land.
    """
    marks: list[tuple[int, float]] = []
    with tempfile.TemporaryDirectory() as td:
        with PdfPages(Path(td) / "dry.pdf") as pdf:
            deck = Deck(pdf)
            walkthrough_pages(deck, g, pick, n_steps, inputs_shown, marks=marks)
            deck.end_page()
    full = [k for k, y in marks if y / (PAGE[1] - MARGIN_B) >= 0.45]
    return max(full) if full and max(full) >= 3 else n_steps


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the per-dataset showcase PDF")
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--steps", type=int, default=6, help="steps walked per trace")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--out", type=Path,
                    default=REPO / "reports" / "figures" / "showcase" / "acg_example_traces.pdf")
    args = ap.parse_args()

    picks = [p for p in PICKS if args.datasets is None or p.dataset in args.datasets]
    metrics = {p.dataset: json.loads((GRAPHS / p.dataset / "metrics.json").read_text())
               for p in PICKS}

    chosen = []
    for p in picks:
        idx, row, how = choose(p, args.seed)
        print(f"[{p.dataset}] record #{idx}  {row['graph_id']}  "
              f"{row['n_nodes']} nodes, depth {row['depth']}")
        chosen.append((p, idx, row, how))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        deck = Deck(pdf)
        cover_page(deck, metrics,
                   {p.dataset: f"{r['graph_id']}  ({r['n_nodes']} nodes, depth {r['depth']})"
                    for p, _, r, _ in chosen})
        for p, idx, row, how in chosen:
            g = load_by_index(p.dataset, idx)
            graph_page(deck, g, p, row, how, idx)
            shown_inputs = prompt_page(deck, g, p)
            steps = fit_steps(g, p, args.steps, shown_inputs)
            walkthrough_pages(deck, g, p, steps, shown_inputs)
            print(f"[{p.dataset}] rendered, {steps} steps walked")
        closing_page(deck, metrics)
        deck.end_page()
        pdf.infodict().update({
            "Title": "Agentic Computation Graph Corpus — example traces",
            "Subject": "One randomly drawn trace per extracted dataset, with per-step content",
        })

    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB, {deck.n_pages} pages)")


if __name__ == "__main__":
    main()
