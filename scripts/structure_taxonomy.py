#!/usr/bin/env python3
"""RQ (supervisor Q2): what structural motifs actually occur in ACGs — is it just
'branch and parallel'?

Classifies every reconstructed run into structural motifs and reports their prevalence.
Motifs are not mutually exclusive (a run can be a linear chain AND contain a repeat
loop), so we report per-motif prevalence plus a single 'primary shape' per run.

  ./.venv/bin/python scripts/structure_taxonomy.py \
     --trace traces/experiment.jsonl:7B --trace traces/qwen14b_fp8.jsonl:14B-FP8
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acg import graph as G
from acg import tracing as T

# Motif definitions (see classify_run) --------------------------------------
# NB: a 2nd call of the SAME TOOL TYPE for a DIFFERENT target (e.g. search hop-1
# then search hop-2) is *normal multi-hop iteration*, NOT a loop. Only an EXACT
# repeat (same tool + same argument) is genuine backtracking / wasted re-work.
MOTIFS = [
    "linear_chain",           # width 1, reaches finish, no exact repeat: the sequential backbone
    "iterative_multihop",     # uses a tool type >=2x for DIFFERENT targets (how depth grows)
    "parallel_fanout",        # width >= 2: >1 tool call issued in a single LLM turn (true branching)
    "redundant_loop",         # EXACT repeat: re-calls same tool on same target (backtracking)
    "degenerate_shortcircuit",# 0 tool calls: answered in one LLM shot
    "truncated_no_finish",    # never called finish (hit max_steps / gave up)
]


def _tool_usage(g) -> Counter:
    c = Counter()
    for n, d in g.nodes(data=True):
        if d.get("type") == T.NODE_TYPE_TOOL:
            c[d.get("tool_name") or "unknown"] += 1
    return c


def classify_run(run) -> dict:
    g = run.graph
    m = run.metrics
    exact = [r for r in G.detect_behavioral_repeats(g) if r.kind == "exact"]

    tools = m.num_tool_calls
    tb = m.tool_breakdown
    usage = _tool_usage(g)
    tags = set()

    if tools == 0:
        tags.add("degenerate_shortcircuit")
    if m.width >= 2:
        tags.add("parallel_fanout")
    if exact:
        tags.add("redundant_loop")
    if tools >= 1 and tb.get("finish", 0) == 0:
        tags.add("truncated_no_finish")
    if any(c >= 2 for name, c in usage.items() if name != "finish"):
        tags.add("iterative_multihop")
    if tools >= 1 and m.width < 2 and not exact and tb.get("finish", 0) >= 1:
        tags.add("linear_chain")

    # single 'primary shape' by salience (rarest/most-notable structure wins)
    for shape in ("degenerate_shortcircuit", "parallel_fanout", "redundant_loop",
                  "truncated_no_finish", "linear_chain"):
        if shape in tags:
            primary = shape
            break
    else:
        primary = "other"

    return {"tags": tags, "primary": primary,
            "n_exact_repeats": len(exact),
            "width": m.width, "depth": m.depth, "nodes": m.node_count,
            "outcome": m.outcome}


def analyze_trace(path: str):
    runs = G.reconstruct_runs(path)
    rows = [classify_run(r) for r in runs]
    n = len(rows)
    motif_counts = Counter()
    for r in rows:
        for t in r["tags"]:
            motif_counts[t] += 1
    primary_counts = Counter(r["primary"] for r in rows)
    # correlate structure with correctness
    correct_by_primary = defaultdict(lambda: [0, 0])
    for r in rows:
        c = correct_by_primary[r["primary"]]
        c[0] += int(r["outcome"] == "correct"); c[1] += 1
    return {
        "n": n,
        "motif_prevalence": {k: round(motif_counts[k] / n, 3) for k in MOTIFS},
        "primary_shape": {k: round(primary_counts[k] / n, 3) for k in sorted(primary_counts)},
        "primary_counts": dict(primary_counts),
        "accuracy_by_primary": {k: round(v[0] / v[1], 2) for k, v in correct_by_primary.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="append", required=True,
                    help="path[:label], repeatable")
    ap.add_argument("--out", default="traces/structure_taxonomy.json")
    args = ap.parse_args()

    results = {}
    for spec in args.trace:
        path, _, label = spec.partition(":")
        label = label or Path(path).stem
        results[label] = analyze_trace(path)

    # print
    from tabulate import tabulate
    labels = list(results)
    print("== Motif prevalence (fraction of runs exhibiting the motif) ==")
    rows = [[mo] + [f'{results[l]["motif_prevalence"][mo]:.2f}' for l in labels] for mo in MOTIFS]
    print(tabulate(rows, headers=["motif"] + [f"{l} (n={results[l]['n']})" for l in labels], tablefmt="github"))

    print("\n== Primary shape distribution ==")
    shapes = sorted({s for l in labels for s in results[l]["primary_shape"]})
    rows = [[s] + [f'{results[l]["primary_shape"].get(s,0):.2f}' for l in labels] for s in shapes]
    print(tabulate(rows, headers=["primary shape"] + labels, tablefmt="github"))

    print("\n== Accuracy by primary shape ==")
    rows = [[s] + [f'{results[l]["accuracy_by_primary"].get(s,"-")}' for l in labels] for s in shapes]
    print(tabulate(rows, headers=["primary shape"] + labels, tablefmt="github"))

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
