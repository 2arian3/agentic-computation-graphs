#!/usr/bin/env python3
"""Cross-cell analysis for the RQ-N2/N8 branch experiment.

Reads every `traces/branch_<model>_<config>.jsonl` produced by the matrix and reports,
per (model x config): accuracy, emitted width vs width_executed, the fraction of runs that
actually fan out (executed width >= 2), and sub_agent usage. This is the table that answers
"does the agent branch when the task invites it and the tools allow it -- as a function of
model capability?" and "does real executed concurrency ever appear?".

  ./.venv/bin/python scripts/analyze_branch.py
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acg import graph as G
from acg import tracing as TR

# (model_tag, precision label) in report order; configs are off/on for sub_agent.
MODELS = [
    ("7b", "Qwen2.5-7B (16-bit)"),
    ("14bfp8", "Qwen2.5-14B FP8 (8-bit)"),
    ("14bawq", "Qwen2.5-14B AWQ (4-bit)"),
    ("llama31", "Llama-3.1-8B (16-bit)"),
]
CONFIGS = [("nosub", "plain"), ("sub", "+sub_agent")]
TRACE_DIR = Path("traces")


def _top_level_subagents(g) -> int:
    """Count sub_agent tool nodes issued at the top level of the run."""
    return sum(
        1 for n, d in g.nodes(data=True)
        if d.get("tool_name") == "sub_agent" and "/" not in str(n)
    )


def _max_calls_per_turn(g) -> int:
    """Max tool calls a single TOP-LEVEL LLM turn emitted -- the honest 'emitted parallelism'.

    Unlike the graph's level-width, this is robust to sub_agent nesting: it counts the tool
    children of each top-level LLM node (one assistant turn) and takes the max. 1 means the
    model always issued tools one at a time; >=2 means it asked for a genuine parallel batch.
    """
    best = 0
    for n, d in g.nodes(data=True):
        if d.get("type") == TR.NODE_TYPE_LLM and "/" not in str(n):
            k = sum(1 for s in g.successors(n) if g.nodes[s].get("type") == TR.NODE_TYPE_TOOL)
            best = max(best, k)
    return best


def _cell_stats(trace: Path) -> dict | None:
    if not trace.exists():
        return None
    runs = G.reconstruct_runs(trace)
    if not runs:
        return None
    n = len(runs)
    acc = sum(r.metrics.outcome == "correct" for r in runs) / n
    emit = [_max_calls_per_turn(r.graph) for r in runs]   # honest emitted parallelism
    execd = [r.metrics.width_executed for r in runs]
    subs = [_top_level_subagents(r.graph) for r in runs]
    errored = sum(bool(r.graph.nodes[G.find_root(r.graph)].get("error")) for r in runs)
    return {
        "n": n,
        "accuracy": round(acc, 3),
        "pct_runs_errored": round(errored / n, 3),   # serving/tool-protocol failures (e.g. parallel-call reject)
        "emit_per_turn_mean": round(st.mean(emit), 2),
        "emit_per_turn_max": max(emit),
        "pct_runs_emit_parallel": round(sum(e >= 2 for e in emit) / n, 3),
        "exec_width_mean": round(st.mean(execd), 2),
        "exec_width_max": max(execd),
        "pct_fanout_executed": round(sum(w >= 2 for w in execd) / n, 3),
        "pct_runs_used_subagent": round(sum(s >= 1 for s in subs) / n, 3),
        "subagent_calls_mean": round(st.mean(subs), 2),
        "nodes_mean": round(st.mean(r.metrics.node_count for r in runs), 1),
        "depth_mean": round(st.mean(r.metrics.depth for r in runs), 1),
        "tokens_mean": round(st.mean(r.metrics.total_tokens for r in runs), 0),
    }


def _served_model(tag: str, label: str) -> str | None:
    prov = TRACE_DIR / f"branch_{tag}_{label}_provenance.json"
    if not prov.exists():
        return None
    d = json.loads(prov.read_text())
    ms = (d.get("server") or {}).get("models") or []
    return ms[0] if ms else None


def main() -> int:
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    rows, out = [], {}
    for tag, prec in MODELS:
        for label, cfg_name in CONFIGS:
            s = _cell_stats(TRACE_DIR / f"branch_{tag}_{label}.jsonl")
            if s is None:
                continue
            out[f"{tag}_{label}"] = {"model": prec, "config": cfg_name,
                                     "served": _served_model(tag, label), **s}
            rows.append([
                prec, cfg_name, s["n"], f'{s["accuracy"]:.2f}', f'{s["pct_runs_errored"]:.2f}',
                f'{s["emit_per_turn_mean"]:.2f}/{s["emit_per_turn_max"]}',
                f'{s["pct_runs_emit_parallel"]:.2f}',
                f'{s["exec_width_mean"]:.2f}/{s["exec_width_max"]}',
                f'{s["pct_fanout_executed"]:.2f}',
                f'{s["pct_runs_used_subagent"]:.2f}',
                f'{s["nodes_mean"]:.1f}', f'{s["depth_mean"]:.1f}',
            ])

    headers = ["model", "config", "n", "acc", "%err", "emit/turn mean/max", "%emit_par",
               "exec_w mean/max", "%fanout(exec>=2)", "%used_subagent", "nodes", "depth"]
    print("\n===============  RQ-N2/N8: branching vs model capability & tool availability  ===============\n")
    if tabulate:
        print(tabulate(rows, headers=headers, tablefmt="github"))
    else:
        print("\t".join(headers))
        for r in rows:
            print("\t".join(map(str, r)))

    outpath = TRACE_DIR / "branch_comparison.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outpath}")
    print("\nRead: emit/turn = max tool calls the model issued in ONE turn (what it *asked* to run "
          "at once); %emit_par = runs with any turn >=2; exec_w = tool spans that actually ran "
          "concurrently; %used_subagent = runs that adopted the branch tool. A model that emits "
          "sub_agents one-per-turn shows high %used_subagent but emit/turn~1 and exec_w=1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
