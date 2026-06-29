#!/usr/bin/env python3
"""Draw human-readable ACG PNGs from a trace file.

For each task, saves:
  * the modal (most common) graph shape from the trace
  * the largest graph in the trace (useful for complex tasks)

  ./.venv/bin/python scripts/draw_graphs.py --trace traces/complex_experiment.jsonl
  ./.venv/bin/python scripts/draw_graphs.py --trace traces/single_T06.jsonl --tasks T06
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acg import graph as G
import scripts.analyze as analyze


def pick_representative_run(runs: list[G.ReconstructedRun]) -> G.ReconstructedRun | None:
    """Pick the modal-signature run, preferring a correct outcome when tied."""
    if not runs:
        return None
    by_sig: dict[tuple, list[G.ReconstructedRun]] = defaultdict(list)
    for run in runs:
        by_sig[analyze.graph_signature(run.graph)].append(run)
    modal_sig = Counter(analyze.graph_signature(r.graph) for r in runs).most_common(1)[0][0]
    candidates = by_sig[modal_sig]
    correct = [r for r in candidates if r.metrics.outcome == "correct"]
    return (correct or candidates)[0]


def pick_largest_run(runs: list[G.ReconstructedRun]) -> G.ReconstructedRun | None:
    if not runs:
        return None
    return max(runs, key=lambda r: r.metrics.node_count)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="JSONL trace file")
    ap.add_argument("--tasks", default="all", help="'all' or comma-separated task ids")
    ap.add_argument("--outdir", default=None, help="output directory (default: trace_dir/figures)")
    args = ap.parse_args()

    trace_file = Path(args.trace)
    runs = G.reconstruct_runs(trace_file)
    if not runs:
        print(f"no runs in {trace_file}")
        return 1

    wanted = None if args.tasks == "all" else set(args.tasks.split(","))
    by_task: dict[str, list[G.ReconstructedRun]] = defaultdict(list)
    for run in runs:
        if wanted is None or run.task_id in wanted:
            by_task[run.task_id].append(run)

    outdir = Path(args.outdir) if args.outdir else trace_file.parent / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    paths = []
    for task_id in sorted(by_task):
        task_runs = by_task[task_id]
        rep = pick_representative_run(task_runs)
        largest = pick_largest_run(task_runs)
        for label, run in (("modal", rep), ("largest", largest)):
            if run is None:
                continue
            m = run.metrics
            title = (
                f"ACG — {task_id} ({label}) — {m.outcome} — "
                f"{m.node_count} nodes, {m.total_tokens} tok"
            )
            out = outdir / f"acg_{task_id}_{label}.png"
            G.draw_png(run.graph, out, title=title)
            paths.append(str(out))
            print(f"  {task_id} {label:7s}: nodes={m.node_count} depth={m.depth} -> {out}")

    print(f"\nWrote {len(paths)} graph(s) to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
