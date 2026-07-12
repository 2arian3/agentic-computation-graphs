#!/usr/bin/env python3
"""Stable-core map across all tasks (next-step #1): for each task, the depth of the
recurring core subgraph (steps every run shares) and where the graph first diverges.

  ./.venv/bin/python scripts/stable_core_map.py --trace traces/experiment.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acg import graph as G
from acg.tasks import load_tasks
from acg.config import load_config
import scripts.branch_points as B


def stable_core(trajs):
    if not trajs:
        return 0
    core = 0
    for i in range(min(len(t) for t in trajs)):
        if len({t[i] for t in trajs}) == 1:
            core += 1
        else:
            break
    return core


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="traces/experiment.jsonl")
    args = ap.parse_args()

    runs = G.reconstruct_runs(args.trace)
    hops = {t.task_id: t.hops for t in load_tasks(load_config().tasks_path)}
    by_task = {}
    for r in runs:
        by_task.setdefault(r.task_id, []).append(r)

    from tabulate import tabulate
    rows, xs_hops, ys_core, ys_nodes = [], [], [], []
    for tid in sorted(by_task):
        rs = by_task[tid]
        trajs = [B.trajectory(r.graph) for r in rs]
        core = stable_core(trajs)
        n_traj = len({tuple(t) for t in trajs})
        mean_nodes = np.mean([r.metrics.node_count for r in rs])
        h = hops.get(tid, 0)
        rows.append([tid, h, len(rs), core, n_traj, f"{mean_nodes:.1f}"])
        xs_hops.append(h); ys_core.append(core); ys_nodes.append(mean_nodes)
    print(tabulate(rows, headers=["task", "hops", "n", "stable_core_depth",
                                  "#trajectories", "mean_nodes"], tablefmt="github"))
    if len(set(xs_hops)) > 1:
        print(f"\ncorr(hops, stable_core_depth) = {np.corrcoef(xs_hops, ys_core)[0,1]:.2f}")
        print(f"corr(hops, mean_nodes)       = {np.corrcoef(xs_hops, ys_nodes)[0,1]:.2f}")
    print("\nNote: stable-core depth is measured at the trace's n; small n overestimates it "
          "(fewer runs agree on a longer prefix by chance). Cross-check high-variance tasks at n=50.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
