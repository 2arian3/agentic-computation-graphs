#!/usr/bin/env python3
"""RQ-E2 / follow-up to RQ-A1: WHERE along the graph is run-to-run variance injected?

Aligns every run's decision trajectory step-by-step and reports:
  * the STABLE CORE = the longest prefix of steps that EVERY run executes identically
    (the recurring subgraph the proposal asks about), and
  * the DIVERGENCE PROFILE = how many distinct decisions occur at each step depth
    (does the graph fan out early = high-leverage, or late = low-leverage?).

Decision key per step: tool name, plus the doc_id for reads (search query wording is
treated as noise). `finish` ends a trajectory.

  ./.venv/bin/python scripts/branch_points.py --trace traces/scale_hivar.jsonl --task T06
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acg import graph as G
from acg import tracing as T


def _decision_key(tool_name, args):
    if tool_name == "read_document":
        return f"read:{str((args or {}).get('doc_id','')).strip().upper()}"
    if tool_name == "finish":
        return "finish"
    return "search"          # query wording treated as noise


def trajectory(g):
    steps = []
    for n, d in g.nodes(data=True):
        if d.get("type") == T.NODE_TYPE_TOOL:
            steps.append((d.get("step"), _decision_key(d.get("tool_name"), d.get("tool_args"))))
    return [k for _, k in sorted(steps)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--task", required=True)
    args = ap.parse_args()

    runs = [r for r in G.reconstruct_runs(args.trace) if r.task_id == args.task]
    trajs = [trajectory(r.graph) for r in runs]
    n = len(trajs)
    if n == 0:
        print("no runs for task"); return 1

    # stable core = longest common prefix across ALL trajectories
    core = 0
    for i in range(min(len(t) for t in trajs)):
        col = {t[i] for t in trajs}
        if len(col) == 1:
            core += 1
        else:
            break
    core_seq = trajs[0][:core]

    print(f"task {args.task}: {n} runs, {len(set(map(tuple,trajs)))} distinct decision-trajectories")
    print(f"STABLE CORE = {core} step(s) identical across ALL runs: {core_seq}")
    print(f"\nDivergence profile (per step depth):")
    print(f"  {'depth':<6}{'#runs':<7}{'#distinct':<11}{'modal decision (fraction)':<34}{'alternatives'}")
    maxlen = max(len(t) for t in trajs)
    for d in range(maxlen):
        col = [t[d] for t in trajs if len(t) > d]
        c = Counter(col)
        modal, mn = c.most_common(1)[0]
        alts = ", ".join(f"{k}×{v}" for k, v in c.most_common()[1:4])
        flag = "  <-- variance injected here" if len(c) > 1 and d == core else ""
        print(f"  {d:<6}{len(col):<7}{len(c):<11}{modal+' ('+format(mn/len(col),'.2f')+')':<34}{alts}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
