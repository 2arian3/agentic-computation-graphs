#!/usr/bin/env python3
"""RQ-E1 (next-step #3): can we PREDICT an ACG's cost from the task before running it?

Aggregates per-task cost (node count, total tokens; mean and p95) from recorded runs,
builds cheap task features (hop count, question length, entity count), fits a linear
predictor, and reports honest leave-one-out cross-validation error (12 tasks is tiny, so
LOO is the right validation).

  ./.venv/bin/python scripts/cost_model.py --trace traces/experiment.jsonl
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acg import graph as G
from acg.tasks import load_tasks
from acg.config import load_config


def _features(task):
    q = task.question
    qlen = len(q.split())
    # entity proxy: capitalized tokens that aren't the first word
    caps = len([w for w in re.findall(r"[A-Za-z][A-Za-z0-9'-]+", q)[1:] if w[0].isupper()])
    return {"hops": task.hops, "qlen": qlen, "entities": caps}


def _loo_linear(X, y):
    """Leave-one-out MAE for ordinary least squares with intercept."""
    n = len(y)
    Xb = np.column_stack([np.ones(n), X])
    errs = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        beta, *_ = np.linalg.lstsq(Xb[idx], y[idx], rcond=None)
        errs.append(abs(Xb[i] @ beta - y[i]))
    # full-fit R^2
    beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    pred = Xb @ beta
    ss_res = np.sum((y - pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return float(np.mean(errs)), float(r2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="traces/experiment.jsonl")
    args = ap.parse_args()

    runs = G.reconstruct_runs(args.trace)
    tasks = {t.task_id: t for t in load_tasks(load_config().tasks_path)}
    by_task = {}
    for r in runs:
        by_task.setdefault(r.task_id, []).append(r)

    feats, targets = [], {"mean_nodes": [], "p95_nodes": [], "mean_tokens": [], "p95_tokens": []}
    tids = sorted(by_task)
    for tid in tids:
        rs = by_task[tid]
        f = _features(tasks[tid])
        feats.append([f["hops"], f["qlen"], f["entities"]])
        nodes = np.array([r.metrics.node_count for r in rs], float)
        toks = np.array([r.metrics.total_tokens for r in rs], float)
        targets["mean_nodes"].append(nodes.mean())
        targets["p95_nodes"].append(np.percentile(nodes, 95))
        targets["mean_tokens"].append(toks.mean())
        targets["p95_tokens"].append(np.percentile(toks, 95))

    F = np.array(feats, float)  # columns: hops, qlen, entities
    print(f"{len(tids)} tasks; features = [hops, qlen, entities]\n")
    from tabulate import tabulate
    rows = []
    for tgt, yv in targets.items():
        y = np.array(yv, float)
        mae_h, r2_h = _loo_linear(F[:, [0]], y)          # hops only
        mae_all, r2_all = _loo_linear(F, y)              # all features
        rows.append([tgt, f"{y.mean():.0f}",
                     f"{r2_h:.2f}", f"{mae_h:.2f}",
                     f"{r2_all:.2f}", f"{mae_all:.2f}"])
    print(tabulate(rows, headers=["target", "actual mean",
                                  "R² (hops)", "LOO-MAE (hops)",
                                  "R² (all feats)", "LOO-MAE (all)"], tablefmt="github"))
    print("\nInterpretation: high R² + low LOO-MAE ⇒ the metric is predictable from cheap task "
          "features BEFORE running the agent — the input a cost/latency planner needs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
