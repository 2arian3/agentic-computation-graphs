#!/usr/bin/env python3
"""Bonus result (proposal §7): separate the TWO sources of run-to-run graph variance.

Because we own the serving stack, we can decompose variance into:
  (a) intended sampling randomness  -- vary the seed at temperature>0,
  (b) incidental serving-batch noise -- FIX the seed (and temperature); any remaining
      run-to-run difference is the serving layer reordering floating-point ops.

This script runs the same task many times under three regimes and reports how many
DISTINCT ACG structures appear in each. A clean result looks like:
  fixed-seed @ temp=0   -> 1 structure   (fully reproducible)
  fixed-seed @ temp>0   -> ~1 structure  (serving noise only; usually still 1)
  varied-seed @ temp>0  -> many          (sampling variance dominates)

  ./.venv/bin/python scripts/determinism_check.py --task T06 --reps 12
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acg import tracing as T
from acg import graph as G
from acg.config import load_config
from acg.corpus import Corpus
from acg.agent import Agent
from acg.tasks import load_tasks
import scripts.analyze as analyze


def _run_regime(cfg, corpus, task, reps, trace_file, *, vary_seed):
    if trace_file.exists():
        trace_file.unlink()
    T.configure_tracing(trace_file)
    agent = Agent(cfg, corpus)
    for rep in range(reps):
        if vary_seed:
            cfg.decode.seed = 2000 + rep
        agent.run(task, run_id=uuid.uuid4().hex[:12])
    T.flush_tracing()
    runs = G.reconstruct_runs(trace_file)
    sigs = {analyze.graph_signature(r.graph) for r in runs}
    nodes = [r.metrics.node_count for r in runs]
    return len(runs), len(sigs), nodes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T06")
    ap.add_argument("--reps", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config()
    corpus = Corpus.load(cfg.corpus_path)
    task = {t.task_id: t for t in load_tasks(cfg.tasks_path)}[args.task]
    td = Path("traces")

    print(f"Task {args.task}: {task.question}\n")
    regimes = [
        ("fixed-seed @ temp=0.0 ", dict(temperature=0.0, seed=1234, vary=False)),
        ("fixed-seed @ temp=0.7 ", dict(temperature=0.7, seed=1234, vary=False)),
        ("varied-seed @ temp=0.7", dict(temperature=0.7, seed=1234, vary=True)),
    ]
    print(f"{'regime':<24} {'runs':>5} {'distinct ACG structures':>26} {'node-count range':>18}")
    for label, r in regimes:
        cfg.decode.temperature = r["temperature"]
        cfg.decode.seed = r["seed"]
        n, sigs, nodes = _run_regime(
            cfg, corpus, task, args.reps, td / f"determinism_{args.task}.jsonl", vary_seed=r["vary"]
        )
        rng = f"{min(nodes)}..{max(nodes)}" if nodes else "-"
        print(f"{label:<24} {n:>5} {sigs:>26} {rng:>18}")

    print("\nInterpretation: structures in the varied-seed row above the fixed-seed row "
          "= sampling variance; any >1 in the fixed-seed rows = serving-batch noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
