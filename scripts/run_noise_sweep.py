#!/usr/bin/env python3
"""RQ-N1: does ACG structure emerge from the MODEL or from the (clean) CORPUS?

Sweeps a controlled retrieval-noise knob (0 = current clean corpus; 1,2 = that many
near-homophone distractor docs injected into every search result) over the 12 tasks, and
measures whether branching / backtracking / re-query / width / size rise as retrieval gets
noisy. If they DON'T, "linear-dominant, parallel-rare" is a real model property; if they
DO, it was a clean-corpus artifact.

  ./.venv/bin/python scripts/run_noise_sweep.py --reps 8
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acg import tracing as T
from acg import graph as G
from acg.config import load_config, DATA_DIR
from acg.corpus import Corpus
from acg.agent import Agent
from acg.tasks import load_tasks
import scripts.analyze as analyze
import scripts.structure_taxonomy as tax


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--levels", default="0,1,2")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    cfg = load_config()
    cfg.decode.temperature = args.temperature
    corpus = Corpus.load(cfg.corpus_path, DATA_DIR / "distractors.json")
    tasks = load_tasks(cfg.tasks_path)
    levels = [int(x) for x in args.levels.split(",")]

    per_level = {}
    for L in levels:
        corpus.noise = L
        trace = Path(f"traces/noise{L}.jsonl")
        if trace.exists():
            trace.unlink()
        T.configure_tracing(trace)
        agent = Agent(cfg, corpus)
        print(f"\n### noise level {L}: {len(tasks)} tasks x {args.reps} reps ###")
        done = correct = 0
        for task in tasks:
            for rep in range(args.reps):
                cfg.decode.seed = 1000 + rep
                res = agent.run(task, run_id=uuid.uuid4().hex[:12])
                done += 1; correct += int(res.correct)
                if done % 24 == 0:
                    print(f"  [{done}/{len(tasks)*args.reps}] running acc={correct/done:.2f}")
        T.flush_tracing()
        per_level[L] = trace

    # ---- compare across noise levels ----
    print("\n\n================  RQ-N1 RESULT: structure vs retrieval noise  ================")
    from tabulate import tabulate
    rows = []
    tax_by_level = {}
    for L, trace in per_level.items():
        runs = G.reconstruct_runs(trace)
        s = analyze.summarize(runs)["overall"]
        t = tax.analyze_trace(str(trace)); tax_by_level[L] = t
        mp = t["motif_prevalence"]
        rows.append([
            L, s["num_runs"], f'{s["accuracy"]:.2f}',
            f'{s["node_count"]["mean"]:.1f}', f'{s["width"]["mean"]:.2f}',
            f'{mp["linear_chain"]:.2f}', f'{mp["parallel_fanout"]:.2f}',
            f'{mp["redundant_loop"]:.2f}', f'{mp["degenerate_shortcircuit"]:.2f}',
            f'{mp["iterative_multihop"]:.2f}',
        ])
    print(tabulate(rows, headers=["noise", "n", "acc", "nodes", "width",
                                  "linear", "parallel", "loop", "shortcut", "iter_multihop"],
                   tablefmt="github"))
    print("\nHypothesis: with noise, expect accuracy ↓, nodes ↑, and loop/iter_multihop/parallel ↑ "
          "(the model must re-query, cross-check, and discriminate). Flat structure ⇒ the clean "
          "result was a model property, not an artifact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
