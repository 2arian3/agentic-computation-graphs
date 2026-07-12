#!/usr/bin/env python3
"""RQ (supervisor Q1): at a FIXED temperature, why does the ACG differ run-to-run for
the same task? Is it the KV cache?

This isolates the candidate sources of run-to-run variance by ablating three factors
and measuring how much the graph (and the exact decision trajectory) changes:

  * SAMPLING   -- the model draws tokens stochastically at temperature>0. Controlled by
                  the request `seed`. Ablate: fixed seed vs per-run seed.
  * KV / PREFIX CACHE -- vLLM reuses KV blocks for shared prefixes. Ablate by serving
                  with `--no-enable-prefix-caching` (a separate server) and comparing.
  * BATCHING   -- concurrent requests change the order of floating-point reductions on
                  the GPU, which can flip a sampled token near a decision boundary.
                  Ablate: sequential (concurrency=1) vs concurrent (concurrency>1).

Two outcome metrics per regime (N reps of ONE task):
  * distinct ACG structures  (graph signature)         -- structural variance
  * distinct decision trajectories (exact tool+args seq) -- token-level reproducibility

Interpretation:
  fixed-seed, sequential            -> should be 1 structure & 1 trajectory (reproducible)
  fixed-seed, sequential, cache OFF -> if identical to cache ON, KV cache is NOT the cause
  fixed-seed, CONCURRENT            -> any >1 here = batching/serving noise (usually tiny)
  varied-seed                       -> many structures = sampling is the dominant cause

  ./.venv/bin/python scripts/variance_sources.py --task T06 --reps 20 \
      --seed-policy fixed --concurrency 1 --label "fixed/seq"
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acg import tracing as T
from acg import graph as G
from acg.config import load_config
from acg.corpus import Corpus
from acg.agent import Agent
from acg.tasks import load_tasks
import scripts.analyze as analyze


def decision_fingerprint(g) -> tuple:
    """The exact tool decision trajectory of a run: ordered (step, tool, canonical-args).
    Two runs with the same fingerprint made byte-identical tool decisions."""
    items = []
    for n, d in g.nodes(data=True):
        if d.get("type") == T.NODE_TYPE_TOOL:
            args = d.get("tool_args") or {}
            canon = json.dumps(args, sort_keys=True)
            items.append((d.get("step"), d.get("tool_name"), canon))
    return tuple(sorted(items))


def _run_one(task, corpus, temperature, seed):
    cfg = load_config()
    cfg.decode.temperature = temperature
    cfg.decode.seed = seed
    agent = Agent(cfg, corpus)      # fresh agent per thread -> no shared mutable seed
    return agent.run(task, run_id=uuid.uuid4().hex[:12])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T06")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed-policy", choices=["fixed", "vary"], default="fixed")
    ap.add_argument("--seed", type=int, default=1234, help="base seed")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--label", default=None)
    ap.add_argument("--trace", default="traces/variance_sources.jsonl")
    args = ap.parse_args()

    label = args.label or f"{args.seed_policy}/conc{args.concurrency}/T{args.temperature}"
    trace_file = Path(args.trace)
    if trace_file.exists():
        trace_file.unlink()
    T.configure_tracing(trace_file)

    cfg0 = load_config()
    corpus = Corpus.load(cfg0.corpus_path)
    task = {t.task_id: t for t in load_tasks(cfg0.tasks_path)}[args.task]

    seeds = [args.seed] * args.reps if args.seed_policy == "fixed" \
        else [args.seed + i for i in range(args.reps)]

    if args.concurrency > 1:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            list(ex.map(lambda s: _run_one(task, corpus, args.temperature, s), seeds))
    else:
        for s in seeds:
            _run_one(task, corpus, args.temperature, s)
    T.flush_tracing()

    runs = G.reconstruct_runs(trace_file)
    structures = {analyze.graph_signature(r.graph) for r in runs}
    fingerprints = {decision_fingerprint(r.graph) for r in runs}
    answers = {(G._run_metadata(r.graph).get("answer") or "").strip().lower() for r in runs}
    nodes = [r.metrics.node_count for r in runs]

    print(f"[{label}] task={args.task} reps={len(runs)} temp={args.temperature} "
          f"seed={args.seed_policy} conc={args.concurrency}")
    print(f"    distinct ACG structures     : {len(structures)}")
    print(f"    distinct decision trajectories: {len(fingerprints)}")
    print(f"    distinct final answers        : {len(answers)}")
    print(f"    node-count range              : {min(nodes)}..{max(nodes)}")
    reproducible = len(fingerprints) == 1
    print(f"    => {'FULLY REPRODUCIBLE (variance not from this regime)' if reproducible else 'VARIES'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
