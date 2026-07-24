#!/usr/bin/env python3
"""Month-2: run a set of multi-hop QA tasks MANY times each, capture every ACG, and
report the per-task distributions + run-to-run structural variance.

  ./.venv/bin/python scripts/run_experiment.py --tasks all --reps 30
  ./.venv/bin/python scripts/run_experiment.py --tasks T02,T06 --reps 50 --temperature 0.7

All runs are written to one trace file (one OTel trace per run). Each run is fully
described by (model, decode params, seed-policy); with temperature>0 the only thing
that changes run-to-run is sampling — which is exactly the variance we characterize.
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acg import tracing as T
from acg.config import load_config
from acg.corpus import Corpus
from acg.agent import Agent
from acg.tasks import load_tasks
from acg import graph as G
from acg import provenance as P
import scripts.analyze as analyze  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="all", help="'all' or comma-separated task_ids")
    ap.add_argument("--tasks-file", default=None,
                    help="path to a tasks .jsonl (default: cfg.tasks_path). e.g. data/tasks_branch.jsonl")
    ap.add_argument("--corpus", default=None,
                    help="path to a corpus .json (default: cfg.corpus_path). e.g. data/corpus_large.json")
    ap.add_argument("--distractors", default=None,
                    help="path to a distractors .json (optional). e.g. data/distractors_large.json")
    ap.add_argument("--reps", type=int, default=30, help="repetitions per task")
    ap.add_argument("--temperature", type=float, default=None, help="override decode temperature")
    ap.add_argument("--vary-seed", action="store_true",
                    help="use a different per-run seed (isolates sampling from serving noise)")
    ap.add_argument("--trace", default="traces/experiment.jsonl")
    ap.add_argument("--outdir", default=None, help="where to write metrics/summary/figures (default: trace dir)")
    args = ap.parse_args()

    cfg = load_config()
    if args.temperature is not None:
        cfg.decode.temperature = args.temperature

    trace_file = Path(args.trace)
    if trace_file.exists():
        trace_file.unlink()
    T.configure_tracing(trace_file)

    corpus_path = args.corpus or cfg.corpus_path
    corpus = Corpus.load(corpus_path, args.distractors)
    tasks_path = args.tasks_file or cfg.tasks_path
    all_tasks = load_tasks(tasks_path)
    if args.tasks != "all":
        wanted = set(args.tasks.split(","))
        all_tasks = [t for t in all_tasks if t.task_id in wanted]

    agent = Agent(cfg, corpus)
    total = len(all_tasks) * args.reps
    print(f"Running {len(all_tasks)} tasks x {args.reps} reps = {total} runs "
          f"@ temperature={cfg.decode.temperature}, vary_seed={args.vary_seed}\n")

    # Provenance: pin config + serving stack + seed policy so this trace is self-describing.
    seed_policy = (f"varied per run: seed = 1000 + rep (reps 0..{args.reps - 1})"
                   if args.vary_seed else f"fixed: seed = {cfg.decode.seed}")
    prov = P.capture(cfg, experiment=trace_file.stem, extra={
        "tasks": [t.task_id for t in all_tasks],
        "reps": args.reps,
        "total_runs": total,
        "temperature": cfg.decode.temperature,
        "vary_seed": args.vary_seed,
        "seed_policy": seed_policy,
        "trace_file": str(trace_file),
        "tasks_file": str(tasks_path),
        "corpus_file": str(corpus_path),
        "max_tool_workers": cfg.max_tool_workers,
    })

    t0 = time.time()
    done = correct = 0
    for task in all_tasks:
        for rep in range(args.reps):
            if args.vary_seed:
                cfg.decode.seed = 1000 + rep
            res = agent.run(task, run_id=uuid.uuid4().hex[:12])
            done += 1
            correct += int(res.correct)
            if done % 10 == 0 or done == total:
                print(f"  [{done:>4}/{total}] {task.task_id} rep{rep:<2} "
                      f"-> {res.outcome:<9} nodes={res.num_llm_calls + res.num_tool_calls} "
                      f"tok={res.total_input_tokens + res.total_output_tokens} "
                      f"(running acc={correct/done:.2f})")
    T.shutdown_tracing()
    print(f"\nfinished {done} runs in {time.time()-t0:.1f}s "
          f"({(time.time()-t0)/max(done,1):.2f}s/run), accuracy={correct/max(done,1):.3f}\n")

    # Reconstruct + summarize.
    runs = G.reconstruct_runs(trace_file)
    summary = analyze.summarize(runs)
    print("== Per-task ACG size & structural variance ==")
    analyze.print_table(summary)

    outdir = Path(args.outdir) if args.outdir else trace_file.parent
    outdir.mkdir(parents=True, exist_ok=True)
    # Derive a filename tag from the trace stem so runs never clobber each other:
    #   "experiment"        -> metrics.csv / summary.json / dist_*.png  (canonical study)
    #   "complex_experiment"-> complex_*                                (preserved)
    #   anything else        -> <stem>_*                                (e.g. scale_hivar_*)
    stem = trace_file.stem
    tag = "" if stem == "experiment" else ("complex_" if "complex" in stem else f"{stem}_")
    metrics_path = outdir / f"{tag}metrics.csv"
    summary_path = outdir / f"{tag}summary.json"
    analyze.write_metrics_csv(runs, metrics_path)
    summary_path.write_text(__import__("json").dumps(summary, indent=2))
    prov_path = outdir / f"{tag}provenance.json"
    P.write(prov, prov_path)
    figs = analyze.make_plots(runs, outdir / "figures", prefix=tag)
    print(f"\nwrote {metrics_path}, {summary_path}, {prov_path}")
    print("figures:", *figs, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
