#!/usr/bin/env python3
"""Month-1 milestone: run ONE task end to end, then draw its graph from the trace.

  ./.venv/bin/python scripts/run_single.py --task T02
  ./.venv/bin/python scripts/run_single.py --task T06 --temperature 0.0

Writes the trace to traces/single_<task>.jsonl, prints the reconstructed ACG and its
metrics, and saves a PNG to traces/figures/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acg import tracing as T
from acg.config import load_config
from acg.corpus import Corpus
from acg.agent import Agent
from acg.tasks import load_tasks
from acg import graph as G


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="T02", help="task_id from data/tasks.jsonl")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    if args.temperature is not None:
        cfg.decode.temperature = args.temperature
    if args.seed is not None:
        cfg.decode.seed = args.seed

    trace_file = cfg.__class__().corpus_path.parent.parent / "traces" / f"single_{args.task}.jsonl"
    if trace_file.exists():
        trace_file.unlink()
    T.configure_tracing(trace_file)

    corpus = Corpus.load(cfg.corpus_path)
    tasks = {t.task_id: t for t in load_tasks(cfg.tasks_path)}
    task = tasks[args.task]

    print(f"== Task {task.task_id} ({task.hops}-hop) ==")
    print(f"Q: {task.question}")
    print(f"gold: {task.answers}\n")

    agent = Agent(cfg, corpus)
    res = agent.run(task)
    T.shutdown_tracing()

    print(f"answer : {res.answer!r}")
    print(f"outcome: {res.outcome}  (correct={res.correct})")
    print(f"steps={res.num_steps} llm_calls={res.num_llm_calls} tool_calls={res.num_tool_calls} "
          f"tools={res.tool_call_names}")
    print(f"tokens : in={res.total_input_tokens} out={res.total_output_tokens} "
          f"wall={res.wall_clock_s:.2f}s\n")

    # Reconstruct the ACG offline from the captured trace.
    runs = G.reconstruct_runs(trace_file)
    assert runs, "no runs reconstructed from trace"
    run = runs[0]
    m = run.metrics
    print("== Reconstructed ACG ==")
    print(G.draw_ascii(run.graph))
    print("\n== Metrics ==")
    for k, v in m.to_row().items():
        print(f"  {k:18s}: {v}")

    fig = trace_file.parent / "figures" / f"acg_{args.task}.png"
    G.draw_png(run.graph, fig, title=f"ACG — {task.task_id} ({task.hops}-hop) — {m.outcome}")
    print(f"\nGraph drawing saved to: {fig}")
    print(f"Trace saved to:         {trace_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
