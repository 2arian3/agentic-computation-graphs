#!/usr/bin/env python3
"""Run single-task traces + variance experiment for complex (3–4 hop) QA tasks.

  ./.venv/bin/python scripts/run_complex.py
  ./.venv/bin/python scripts/run_complex.py --reps 12 --skip-experiment
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "bin" / "python"

# 3-hop and 4-hop tasks from data/tasks.jsonl
COMPLEX_TASKS = ["T02", "T03", "T05", "T06", "T07", "T09", "T11", "T12"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=8, help="reps per task in the experiment")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--skip-single", action="store_true")
    ap.add_argument("--skip-experiment", action="store_true")
    args = ap.parse_args()

    tasks_csv = ",".join(COMPLEX_TASKS)
    trace = ROOT / "traces" / "complex_experiment.jsonl"

    if not args.skip_single:
        print(f"== Running each complex task once (fresh trace + graph PNG) ==")
        for tid in COMPLEX_TASKS:
            print(f"\n--- {tid} ---")
            rc = subprocess.call([str(PY), str(ROOT / "scripts" / "run_single.py"),
                                  "--task", tid, "--temperature", str(args.temperature)])
            if rc != 0:
                return rc

    if not args.skip_experiment:
        print(f"\n== Variance experiment: {len(COMPLEX_TASKS)} tasks x {args.reps} reps =="
              f" {len(COMPLEX_TASKS) * args.reps} runs ==")
        rc = subprocess.call([
            str(PY), str(ROOT / "scripts" / "run_experiment.py"),
            "--tasks", tasks_csv,
            "--reps", str(args.reps),
            "--temperature", str(args.temperature),
            "--vary-seed",
            "--trace", str(trace),
        ])
        if rc != 0:
            return rc

        print(f"\n== Drawing representative graphs from {trace.name} ==")
        rc = subprocess.call([
            str(PY), str(ROOT / "scripts" / "draw_graphs.py"),
            "--trace", str(trace),
            "--tasks", tasks_csv,
        ])
        if rc != 0:
            return rc

    print("\nDone. Graphs in traces/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
