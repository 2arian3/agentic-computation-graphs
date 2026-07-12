#!/usr/bin/env python3
"""Side-by-side comparison of two model runs over the SAME task set.

Reads two summary.json files (produced by run_experiment.py / analyze.py) and prints an
overall + per-task diff of accuracy and the ACG size/structure metrics. Useful for the
"second model" comparison (next-steps item 6): does a larger/quantized model change
accuracy and/or the shape of the graphs it generates?

  ./.venv/bin/python scripts/compare_models.py \
      --a traces/summary.json            --a-label 7B-bf16 \
      --b traces/qwen14b_fp8_summary.json --b-label 14B-fp8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tabulate import tabulate


def _load(p):
    return json.loads(Path(p).read_text())


def _row(task, ea, eb):
    def acc(e):
        lo, hi = e["accuracy_ci95"]
        return f'{e["accuracy"]:.2f} [{lo:.2f},{hi:.2f}]'

    def m(e, field):
        return e["distributions"][field]["mean"]

    def sv(e, key):
        return e["structural_variance"].get(key)

    da = m(ea, "total_tokens"); db = m(eb, "total_tokens")
    return [
        task,
        acc(ea), acc(eb),
        f'{m(ea,"node_count"):.1f}', f'{m(eb,"node_count"):.1f}',
        f'{m(ea,"width"):.2f}', f'{m(eb,"width"):.2f}',
        f'{da:.0f}', f'{db:.0f}', f'{(db-da):+.0f}',
        f'{sv(ea,"distinct_signatures")}/{sv(ea,"modal_signature_fraction"):.2f}',
        f'{sv(eb,"distinct_signatures")}/{sv(eb,"modal_signature_fraction"):.2f}',
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="summary.json for model A")
    ap.add_argument("--b", required=True, help="summary.json for model B")
    ap.add_argument("--a-label", default="A")
    ap.add_argument("--b-label", default="B")
    args = ap.parse_args()

    A, B = _load(args.a), _load(args.b)
    la, lb = args.a_label, args.b_label

    oa, ob = A["overall"], B["overall"]
    print(f"\n== Overall ({la} vs {lb}) ==")
    print(f"  runs:      {oa['num_runs']} vs {ob['num_runs']}   tasks: {oa['num_tasks']}")
    print(f"  accuracy:  {oa['accuracy']:.3f} {oa['accuracy_ci95']}  vs  "
          f"{ob['accuracy']:.3f} {ob['accuracy_ci95']}   (Δ {ob['accuracy']-oa['accuracy']:+.3f})")
    print(f"  nodes/run: {oa['node_count']['mean']:.1f} vs {ob['node_count']['mean']:.1f}"
          f"   tokens/run: {oa['total_tokens']['mean']:.0f} vs {ob['total_tokens']['mean']:.0f}"
          f"   width: {oa['width']['mean']:.2f} vs {ob['width']['mean']:.2f}")

    rows = []
    for task in sorted(A["per_task"]):
        if task in B["per_task"]:
            rows.append(_row(task, A["per_task"][task], B["per_task"][task]))
    headers = ["task",
               f"acc {la}", f"acc {lb}",
               f"nodes {la}", f"nodes {lb}",
               f"width {la}", f"width {lb}",
               f"tok {la}", f"tok {lb}", "Δtok",
               f"#sig/modal {la}", f"#sig/modal {lb}"]
    print()
    print(tabulate(rows, headers=headers, tablefmt="github"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
