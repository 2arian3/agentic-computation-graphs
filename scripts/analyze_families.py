#!/usr/bin/env python3
"""Per-FAMILY structural characterization of the enriched benchmark.

Groups reconstructed ACGs by task family (from a tasks .jsonl carrying a "family" field,
e.g. data/tasks_families.jsonl) and reports, per family: accuracy, ACG size
(nodes/depth), emitted & executed width (and the fraction of runs with width>1),
structural diversity (#distinct shapes + modal-shape fraction), token cost, and which
tool node types appear. First cut of docs/10 RQ1 -- "what graph properties differ across
application types".

  ./.venv/bin/python scripts/analyze_families.py --trace traces/families.jsonl \
      --tasks data/tasks_families.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acg import graph as G
from scripts.analyze import graph_signature

EXT = ["calculator", "compare", "verify_claim", "decompose", "sub_agent"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="traces/families.jsonl")
    ap.add_argument("--tasks", default="data/tasks_families.jsonl")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fam = {}
    for line in Path(args.tasks).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            fam[d["task_id"]] = d.get("family", "?")

    runs = G.reconstruct_runs(args.trace)
    by_fam: dict[str, list] = defaultdict(list)
    for r in runs:
        by_fam[fam.get(r.task_id, "?")].append(r)

    from tabulate import tabulate
    rows, report = [], {}
    for family in sorted(by_fam):
        rs = by_fam[family]
        ms = [r.metrics for r in rs]
        n = len(rs)
        acc = sum(m.outcome == "correct" for m in ms) / n
        nodes = np.array([m.node_count for m in ms], float)
        depth = np.array([m.depth for m in ms], float)
        width = np.array([m.width for m in ms], float)
        wexec = np.array([m.width_executed for m in ms], float)
        tok = np.array([m.total_tokens for m in ms], float)
        sigs = Counter(graph_signature(r.graph) for r in rs)
        modal = sigs.most_common(1)[0][1] / n
        present = {t: round(sum(1 for m in ms if m.tool_breakdown.get(t, 0) > 0) / n, 2) for t in EXT}
        rows.append([
            family, n, f"{acc:.2f}",
            f"{nodes.mean():.1f}±{nodes.std():.1f}",
            f"{depth.mean():.1f}",
            f"{width.mean():.2f}/{int(width.max())}",
            f"{(width > 1).mean():.2f}",
            len(sigs), f"{modal:.2f}",
            f"{tok.mean():.0f}",
            ",".join(t for t in EXT if present[t] > 0) or "-",
        ])
        report[family] = dict(
            n=n, accuracy=round(acc, 3),
            nodes_mean=round(float(nodes.mean()), 2), nodes_std=round(float(nodes.std()), 2),
            depth_mean=round(float(depth.mean()), 2),
            width_mean=round(float(width.mean()), 2), width_max=int(width.max()),
            width_gt1_frac=round(float((width > 1).mean()), 3),
            width_exec_gt1_frac=round(float((wexec > 1).mean()), 3),
            distinct_shapes=len(sigs), modal_shape_frac=round(modal, 3),
            tokens_mean=round(float(tok.mean()), 1), ext_tool_usage=present,
        )
    headers = ["family", "n", "acc", "nodes mean±sd", "depth", "width mean/max",
               "width>1 frac", "#shapes", "modal frac", "tok mean", "ext tools used"]
    print(tabulate(rows, headers=headers, tablefmt="github"))
    out = Path(args.out) if args.out else Path(args.trace).with_name("families_by_family.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
