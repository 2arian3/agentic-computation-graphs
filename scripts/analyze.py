#!/usr/bin/env python3
"""Aggregate reconstructed ACGs into the distributions + structural-variance numbers
that are the actual contribution (Month-2 milestone).

Importable functions are used by run_experiment.py; it can also be run standalone on
any trace file:

  ./.venv/bin/python scripts/analyze.py --trace traces/experiment.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acg import graph as G

_METRIC_FIELDS = [
    "node_count", "num_llm_calls", "num_tool_calls",
    "edge_count", "depth", "width", "width_executed",
    "total_tokens", "input_tokens", "output_tokens", "wall_clock_s",
]


def _dist(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 3),
        "std": round(float(a.std(ddof=0)), 3),
        "min": round(float(a.min()), 3),
        "median": round(float(np.median(a)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "max": round(float(a.max()), 3),
    }


def _wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score interval for a binomial proportion k/n.

    Used for the stable-core (modal-signature) fraction and for accuracy: with small
    rep counts a bare fraction is misleading, so we report an honest interval."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / denom
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def graph_signature(g) -> tuple:
    """A compact structural fingerprint of a run's ACG. Two runs with the same
    signature have the same shape up to node identity."""
    m = G.compute_metrics(g)
    tb = tuple(sorted(m.tool_breakdown.items()))
    return (m.num_llm_calls, m.num_tool_calls, m.depth, m.width, tb)


def structural_variance(
    runs: list, *, ged_sample_pairs: int = 200, ged_time_budget_s: float = 25.0,
) -> dict:
    """Quantify how much the structure varies across runs of one task.

    Reports, for the same task+model across many runs:
      * distinct_signatures + modal_signature_fraction (stable-core proxy) WITH a 95%
        Wilson CI, so the stable-core claim is honest at small rep counts;
      * a real pairwise graph-edit-distance DISTRIBUTION (labelled by node type + tool),
        both raw and normalized by graph size so tasks of different size are comparable.
    """
    n = len(runs)
    sigs = [graph_signature(r.graph) for r in runs]
    counts = Counter(sigs)
    modal_sig, modal_n = counts.most_common(1)[0]
    out = {
        "n": n,
        "distinct_signatures": len(counts),
        "modal_signature_fraction": round(modal_n / n, 3),           # proxy for a stable core
        "modal_signature_fraction_ci95": _wilson_ci(modal_n, n),     # honest interval
        "modal_signature": {
            "num_llm_calls": modal_sig[0], "num_tool_calls": modal_sig[1],
            "depth": modal_sig[2], "width": modal_sig[3], "tools": dict(modal_sig[4]),
        },
    }

    # Pairwise graph edit distance (nodes labelled by type + tool). Exact GED is NP-hard
    # and times out on the larger graphs, so we use the first upper bound yielded by
    # networkx's optimizer -- fast (~10 ms/pair), deterministic, and never None. Sample up
    # to ged_sample_pairs pairs, bounded by a wall-clock budget so analysis never stalls.
    import itertools
    import time
    import networkx as nx

    def nmatch(a, b):
        return a.get("type") == b.get("type") and a.get("tool_name") == b.get("tool_name")

    def ged_upper_bound(g1, g2):
        try:
            return float(next(nx.optimize_graph_edit_distance(g1, g2, node_match=nmatch)))
        except (StopIteration, Exception):
            return None

    pairs = list(itertools.combinations(range(n), 2))
    if pairs:
        rng = np.random.default_rng(0)
        if len(pairs) > ged_sample_pairs:
            idx = rng.choice(len(pairs), size=ged_sample_pairs, replace=False)
            pairs = [pairs[i] for i in idx]
        geds, geds_norm = [], []
        t_start = time.time()
        for i, j in pairs:
            if time.time() - t_start > ged_time_budget_s:
                break
            d = ged_upper_bound(runs[i].graph, runs[j].graph)
            if d is not None:
                geds.append(d)
                size_avg = (runs[i].graph.number_of_nodes()
                            + runs[j].graph.number_of_nodes()) / 2.0
                geds_norm.append(d / size_avg if size_avg else 0.0)
        if geds:
            out["graph_edit_distance"] = {
                "method": "optimize_first_upper_bound",  # fast approximate GED (upper bound)
                "pairs_computed": len(geds),
                "pairs_possible": n * (n - 1) // 2,
                "mean": round(float(np.mean(geds)), 3),
                "std": round(float(np.std(geds)), 3),
                "median": round(float(np.median(geds)), 3),
                "p95": round(float(np.percentile(geds, 95)), 3),
                "max": round(float(np.max(geds)), 3),
                "normalized_mean": round(float(np.mean(geds_norm)), 3),   # GED / avg graph size
                "normalized_median": round(float(np.median(geds_norm)), 3),
            }
    return out


def summarize(runs: list) -> dict:
    by_task: dict[str, list] = defaultdict(list)
    for r in runs:
        by_task[r.task_id].append(r)

    summary = {"overall": {}, "per_task": {}}
    all_metrics = [r.metrics for r in runs]
    n_correct = sum(1 for m in all_metrics if m.outcome == "correct")
    summary["overall"] = {
        "num_runs": len(runs),
        "num_tasks": len(by_task),
        "accuracy": round(n_correct / max(len(runs), 1), 3),
        "accuracy_ci95": _wilson_ci(n_correct, len(runs)),
    }
    for field in _METRIC_FIELDS:
        summary["overall"][field] = _dist([getattr(m, field) for m in all_metrics])

    for task_id, task_runs in sorted(by_task.items()):
        ms = [r.metrics for r in task_runs]
        k = sum(1 for m in ms if m.outcome == "correct")
        entry = {
            "num_runs": len(task_runs),
            "accuracy": round(k / max(len(ms), 1), 3),
            "accuracy_ci95": _wilson_ci(k, len(ms)),
            "distributions": {f: _dist([getattr(m, f) for m in ms]) for f in _METRIC_FIELDS},
            "structural_variance": structural_variance(task_runs),
        }
        summary["per_task"][task_id] = entry
    return summary


def write_metrics_csv(runs: list, path: str | Path) -> None:
    rows = [r.metrics.to_row() for r in runs]
    keys = sorted({k for row in rows for k in row})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def make_plots(runs: list, outdir: str | Path, prefix: str = "") -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    by_task: dict[str, list] = defaultdict(list)
    for r in runs:
        by_task[r.task_id].append(r)
    tasks = sorted(by_task)
    saved = []

    # (1) Distribution of total tokens per task (cost distribution; tail matters).
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(tasks)), 4.5))
    data = [[r.metrics.total_tokens for r in by_task[t]] for t in tasks]
    ax.boxplot(data, tick_labels=tasks, showmeans=True)
    ax.set_title("ACG cost distribution per task — total tokens")
    ax.set_ylabel("total tokens (input+output)")
    ax.set_xlabel("task")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    p = outdir / f"{prefix}dist_total_tokens.png"
    plt.savefig(p, dpi=130); plt.close(); saved.append(str(p))

    # (2) Distribution of node counts per task (graph SIZE distribution).
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(tasks)), 4.5))
    data = [[r.metrics.node_count for r in by_task[t]] for t in tasks]
    ax.boxplot(data, tick_labels=tasks, showmeans=True)
    ax.set_title("ACG size distribution per task — node count")
    ax.set_ylabel("node count (LLM + tool)")
    ax.set_xlabel("task")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    p = outdir / f"{prefix}dist_node_count.png"
    plt.savefig(p, dpi=130); plt.close(); saved.append(str(p))
    return saved


def print_table(summary: dict) -> None:
    from tabulate import tabulate
    rows = []
    for task_id, e in summary["per_task"].items():
        d = e["distributions"]
        sv = e["structural_variance"]
        tok = d["total_tokens"]
        mf = sv["modal_signature_fraction"]
        lo, hi = sv["modal_signature_fraction_ci95"]
        ged = sv.get("graph_edit_distance", {})
        ged_cell = (f'{ged.get("mean", 0):.1f} ({ged.get("normalized_mean", 0):.2f})'
                    if ged else "-")
        rows.append([
            task_id, e["num_runs"], f'{e["accuracy"]:.2f}',
            f'{d["node_count"]["mean"]:.1f}±{d["node_count"]["std"]:.1f}',
            f'{d["node_count"]["median"]:.0f}/{d["node_count"]["p95"]:.0f}/{d["node_count"]["max"]:.0f}',
            f'{d["depth"]["mean"]:.1f}', f'{d["width"]["mean"]:.2f}',
            f'{tok["mean"]:.0f}', f'{tok["p95"]:.0f}/{tok["p99"]:.0f}',
            sv["distinct_signatures"],
            f'{mf:.2f} [{lo:.2f},{hi:.2f}]',
            ged_cell,
        ])
    headers = ["task", "n", "acc", "nodes mean±sd", "nodes med/p95/max",
               "depth", "width", "tok mean", "tok p95/p99", "#sigs",
               "modal frac [95% CI]", "GED mean (norm)"]
    print(tabulate(rows, headers=headers, tablefmt="github"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="traces/experiment.jsonl")
    ap.add_argument("--outdir", default="traces")
    args = ap.parse_args()

    runs = G.reconstruct_runs(args.trace)
    print(f"reconstructed {len(runs)} runs from {args.trace}\n")
    summary = summarize(runs)
    print_table(summary)

    outdir = Path(args.outdir)
    write_metrics_csv(runs, outdir / "metrics.csv")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    figs = make_plots(runs, outdir / "figures")
    print(f"\nwrote {outdir/'metrics.csv'}, {outdir/'summary.json'}")
    print("figures:", *figs, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
