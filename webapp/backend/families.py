"""Per-family × per-backbone structural rollup for the dashboard.

Reconstructs runs from one or more archived trace files (e.g. families_7b, families_fp8),
groups them by task family (from data/tasks_families.jsonl), and computes the docs/12
characterization table plus the short-circuit rate. No agent is run — archived traces only.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from acg import graph as G

from . import paths

EXT_TOOLS = ["calculator", "compare", "verify_claim", "decompose", "sub_agent"]
FAMILY_ORDER = ["linear_bridge", "numeric_diff", "counting", "fan_out_superlative",
                "unanswerable", "constraint_satisfaction", "conditional"]


def _family_map() -> dict[str, str]:
    fmap: dict[str, str] = {}
    p = paths.TASKS_FAMILIES_PATH
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                fmap[d["task_id"]] = d.get("family", "?")
    return fmap


def _signature(m) -> tuple:
    tb = tuple(sorted((m.tool_breakdown or {}).items()))
    return (m.num_llm_calls, m.num_tool_calls, m.depth, m.width, tb)


def _resolve(file: str) -> Path:
    p = (paths.REPO_ROOT / file).resolve()
    if not str(p).startswith(str(paths.TRACES_DIR.resolve())):
        raise ValueError("trace file must be under traces/")
    return p


def default_files() -> list[str]:
    files = []
    for name in ("families_7b.jsonl", "families_fp8.jsonl", "families.jsonl"):
        if (paths.TRACES_DIR / name).exists():
            files.append(f"traces/{name}")
    if not files:
        files = [str(f.relative_to(paths.REPO_ROOT))
                 for f in sorted(paths.TRACES_DIR.glob("families*.jsonl"))]
    return files


def _family_stats(rs: list) -> dict[str, Any]:
    ms = [r.metrics for r in rs]
    n = len(rs)

    def arr(f):
        return np.array([getattr(m, f) for m in ms], dtype=float)

    nodes, depth, width = arr("node_count"), arr("depth"), arr("width")
    wexec, tok = arr("width_executed"), arr("total_tokens")
    tc = np.array([m.num_tool_calls for m in ms], dtype=float)
    sigs = Counter(_signature(m) for m in ms)
    present = {t: sum(1 for m in ms if (m.tool_breakdown or {}).get(t, 0) > 0) / n for t in EXT_TOOLS}
    return {
        "n": n,
        "accuracy": round(sum(m.outcome == "correct" for m in ms) / n, 3),
        "short_circuit_frac": round(float((tc <= 1).mean()), 3),
        "nodes_mean": round(float(nodes.mean()), 2),
        "nodes_std": round(float(nodes.std()), 2),
        "depth_mean": round(float(depth.mean()), 2),
        "width_mean": round(float(width.mean()), 2),
        "width_max": int(width.max()),
        "width_gt1_frac": round(float((width > 1).mean()), 3),
        "width_exec_gt1_frac": round(float((wexec > 1).mean()), 3),
        "tokens_mean": round(float(tok.mean()), 1),
        "distinct_shapes": len(sigs),
        "modal_shape_frac": round(sigs.most_common(1)[0][1] / n, 3),
        "ext_tools": [t for t in EXT_TOOLS if present[t] > 0],
    }


def rollup(files: list[str] | None = None) -> dict[str, Any]:
    files = [f for f in (files or default_files()) if f]
    fmap = _family_map()
    by_file: dict[str, dict[str, Any]] = {}
    families_seen: set[str] = set()

    for file in files:
        try:
            p = _resolve(file)
        except ValueError:
            continue
        if not p.exists():
            continue
        runs = G.reconstruct_runs(p)
        groups: dict[str, list] = defaultdict(list)
        for r in runs:
            groups[fmap.get(r.task_id, "?")].append(r)
        stats = {}
        for fam, rs in groups.items():
            if not rs:
                continue
            families_seen.add(fam)
            stats[fam] = _family_stats(rs)
        by_file[Path(file).stem] = stats

    families = [f for f in FAMILY_ORDER if f in families_seen] + \
               sorted(families_seen - set(FAMILY_ORDER))
    return {
        "files": [Path(f).stem for f in files],
        "families": families,
        "by_file": by_file,
        "metric_help": {
            "short_circuit_frac": "fraction of runs finishing with ≤1 tool call (a ≤1-node ACG) — "
                                  "the failure signature; accuracy ≈ 1 − short_circuit_frac on "
                                  "tool-composition tasks",
            "width_gt1_frac": "fraction of runs whose emitted width exceeds 1 (fan-out)",
        },
    }
