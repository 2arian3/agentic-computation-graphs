"""Structural characterization of canonical agentic computation graphs.

Reads data/graphs/<dataset_id>/graphs.jsonl, computes per-graph metrics, and
writes distributions (not just means) to data/graphs/<dataset_id>/metrics.json.

Metric definitions -- stated here because the report quotes them:

  n_nodes / n_edges       counts of the emitted graph.
  depth                   longest path in the DAG, measured in edges.
  max_fan_out             maximum out-degree over nodes.
  fan_out_nodes           how many nodes have out-degree > 1.
  loop_iterations         times the agent went around the act->observe loop,
                          i.e. llm nodes with >=1 incoming `data` edge from a
                          tool node. Loops are NOT unrolled; a revisited step
                          shows up as a later turn_id.
  revisited_turns         turn_id values used by more than one llm node.
  n_branches              distinct branch_id values.
  abandoned_nodes         nodes with committed == False (explored-but-not-taken).
  candidate_parallel_width
                          max, over parents, of the largest set of that parent's
                          children with no dependency path among themselves --
                          the spec's "sibling nodes with no mutual data
                          dependency". This is a *candidate*: structure permits
                          concurrency, nothing says it happened.
  measured_parallel_width
                          max number of nodes whose [start_ts, end_ts] intervals
                          are simultaneously open. Only computable where the
                          source ships real timestamps; null otherwise. This is
                          observed concurrency, not inferred.

Coverage fractions report the share of nodes carrying non-null tokens, latency,
timestamps, and KV/prefix fields. For most public datasets these are ~0, which
is a headline finding rather than a defect.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx

from src.common import GRAPHS, read_graphs_jsonl
from src.schema import (
    IO_FIELDS,
    KV_FIELDS,
    LATENCY_FIELDS,
    NODE_TYPES,
    REASONING_FIELDS,
    TIMESTAMP_FIELDS,
    TOKEN_FIELDS,
)

_PCTS = (10, 25, 50, 75, 90, 99)


def _parse_ts(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _candidate_parallel_width(g: nx.DiGraph, topo: dict[str, int] | None) -> int:
    """Largest sibling set with no dependency path among its members.

    Naively this walks the whole downstream graph per child, which is O(n^2) on
    the long linear sessions in TraceLab (up to 7,610 rounds). Topological order
    increases monotonically along any path, so a path from one sibling to another
    can only pass through nodes indexed below the highest sibling -- pruning
    there is exact and collapses the search to a local neighbourhood. Without a
    topo order (cyclic graph) we fall back to a visit-capped walk.
    """
    best = 0
    for parent in g.nodes:
        kids = list(g.successors(parent))
        if len(kids) <= 1:
            best = max(best, len(kids))
            continue
        kidset = set(kids)
        limit = max((topo[k] for k in kids), default=0) if topo else None
        dependent = set()
        for k in kids:
            stack, seen, visits = [k], set(), 0
            while stack:
                cur = stack.pop()
                for nxt in g.successors(cur):
                    if nxt in seen:
                        continue
                    if topo is not None and topo[nxt] > limit:
                        continue  # exact prune: cannot reach a sibling from here
                    visits += 1
                    if visits > 20000:  # cyclic-graph safety valve
                        stack = []
                        break
                    seen.add(nxt)
                    if nxt in kidset:
                        dependent.add(nxt)
                    stack.append(nxt)
        best = max(best, len(kidset - dependent))
    return best


def _measured_parallel_width(nodes: list[dict]) -> int | None:
    """Max simultaneously-open [start_ts, end_ts] intervals. None if untimed."""
    iv = []
    for n in nodes:
        a, b = _parse_ts(n.get("start_ts")), _parse_ts(n.get("end_ts"))
        if a is not None and b is not None and b >= a:
            iv.append((a, b))
    if not iv:
        return None
    events = sorted([(a, 1) for a, _ in iv] + [(b, -1) for _, b in iv])
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best


def graph_metrics(gd: dict[str, Any]) -> dict[str, Any]:
    nodes = gd.get("nodes", [])
    edges = gd.get("edges", [])
    g = nx.DiGraph()
    for n in nodes:
        g.add_node(n["node_id"])
    for e in edges:
        g.add_edge(e["src"], e["dst"], edge_type=e["edge_type"])

    types = Counter(n["node_type"] for n in nodes)
    out_deg = [d for _, d in g.out_degree()] or [0]

    is_dag = nx.is_directed_acyclic_graph(g)
    depth = nx.dag_longest_path_length(g) if is_dag else None
    topo = {n: i for i, n in enumerate(nx.topological_sort(g))} if is_dag else None

    # loop iterations: llm nodes fed back by an action result. `agent` counts
    # alongside `tool` -- delegating to a sub-agent and consuming its result is
    # a turn of the same act->observe loop.
    tool_ids = {n["node_id"] for n in nodes if n["node_type"] in ("tool", "agent")}
    llm_ids = {n["node_id"] for n in nodes if n["node_type"] == "llm"}
    looped = {e["dst"] for e in edges
              if e["edge_type"] == "data" and e["src"] in tool_ids and e["dst"] in llm_ids}

    turn_counts = Counter(n.get("turn_id") for n in nodes
                          if n["node_type"] == "llm" and n.get("turn_id") is not None)

    def _cov(fields: tuple[str, ...]) -> float:
        if not nodes:
            return 0.0
        return sum(1 for n in nodes if any(n.get(f) is not None for f in fields)) / len(nodes)

    return {
        "graph_id": gd["graph_id"],
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "types": {t: types.get(t, 0) for t in sorted(NODE_TYPES)},
        "edge_types": dict(Counter(e["edge_type"] for e in edges)),
        "is_dag": is_dag,
        "depth": depth,
        "max_fan_out": max(out_deg),
        "fan_out_nodes": sum(1 for d in out_deg if d > 1),
        "loop_iterations": len(looped),
        "revisited_turns": sum(1 for c in turn_counts.values() if c > 1),
        "n_branches": len({n.get("branch_id") for n in nodes if n.get("branch_id") is not None}),
        "abandoned_nodes": sum(1 for n in nodes if n.get("committed") is False),
        "candidate_parallel_width": _candidate_parallel_width(g, topo),
        "measured_parallel_width": _measured_parallel_width(nodes),
        "cov_tokens": _cov(TOKEN_FIELDS),
        "cov_latency": _cov(LATENCY_FIELDS),
        "cov_timestamps": _cov(TIMESTAMP_FIELDS),
        "cov_kv": _cov(KV_FIELDS),
        # Semantic coverage is measured against the nodes that *could* carry it:
        # reasoning against llm nodes, tool i/o against tool/agent nodes.
        # Measuring reasoning over all nodes would just report the llm fraction.
        "cov_reasoning": _cov_where(nodes, REASONING_FIELDS, ("llm",)),
        "cov_tool_io": _cov_where(nodes, IO_FIELDS, ("tool", "agent", "retrieval")),
    }


def _cov_where(nodes: list[dict], fields: tuple[str, ...], types: tuple[str, ...]) -> float:
    """Fraction of nodes *of the given types* carrying any of `fields`."""
    elig = [n for n in nodes if n["node_type"] in types]
    if not elig:
        return 0.0
    return sum(1 for n in elig if any(n.get(f) is not None for f in fields)) / len(elig)


def _dist(vals: list[float]) -> dict[str, Any]:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    d = {
        "n": len(s),
        "min": s[0],
        "max": s[-1],
        "mean": round(st.fmean(s), 3),
        "median": st.median(s),
    }
    for p in _PCTS:
        d[f"p{p}"] = s[min(int(p / 100 * len(s)), len(s) - 1)]
    return d


def aggregate(per_graph: list[dict[str, Any]], dataset: str, domain: str) -> dict[str, Any]:
    if not per_graph:
        return {"dataset": dataset, "n_graphs": 0}
    tot_nodes = sum(m["n_nodes"] for m in per_graph)
    types = Counter()
    edge_types = Counter()
    for m in per_graph:
        types.update(m["types"])
        edge_types.update(m["edge_types"])

    def col(k: str) -> list:
        return [m[k] for m in per_graph]

    # Coverage weighted by node, not by graph: "what share of all nodes carry X".
    def node_weighted(k: str) -> float:
        return sum(m[k] * m["n_nodes"] for m in per_graph) / tot_nodes if tot_nodes else 0.0

    # Semantic coverage is already a per-eligible-node ratio, so weight it by the
    # eligible population rather than by total nodes.
    def weighted_by(k: str, types: tuple[str, ...]) -> float:
        num = den = 0.0
        for m in per_graph:
            e = sum(m["types"].get(t, 0) for t in types)
            num += m[k] * e
            den += e
        return num / den if den else 0.0

    return {
        "dataset": dataset,
        "source_domain": domain,
        "n_graphs": len(per_graph),
        "total_nodes": tot_nodes,
        "total_edges": sum(m["n_edges"] for m in per_graph),
        "node_type_histogram": dict(types),
        "edge_type_histogram": dict(edge_types),
        "non_dag_graphs": sum(1 for m in per_graph if not m["is_dag"]),
        "dist": {
            k: _dist(col(k))
            for k in (
                "n_nodes", "n_edges", "depth", "max_fan_out", "fan_out_nodes",
                "loop_iterations", "revisited_turns", "n_branches",
                "abandoned_nodes", "candidate_parallel_width",
                "measured_parallel_width",
            )
        },
        "coverage_node_weighted": {
            "tokens": round(node_weighted("cov_tokens"), 4),
            "latency": round(node_weighted("cov_latency"), 4),
            "timestamps": round(node_weighted("cov_timestamps"), 4),
            "kv_prefix": round(node_weighted("cov_kv"), 4),
        },
        "coverage_semantic": {
            "reasoning_of_llm_nodes": round(weighted_by("cov_reasoning", ("llm",)), 4),
            "tool_io_of_action_nodes": round(
                weighted_by("cov_tool_io", ("tool", "agent", "retrieval")), 4
            ),
        },
    }


def characterize(dataset: str, force: bool = False) -> dict[str, Any]:
    path = GRAPHS / dataset / "graphs.jsonl"
    if not path.exists():
        raise SystemExit(f"no graphs at {path}")
    out = GRAPHS / dataset / "metrics.json"
    if out.exists() and not force:
        print(f"[{dataset}] metrics exist; use --force to recompute")
        return json.loads(out.read_text())

    per_graph: list[dict[str, Any]] = []
    domain = "unknown"
    for gd in read_graphs_jsonl(path):
        domain = gd.get("source_domain", domain)
        per_graph.append(graph_metrics(gd))

    agg = aggregate(per_graph, dataset, domain)
    out.write_text(json.dumps(agg, indent=2, default=str))
    (GRAPHS / dataset / "per_graph_metrics.jsonl").write_text(
        "".join(json.dumps(m, default=str) + "\n" for m in per_graph)
    )
    print(f"[{dataset}] characterized {len(per_graph)} graphs -> {out}")
    return agg


def main() -> None:
    ap = argparse.ArgumentParser(description="Characterize extracted graphs")
    ap.add_argument("dataset")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    agg = characterize(args.dataset, args.force)
    print(json.dumps(agg, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
