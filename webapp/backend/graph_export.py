"""Serialize a reconstructed ACG (networkx DiGraph) into UI-friendly JSON.

This is a pure adapter: it reuses acg.graph for the graph, the metrics, the level
layout, and the behavioral-repeat detection, and reshapes the result into plain dicts.
Nothing here recomputes structure — it only reads what acg.graph already produced.
"""
from __future__ import annotations

from typing import Any

import networkx as nx

from acg import graph as G
from acg import tracing as T


def _node_label(g: nx.DiGraph, n: str) -> str:
    d = g.nodes[n]
    t = d.get("type")
    if t == T.NODE_TYPE_LLM:
        return f"LLM #{d.get('step')}"
    if t == T.NODE_TYPE_TOOL:
        name = d.get("tool_name") or "tool"
        args = d.get("tool_args") or {}
        if name == "read_document" and args.get("doc_id"):
            return f"read {args['doc_id']}"
        if name == "search" and args.get("query"):
            q = str(args["query"])
            return f"search: {q[:40]}"
        if name == "sub_agent" and args.get("question"):
            return "sub_agent"
        return name.replace("_", " ")
    if t == T.NODE_TYPE_AGENT_RUN:
        return "START"
    return str(n)


def graph_to_json(g: nx.DiGraph) -> dict[str, Any]:
    """Return {nodes, edges, levels, metrics, behavioral_repeats} for one run's ACG."""
    if g.number_of_nodes() == 0:
        return {"nodes": [], "edges": [], "levels": {}, "metrics": {}, "behavioral_repeats": []}

    root = G.find_root(g)
    levels = G._levels_from_root(g, root) if root is not None else {}
    repeats = G.detect_behavioral_repeats(g)  # also annotates is_repeat on nodes

    nodes = []
    for n, d in g.nodes(data=True):
        nodes.append({
            "id": n,
            "type": d.get("type"),
            "label": _node_label(g, n),
            "name": d.get("name"),
            "step": d.get("step"),
            "level": levels.get(n, 0),
            "tool_name": d.get("tool_name"),
            "tool_args": d.get("tool_args") or {},
            "input_tokens": d.get("input_tokens", 0),
            "output_tokens": d.get("output_tokens", 0),
            "duration_ns": d.get("duration_ns", 0),
            "start_time_ns": d.get("start_time_ns"),
            "end_time_ns": d.get("end_time_ns"),
            "outcome": d.get("outcome"),
            "answer": d.get("answer"),
            "question": d.get("question"),
            "error": d.get("error"),
            "is_repeat": bool(d.get("is_repeat")),
            "repeat_labels": d.get("repeat_labels") or [],
            "is_nested": "/" in str(n),
        })

    edges = [{"source": u, "target": v} for u, v in g.edges()]

    return {
        "nodes": nodes,
        "edges": edges,
        "levels": {str(k): v for k, v in levels.items()},
        "metrics": metrics_to_json(g),
        "behavioral_repeats": [
            {
                "kind": r.kind,
                "from_node": r.from_node,
                "to_node": r.to_node,
                "tool_name": r.tool_name,
                "detail": r.detail,
            }
            for r in repeats
        ],
    }


def metrics_to_json(g: nx.DiGraph, *, run_id: str = "", task_id: str = "", trace_id: str = "") -> dict[str, Any]:
    m = G.compute_metrics(g, run_id=run_id, task_id=task_id, trace_id=trace_id)
    row = m.to_row()
    # Expose tool_breakdown as a nested dict too (to_row flattens it into tool_* columns).
    row["tool_breakdown"] = dict(m.tool_breakdown)
    return row
