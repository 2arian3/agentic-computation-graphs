"""Replay an archived trace as if it were executing live.

Reads spans from a traces/*.jsonl file, re-emits them (in finish order, with capped
inter-event delays that mimic the real timing) through the same event schema as a live
run, then reconstructs the ACG. This makes the whole visualization usable with the model
server down and turns the archived traces into a browsable gallery — no agent is run.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional

from acg import graph as G

from . import paths
from .graph_export import graph_to_json
from .streaming import normalize_span


def list_traces() -> list[dict[str, Any]]:
    """Every *.jsonl under traces/ (recursively), with per-file run counts."""
    out = []
    for f in sorted(paths.TRACES_DIR.rglob("*.jsonl")):
        try:
            spans = G.load_spans(f)
        except Exception:
            continue
        by_trace = G.group_by_trace(spans)
        # Peek at the first run's task/outcome for a friendly label.
        tasks = sorted({(s.get("attributes") or {}).get("acg.task_id") for s in spans} - {None})
        out.append({
            "file": str(f.relative_to(paths.REPO_ROOT)),
            "name": f.stem,
            "num_spans": len(spans),
            "num_runs": len(by_trace),
            "tasks": tasks,
            "size_bytes": f.stat().st_size,
        })
    return out


def _resolve(file: str) -> Path:
    p = (paths.REPO_ROOT / file).resolve()
    # Keep replay inside the repo's traces dir.
    if not str(p).startswith(str(paths.TRACES_DIR.resolve())):
        raise ValueError("trace file must be under traces/")
    if not p.exists():
        raise FileNotFoundError(file)
    return p


def trace_runs(file: str) -> list[dict[str, Any]]:
    """Reconstruct every run in a trace file (metrics only) for a chooser list."""
    p = _resolve(file)
    runs = G.reconstruct_runs(p)
    out = []
    for r in runs:
        gj = graph_to_json(r.graph)
        out.append({
            "trace_id": r.trace_id,
            "run_id": r.run_id,
            "task_id": r.task_id,
            "outcome": gj["metrics"].get("outcome"),
            "node_count": gj["metrics"].get("node_count"),
            "total_tokens": gj["metrics"].get("total_tokens"),
        })
    return out


def replay_run(file: str, trace_id: Optional[str], on_event, *, speed: float = 1.0,
               max_delay_s: float = 0.6) -> dict[str, Any]:
    """Stream one trace's spans progressively, then emit the reconstructed run."""
    p = _resolve(file)
    spans = G.load_spans(p)
    by_trace = G.group_by_trace(spans)
    tid = trace_id if trace_id in by_trace else next(iter(by_trace))
    run_spans = by_trace[tid]

    run_id = next(((s.get("attributes") or {}).get("acg.run_id") for s in run_spans
                   if (s.get("attributes") or {}).get("acg.run_id")), tid[:12])
    task_id = next(((s.get("attributes") or {}).get("acg.task_id") for s in run_spans
                    if (s.get("attributes") or {}).get("acg.task_id")), "")
    question = next(((s.get("attributes") or {}).get("acg.question") for s in run_spans
                     if (s.get("attributes") or {}).get("acg.question")), "")

    on_event({
        "kind": "run_started", "run_id": run_id, "task_id": task_id,
        "question": question, "graded": True, "mode": "replay",
    })

    # Emit in finish order (that is the order a live SimpleSpanProcessor would export).
    ordered = sorted(run_spans, key=lambda s: (s.get("end_time_ns") or 0))
    prev_end = None
    for s in ordered:
        end = s.get("end_time_ns")
        if prev_end is not None and end is not None and speed > 0:
            gap = max(0.0, min(max_delay_s, (end - prev_end) / 1e9 / speed))
            if gap:
                time.sleep(gap)
        prev_end = end
        on_event(normalize_span(s))

    # Reconstruct exactly as the offline analysis does.
    g = G.build_graph(run_spans)
    graph_json = graph_to_json(g)
    metrics = graph_json.get("metrics", {})
    result = {
        "kind": "run_finished",
        "run_id": run_id,
        "task_id": task_id,
        "question": question,
        "answer": next((n.get("answer") for n in graph_json["nodes"]
                        if n.get("type") == "agent_run" and n.get("answer")), None),
        "outcome": metrics.get("outcome", "unknown"),
        "graded": True,
        "graph": graph_json,
        "report": _replay_report(metrics),
        "trace_file": file,
        "trace_id": tid,
        "mode": "replay",
        "created_at": time.time(),
    }
    on_event(result)
    return result


def _replay_report(m: dict[str, Any]) -> dict[str, Any]:
    tb = m.get("tool_breakdown", {}) or {}
    llm_s = float(m.get("llm_time_s", 0) or 0)
    tool_s = float(m.get("tool_time_s", 0) or 0)
    wall = float(m.get("wall_clock_s", 0) or 0)
    in_tok = int(m.get("input_tokens", 0) or 0)
    out_tok = int(m.get("output_tokens", 0) or 0)
    return {
        "wall_clock_s": round(wall, 4),
        "stage_times": {
            "llm_s": round(llm_s, 4),
            "tool_s": round(tool_s, 4),
            "overhead_s": round(max(wall - llm_s - tool_s, 0.0), 4),
        },
        "num_llm_calls": m.get("num_llm_calls", 0),
        "num_tool_calls": m.get("num_tool_calls", 0),
        "num_searches": tb.get("search", 0),
        "num_reads": tb.get("read_document", 0),
        "num_sub_agents": tb.get("sub_agent", 0),
        "reasoning_iterations": m.get("num_llm_calls", 0),
        "node_count": m.get("node_count", 0),
        "edge_count": m.get("edge_count", 0),
        "depth": m.get("depth", 0),
        "width": m.get("width", 0),
        "width_executed": m.get("width_executed", 0),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "tool_breakdown": tb,
        "cost": {"usd": 0.0, "note": "Replay of an archived trace."},
        "memory": None,
    }
