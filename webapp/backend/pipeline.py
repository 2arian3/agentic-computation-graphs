"""Adapter that drives the existing acg pipeline from a dashboard request.

This is the ONLY place the experiment is run. It builds an `acg.config.Config` from the
request, resolves a `Task`, calls `Agent.run()` unchanged, and reconstructs the ACG with
`acg.graph`. No experimental behavior is added or altered here.
"""
from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from acg import graph as G
from acg import provenance as P
from acg import tracing as T
from acg.agent import Agent
from acg.config import Config
from acg.corpus import Corpus
from acg.tasks import Task, load_tasks

from . import paths
from .graph_export import graph_to_json

# Live runs share one global OTel exporter path, so run them one at a time. The single
# GPU model server is a serial resource anyway; replay is unaffected (it runs no agent).
LIVE_RUN_LOCK = threading.Lock()

# Labeled, editable token-price table (USD per 1M tokens). Local models cost $0; the
# estimate exists so the report can show a number when a priced model is pointed at.
PRICE_TABLE = {
    # substring match on model id -> (input_per_1m, output_per_1m)
    "gpt-4o": (2.5, 10.0),
    "gpt-4": (30.0, 60.0),
    "claude": (3.0, 15.0),
    "default": (0.0, 0.0),
}


def build_config(req: dict[str, Any]) -> Config:
    """Map a request's model + parameters onto a fresh Config (all existing fields)."""
    cfg = Config()
    if req.get("model"):
        cfg.model = req["model"]
    if req.get("base_url"):
        cfg.base_url = req["base_url"]
    if req.get("gen_ai_system"):
        cfg.gen_ai_system = req["gen_ai_system"]

    d = cfg.decode
    if req.get("temperature") is not None:
        d.temperature = float(req["temperature"])
    if req.get("top_p") is not None:
        d.top_p = float(req["top_p"])
    if req.get("max_tokens") is not None:
        d.max_tokens = int(req["max_tokens"])
    if "seed" in req:
        d.seed = None if req["seed"] is None else int(req["seed"])

    if req.get("max_steps") is not None:
        cfg.max_steps = int(req["max_steps"])
    if req.get("search_top_k") is not None:
        cfg.search_top_k = int(req["search_top_k"])
    if req.get("max_tool_workers") is not None:
        cfg.max_tool_workers = int(req["max_tool_workers"])
    if req.get("elicit_reasoning") is not None:
        cfg.elicit_reasoning = bool(req["elicit_reasoning"])
    if req.get("enable_sub_agent") is not None:
        cfg.enable_sub_agent = bool(req["enable_sub_agent"])
    if req.get("sub_agent_max_steps") is not None:
        cfg.sub_agent_max_steps = int(req["sub_agent_max_steps"])
    # Extended tool alphabet: Config() already reads ACG_EXTRA_TOOLS from the env; a request
    # may additionally override it (list of tool names).
    if req.get("extra_tools") is not None:
        cfg.extra_tools = tuple(req["extra_tools"])
    return cfg


def resolve_task(req: dict[str, Any]) -> Task:
    """A preset task (graded, from the jsonl files) or a custom, ungraded prompt."""
    task_id = req.get("task_id")
    if task_id and task_id != "CUSTOM":
        for path in (paths.TASKS_PATH, paths.TASKS_BRANCH_PATH):
            if Path(path).exists():
                for t in load_tasks(path):
                    if t.task_id == task_id:
                        return t
    # Custom prompt -> ungraded task (no gold answer).
    question = (req.get("prompt") or req.get("question") or "").strip()
    return Task(task_id="CUSTOM", question=question, answers=[], hops=0, supporting=[])


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> dict[str, Any]:
    key = "default"
    for k in PRICE_TABLE:
        if k != "default" and k in (model or "").lower():
            key = k
            break
    in_price, out_price = PRICE_TABLE[key]
    usd = input_tokens / 1e6 * in_price + output_tokens / 1e6 * out_price
    return {
        "usd": round(usd, 6),
        "price_key": key,
        "input_per_1m": in_price,
        "output_per_1m": out_price,
        "note": "Local model — $0." if key == "default" else f"Estimate using {key} list prices.",
    }


def _report(graph_json: dict, res, cfg: Config, wall_s: float) -> dict[str, Any]:
    m = graph_json.get("metrics", {})
    tb = m.get("tool_breakdown", {}) or {}
    llm_s = float(m.get("llm_time_s", 0) or 0)
    tool_s = float(m.get("tool_time_s", 0) or 0)
    wall = float(m.get("wall_clock_s", wall_s) or wall_s)
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
        "cost": estimate_cost(cfg.model, in_tok, out_tok),
        "memory": None,  # not exposed by the instrument; shown as N/A in the UI
    }


def run_live(run_id: str, req: dict[str, Any], on_event) -> dict[str, Any]:
    """Execute one live run, streaming events via on_event(dict). Returns the final result.

    on_event is called with lifecycle events; individual span events are streamed by the
    global streaming processor (streaming.py) straight from the tracing layer.
    """
    from .streaming import attach_streaming_processor

    cfg = build_config(req)
    task = resolve_task(req)
    trace_file = paths.RUN_TRACES_DIR / f"run_{run_id}.jsonl"
    if trace_file.exists():
        trace_file.unlink()

    with LIVE_RUN_LOCK:
        T.configure_tracing(trace_file)
        attach_streaming_processor()

        # Use the app's configured corpus path (env-overridable → enriched benchmark).
        # Corpus.load honors ACG_RETRIEVAL (overlap|bm25) from the environment.
        corpus = Corpus.load(paths.CORPUS_PATH, paths.DISTRACTORS_PATH)
        if req.get("noise"):
            corpus.noise = int(req["noise"])

        on_event({
            "kind": "run_started",
            "run_id": run_id,
            "task_id": task.task_id,
            "question": task.question,
            "graded": bool(task.answers),
            "config": _public_config(cfg),
        })

        agent = Agent(cfg, corpus)
        t0 = time.time()
        error = None
        try:
            res = agent.run(task, run_id=run_id)
        except Exception as e:  # serving down, bad model id, etc.
            T.flush_tracing()
            error = f"{type(e).__name__}: {e}"
            on_event({"kind": "error", "run_id": run_id, "error": error})
            raise
        finally:
            T.flush_tracing()
        wall_s = time.time() - t0

    # Offline reconstruction (identical to the CLI path).
    graph_json = _reconstruct_run(trace_file, run_id)
    report = _report(graph_json, res, cfg, wall_s)
    prov = P.capture(cfg, experiment=f"webapp_{run_id}", extra={
        "source": "webapp", "task_id": task.task_id, "graded": bool(task.answers),
    })

    result = {
        "kind": "run_finished",
        "run_id": run_id,
        "task_id": task.task_id,
        "question": task.question,
        "answer": res.answer,
        "outcome": ("ungraded" if not task.answers else res.outcome),
        "graded": bool(task.answers),
        "config": _public_config(cfg),
        "graph": graph_json,
        "report": report,
        "provenance": prov,
        "trace_file": str(trace_file.relative_to(paths.REPO_ROOT)),
        "created_at": time.time(),
        "mode": "live",
    }
    on_event(result)
    return result


def _reconstruct_run(trace_file: Path, run_id: str) -> dict[str, Any]:
    runs = G.reconstruct_runs(trace_file)
    if not runs:
        return {"nodes": [], "edges": [], "levels": {}, "metrics": {}, "behavioral_repeats": []}
    run = next((r for r in runs if r.run_id == run_id), runs[0])
    return graph_to_json(run.graph)


def _public_config(cfg: Config) -> dict[str, Any]:
    d = cfg.decode
    return {
        "model": cfg.model,
        "base_url": cfg.base_url,
        "gen_ai_system": cfg.gen_ai_system,
        "temperature": d.temperature,
        "top_p": d.top_p,
        "max_tokens": d.max_tokens,
        "seed": d.seed,
        "max_steps": cfg.max_steps,
        "search_top_k": cfg.search_top_k,
        "max_tool_workers": cfg.max_tool_workers,
        "elicit_reasoning": cfg.elicit_reasoning,
        "enable_sub_agent": cfg.enable_sub_agent,
        "sub_agent_max_steps": cfg.sub_agent_max_steps,
        "extra_tools": list(cfg.extra_tools),
    }
