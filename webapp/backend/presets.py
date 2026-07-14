"""Model + prompt presets and server health.

Model list = whatever the live server advertises at /v1/models (when up) merged with a
few known presets from the repo's docs/env. Prompt presets = the graded tasks in
data/tasks.jsonl and data/tasks_branch.jsonl (loaded via acg.tasks).
"""
from __future__ import annotations

import json
from typing import Any
from urllib.request import urlopen

from acg.config import Config
from acg.tasks import load_tasks

from . import paths

# Known models this repo has been run with (see docs/ + .env.example). `served` is the
# --served-model-name the OpenAI-compatible endpoint expects; `parser` is informational.
MODEL_PRESETS = [
    {"served": "qwen2.5-7b-instruct", "label": "Qwen2.5-7B-Instruct (BF16)", "system": "qwen", "parser": "hermes"},
    {"served": "qwen2.5-14b-instruct-awq", "label": "Qwen2.5-14B-Instruct (AWQ)", "system": "qwen", "parser": "hermes"},
    {"served": "qwen2.5-14b-instruct-fp8", "label": "Qwen2.5-14B-Instruct (FP8)", "system": "qwen", "parser": "hermes"},
    {"served": "llama3.1-8b-instruct", "label": "Llama-3.1-8B-Instruct", "system": "llama", "parser": "llama3_json"},
]


def server_models(base_url: str) -> list[str]:
    try:
        with urlopen(base_url.rstrip("/") + "/models", timeout=3) as r:
            return [m.get("id") for m in json.load(r).get("data", [])]
    except Exception:
        return []


def health(base_url: str | None = None) -> dict[str, Any]:
    cfg = Config()
    bu = base_url or cfg.base_url
    live = server_models(bu)
    return {
        "base_url": bu,
        "server_up": bool(live),
        "served_models": live,
        "default_model": cfg.model,
    }


def models(base_url: str | None = None) -> dict[str, Any]:
    cfg = Config()
    bu = base_url or cfg.base_url
    live = server_models(bu)
    presets = {p["served"]: dict(p, source="preset") for p in MODEL_PRESETS}
    for mid in live:
        presets.setdefault(mid, {"served": mid, "label": mid, "system": cfg.gen_ai_system, "parser": None})
        presets[mid]["source"] = "live"
        presets[mid]["available"] = True
    for mid, p in presets.items():
        p.setdefault("available", mid in live)
    return {
        "base_url": bu,
        "server_up": bool(live),
        "default_model": cfg.model,
        "models": list(presets.values()),
    }


def prompt_presets() -> list[dict[str, Any]]:
    out = []
    for path, group in ((paths.TASKS_PATH, "multi-hop"), (paths.TASKS_BRANCH_PATH, "branching")):
        if not path.exists():
            continue
        for t in load_tasks(path):
            out.append({
                "task_id": t.task_id,
                "question": t.question,
                "answers": t.answers,
                "hops": t.hops,
                "supporting": t.supporting or [],
                "group": group,
                "branch": group == "branching",
            })
    return out


def defaults() -> dict[str, Any]:
    """The canonical Config defaults, for pre-filling the parameter form."""
    cfg = Config()
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
    }
