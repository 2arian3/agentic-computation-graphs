"""Experiment history — a small JSON store under webapp/data/history/.

One file per run (the full run_finished payload) plus a compact index for listing.
"Rerun" is just reading back an entry's config + prompt and re-submitting it.
"""
from __future__ import annotations

import json
import time
from typing import Any

from . import paths


def _index_path():
    return paths.HISTORY_DIR / "index.json"


def _load_index() -> list[dict[str, Any]]:
    p = _index_path()
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_index(idx: list[dict[str, Any]]) -> None:
    _index_path().write_text(json.dumps(idx, indent=2), encoding="utf-8")


def save(result: dict[str, Any]) -> dict[str, Any]:
    run_id = result.get("run_id") or f"run{int(time.time())}"
    entry_id = f"{int(time.time()*1000)}_{run_id}"
    (paths.HISTORY_DIR / f"{entry_id}.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8")

    report = result.get("report", {}) or {}
    summary = {
        "id": entry_id,
        "run_id": run_id,
        "timestamp": result.get("created_at") or time.time(),
        "mode": result.get("mode", "live"),
        "task_id": result.get("task_id"),
        "question": result.get("question"),
        "model": (result.get("config", {}) or {}).get("model"),
        "outcome": result.get("outcome"),
        "answer_preview": (result.get("answer") or "")[:120],
        "node_count": report.get("node_count"),
        "total_tokens": report.get("total_tokens"),
        "wall_clock_s": report.get("wall_clock_s"),
        "config": result.get("config", {}),
    }
    idx = _load_index()
    idx.insert(0, summary)
    _save_index(idx)
    return summary


def list_entries() -> list[dict[str, Any]]:
    return _load_index()


def get(entry_id: str) -> dict[str, Any] | None:
    p = paths.HISTORY_DIR / f"{entry_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def delete(entry_id: str) -> bool:
    p = paths.HISTORY_DIR / f"{entry_id}.json"
    existed = p.exists()
    if existed:
        p.unlink()
    idx = [e for e in _load_index() if e.get("id") != entry_id]
    _save_index(idx)
    return existed
