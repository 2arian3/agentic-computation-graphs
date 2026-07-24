"""Filesystem locations, resolved once, relative to the repo root.

Keeping every path in one place means the rest of the backend never guesses where the
corpus, traces, or history store live — and makes the app relocatable.

The corpus / distractors / tasks paths are env-overridable so the app can be pointed at
the *enriched* benchmark (corpus_large.json + tasks_families.jsonl) without code changes:
    ACG_CORPUS_PATH, ACG_DISTRACTORS_PATH, ACG_TASKS_PATH, ACG_TASKS_BRANCH_PATH
A relative value is resolved against the repo root. Retrieval mode (overlap vs bm25) and
the extended tool alphabet flow through the instrument's own env vars (ACG_RETRIEVAL,
ACG_EXTRA_TOOLS), so they need no path here.
"""
from __future__ import annotations

import os
from pathlib import Path

# webapp/backend/paths.py -> repo root is two parents up from this file's dir.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
TRACES_DIR = REPO_ROOT / "traces"


def _env_path(env: str, default: Path) -> Path:
    v = os.environ.get(env)
    if not v:
        return default
    p = Path(v)
    return p if p.is_absolute() else (REPO_ROOT / p)


CORPUS_PATH = _env_path("ACG_CORPUS_PATH", DATA_DIR / "corpus.json")
DISTRACTORS_PATH = _env_path("ACG_DISTRACTORS_PATH", DATA_DIR / "distractors.json")
TASKS_PATH = _env_path("ACG_TASKS_PATH", DATA_DIR / "tasks.jsonl")
TASKS_BRANCH_PATH = _env_path("ACG_TASKS_BRANCH_PATH", DATA_DIR / "tasks_branch.jsonl")

# The families tasks file carries the "family" label used by the family-rollup panel.
TASKS_FAMILIES_PATH = DATA_DIR / "tasks_families.jsonl"

# App-owned state (never mixed with the instrument's own outputs).
WEBAPP_DIR = REPO_ROOT / "webapp"
APP_DATA_DIR = WEBAPP_DIR / "data"
HISTORY_DIR = APP_DATA_DIR / "history"
RUN_TRACES_DIR = TRACES_DIR / "webapp"          # per-run live traces land here
FRONTEND_DIST = WEBAPP_DIR / "frontend" / "dist"

for _d in (APP_DATA_DIR, HISTORY_DIR, RUN_TRACES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
