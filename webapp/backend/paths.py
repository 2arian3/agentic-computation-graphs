"""Filesystem locations, resolved once, relative to the repo root.

Keeping every path in one place means the rest of the backend never guesses where the
corpus, traces, or history store live — and makes the app relocatable.
"""
from __future__ import annotations

from pathlib import Path

# webapp/backend/paths.py -> repo root is two parents up from this file's dir.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
TRACES_DIR = REPO_ROOT / "traces"

CORPUS_PATH = DATA_DIR / "corpus.json"
DISTRACTORS_PATH = DATA_DIR / "distractors.json"
TASKS_PATH = DATA_DIR / "tasks.jsonl"
TASKS_BRANCH_PATH = DATA_DIR / "tasks_branch.jsonl"

# App-owned state (never mixed with the instrument's own outputs).
WEBAPP_DIR = REPO_ROOT / "webapp"
APP_DATA_DIR = WEBAPP_DIR / "data"
HISTORY_DIR = APP_DATA_DIR / "history"
RUN_TRACES_DIR = TRACES_DIR / "webapp"          # per-run live traces land here
FRONTEND_DIST = WEBAPP_DIR / "frontend" / "dist"

for _d in (APP_DATA_DIR, HISTORY_DIR, RUN_TRACES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
