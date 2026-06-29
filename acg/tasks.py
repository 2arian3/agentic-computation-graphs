"""Task loading + answer checking for the multi-hop QA benchmark."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Task:
    task_id: str
    question: str
    answers: list[str]      # acceptable gold answers / aliases
    hops: int = 0
    supporting: list[str] = None  # supporting doc ids (for reference / analysis)

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            task_id=d["task_id"],
            question=d["question"],
            answers=d.get("answers", []),
            hops=d.get("hops", 0),
            supporting=d.get("supporting", []),
        )


def load_tasks(path: str | Path) -> list[Task]:
    tasks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            tasks.append(Task.from_dict(json.loads(line)))
    return tasks


_NORM = re.compile(r"[^a-z0-9 ]+")
_ARTICLES = re.compile(r"\b(the|a|an)\b")


def _normalize(s: str) -> str:
    s = (s or "").lower()
    s = _NORM.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def check_answer(predicted: str | None, gold_answers: list[str]) -> bool:
    """Graded substring match after normalization (drops articles/punctuation/case).

    A prediction counts as correct if any gold alias appears as a token-substring of
    the prediction (so 'The currency is the drell.' matches gold 'drell')."""
    if not predicted:
        return False
    p = _normalize(predicted)
    for gold in gold_answers:
        g = _normalize(gold)
        if g and (g in p or p in g):
            return True
    return False
