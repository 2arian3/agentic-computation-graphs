#!/usr/bin/env python3
"""Closed-book baseline: answer WITHOUT tools/retrieval, to measure memorization.

The retrieval-necessity gap = open_book_acc - closed_book_acc quantifies how much the
agentic graph actually contributes vs the model's parametric memory. On the owned
fictional corpus this should be ~0 closed-book (the facts cannot be memorized) -- which
is exactly why the fictional corpus is a valid controlled substrate. On a real dataset a
small gap warns that the apparent result is about memory, not structure, so this baseline
must accompany every external-dataset run.

  ./.venv/bin/python scripts/closed_book.py --tasks data/tasks_families.jsonl --limit 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from acg.config import load_config
from acg.tasks import load_tasks, check_answer

SYS = ("Answer the question with a short, concise answer using ONLY your own knowledge. "
       "If you do not know the answer, reply exactly 'insufficient information'.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/tasks_families.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    tasks = load_tasks(args.tasks)
    if args.limit:
        tasks = tasks[: args.limit]

    correct = 0
    for t in tasks:
        r = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "system", "content": SYS}, {"role": "user", "content": t.question}],
            temperature=0.0, max_tokens=64, seed=cfg.decode.seed,
        )
        ans = (r.choices[0].message.content or "").strip()
        ok = check_answer(ans, t.answers)
        correct += int(ok)
        print(f"[{'OK ' if ok else '   '}] {t.task_id}: {ans[:60]!r}  gold={t.answers}")

    n = len(tasks) or 1
    print(f"\nclosed-book accuracy: {correct}/{len(tasks)} = {correct/n:.2f}  (model={cfg.model})")
    print("open-book accuracy MINUS this = retrieval-necessity gap; ~0 closed-book => the "
          "corpus/graph is doing the work (good for a controlled study).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
