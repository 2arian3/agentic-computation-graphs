"""Load external multi-hop QA datasets into (tasks, corpus) for external validity.

Real datasets (HotpotQA, 2WikiMultiHopQA, MuSiQue, FanOutQA) reintroduce the
memorization confound that the owned fictional corpus was built to avoid. They are
therefore for EXTERNAL VALIDITY ONLY, and must always be paired with the closed-book
baseline (scripts/closed_book.py):

    retrieval-necessity gap = open_book_accuracy - closed_book_accuracy

If that gap is ~0 the model is answering from parametric memory and the *graph is not
doing the work* -- the result is about memorization, not structure. Report the gap.

Canonical intermediate schema (one JSON object per line), which matches the HotpotQA /
2Wiki "context" field:

    {"id": "...", "question": "...", "answers": ["..."],
     "context": [[title, [sentence, ...]], ...],
     "supporting": [title, ...]}          # optional gold supporting titles

Each context paragraph becomes a Document, so the SAME search / read_document / EXT
tools operate unchanged; only the corpus contents differ.
"""
from __future__ import annotations

import json
from pathlib import Path

from .corpus import Corpus, Document
from .tasks import Task


def _para_id(i: int) -> str:
    return f"E{i:04d}"


def load_context_qa(path: str | Path, scoring: str = "bm25") -> tuple[list[Task], Corpus]:
    """Load a JSONL file in the canonical context-QA schema into (tasks, corpus).

    All paragraphs across all records are pooled into ONE corpus (the fullwiki-style
    setting) so that retrieval is non-trivial and distractor paragraphs are present.
    Returns (tasks, corpus). `supporting` on each Task holds the doc_ids of the gold
    paragraphs when the dataset provides gold titles.
    """
    records = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]

    docs: list[Document] = []
    title_to_id: dict[str, str] = {}
    for rec in records:
        for title, sents in rec.get("context", []):
            if title in title_to_id:
                continue
            did = _para_id(len(docs) + 1)
            title_to_id[title] = did
            docs.append(Document(id=did, title=title, text=" ".join(sents)))

    corpus = Corpus(docs, scoring=scoring)

    tasks: list[Task] = []
    for i, rec in enumerate(records):
        support = [title_to_id[t] for t in rec.get("supporting", []) if t in title_to_id]
        tasks.append(Task(
            task_id=rec.get("id", f"X{i:04d}"),
            question=rec["question"],
            answers=rec.get("answers", []),
            hops=rec.get("hops", 0),
            supporting=support,
        ))
    return tasks, corpus


# --- normalizers from raw dataset formats into the canonical schema -----------
def hotpot_to_context(rec: dict) -> dict:
    """HotpotQA / 2WikiMultiHopQA record -> canonical context-QA record.

    Both use `context: [[title, [sentences]], ...]`, an `answer` string, and
    `supporting_facts: [[title, sent_idx], ...]`.
    """
    ans = rec.get("answer")
    supporting = sorted({t for t, _ in rec.get("supporting_facts", [])})
    return {
        "id": rec.get("_id") or rec.get("id"),
        "question": rec["question"],
        "answers": [ans] if isinstance(ans, str) else (ans or []),
        "context": rec.get("context", []),
        "supporting": supporting,
    }


def musique_to_context(rec: dict) -> dict:
    """MuSiQue record -> canonical context-QA record.

    MuSiQue uses `paragraphs: [{title, paragraph_text, is_supporting}, ...]` and
    `answer` (+ `answer_aliases`). Sentences are approximated by the whole paragraph.
    """
    paras = rec.get("paragraphs", [])
    context = [[p.get("title", f"p{j}"), [p.get("paragraph_text", "")]] for j, p in enumerate(paras)]
    supporting = [p.get("title", f"p{j}") for j, p in enumerate(paras) if p.get("is_supporting")]
    answers = [rec.get("answer")] + list(rec.get("answer_aliases", []))
    return {
        "id": rec.get("id"),
        "question": rec["question"],
        "answers": [a for a in answers if a],
        "context": context,
        "supporting": supporting,
    }
