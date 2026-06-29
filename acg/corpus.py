"""The owned corpus + a deterministic retrieval function.

A small, fully-owned mini-wiki (data/corpus.json) over a self-contained *fictional*
world. Fictional facts are a deliberate choice: the model cannot answer from
parametric memory, so it is forced to actually decompose the question and chain
search/read calls -- which is exactly the branching, multi-hop structure we want to
measure (and it avoids the "degenerate trivial graph" failure mode).

Retrieval is plain, deterministic keyword overlap so that the *only* stochastic part
of the system is the model's sampling -- never the tools.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_WORD = re.compile(r"[a-z0-9']+")
_STOP = {
    "the", "a", "an", "of", "in", "on", "is", "are", "was", "were", "to", "and",
    "for", "by", "with", "that", "which", "where", "who", "what", "it", "its",
    "from", "at", "as", "this", "be", "or",
}


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


@dataclass
class Document:
    id: str
    title: str
    text: str


class Corpus:
    def __init__(self, docs: list[Document]):
        self.docs = {d.id: d for d in docs}
        self._index = {d.id: set(_tokens(d.title + " " + d.text)) for d in docs}

    @classmethod
    def load(cls, path: str | Path) -> "Corpus":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([Document(**d) for d in data])

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return up to top_k docs ranked by keyword overlap with the query.

        Ties (and zero-overlap fallbacks) are broken by document id so retrieval is
        fully deterministic.
        """
        q = set(_tokens(query))
        scored = []
        for doc_id, toks in self._index.items():
            overlap = len(q & toks)
            scored.append((overlap, doc_id))
        # higher overlap first, then stable by id
        scored.sort(key=lambda s: (-s[0], s[1]))
        results = []
        for overlap, doc_id in scored[:top_k]:
            if overlap == 0:
                continue
            d = self.docs[doc_id]
            snippet = d.text[:160] + ("..." if len(d.text) > 160 else "")
            results.append({"doc_id": d.id, "title": d.title, "score": overlap, "snippet": snippet})
        return results

    def read(self, doc_id: str) -> dict:
        doc_id = (doc_id or "").strip()
        d = self.docs.get(doc_id)
        if d is None:
            # tolerate the model passing a title instead of an id
            for cand in self.docs.values():
                if cand.title.lower() == doc_id.lower():
                    d = cand
                    break
        if d is None:
            return {"error": f"No document with id '{doc_id}'. Use search() to find valid doc_ids."}
        return {"doc_id": d.id, "title": d.title, "text": d.text}
