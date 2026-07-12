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
    def __init__(self, docs: list[Document], distractors: list[Document] | None = None):
        self.docs = {d.id: d for d in docs}
        self._index = {d.id: set(_tokens(d.title + " " + d.text)) for d in docs}
        # Distractor pool (RQ-N1). Distractors are READABLE (added to self.docs so the
        # model can open them) but only enter SEARCH results when `noise` > 0.
        distractors = distractors or []
        self.distractor_ids = {d.id for d in distractors}
        for d in distractors:
            self.docs[d.id] = d
        self._dindex = {d.id: set(_tokens(d.title + " " + d.text)) for d in distractors}
        # noise = number of distractors guaranteed into each search result list
        # (capped to keep >=1 real slot). Set by the experiment; 0 = clean baseline.
        self.noise = 0

    @classmethod
    def load(cls, path: str | Path, distractors_path: str | Path | None = None) -> "Corpus":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        dist = []
        if distractors_path and Path(distractors_path).exists():
            dist = [Document(**d) for d in json.loads(Path(distractors_path).read_text(encoding="utf-8"))]
        return cls([Document(**d) for d in data], dist)

    def _rank(self, index: dict, q: set) -> list[tuple[int, str]]:
        scored = [(len(q & toks), doc_id) for doc_id, toks in index.items()]
        scored.sort(key=lambda s: (-s[0], s[1]))          # overlap desc, then stable by id
        return [(o, i) for o, i in scored if o > 0]

    def _result(self, doc_id: int, overlap: int) -> dict:
        d = self.docs[doc_id]
        snippet = d.text[:160] + ("..." if len(d.text) > 160 else "")
        return {"doc_id": d.id, "title": d.title, "score": overlap, "snippet": snippet}

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return up to top_k docs ranked by keyword overlap with the query.

        Deterministic (ties broken by id). When self.noise > 0, that many distractor
        documents are injected into the result list, displacing lower-ranked real docs
        (RQ-N1 retrieval-noise knob) — but at least one real slot is kept so tasks stay
        solvable by a careful agent.
        """
        q = set(_tokens(query))
        real = self._rank(self._index, q)
        if self.noise <= 0 or not self._dindex:
            return [self._result(i, o) for o, i in real[:top_k]]

        n_d = min(self.noise, max(top_k - 1, 0))
        dist = self._rank(self._dindex, q)[:n_d]
        chosen = dist + real[: max(top_k - len(dist), 0)]
        chosen.sort(key=lambda s: (-s[0], s[1]))          # present as one ranked list
        return [self._result(i, o) for o, i in chosen[:top_k]]

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
