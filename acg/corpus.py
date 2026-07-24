"""The owned corpus + a deterministic retrieval function.

A fully-owned mini-wiki over a self-contained *fictional* world. Fictional facts are a
deliberate choice: the model cannot answer from parametric memory, so it is forced to
actually decompose the question and chain search/read calls -- which is exactly the
branching, multi-hop structure we want to measure (and it avoids the "degenerate trivial
graph" failure mode).

Retrieval is deterministic so that the *only* stochastic part of the system is the
model's sampling -- never the tools. Two scoring modes:

  * "overlap" (default) -- keyword-overlap count; ideal for the tiny 16-doc canonical
    corpus, and keeps prior results byte-stable.
  * "bm25"              -- Okapi BM25; needed once the corpus is large (hundreds/thousands
    of docs) so ranking stays meaningful. Select with ACG_RETRIEVAL=bm25 (no dependency).
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_WORD = re.compile(r"[a-z0-9']+")
_STOP = {
    "the", "a", "an", "of", "in", "on", "is", "are", "was", "were", "to", "and",
    "for", "by", "with", "that", "which", "where", "who", "what", "it", "its",
    "from", "at", "as", "this", "be", "or",
}

_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


@dataclass
class Document:
    id: str
    title: str
    text: str


class _Pool:
    """A rankable pool of documents (real docs or distractors), scored by 'overlap' or 'bm25'.

    `.rank(query_tokens)` returns [(score, doc_id), ...] with score > 0, sorted by score
    desc then doc_id asc (deterministic tie-break)."""

    def __init__(self, docs: list[Document], scoring: str):
        self.scoring = scoring
        self._tokset = {d.id: set(_tokens(d.title + " " + d.text)) for d in docs}
        if scoring == "bm25":
            self._tf = {d.id: Counter(_tokens(d.title + " " + d.text)) for d in docs}
            self._len = {i: sum(c.values()) for i, c in self._tf.items()}
            self._n = max(len(docs), 1)
            self._avgdl = (sum(self._len.values()) / self._n) if self._n else 0.0
            df: Counter = Counter()
            for toks in self._tokset.values():
                df.update(toks)
            # BM25+ idf (always positive): ln(1 + (N - df + 0.5)/(df + 0.5))
            self._idf = {t: math.log(1 + (self._n - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def rank(self, q: set) -> list[tuple[float, str]]:
        if self.scoring == "bm25":
            scored = []
            for doc_id, tf in self._tf.items():
                dl = self._len[doc_id]
                denom_norm = _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / (self._avgdl or 1.0))
                s = 0.0
                for t in q:
                    f = tf.get(t, 0)
                    if f:
                        s += self._idf.get(t, 0.0) * (f * (_BM25_K1 + 1)) / (f + denom_norm)
                if s > 0:
                    scored.append((s, doc_id))
        else:  # overlap
            scored = [(float(len(q & toks)), doc_id) for doc_id, toks in self._tokset.items()]
            scored = [(s, i) for s, i in scored if s > 0]
        scored.sort(key=lambda p: (-p[0], p[1]))
        return scored


class Corpus:
    def __init__(self, docs: list[Document], distractors: list[Document] | None = None,
                 scoring: str = "overlap"):
        self.scoring = scoring
        self.docs = {d.id: d for d in docs}
        self._real = _Pool(docs, scoring)
        # Distractor pool (RQ-N1). Distractors are READABLE (added to self.docs so the model
        # can open them) but only enter SEARCH results when `noise` > 0.
        distractors = distractors or []
        self.distractor_ids = {d.id for d in distractors}
        for d in distractors:
            self.docs[d.id] = d
        self._dist = _Pool(distractors, scoring) if distractors else None
        # noise = number of distractors guaranteed into each search result list
        # (capped to keep >=1 real slot). Set by the experiment; 0 = clean baseline.
        self.noise = 0

    @classmethod
    def load(cls, path: str | Path, distractors_path: str | Path | None = None,
             scoring: str | None = None) -> "Corpus":
        scoring = scoring or os.environ.get("ACG_RETRIEVAL", "overlap")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        dist = []
        if distractors_path and Path(distractors_path).exists():
            dist = [Document(**d) for d in json.loads(Path(distractors_path).read_text(encoding="utf-8"))]
        return cls([Document(**d) for d in data], dist, scoring=scoring)

    def _result(self, doc_id: str, score: float) -> dict:
        d = self.docs[doc_id]
        snippet = d.text[:160] + ("..." if len(d.text) > 160 else "")
        # keep integer scores tidy for the overlap mode
        s = int(score) if self.scoring != "bm25" else round(score, 3)
        return {"doc_id": d.id, "title": d.title, "score": s, "snippet": snippet}

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Return up to top_k docs ranked by the configured scoring.

        Deterministic (ties broken by id). When self.noise > 0, that many distractor
        documents are injected into the result list, displacing lower-ranked real docs
        (RQ-N1 retrieval-noise knob) — but at least one real slot is kept so tasks stay
        solvable by a careful agent.
        """
        q = set(_tokens(query))
        real = self._real.rank(q)
        if self.noise <= 0 or not self._dist:
            return [self._result(i, s) for s, i in real[:top_k]]

        n_d = min(self.noise, max(top_k - 1, 0))
        dist = self._dist.rank(q)[:n_d]
        chosen = dist + real[: max(top_k - len(dist), 0)]
        chosen.sort(key=lambda p: (-p[0], p[1]))          # present as one ranked list
        return [self._result(i, s) for s, i in chosen[:top_k]]

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
