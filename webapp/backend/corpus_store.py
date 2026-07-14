"""Document manager over the instrument's owned corpus (data/corpus.json).

CRUD + reindex. Editing the corpus is the app's stated job, so these are the only writes
the backend makes to the repo's data. Every write keeps a single .bak sibling. "Reindex"
is a round-trip through acg.corpus.Corpus.load — the same load the agent does per run —
so it validates the file and reports the rebuilt index size.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from acg.corpus import Corpus

from . import paths


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, docs: list[dict[str, Any]]) -> None:
    if path.exists():
        path.with_suffix(path.suffix + ".bak").write_text(
            path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(json.dumps(docs, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _which(kind: str) -> Path:
    return paths.DISTRACTORS_PATH if kind == "distractors" else paths.CORPUS_PATH


def list_docs(kind: str = "corpus") -> list[dict[str, Any]]:
    return _read(_which(kind))


def get_doc(doc_id: str, kind: str = "corpus") -> dict[str, Any] | None:
    return next((d for d in _read(_which(kind)) if d.get("id") == doc_id), None)


def _validate(doc: dict[str, Any]) -> dict[str, Any]:
    out = {"id": str(doc.get("id", "")).strip(),
           "title": str(doc.get("title", "")).strip(),
           "text": str(doc.get("text", "")).strip()}
    if not out["id"]:
        raise ValueError("document 'id' is required")
    return out


def create_doc(doc: dict[str, Any], kind: str = "corpus") -> dict[str, Any]:
    path = _which(kind)
    docs = _read(path)
    doc = _validate(doc)
    if any(d.get("id") == doc["id"] for d in docs):
        raise ValueError(f"document id '{doc['id']}' already exists")
    docs.append(doc)
    _write(path, docs)
    return doc


def update_doc(doc_id: str, doc: dict[str, Any], kind: str = "corpus") -> dict[str, Any]:
    path = _which(kind)
    docs = _read(path)
    for i, d in enumerate(docs):
        if d.get("id") == doc_id:
            merged = _validate({**d, **doc, "id": doc_id})
            docs[i] = merged
            _write(path, docs)
            return merged
    raise KeyError(doc_id)


def delete_doc(doc_id: str, kind: str = "corpus") -> None:
    path = _which(kind)
    docs = _read(path)
    new = [d for d in docs if d.get("id") != doc_id]
    if len(new) == len(docs):
        raise KeyError(doc_id)
    _write(path, new)


def reindex() -> dict[str, Any]:
    """Reload the corpus the way the agent does; report the rebuilt index size."""
    corpus = Corpus.load(paths.CORPUS_PATH, paths.DISTRACTORS_PATH)
    total_terms = sum(len(toks) for toks in corpus._index.values())
    return {
        "num_docs": len(corpus._index),
        "num_distractors": len(corpus.distractor_ids),
        "total_indexed_terms": total_terms,
        "ok": True,
    }


def preview_search(query: str, top_k: int = 3, noise: int = 0) -> list[dict[str, Any]]:
    """Run the real retrieval function so the UI can preview what the agent would see."""
    corpus = Corpus.load(paths.CORPUS_PATH, paths.DISTRACTORS_PATH)
    corpus.noise = int(noise)
    return corpus.search(query, top_k=top_k)
