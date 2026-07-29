"""Shared IO helpers for extractors: JSONL writing, checksums, run manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import orjson

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "data" / "raw"
GRAPHS = REPO / "data" / "graphs"
REPORTS = REPO / "reports"


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_graphs_jsonl(path: Path, graphs: Iterable[dict[str, Any]]) -> int:
    """Write one graph per line. Returns the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    tmp = path.with_suffix(path.suffix + ".partial")
    with open(tmp, "wb") as f:
        for g in graphs:
            f.write(orjson.dumps(g))
            f.write(b"\n")
            n += 1
    os.replace(tmp, path)  # atomic: a crashed run never leaves a half file
    return n


def read_graphs_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                yield orjson.loads(line)


def write_manifest(dataset_id: str, manifest: dict[str, Any]) -> None:
    """Record download URLs, sizes, checksums, record counts -- audit trail."""
    p = RAW / dataset_id / "_manifest.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))


def already_done(path: Path, force: bool) -> bool:
    """Idempotence guard: skip a step whose output exists unless --force."""
    return path.exists() and path.stat().st_size > 0 and not force
