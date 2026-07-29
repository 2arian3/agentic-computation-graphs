"""Render a representative gallery of ACGs for one dataset.

Selection runs off the precomputed per_graph_metrics.jsonl (cheap), then the
chosen graphs are pulled in a single pass over graphs.jsonl (which is large).
Big sessions are windowed to their first rounds so the figure stays legible --
the header still reports whole-session totals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import GRAPHS, REPO  # noqa: E402
from src.visualize import render  # noqa: E402

WINDOW_ABOVE = 90  # nodes; larger sessions get a first-rounds window
WINDOW = "0:8"


def pick(dataset: str) -> dict[str, str]:
    mp = GRAPHS / dataset / "per_graph_metrics.jsonl"
    if not mp.exists():
        raise SystemExit(f"missing {mp} -- run src.characterize first")
    rows = [json.loads(l) for l in mp.read_text().splitlines() if l.strip()]
    if not rows:
        raise SystemExit("no metrics rows")

    med = sorted(r["n_nodes"] for r in rows)[len(rows) // 2]

    taken: set[str] = set()

    def best(key, where=lambda r: True):
        """Highest-scoring row not already claimed, so picks stay distinct."""
        c = [r for r in rows if where(r) and r["graph_id"] not in taken]
        if not c:
            return None
        gid = max(c, key=key)["graph_id"]
        taken.add(gid)
        return gid

    selectors = [
        ("median_session", lambda r: -abs(r["n_nodes"] - med), lambda r: True),
        ("most_parallel",
         lambda r: (r["measured_parallel_width"] or 0, -r["n_nodes"]), lambda r: True),
        ("widest_fan_out", lambda r: (r["max_fan_out"], -r["n_nodes"]), lambda r: True),
        ("deepest", lambda r: r["depth"] or 0, lambda r: True),
        # provider slices are TraceLab-specific but harmless elsewhere
        ("typical_claude", lambda r: -abs(r["n_nodes"] - med),
         lambda r: ":claude:" in r["graph_id"]),
        ("typical_codex", lambda r: -abs(r["n_nodes"] - med),
         lambda r: ":codex:" in r["graph_id"]),
    ]
    sel = {name: gid for name, key, where in selectors if (gid := best(key, where))}
    return sel


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a representative ACG gallery")
    ap.add_argument("dataset")
    ap.add_argument("--outdir", type=Path, default=None)
    args = ap.parse_args()

    sel = pick(args.dataset)
    wanted = {gid: name for name, gid in sel.items()}
    print(f"selected {len(sel)} sessions:")
    for name, gid in sel.items():
        print(f"  {name:18s} {gid}")

    outdir = args.outdir or (REPO / "reports" / "figures" / args.dataset)
    outdir.mkdir(parents=True, exist_ok=True)

    found = 0
    with open(GRAPHS / args.dataset / "graphs.jsonl", "rb") as f:
        for line in f:
            if not line.strip():
                continue
            g = json.loads(line)
            name = wanted.get(g["graph_id"])
            if name is None:
                continue
            rounds = WINDOW if len(g["nodes"]) > WINDOW_ABOVE else None
            out = outdir / f"{name}.png"
            summary = render(g, out, rounds=rounds)
            out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))
            print(f"  wrote {out.name:24s} ({summary['n_nodes']:,} nodes"
                  f"{', windowed ' + rounds if rounds else ''})")
            found += 1
            if found == len(wanted):
                break


if __name__ == "__main__":
    main()
