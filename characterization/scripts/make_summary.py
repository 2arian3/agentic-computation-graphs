"""Regenerate reports/SUMMARY.md from registry.yaml + each dataset's metrics.json.

The comparison table is the primary deliverable, so this is rerun after every
dataset and never hand-edited.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import GRAPHS, REPORTS, REPO  # noqa: E402

FINDINGS = REPORTS / "_findings.md"  # hand-written prose, prepended if present


def main() -> None:
    reg = yaml.safe_load((REPO / "registry.yaml").read_text())["datasets"]
    rows = []
    for d in sorted(reg, key=lambda x: x["order"]):
        m_path = GRAPHS / d["id"] / "metrics.json"
        if not m_path.exists():
            rows.append((d, None))
            continue
        rows.append((d, json.loads(m_path.read_text())))

    out = ["# Cross-dataset summary", ""]
    if FINDINGS.exists():
        out += [FINDINGS.read_text().strip(), ""]

    out += [
        "## Comparison table",
        "",
        "| # | dataset | status | domain | #graphs | med nodes | med depth | med max-fan-out | %nodes tokens | %nodes timestamps |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for d, m in rows:
        if m is None or not m.get("n_graphs"):
            out.append(
                f"| {d['order']} | {d['id']} | `{d['status']}` | {d.get('source_domain','—')} "
                f"| — | — | — | — | — | — |"
            )
            continue
        dist = m["dist"]
        cov = m["coverage_node_weighted"]
        out.append(
            f"| {d['order']} | {d['id']} | `{d['status']}` | {m['source_domain']} "
            f"| {m['n_graphs']:,} | {dist['n_nodes']['median']:g} | {dist['depth']['median']:g} "
            f"| {dist['max_fan_out']['median']:g} "
            f"| {cov['tokens'] * 100:.1f}% | {cov['timestamps'] * 100:.1f}% |"
        )

    out += ["", "## Coverage of cost fields (node-weighted)", "",
            "| dataset | tokens | latency | timestamps | KV/prefix |", "|---|---|---|---|---|"]
    for d, m in rows:
        if m is None or not m.get("n_graphs"):
            continue
        c = m["coverage_node_weighted"]
        out.append(
            f"| {d['id']} | {c['tokens']*100:.1f}% | {c['latency']*100:.1f}% "
            f"| {c['timestamps']*100:.1f}% | {c['kv_prefix']*100:.1f}% |"
        )

    # The second axis: can the graph say *why* a node ran, not just its shape.
    out += ["", "## Coverage of semantic fields", "",
            "Reasoning is measured against `llm` nodes and tool i/o against",
            "action (`tool`/`agent`/`retrieval`) nodes — the nodes that could",
            "carry them. A source that strips text reports 0% here even when it",
            "scores 100% on cost fields.", "",
            "| dataset | reasoning (of llm nodes) | tool input/output (of action nodes) |",
            "|---|---|---|"]
    for d, m in rows:
        if m is None or not m.get("n_graphs"):
            continue
        s = m.get("coverage_semantic")
        if not s:
            out.append(f"| {d['id']} | — | — |")
            continue
        out.append(
            f"| {d['id']} | {s['reasoning_of_llm_nodes']*100:.1f}% "
            f"| {s['tool_io_of_action_nodes']*100:.1f}% |"
        )

    out += ["", "## Registry status", "", "| # | dataset | mode | status | source verified |",
            "|---|---|---|---|---|"]
    for d, _ in rows:
        out.append(
            f"| {d['order']} | {d['id']} | `{d['mode']}` | `{d['status']}` "
            f"| {'yes' if d.get('source_verified') else 'no'} |"
        )
    out.append("")

    (REPORTS / "SUMMARY.md").write_text("\n".join(out))
    print(f"wrote {REPORTS / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
