"""OSWorld computer-use trajectories -> canonical agentic computation graphs.

Source: HF `mlfoundations/gelato-osworld-agent-trajectories` (MIT).
2,166 episodes of a GUI agent driving **real desktop applications** -- Chrome,
LibreOffice Calc/Impress/Writer, GIMP, VS Code, VLC, Thunderbird -- inside an
Ubuntu VM. Two agents (`gelato-30b`, `gta1-32b_baseline`) over two runs.

The applications and the OS are real software, not a simulated world model, but
the task set is a benchmark harness -- registered as `real-controlled` rather
than `real`, and it is the only non-coding domain in the semantic corpus.

Layout: `<agent>/<run>/<app>/<task_id>/{traj.jsonl,result.txt}`.
`traj.jsonl` holds one JSON array of steps; `result.txt` is the graded score.

Record structure is already almost canonical -- each step carries the decision
and its consequence together, so this needs the least parsing of any source:

  reasoning   -> `llm` node reasoning_text
  name        -> `tool` node tool_name        (click, hotkey, type, scroll, ...)
  arguments   -> `tool` node tool_input       (structured, verbatim)
  command     -> extra.command                (the executed pyautogui line)
  tool_output -> `tool` node tool_output
  call_id     -> extra.call_id

One step therefore becomes an llm node plus its tool node; the tool feeds the
next step's llm node with a `data` edge. A terminal `{"type": "TERMINATE"}`
step carries no decision and produces no node.

There is **no user/prompt node**: the dataset ships only the OSWorld task UUID,
not the instruction text, so the task statement is not recoverable from this
release. The UUID is kept in provenance so it can be joined against OSWorld
upstream if the instructions are wanted.

All cost fields are null -- no tokens, no timestamps, no latency.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from src.common import GRAPHS, RAW, already_done, write_graphs_jsonl, write_manifest
from src.schema import Edge, Graph, Node, graph_to_dict, validate_graph

DATASET_ID = "osworld_gelato"
SOURCE_DOMAIN = "web"  # closest canonical domain for GUI/computer-use


def _text(v: Any) -> str | None:
    if v is None:
        return None
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = s.strip()
    return s or None


def _episode_to_graph(path: Path, root: Path, max_text: int,
                      reasons: Counter) -> Graph | None:
    """Returns None for episodes with no action; `reasons` records why.

    A zero-action episode is not a parsing failure. `INFEASIBLE` means the agent
    judged the task impossible and refused, `TERMINATE` that it stopped without
    acting -- both are real outcomes and are counted separately so they never
    look like extraction loss.
    """
    try:
        raw = path.read_text().strip()
        if not raw:
            reasons["empty_file"] += 1
            return None
        steps = json.loads(raw) if raw.startswith("[") else [
            json.loads(l) for l in raw.splitlines() if l.strip()
        ]
    except (json.JSONDecodeError, OSError):
        reasons["unparseable"] += 1
        return None
    # a file holding a single array parses to [[...]] under the line reader
    if len(steps) == 1 and isinstance(steps[0], list):
        steps = steps[0]
    if not steps:
        reasons["empty_file"] += 1
        return None
    if not any(isinstance(s, dict) and s.get("name") for s in steps):
        # zero-action episode: record the agent's declared outcome
        marks = {s.get("type") for s in steps if isinstance(s, dict) and s.get("type")}
        reasons[f"no_action:{'+'.join(sorted(m for m in marks if m)) or 'unknown'}"] += 1
        return None

    rel = path.relative_to(root)
    parts = rel.parts  # <agent>/<run>/<app>/<task_id>/traj.jsonl
    agent = parts[0] if len(parts) > 4 else "unknown"
    run = parts[1] if len(parts) > 4 else None
    app = parts[-3] if len(parts) >= 3 else None
    task_id = parts[-2] if len(parts) >= 2 else path.parent.name

    score: float | None = None
    rp = path.parent / "result.txt"
    if rp.exists():
        try:
            score = float(rp.read_text().strip())
        except ValueError:
            score = None

    def _clip(s: str | None) -> tuple[str | None, int | None]:
        if s is None:
            return None, None
        if max_text and len(s) > max_text:
            return s[:max_text] + f"\n...[truncated, {len(s)} chars total]", len(s)
        return s, len(s)

    gid = f"{DATASET_ID}:{agent}:{run}:{app}:{task_id}"
    nodes: list[Node] = []
    edges: list[Edge] = []
    prev_tool: str | None = None
    prev_llm: str | None = None
    step = 0

    for s in steps:
        if not isinstance(s, dict) or not s.get("name"):
            continue  # TERMINATE marker or malformed -- no decision to record

        llm_id = f"{task_id}:s{step}:llm"
        parents: list[str] = []
        if prev_tool is not None:
            edges.append(Edge(src=prev_tool, dst=llm_id, edge_type="data"))
            parents.append(prev_tool)
        elif prev_llm is not None:
            edges.append(Edge(src=prev_llm, dst=llm_id, edge_type="order"))
            parents.append(prev_llm)

        reason, rn = _clip(_text(s.get("reasoning")))
        nodes.append(Node(node_id=llm_id, node_type="llm", parent_ids=parents,
                          agent_id=agent, turn_id=step, model=agent,
                          reasoning_text=reason, branch_id="b0", committed=True,
                          extra={"reasoning_chars": rn}))

        t_id = f"{task_id}:s{step}:tool"
        arg, an = _clip(_text(s.get("arguments")))
        out, on = _clip(_text(s.get("tool_output")))
        cmd, cn = _clip(_text(s.get("command")))
        nodes.append(Node(node_id=t_id, node_type="tool", parent_ids=[llm_id],
                          agent_id=agent, turn_id=step,
                          tool_name=s.get("name"),
                          tool_input=arg, tool_output=out,
                          tool_status="unknown",  # no per-call success flag
                          branch_id="b0", committed=True,
                          extra={"arg_chars": an, "result_chars": on,
                                 "command": cmd, "command_chars": cn,
                                 "call_id": s.get("call_id")}))
        edges.append(Edge(src=llm_id, dst=t_id, edge_type="control"))

        prev_tool, prev_llm = t_id, llm_id
        step += 1

    if not nodes:
        return None

    return Graph(
        graph_id=gid, dataset=DATASET_ID, source_domain=SOURCE_DOMAIN,
        nodes=nodes, edges=edges,
        provenance={
            "file": str(rel),
            "agent": agent, "run": run,
            "application": app,
            "task_id": task_id,          # OSWorld task UUID; instruction not shipped
            "result_score": score,       # graded outcome, 1.0 = solved
            "steps": step,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract OSWorld computer-use trajectories")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-text", type=int, default=0)
    ap.add_argument("--out", type=Path, default=GRAPHS / DATASET_ID / "graphs.jsonl")
    args = ap.parse_args()

    root = RAW / DATASET_ID
    files = sorted(root.rglob("traj.jsonl"))
    if not files:
        raise SystemExit(f"no traj.jsonl under {root}")
    if already_done(args.out, args.force) and args.limit is None:
        print(f"[{DATASET_ID}] {args.out} exists; use --force")
        return

    stats = Counter()
    reasons: Counter[str] = Counter()

    def _gen() -> Iterator[dict[str, Any]]:
        for p in files:
            if args.limit is not None and stats["graphs"] >= args.limit:
                return
            g = _episode_to_graph(p, root, args.max_text, reasons)
            if g is None:
                stats["skipped"] += 1
                continue
            validate_graph(g)
            stats["graphs"] += 1
            stats["nodes"] += len(g.nodes)
            stats["edges"] += len(g.edges)
            yield graph_to_dict(g)

    n = write_graphs_jsonl(args.out, _gen())
    print(f"[{DATASET_ID}] wrote {n} graphs, {stats['nodes']:,} nodes, "
          f"{stats['edges']:,} edges, {stats['skipped']} skipped -> {args.out}")
    if reasons:
        print(f"[{DATASET_ID}] zero-action episodes: {dict(reasons)}")

    if args.limit is None:
        write_manifest(DATASET_ID, {
            "dataset_id": DATASET_ID,
            "hf_dataset": "mlfoundations/gelato-osworld-agent-trajectories",
            "data_license": "MIT",
            "realness": "real-controlled -- real desktop apps in a VM, benchmark task set",
            "episodes": len(files),
            "graphs": stats["graphs"], "nodes": stats["nodes"],
            "edges": stats["edges"], "skipped": stats["skipped"],
            "skip_reasons": dict(reasons),
        })


if __name__ == "__main__":
    main()
