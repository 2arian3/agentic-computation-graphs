"""Open-SWE-Traces -> canonical agentic computation graphs.

Source: HF `nvidia/Open-SWE-Traces` (CC BY 4.0), paper arXiv:2606.16038.
207,489 SWE-agent / OpenHands trajectories over SWE-rebench-V2 tasks, produced
by Minimax-M2.5 (with thinking) and Qwen3.5-122B (without).

One row = one trajectory = one emitted graph.

Record structure: a **message list with tool_calls** -- roles `system`, `user`,
`assistant` (reasoning + tool_calls) and `tool` (environment observations).

Raw -> canonical field mapping
------------------------------
  assistant message -> `llm` node       turn_id = assistant step index
  tool_calls[]      -> `tool` node      tool_name = function.name
  `tool` message    -> result attached to the matching tool node
  `user` message    -> `user` node
  `system` message  -> **no node**: a static prompt, not a runtime operation.

  model     <- the subset the row came from (minimax_m25 | qwen35_122b)
  agent_id  <- the scaffold the row came from (openhands | sweagent)

Result linkage is **positional, not by id**: unlike TraceLab, `tool` messages
here carry no `tool_call_id`, so a tool result is matched to the tool call it
follows. This is safe because every assistant message in this corpus emits
exactly one tool call (verified: 17,730/17,730 in the sample), and the only
assistant blocks without a following `tool` message are the terminal
`finish`/`submit` calls.

Every cost field is null. The source ships no tokens, no timestamps, no
latency and no KV/prefix data -- it is a text trajectory dump, not a serving
trace. That absence is preserved rather than filled in.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from src.common import GRAPHS, RAW, already_done, write_graphs_jsonl, write_manifest
from src.schema import Edge, Graph, Node, graph_to_dict, validate_graph

DATASET_ID = "open_swe_traces"
SOURCE_DOMAIN = "coding"

# subset directory name -> (model, scaffold). The subset is the only evidence of
# which model/scaffold produced a row; the rows themselves do not say.
_SUBSETS = {
    "minimax_m25_openhands_trajectories": ("minimax_m25", "openhands"),
    "minimax_m25_sweagent_trajectories": ("minimax_m25", "sweagent"),
    "qwen35_openhands_trajectories": ("qwen35_122b", "openhands"),
    "qwen35_sweagent_trajectories": ("qwen35_122b", "sweagent"),
}


def _row_to_graph(row: dict, subset: str, shard: str, idx: int) -> Graph | None:
    traj = row.get("trajectory") or []
    if not traj:
        return None
    model, scaffold = _SUBSETS[subset]
    tid = row.get("trajectory_id") or f"{shard}:{idx}"
    gid = f"{DATASET_ID}:{subset}:{tid}"

    nodes: list[Node] = []
    edges: list[Edge] = []
    by_id: dict[str, Node] = {}
    prev_llm: str | None = None
    # tool nodes of the last assistant step: `emitted` feeds the next llm node,
    # `unresolved` is consumed positionally as `tool` messages arrive.
    emitted: list[str] = []
    unresolved: list[str] = []
    pending_user: list[str] = []
    step = 0

    for m in traj:
        role = m.get("role")

        if role == "system":
            continue  # static prompt, not a runtime operation

        if role == "user":
            uid = f"{tid}:u{step}"
            n = Node(node_id=uid, node_type="user", turn_id=step,
                     agent_id=scaffold, branch_id="b0", committed=True,
                     extra={"content_chars": len(m.get("content") or "")})
            nodes.append(n)
            by_id[uid] = n
            pending_user.append(uid)
            step += 1
            continue

        if role == "assistant":
            llm_id = f"{tid}:a{step}"
            parents: list[str] = []
            # tool results produced since the last assistant message are data in
            for t in emitted:
                edges.append(Edge(src=t, dst=llm_id, edge_type="data"))
                parents.append(t)
            for u in pending_user:
                edges.append(Edge(src=u, dst=llm_id, edge_type="agent_msg"))
                parents.append(u)
            pending_user.clear()
            if not parents and prev_llm is not None:
                edges.append(Edge(src=prev_llm, dst=llm_id, edge_type="order"))
                parents.append(prev_llm)

            n = Node(node_id=llm_id, node_type="llm", parent_ids=parents,
                     agent_id=scaffold, turn_id=step, model=model,
                     branch_id="b0", committed=True,
                     extra={
                         "content_chars": len(m.get("content") or ""),
                         "reasoning_chars": len(m.get("reasoning_content") or ""),
                     })
            nodes.append(n)
            by_id[llm_id] = n

            emitted, unresolved = [], []
            for ti, tc in enumerate(m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                t_id = f"{tid}:a{step}:t{ti}"
                tn = Node(node_id=t_id, node_type="tool", parent_ids=[llm_id],
                          agent_id=scaffold, turn_id=step,
                          tool_name=fn.get("name"),
                          tool_status="unknown",  # source records no success flag
                          branch_id="b0", committed=True,
                          extra={"arg_chars": len(fn.get("arguments") or ""),
                                 "tool_call_id": tc.get("id")})
                nodes.append(tn)
                by_id[t_id] = tn
                edges.append(Edge(src=llm_id, dst=t_id, edge_type="control"))
                emitted.append(t_id)
                unresolved.append(t_id)

            prev_llm = llm_id
            step += 1
            continue

        if role == "tool":
            # Positional match: results arrive in call order (no tool_call_id).
            if unresolved:
                by_id[unresolved.pop(0)].extra["result_chars"] = len(m.get("content") or "")
            continue

    if not nodes:
        return None

    md = row.get("metadata") or {}
    return Graph(
        graph_id=gid,
        dataset=DATASET_ID,
        source_domain=SOURCE_DOMAIN,
        nodes=nodes,
        edges=edges,
        provenance={
            "file": shard,
            "record_index": idx,
            "subset": subset,
            "model": model,
            "scaffold": scaffold,
            "instance_id": row.get("instance_id"),
            "repo": row.get("repo"),
            "repo_license": row.get("license"),
            "language": row.get("language"),
            "resolved": row.get("resolved"),  # 1 solved / 0 unsolved / -1 unknown
            "category": md.get("category"),
            "hf_dataset_name": row.get("hf_dataset_name"),
        },
    )



def iter_graphs(root: Path) -> Iterator[Graph]:
    for subset in sorted(_SUBSETS):
        d = root / "data" / subset
        if not d.is_dir():
            continue
        for shard in sorted(d.glob("*.parquet")):
            table = pq.read_table(
                shard, columns=["instance_id", "repo", "license", "language",
                                "trajectory_id", "trajectory", "resolved",
                                "metadata", "hf_dataset_name"]
            )
            rel = str(shard.relative_to(root.parent.parent.parent))
            for i, row in enumerate(table.to_pylist()):
                g = _row_to_graph(row, subset, rel, i)
                if g is not None:
                    yield g
            del table


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract Open-SWE-Traces to canonical graphs")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=GRAPHS / DATASET_ID / "graphs.jsonl")
    args = ap.parse_args()

    root = RAW / DATASET_ID
    if not (root / "data").is_dir():
        raise SystemExit(f"missing shards under {root}/data")
    if already_done(args.out, args.force) and args.limit is None:
        print(f"[{DATASET_ID}] {args.out} exists; use --force to re-extract")
        return

    stats = Counter()
    shards = sorted((root / "data").glob("*/*.parquet"))

    def _gen() -> Iterator[dict[str, Any]]:
        for i, g in enumerate(iter_graphs(root)):
            if args.limit is not None and i >= args.limit:
                break
            validate_graph(g)
            stats["graphs"] += 1
            stats["nodes"] += len(g.nodes)
            stats["edges"] += len(g.edges)
            yield graph_to_dict(g)

    n = write_graphs_jsonl(args.out, _gen())
    print(f"[{DATASET_ID}] wrote {n} graphs, {stats['nodes']} nodes, {stats['edges']} edges -> {args.out}")

    if args.limit is None:
        write_manifest(DATASET_ID, {
            "dataset_id": DATASET_ID,
            "hf_dataset": "nvidia/Open-SWE-Traces",
            "paper": "https://arxiv.org/abs/2606.16038",
            "data_license": "CC BY 4.0",
            "shards_present": len(shards),
            "bytes": sum(s.stat().st_size for s in shards),
            "subsets": sorted({s.parent.name for s in shards}),
            "graphs": stats["graphs"], "nodes": stats["nodes"], "edges": stats["edges"],
        })


if __name__ == "__main__":
    main()
