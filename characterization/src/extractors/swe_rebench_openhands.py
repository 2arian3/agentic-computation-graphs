"""SWE-rebench OpenHands trajectories -> canonical agentic computation graphs.

Source: HF `nebius/SWE-rebench-openhands-trajectories` (CC BY 4.0).
67,074 OpenHands runs over **1,823 real GitHub repositories** (sqlglot, Pillow,
dvc, dask, conan, ...). Real issues, real repos, real agent executions --
nothing simulated and nothing synthetically generated.

This is the first extractor that populates the semantic fields, so the graphs
answer *why* a node ran, not only what shape the run had.

One row = one trajectory = one emitted graph.

Raw -> canonical field mapping
------------------------------
  assistant message -> `llm` node
      reasoning_text <- reasoning_content, falling back to content. OpenHands
                        puts the rationale in reasoning_content when the model
                        emits thinking, and in content otherwise; whichever is
                        present is the model's stated reason for the next call.
  tool_calls[]      -> `tool` node (or `agent`, see _SUBAGENT_TOOLS)
      tool_name      <- function.name
      tool_input     <- function.arguments, verbatim
  `tool` message    -> tool_output on the matching tool node, verbatim
  `user` message    -> `user` node (carries the GitHub issue text)
  `system` message  -> no node. It is a static prompt, not a runtime operation;
                        it is deduplicated into prompts.json instead (see below).

Edges follow the same rules as the other ReAct-style sources: user -> llm is
`agent_msg`, llm -> tool is `control`, tool -> next llm is `data`, and
consecutive llm nodes with no tool between them get an `order` edge.

Result linkage is **positional**: `tool` messages carry no tool_call_id, so a
result is matched to the call it follows. Safe here because OpenHands emits one
tool call per assistant message (measured: 10,000/10,000 in a 150-trajectory
sample); the only unmatched calls are terminal `finish`.

System prompts are deduplicated
-------------------------------
All 67,074 trajectories share a handful of system prompts. Storing a copy per
graph would add gigabytes of identical text, so each graph records
`provenance.system_prompt_sha256` and the distinct prompts are written once to
`data/graphs/<id>/prompts.json`. The same is done for the per-trajectory tool
schemas. Nothing is lost; it is a join instead of a copy.

Cost fields are all null: this source ships no tokens, no timestamps, no
latency. That absence is preserved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from src.common import GRAPHS, RAW, already_done, write_graphs_jsonl, write_manifest
from src.schema import Edge, Graph, Node, graph_to_dict, validate_graph

DATASET_ID = "swe_rebench_openhands"
SOURCE_DOMAIN = "coding"
SOURCE_FILE = "trajectories.parquet"

# The parquet carries no model column -- the model is dataset-level metadata.
# The HF dataset card states every trajectory was produced by this model under
# OpenHands v0.54.0, so it is recorded per node with that provenance rather than
# left null. `model_source` in provenance marks it as card-level, not per-row.
MODEL = "Qwen3-Coder-480B-A35B-Instruct"
SCAFFOLD_VERSION = "openhands-v0.54.0"

# OpenHands has no sub-agent tool, but keep the check so the node-type rule is
# identical across extractors and picks them up if the scaffold gains one.
_SUBAGENT_TOOLS = frozenset({"spawn_agent", "wait_agent", "close_agent", "resume_agent",
                             "Agent", "Task", "delegate", "SendMessage"})


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()


def _text(v: Any) -> str | None:
    """Normalise a content field to non-empty text, else None (never '')."""
    if v is None:
        return None
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = s.strip()
    return s or None


def _row_to_graph(row: dict, idx: int, prompts: dict[str, str],
                  max_text: int) -> Graph | None:
    traj = row.get("trajectory")
    if isinstance(traj, str):
        try:
            traj = json.loads(traj)
        except json.JSONDecodeError:
            return None
    if not traj:
        return None

    tid = row.get("trajectory_id") or f"row{idx}"
    gid = f"{DATASET_ID}:{tid}"

    def _clip(s: str | None) -> tuple[str | None, int | None]:
        """Truncate only if asked; always report the true original length."""
        if s is None:
            return None, None
        if max_text and len(s) > max_text:
            return s[:max_text] + f"\n...[truncated, {len(s)} chars total]", len(s)
        return s, len(s)

    nodes: list[Node] = []
    edges: list[Edge] = []
    by_id: dict[str, Node] = {}
    prev_llm: str | None = None
    emitted: list[str] = []      # tool nodes of the last step -> feed next llm
    unresolved: list[str] = []   # awaiting their positional result
    pending_user: list[str] = []
    sys_sha: str | None = None
    step = 0

    for m in traj:
        role = m.get("role")

        if role == "system":
            s = _text(m.get("content"))
            if s:
                sys_sha = _sha(s)
                prompts.setdefault(sys_sha, s)
            continue

        if role == "user":
            uid = f"{tid}:u{step}"
            body, n = _clip(_text(m.get("content")))
            node = Node(node_id=uid, node_type="user", turn_id=step,
                        agent_id="openhands", branch_id="b0", committed=True,
                        # the issue text is the task statement, i.e. the reason
                        # the whole trajectory exists
                        reasoning_text=body,
                        extra={"content_chars": n})
            nodes.append(node)
            by_id[uid] = node
            pending_user.append(uid)
            step += 1
            continue

        if role == "assistant":
            llm_id = f"{tid}:a{step}"
            parents: list[str] = []
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

            # OpenHands puts the rationale in reasoning_content when the model
            # emits thinking, otherwise in content. Take whichever is present.
            reason = _text(m.get("reasoning_content")) or _text(m.get("content"))
            reason, rn = _clip(reason)

            node = Node(node_id=llm_id, node_type="llm", parent_ids=parents,
                        agent_id="openhands", turn_id=step,
                        model=MODEL,
                        reasoning_text=reason,
                        branch_id="b0", committed=True,
                        extra={"reasoning_chars": rn,
                               "from_reasoning_content":
                                   _text(m.get("reasoning_content")) is not None})
            nodes.append(node)
            by_id[llm_id] = node

            emitted, unresolved = [], []
            for ti, tc in enumerate(m.get("tool_calls") or []):
                fn = tc.get("function") or tc
                name = fn.get("name")
                t_id = f"{tid}:a{step}:t{ti}"
                arg, an = _clip(_text(fn.get("arguments")))
                tn = Node(node_id=t_id,
                          node_type="agent" if name in _SUBAGENT_TOOLS else "tool",
                          parent_ids=[llm_id], agent_id="openhands", turn_id=step,
                          tool_name=name,
                          tool_input=arg,
                          tool_status="unknown",  # no per-call success flag
                          branch_id="b0", committed=True,
                          extra={"arg_chars": an, "tool_call_id": tc.get("id")})
                nodes.append(tn)
                by_id[t_id] = tn
                edges.append(Edge(src=llm_id, dst=t_id, edge_type="control"))
                emitted.append(t_id)
                unresolved.append(t_id)

            prev_llm = llm_id
            step += 1
            continue

        if role == "tool":
            if unresolved:
                n = by_id[unresolved.pop(0)]
                out, on = _clip(_text(m.get("content")))
                n.tool_output = out
                n.extra["result_chars"] = on
            continue

    if not nodes:
        return None

    tools_schema = row.get("tools")
    tools_sha = None
    if tools_schema:
        s = tools_schema if isinstance(tools_schema, str) else json.dumps(
            tools_schema, ensure_ascii=False, sort_keys=True)
        tools_sha = _sha(s)
        prompts.setdefault(tools_sha, s)

    return Graph(
        graph_id=gid, dataset=DATASET_ID, source_domain=SOURCE_DOMAIN,
        nodes=nodes, edges=edges,
        provenance={
            "file": f"data/raw/{DATASET_ID}/{SOURCE_FILE}",
            "record_index": idx,
            "trajectory_id": tid,
            "instance_id": row.get("instance_id"),
            "repo": row.get("repo"),            # real GitHub repository
            "resolved": row.get("resolved"),    # 1 solved / 0 not
            "exit_status": row.get("exit_status"),
            "gen_tests_correct": row.get("gen_tests_correct"),
            "pred_passes_gen_tests": row.get("pred_passes_gen_tests"),
            "model_source": "hf dataset card (dataset-level, not per-row)",
            "scaffold": SCAFFOLD_VERSION,
            "system_prompt_sha256": sys_sha,    # join key into prompts.json
            "tool_schemas_sha256": tools_sha,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract SWE-rebench OpenHands trajectories")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-text", type=int, default=0,
                    help="truncate stored text at N chars (0 = keep verbatim); "
                         "the true length is always recorded in extra")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--out", type=Path, default=GRAPHS / DATASET_ID / "graphs.jsonl")
    args = ap.parse_args()

    src = RAW / DATASET_ID / SOURCE_FILE
    if not src.exists():
        raise SystemExit(f"missing {src}")
    if already_done(args.out, args.force) and args.limit is None:
        print(f"[{DATASET_ID}] {args.out} exists; use --force to re-extract")
        return

    prompts: dict[str, str] = {}
    stats = Counter()
    cols = ["trajectory_id", "instance_id", "repo", "trajectory", "tools",
            "exit_status", "resolved", "gen_tests_correct", "pred_passes_gen_tests"]

    def _gen() -> Iterator[dict[str, Any]]:
        pf = pq.ParquetFile(src)
        i = 0
        for batch in pf.iter_batches(batch_size=args.batch, columns=cols):
            for row in batch.to_pylist():
                if args.limit is not None and stats["graphs"] >= args.limit:
                    return
                g = _row_to_graph(row, i, prompts, args.max_text)
                i += 1
                if g is None:
                    stats["skipped"] += 1
                    continue
                validate_graph(g)
                stats["graphs"] += 1
                stats["nodes"] += len(g.nodes)
                stats["edges"] += len(g.edges)
                yield graph_to_dict(g)

    n = write_graphs_jsonl(args.out, _gen())
    (args.out.parent / "prompts.json").write_text(
        json.dumps(prompts, indent=2, ensure_ascii=False))
    print(f"[{DATASET_ID}] wrote {n} graphs, {stats['nodes']:,} nodes, "
          f"{stats['edges']:,} edges, {stats['skipped']} skipped -> {args.out}")
    print(f"[{DATASET_ID}] {len(prompts)} distinct system prompts / tool schemas "
          f"-> {args.out.parent / 'prompts.json'}")

    if args.limit is None:
        write_manifest(DATASET_ID, {
            "dataset_id": DATASET_ID,
            "hf_dataset": "nebius/SWE-rebench-openhands-trajectories",
            "data_license": "CC BY 4.0",
            "realness": "real -- real GitHub repos, real issues, real agent runs",
            "bytes": src.stat().st_size,
            "graphs": stats["graphs"], "nodes": stats["nodes"],
            "edges": stats["edges"], "skipped": stats["skipped"],
            "distinct_prompts": len(prompts),
            "max_text": args.max_text,
        })


if __name__ == "__main__":
    main()
