"""SWE-agent trajectories -> canonical agentic computation graphs.

Source: HF `nebius/SWE-agent-trajectories` (CC BY 4.0). 80,036 SWE-agent runs
over real GitHub repositories (pydantic, dvc, sqlglot, textual, sqlfluff, ...).
Real issues, real repos, real agent executions.

Paired deliberately with `swe_rebench_openhands`: same domain, same kind of
task, **different scaffold**. Any structural difference between the two is
attributable to the harness rather than to the workload.

Record structure: a flat step list, roles `system` / `user` / `ai`, alternating
`user, ai, user, ai, ...` after a single `system` step.

  system  -> no node. `system_prompt` holds the full prompt; deduplicated into
             prompts.json and joined by sha256 (see swe_rebench_openhands).
  user[0] -> `user` node. This is the GitHub issue text, i.e. the task.
  ai[k]   -> `llm` node + one `tool` node.
  user[k] -> for k >= 1, the observation produced by ai[k-1]'s command.

Unlike OpenHands, SWE-agent does **not** emit structured tool calls. Each `ai`
step is prose followed by a single fenced block holding the command:

    The error we received is an HTTPError ... Let's find the file.

    ```
    find_file "memset.py" src
    ```

so the split is: reasoning = text before the first fence, tool_input = the last
fenced block verbatim, tool_name = the command's first token (`find_file`,
`open`, `edit`, `python`, `scroll_up`, ...). Commands are frequently multi-line
(`edit 20:24` followed by replacement source); only the first token is the
tool, and the whole block is preserved as tool_input.

Every cost field is null -- this source ships no tokens, timestamps or latency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import pyarrow.parquet as pq

from src.common import GRAPHS, RAW, already_done, write_graphs_jsonl, write_manifest
from src.schema import Edge, Graph, Node, graph_to_dict, validate_graph

DATASET_ID = "swe_agent_traj"
SOURCE_DOMAIN = "coding"

_FENCE = re.compile(r"```(?:[\w.+-]*\n)?(.*?)```", re.S)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "replace")).hexdigest()


def _text(v: Any) -> str | None:
    if v is None:
        return None
    s = v if isinstance(v, str) else str(v)
    s = s.strip()
    return s if s and s != "None" else None


def _split_action(text: str) -> tuple[str | None, str | None, str | None]:
    """(reasoning, command, tool_name) from one SWE-agent `ai` step."""
    blocks = _FENCE.findall(text)
    if not blocks:
        # The model narrated without issuing a command -- keep the prose as
        # reasoning and record no tool rather than inventing one.
        return _text(text), None, None
    cmd = blocks[-1].strip()
    reasoning = _text(text.split("```")[0])
    first = cmd.split(maxsplit=1)[0] if cmd.split() else None
    return reasoning, (cmd or None), first


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

    tid = f"{row.get('instance_id') or 'row'}#{idx}"
    model = row.get("model_name")

    def _clip(s: str | None) -> tuple[str | None, int | None]:
        if s is None:
            return None, None
        if max_text and len(s) > max_text:
            return s[:max_text] + f"\n...[truncated, {len(s)} chars total]", len(s)
        return s, len(s)

    nodes: list[Node] = []
    edges: list[Edge] = []
    sys_sha: str | None = None
    prev_llm: str | None = None
    last_tool: str | None = None   # awaiting its observation from the next user step
    prev_tools: list[str] = []     # tools of the previous ai step -> feed this one
    pending_user: list[str] = []
    seen_user = 0
    step = 0

    for m in traj:
        role = m.get("role")

        if role == "system":
            s = _text(m.get("system_prompt")) or _text(m.get("text"))
            if s:
                sys_sha = _sha(s)
                prompts.setdefault(sys_sha, s)
            continue

        if role == "user":
            body = _text(m.get("text"))
            if seen_user == 0:
                # first user step is the issue -- the task, not an observation
                uid = f"{tid}:u0"
                b, n = _clip(body)
                node = Node(node_id=uid, node_type="user", turn_id=step,
                            agent_id="swe-agent", branch_id="b0", committed=True,
                            reasoning_text=b, extra={"content_chars": n})
                nodes.append(node)
                pending_user.append(uid)
                step += 1
            elif last_tool is not None:
                # observation of the previous command
                tn = next(x for x in nodes if x.node_id == last_tool)
                out, n = _clip(body)
                tn.tool_output = out
                tn.extra["result_chars"] = n
                last_tool = None
            seen_user += 1
            continue

        if role == "ai":
            text = m.get("text") or ""
            reasoning, cmd, tool_name = _split_action(text)
            llm_id = f"{tid}:a{step}"
            parents: list[str] = []
            # The previous step's command feeds this one, whether or not its
            # observation arrived (a truncated run can end mid-command).
            for prev_t in prev_tools:
                edges.append(Edge(src=prev_t, dst=llm_id, edge_type="data"))
                parents.append(prev_t)
            prev_tools = []
            for u in pending_user:
                edges.append(Edge(src=u, dst=llm_id, edge_type="agent_msg"))
                parents.append(u)
            pending_user.clear()
            if not parents and prev_llm is not None:
                edges.append(Edge(src=prev_llm, dst=llm_id, edge_type="order"))
                parents.append(prev_llm)

            r_, rn = _clip(reasoning)
            nodes.append(Node(node_id=llm_id, node_type="llm", parent_ids=parents,
                              agent_id="swe-agent", turn_id=step, model=model,
                              reasoning_text=r_, branch_id="b0", committed=True,
                              extra={"reasoning_chars": rn}))

            if cmd is not None:
                t_id = f"{tid}:a{step}:t0"
                c_, cn = _clip(cmd)
                nodes.append(Node(node_id=t_id, node_type="tool",
                                  parent_ids=[llm_id], agent_id="swe-agent",
                                  turn_id=step, tool_name=tool_name,
                                  tool_input=c_, tool_status="unknown",
                                  branch_id="b0", committed=True,
                                  extra={"command_chars": cn}))
                edges.append(Edge(src=llm_id, dst=t_id, edge_type="control"))
                last_tool = t_id
                prev_tools = [t_id]

            prev_llm = llm_id
            step += 1
            continue

    if not nodes:
        return None

    return Graph(
        graph_id=f"{DATASET_ID}:{tid}", dataset=DATASET_ID,
        source_domain=SOURCE_DOMAIN, nodes=nodes, edges=edges,
        provenance={
            "file": "data/raw/swe_agent_traj/",
            "record_index": idx,
            "instance_id": row.get("instance_id"),
            "model_name": model,
            "exit_status": row.get("exit_status"),
            "has_generated_patch": bool(row.get("generated_patch")),
            "system_prompt_sha256": sys_sha,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract SWE-agent trajectories")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-text", type=int, default=0)
    ap.add_argument("--out", type=Path, default=GRAPHS / DATASET_ID / "graphs.jsonl")
    args = ap.parse_args()

    shards = sorted((RAW / DATASET_ID).rglob("*.parquet"))
    if not shards:
        raise SystemExit(f"no parquet under {RAW / DATASET_ID}")
    if already_done(args.out, args.force) and args.limit is None:
        print(f"[{DATASET_ID}] {args.out} exists; use --force")
        return

    prompts: dict[str, str] = {}
    stats = Counter()
    cols = ["instance_id", "model_name", "trajectory", "exit_status", "generated_patch"]

    def _gen() -> Iterator[dict[str, Any]]:
        i = 0
        for sh in shards:
            pf = pq.ParquetFile(sh)
            for batch in pf.iter_batches(batch_size=400, columns=cols):
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
    print(f"[{DATASET_ID}] {len(prompts)} distinct system prompts")

    if args.limit is None:
        write_manifest(DATASET_ID, {
            "dataset_id": DATASET_ID,
            "hf_dataset": "nebius/SWE-agent-trajectories",
            "data_license": "CC BY 4.0",
            "realness": "real -- real GitHub repos, real issues, real agent runs",
            "shards": len(shards),
            "graphs": stats["graphs"], "nodes": stats["nodes"],
            "edges": stats["edges"], "skipped": stats["skipped"],
            "distinct_prompts": len(prompts),
        })


if __name__ == "__main__":
    main()
