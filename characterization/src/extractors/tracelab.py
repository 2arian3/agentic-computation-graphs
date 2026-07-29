"""TraceLab -> canonical agentic computation graphs.

Source: https://github.com/uw-syfi/TraceLab release v0.0.1 asset
`syfi_coding_trace.jsonl.gz` (CC BY 4.0). 357,161 LLM rounds / 432,510 tool
records / 4,265 sessions from 43 developers' real Claude Code and Codex sessions.

One raw row = one LLM invocation ("round"). One emitted graph = one session.

Raw -> canonical field mapping
------------------------------
llm node (one per round)
  turn_id          <- round_index                (session-local ordering)
  model            <- model
  agent_id         <- provider                   ("claude" | "codex")
  input_tokens     <- input_tokens_total
  output_tokens    <- output_tokens
  prefill_tokens   <- newly_append_tokens        tokens NOT served from the prefix
                                                 cache, i.e. the ones actually
                                                 prefilled. Per the repo's
                                                 docs/prompt_cache_accounting.md,
                                                 input_tokens_total = prefix_tokens
                                                 + newly_append_tokens.
  decode_tokens    <- output_tokens              identity: one decode step per
                                                 output token.
  cache_hit_tokens <- prefix_tokens              = claude cache_read_input_tokens
                                                 / codex cached_input_tokens.
  prompt_prefix_id <- null                       the trace carries no prefix identity.
  start_ts/end_ts  <- first/last timing_event timestamp of the round.
  wall_latency_ms  <- the repo's documented observable-generation-time proxy:
                      (first tool_call ts) - (latest user_message|tool_result ts
                      before it). null when the round emits no tool call.

tool node (one per tools[] entry)
  tool_name        <- tool_name
  start_ts/end_ts  <- emitted_at / result_at
  wall_latency_ms  <- tool_wall_latency_ms       (result_at - emitted_at)
  tool_latency_ms  <- tool_internal_latency_ms, falling back to
                      tool_wall_latency_ms -- the precedence the repo's own
                      analyses use.
  tool_status      <- is_error: True->"error", False->"ok", null->"unknown"

user node (one per user_message timing event)
  start_ts         <- event timestamp

Edges
  user   -> llm       agent_msg  human turn entering the round
  llm    -> tool      control    the round decided to call this tool
  tool   -> llm       data       EXACT linkage: a tool_result event in round r
                                 carries the tool_call_id of the tool node that
                                 produced it. Verified to resolve for 100% of
                                 431,818 tool_result events (99.8% to the
                                 immediately-preceding round, 0.2% older).
  llm    -> llm       order      consecutive rounds with no tool_result linkage,
                                 so ordering is still recoverable.

Not populated (absent from the source, left null by design): retrieval_k,
retrieved_ids, retries, prompt_prefix_id. TraceLab traces are linear -- no
explored-but-abandoned branches -- so every node is branch_id "b0",
committed=True (these operations all really executed).
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from src.common import GRAPHS, RAW, already_done, sha256, write_graphs_jsonl, write_manifest
from src.schema import Edge, Graph, Node, graph_to_dict, validate_graph

DATASET_ID = "tracelab"
SOURCE_DOMAIN = "coding"
SOURCE_FILE = "syfi_coding_trace.jsonl.gz"

# Input-side events: the round is "ready to generate" after the latest of these.
_INPUT_EVENTS = frozenset({"user_message", "tool_result"})

# Tools that cross an agent boundary rather than doing work themselves: they
# spawn, message, wait on or close a sub-agent. These become `agent` nodes so
# the corpus can tell delegation apart from ordinary tool use. Classification is
# by tool name only -- evidenced, not inferred from behaviour. Note the Claude
# `Task*` family (TaskCreate/TaskUpdate/TaskList/TaskGet) is the todo list, NOT
# sub-agents, and is deliberately excluded.
_SUBAGENT_TOOLS = frozenset({
    # Codex sub-agent lifecycle
    "spawn_agent", "wait_agent", "close_agent", "resume_agent",
    # Claude Code sub-agent invocation
    "Agent", "Task", "Explore", "SendMessage",
})


def _ts_ms(a: str | None, b: str | None) -> float | None:
    """Milliseconds between two ISO-8601 timestamps; None if either is absent."""
    if not a or not b:
        return None
    try:
        ta = datetime.fromisoformat(a.replace("Z", "+00:00"))
        tb = datetime.fromisoformat(b.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (tb - ta).total_seconds() * 1000.0


def _generation_ms(events: list[dict]) -> float | None:
    """TraceLab's documented observable-generation-time proxy.

    README: "the usual proxy for input ready -> next tool input is the latest
    user_message or tool_result event before the first tool_call, subtracted
    from that tool_call timestamp." Returns None when the round emits no tool
    call or has no preceding input event -- we do not substitute a guess.
    """
    first_call = next((e for e in events if e.get("event_type") == "tool_call"), None)
    if first_call is None:
        return None
    prior = [
        e for e in events
        if e.get("event_type") in _INPUT_EVENTS
        and e.get("timestamp")
        and e["timestamp"] <= first_call.get("timestamp", "")
    ]
    if not prior:
        return None
    return _ts_ms(max(e["timestamp"] for e in prior), first_call.get("timestamp"))


def _session_to_graph(session_id: str, rows: list[dict], provenance: dict) -> Graph:
    """Build one canonical graph from all rounds of one session.

    Providers place tool results differently, and the placement decides where a
    data edge points:

    * Claude splits the loop across rounds -- round r emits the call, round r+1
      carries the tool_result. The result is genuinely round r+1's input.
    * Codex bundles them -- round r carries both its own tool_calls and their
      tool_results, because TraceLab closes a round on the `token_count` event
      which fires after the results land (see the repo's extract_codex_rounds.py
      line ~690, `timing_events = [*pending_input_events, *segment_timing_events]`).
      The result timestamps are strictly after that round's own generation, so
      round r's LLM cannot have consumed them; round r+1 does. Token accounting
      confirms it: round r+1's input = round r's input + round r's output + the
      result tokens.

    So the rule is provider-independent and timestamp-correct: a tool result
    flows into the next LLM invocation that could have seen it.
    """
    # 16/4265 sessions repeat a round_index, so position (not round_index) is
    # what makes node ids unique. turn_id keeps the raw, evidenced round_index.
    rows = sorted(rows, key=lambda r: (r.get("round_index", 0), r.get("_seq", 0)))

    nodes: list[Node] = []
    edges: list[Edge] = []
    parents: defaultdict[str, list[str]] = defaultdict(list)
    # tool_call_id -> (tool node id, round position that emitted it)
    tool_node_by_call_id: dict[str, tuple[str, int]] = {}
    # (tool node id, target round position) for data edges resolved after the
    # whole session is laid out, since a Codex result targets the *next* round.
    pending_data: list[tuple[str, int]] = []
    n_rounds = len(rows)
    unconsumed = 0

    for pos, r in enumerate(rows):
        llm_id = f"{session_id}:r{pos}:llm"
        events = r.get("timing_events") or []
        provider = r.get("provider")

        # --- user nodes: each user_message event entering this round ---------
        for ui, ev in enumerate(e for e in events if e.get("event_type") == "user_message"):
            uid = f"{session_id}:r{pos}:user{ui}"
            nodes.append(
                Node(
                    node_id=uid,
                    node_type="user",
                    turn_id=r.get("round_index"),
                    agent_id=provider,
                    start_ts=ev.get("timestamp"),
                    end_ts=ev.get("timestamp"),
                    branch_id="b0",
                    committed=True,
                    extra={"content_chars": ev.get("content_chars")},
                )
            )
            edges.append(Edge(src=uid, dst=llm_id, edge_type="agent_msg"))
            parents[llm_id].append(uid)

        ts = [e.get("timestamp") for e in events if e.get("timestamp")]
        nodes.append(
            Node(
                node_id=llm_id,
                node_type="llm",
                parent_ids=parents[llm_id],
                agent_id=provider,
                turn_id=r.get("round_index"),
                model=r.get("model"),
                start_ts=min(ts) if ts else None,
                end_ts=max(ts) if ts else None,
                wall_latency_ms=_generation_ms(events),
                input_tokens=r.get("input_tokens_total"),
                output_tokens=r.get("output_tokens"),
                prefill_tokens=r.get("newly_append_tokens"),
                decode_tokens=r.get("output_tokens"),
                cache_hit_tokens=r.get("prefix_tokens"),
                prompt_prefix_id=None,  # not present in the source
                branch_id="b0",
                committed=True,
                extra={
                    "round_id": r.get("round_id"),
                    "reasoning_output_tokens": r.get("reasoning_output_tokens"),
                    "claude_cache_creation_input_tokens": r.get(
                        "claude_cache_creation_input_tokens"
                    ),
                    "claude_uncached_input_tokens": r.get("claude_uncached_input_tokens"),
                    "current_input_chars": r.get("current_input_chars"),
                    "codex_turn_id": r.get("turn_id"),
                },
            )
        )

        # --- tool nodes emitted by this round --------------------------------
        for t in r.get("tools") or []:
            tid = f"{session_id}:r{pos}:tool{t.get('tool_index')}"
            is_err = t.get("is_error")
            internal, wall = t.get("tool_internal_latency_ms"), t.get("tool_wall_latency_ms")
            nodes.append(
                Node(
                    node_id=tid,
                    node_type=("agent" if t.get("tool_name") in _SUBAGENT_TOOLS else "tool"),
                    parent_ids=[llm_id],
                    agent_id=provider,
                    turn_id=r.get("round_index"),
                    start_ts=t.get("emitted_at"),
                    end_ts=t.get("result_at"),
                    wall_latency_ms=wall,
                    # repo precedence: internal duration when reported, else wall
                    tool_latency_ms=internal if internal is not None else wall,
                    tool_name=t.get("tool_name"),
                    tool_status=("unknown" if is_err is None else ("error" if is_err else "ok")),
                    branch_id="b0",
                    committed=True,
                    extra={
                        "input_chars": t.get("input_chars"),
                        "result_chars": t.get("result_chars"),
                        "tool_wall_latency_ms": wall,
                        "tool_internal_latency_ms": internal,
                    },
                )
            )
            edges.append(Edge(src=llm_id, dst=tid, edge_type="control"))
            if t.get("tool_call_id"):
                tool_node_by_call_id[t["tool_call_id"]] = (tid, pos)

        # --- data edges: a tool_result event names the node that produced it --
        # Registered after this round's tools so same-round (Codex) results
        # resolve too. Target = this round for a result of an earlier round's
        # call; the next round when the call was made in this same round.
        for ev in events:
            if ev.get("event_type") != "tool_result":
                continue
            hit = tool_node_by_call_id.get(ev.get("tool_call_id"))
            if hit is None:
                continue
            src, emitted_pos = hit
            target = pos if emitted_pos < pos else pos + 1
            if target < n_rounds:
                pending_data.append((src, target))
            else:
                unconsumed += 1  # produced by the last round; nothing consumes it

    # Resolve deferred data edges, then connect any round left without one.
    fed: set[int] = set()
    for src, target in dict.fromkeys(pending_data):  # a repeated result -> one edge
        dst = f"{session_id}:r{target}:llm"
        edges.append(Edge(src=src, dst=dst, edge_type="data"))
        parents[dst].append(src)
        fed.add(target)
    for pos in range(1, n_rounds):
        if pos not in fed:
            dst, prev = f"{session_id}:r{pos}:llm", f"{session_id}:r{pos - 1}:llm"
            edges.append(Edge(src=prev, dst=dst, edge_type="order"))
            parents[dst].append(prev)

    # parent_ids were captured by reference before these appends, except for the
    # llm nodes built earlier in the loop -- refresh them all from `parents`.
    for n in nodes:
        if n.node_type == "llm":
            n.parent_ids = parents[n.node_id]

    provenance = {**provenance, "tool_results_unconsumed": unconsumed}

    return Graph(
        graph_id=f"{DATASET_ID}:{session_id}",
        dataset=DATASET_ID,
        source_domain=SOURCE_DOMAIN,
        nodes=nodes,
        edges=edges,
        provenance=provenance,
    )


def iter_graphs(src: Path) -> Iterator[Graph]:
    """Stream sessions out of the gzipped JSONL with bounded memory.

    Pass 1 counts rows per session; pass 2 buffers a session only until its
    last row arrives, then emits and frees it. Sessions are mostly contiguous
    (16/4265 re-enter later), so only a handful are ever open at once.
    """
    counts: Counter[str] = Counter()
    with gzip.open(src, "rt") as f:
        for line in f:
            counts[json.loads(line)["session_id"]] += 1

    buf: defaultdict[str, list[dict]] = defaultdict(list)
    with gzip.open(src, "rt") as f:
        for seq, line in enumerate(f):
            r = json.loads(line)
            r["_seq"] = seq
            sid = r["session_id"]
            buf[sid].append(r)
            if len(buf[sid]) == counts[sid]:
                rows = buf.pop(sid)
                yield _session_to_graph(
                    sid,
                    rows,
                    {
                        "file": f"data/raw/{DATASET_ID}/{SOURCE_FILE}",
                        "record_index": rows[0]["_seq"],
                        "n_rounds": len(rows),
                        "provider": rows[0].get("provider"),
                        "project": rows[0].get("project"),
                        "release": "v0.0.1",
                    },
                )
    assert not buf, f"unflushed sessions: {list(buf)[:5]}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract TraceLab sessions to canonical graphs")
    ap.add_argument("--force", action="store_true", help="re-extract even if output exists")
    ap.add_argument("--limit", type=int, default=None, help="stop after N sessions (smoke test)")
    ap.add_argument("--out", type=Path, default=GRAPHS / DATASET_ID / "graphs.jsonl")
    args = ap.parse_args()

    src = RAW / DATASET_ID / SOURCE_FILE
    if not src.exists():
        raise SystemExit(f"missing source {src}; download it first")
    if already_done(args.out, args.force) and args.limit is None:
        print(f"[{DATASET_ID}] {args.out} exists; use --force to re-extract")
        return

    stats = Counter()

    def _gen() -> Iterator[dict[str, Any]]:
        for i, g in enumerate(iter_graphs(src)):
            if args.limit is not None and i >= args.limit:
                break
            validate_graph(g)  # fail loudly rather than emit junk
            stats["graphs"] += 1
            stats["nodes"] += len(g.nodes)
            stats["edges"] += len(g.edges)
            yield graph_to_dict(g)

    n = write_graphs_jsonl(args.out, _gen())
    print(f"[{DATASET_ID}] wrote {n} graphs, {stats['nodes']} nodes, {stats['edges']} edges -> {args.out}")

    if args.limit is None:
        write_manifest(
            DATASET_ID,
            {
                "dataset_id": DATASET_ID,
                "source_repo": "https://github.com/uw-syfi/TraceLab",
                "release": "v0.0.1",
                "asset": SOURCE_FILE,
                "url": f"https://github.com/uw-syfi/TraceLab/releases/download/v0.0.1/{SOURCE_FILE}",
                "bytes": src.stat().st_size,
                "sha256": sha256(src),
                "sha256_published": "9d265eae69a31cae203848bea936f018148eed7ca8bf56050c5abe96da0b4e6b",
                "code_license": "Apache-2.0",
                "data_license": "CC BY 4.0",
                "graphs": stats["graphs"],
                "nodes": stats["nodes"],
                "edges": stats["edges"],
            },
        )


if __name__ == "__main__":
    main()
