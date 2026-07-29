"""Canonical agentic computation graph schema.

One graph = one agent session / task episode.
Nodes = runtime operations (LLM call, tool call, retrieval, memory op, agent boundary).
Edges = data or control dependencies.

Design rule that the whole corpus depends on: **absent means null, never a guess.**
`Node` defaults every cost field to None, so an extractor has to go out of its way
to populate one. `validate_graph` then re-checks that nothing structurally
impossible slipped through (dangling edges, unknown enum values, wrong types).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------- enumerations

NODE_TYPES = frozenset(
    {"llm", "tool", "retrieval", "memory", "agent", "verifier", "user"}
)
EDGE_TYPES = frozenset({"data", "control", "order", "agent_msg"})
SOURCE_DOMAINS = frozenset(
    {"coding", "qa", "rag", "tool_use", "web", "multi_agent"}
)
# `ok` / `error` are evidenced; `unknown` is used when the source records a tool
# call but says nothing about whether it succeeded.
TOOL_STATUSES = frozenset({"ok", "error", "unknown"})

# Cost fields, grouped the way the coverage report slices them. These are the
# fields that are null for almost every public dataset -- measuring that
# emptiness honestly is a primary deliverable, so they are named in one place.
TOKEN_FIELDS = ("input_tokens", "output_tokens", "prefill_tokens", "decode_tokens")
LATENCY_FIELDS = ("wall_latency_ms", "tool_latency_ms")
TIMESTAMP_FIELDS = ("start_ts", "end_ts")
KV_FIELDS = ("prompt_prefix_id", "cache_hit_tokens")

# Semantic fields answer "*why* was this node called", which the cost fields
# above cannot. They are tracked as their own group for exactly the same reason:
# so a source that ships no reasoning (TraceLab, whose sanitizer replaces text
# with character counts) reports 0% rather than quietly hiding it in `extra`.
REASONING_FIELDS = ("reasoning_text",)
IO_FIELDS = ("tool_input", "tool_output")
SEMANTIC_FIELDS = REASONING_FIELDS + IO_FIELDS


# --------------------------------------------------------------------- records


@dataclass
class Node:
    """One runtime operation.

    Only `node_id` and `node_type` are required. Everything else defaults to
    null/empty precisely so that a missing source field stays missing.
    """

    node_id: str
    node_type: str
    parent_ids: list[str] = field(default_factory=list)

    agent_id: str | None = None
    turn_id: int | None = None
    model: str | None = None

    start_ts: str | None = None
    end_ts: str | None = None
    wall_latency_ms: float | None = None

    input_tokens: int | None = None
    output_tokens: int | None = None
    prefill_tokens: int | None = None
    decode_tokens: int | None = None

    prompt_prefix_id: str | None = None
    cache_hit_tokens: int | None = None

    tool_name: str | None = None
    tool_latency_ms: float | None = None
    tool_status: str | None = None

    retrieval_k: int | None = None
    retrieved_ids: list[str] = field(default_factory=list)

    retries: int | None = None
    branch_id: str | None = None
    committed: bool | None = None

    # --- semantic content: the evidence for *why* this node ran -------------
    # reasoning_text: the model's stated rationale for this step (thinking /
    #   reasoning_content / "Thought:" prose). Null where the source strips it.
    # tool_input:     the arguments actually passed to the call, verbatim.
    # tool_output:    the observation that came back, verbatim.
    # Verbatim means verbatim -- never summarised, never reformatted. When a
    # source truncates, the extractor records the original length in `extra`.
    reasoning_text: str | None = None
    tool_input: str | None = None
    tool_output: str | None = None

    # Free-form, dataset-specific evidence that does not fit the canonical
    # fields (e.g. TraceLab char counts, tau2 reward). Never read by metrics.
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    edge_type: str


@dataclass
class Graph:
    graph_id: str
    dataset: str
    source_domain: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


class SchemaError(ValueError):
    """Raised loudly on any violation; extraction must not silently emit junk."""


# ------------------------------------------------------------------ validation

_INT_FIELDS = (
    "turn_id",
    "input_tokens",
    "output_tokens",
    "prefill_tokens",
    "decode_tokens",
    "cache_hit_tokens",
    "retrieval_k",
    "retries",
)
_NUM_FIELDS = ("wall_latency_ms", "tool_latency_ms")
_STR_FIELDS = (
    "agent_id",
    "model",
    "start_ts",
    "end_ts",
    "prompt_prefix_id",
    "tool_name",
    "branch_id",
    "reasoning_text",
    "tool_input",
    "tool_output",
)


def validate_graph(g: Graph) -> None:
    """Raise SchemaError on the first violation found."""
    where = f"graph {g.graph_id!r}"

    if not isinstance(g.graph_id, str) or not g.graph_id:
        raise SchemaError(f"{where}: graph_id must be a non-empty str")
    if not isinstance(g.dataset, str) or not g.dataset:
        raise SchemaError(f"{where}: dataset must be a non-empty str")
    if g.source_domain not in SOURCE_DOMAINS:
        raise SchemaError(
            f"{where}: source_domain {g.source_domain!r} not in {sorted(SOURCE_DOMAINS)}"
        )
    if not isinstance(g.provenance, dict):
        raise SchemaError(f"{where}: provenance must be a dict")

    seen: set[str] = set()
    for n in g.nodes:
        if not isinstance(n, Node):
            raise SchemaError(f"{where}: node is not a Node instance: {n!r}")
        if not isinstance(n.node_id, str) or not n.node_id:
            raise SchemaError(f"{where}: node_id must be a non-empty str")
        if n.node_id in seen:
            raise SchemaError(f"{where}: duplicate node_id {n.node_id!r}")
        seen.add(n.node_id)

        nw = f"{where}, node {n.node_id!r}"
        if n.node_type not in NODE_TYPES:
            raise SchemaError(f"{nw}: node_type {n.node_type!r} not in {sorted(NODE_TYPES)}")
        if n.tool_status is not None and n.tool_status not in TOOL_STATUSES:
            raise SchemaError(f"{nw}: tool_status {n.tool_status!r} not in {sorted(TOOL_STATUSES)}")
        if not isinstance(n.parent_ids, list):
            raise SchemaError(f"{nw}: parent_ids must be a list")
        if not isinstance(n.retrieved_ids, list):
            raise SchemaError(f"{nw}: retrieved_ids must be a list")
        if n.committed is not None and not isinstance(n.committed, bool):
            raise SchemaError(f"{nw}: committed must be bool or None")

        for f_ in _INT_FIELDS:
            v = getattr(n, f_)
            # bool is an int subclass in Python; reject it explicitly.
            if v is not None and (not isinstance(v, int) or isinstance(v, bool)):
                raise SchemaError(f"{nw}: {f_} must be int or None, got {type(v).__name__}")
        for f_ in _NUM_FIELDS:
            v = getattr(n, f_)
            if v is not None and (not isinstance(v, (int, float)) or isinstance(v, bool)):
                raise SchemaError(f"{nw}: {f_} must be numeric or None, got {type(v).__name__}")
        for f_ in _STR_FIELDS:
            v = getattr(n, f_)
            if v is not None and not isinstance(v, str):
                raise SchemaError(f"{nw}: {f_} must be str or None, got {type(v).__name__}")

    # Edge endpoints and parent_ids must both resolve to real nodes.
    for e in g.edges:
        if not isinstance(e, Edge):
            raise SchemaError(f"{where}: edge is not an Edge instance: {e!r}")
        if e.edge_type not in EDGE_TYPES:
            raise SchemaError(f"{where}: edge_type {e.edge_type!r} not in {sorted(EDGE_TYPES)}")
        if e.src not in seen:
            raise SchemaError(f"{where}: dangling edge src {e.src!r}")
        if e.dst not in seen:
            raise SchemaError(f"{where}: dangling edge dst {e.dst!r}")

    for n in g.nodes:
        for p in n.parent_ids:
            if p not in seen:
                raise SchemaError(f"{where}: node {n.node_id!r} has unknown parent {p!r}")


# ----------------------------------------------------------------- (de)serialize


def graph_to_dict(g: Graph) -> dict[str, Any]:
    d = dataclasses.asdict(g)
    # Drop empty `extra` bags so the JSONL stays readable.
    for n in d["nodes"]:
        if not n.get("extra"):
            n.pop("extra", None)
    return d


def graph_from_dict(d: dict[str, Any]) -> Graph:
    return Graph(
        graph_id=d["graph_id"],
        dataset=d["dataset"],
        source_domain=d["source_domain"],
        nodes=[Node(**n) for n in d.get("nodes", [])],
        edges=[Edge(**e) for e in d.get("edges", [])],
        provenance=d.get("provenance", {}),
    )
