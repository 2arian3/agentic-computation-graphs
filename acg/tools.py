"""The fixed tool alphabet.

Decision 2: because our entire question is about the structure the *model* generates,
we give it a small, fixed, countable tool set and let the loop be emergent. Fixing the
tool alphabet is a precondition for comparing graphs across runs -- every tool node in
every graph is one of a known, finite set of types.

Tools are pure functions of (args, corpus). They contain NO tracing and NO control
flow; the agent loop (acg/agent.py) owns the loop and the spans. This keeps the node
alphabet and the execution structure cleanly separated.
"""
from __future__ import annotations

import json
from typing import Any

from .corpus import Corpus

# Tool names are the *node types* for tool nodes in the ACG.
SEARCH = "search"
READ_DOCUMENT = "read_document"
FINISH = "finish"
SUB_AGENT = "sub_agent"

# The core alphabet used by the canonical experiments.
TOOL_NAMES = (SEARCH, READ_DOCUMENT, FINISH)
# The extended alphabet including the opt-in branch tool (RQ-N2).
TOOL_NAMES_EXT = (SEARCH, READ_DOCUMENT, FINISH, SUB_AGENT)


def tool_schemas(search_top_k: int = 3, elicit_reasoning: bool = False,
                 include_sub_agent: bool = False) -> list[dict]:
    """OpenAI-style function/tool schemas advertised to the model.

    If `elicit_reasoning` is set, every tool gains a required `thought` argument so the
    model must verbalize WHY it takes each step (RQ Q3). This makes the per-step
    reasoning observable in the trace (tool_args) without changing the graph structure.

    If `include_sub_agent` is set, the model is additionally offered `sub_agent` (RQ-N2):
    an opt-in branch tool that delegates a self-contained sub-question to a nested
    assistant. Emitting several `sub_agent` calls in one turn lets the graph fan out into
    a real tree; the executor runs them concurrently so `width_executed` can exceed 1.
    """
    schemas = [
        {
            "type": "function",
            "function": {
                "name": SEARCH,
                "description": (
                    "Search the document store for documents relevant to a query. "
                    f"Returns up to {search_top_k} results, each with a doc_id, title, and snippet."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Keywords or a short question."}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": READ_DOCUMENT,
                "description": "Read the full text of one document by its doc_id (e.g. 'D03').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string", "description": "A doc_id returned by search."}
                    },
                    "required": ["doc_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": FINISH,
                "description": (
                    "Provide the final answer once you have gathered enough information. "
                    "This ends the task."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "The final, concise answer."}
                    },
                    "required": ["answer"],
                },
            },
        },
    ]
    if include_sub_agent:
        schemas.append({
            "type": "function",
            "function": {
                "name": SUB_AGENT,
                "description": (
                    "Delegate a self-contained sub-question to a fresh research assistant that "
                    "has the same search/read tools and returns a short answer. Use it to "
                    "investigate several entities independently: emit ONE sub_agent call per "
                    "entity IN THE SAME TURN and they are researched in parallel, then combine "
                    "the answers."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "A single, self-contained sub-question to research.",
                        }
                    },
                    "required": ["question"],
                },
            },
        })
    if elicit_reasoning:
        for s in schemas:
            params = s["function"]["parameters"]
            # put `thought` first so the model reasons BEFORE choosing arguments
            params["properties"] = {
                "thought": {
                    "type": "string",
                    "description": "Briefly, why you are taking this step given what you know so far.",
                },
                **params["properties"],
            }
            params["required"] = ["thought"] + params["required"]
    return schemas


def parse_arguments(raw: str) -> dict:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}


def execute(name: str, args: dict, corpus: Corpus, *, search_top_k: int = 3) -> dict:
    """Run one tool. Returns a JSON-serializable result dict.

    `finish` is handled by the agent loop (it terminates the run); we still return a
    structured echo so the result can be traced uniformly.
    """
    if name == SEARCH:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "search requires a non-empty 'query'."}
        return {"results": corpus.search(query, top_k=search_top_k)}
    if name == READ_DOCUMENT:
        return corpus.read(str(args.get("doc_id", "")))
    if name == FINISH:
        return {"answer": str(args.get("answer", "")).strip()}
    return {"error": f"Unknown tool '{name}'. Valid tools: {', '.join(TOOL_NAMES)}."}
