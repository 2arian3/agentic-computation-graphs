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

TOOL_NAMES = (SEARCH, READ_DOCUMENT, FINISH)


def tool_schemas(search_top_k: int = 3) -> list[dict]:
    """OpenAI-style function/tool schemas advertised to the model."""
    return [
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
