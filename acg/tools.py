"""The tool alphabet (node types for tool nodes in the ACG).

Decision 2: because our entire question is about the structure the *model* generates,
we give it a small, fixed, countable tool set and let the loop be emergent. Fixing the
tool alphabet is a precondition for comparing graphs across runs -- every tool node in
every graph is one of a known, finite set of types.

Tools are pure functions of (args, corpus). They contain NO tracing and NO control
flow; the agent loop (acg/agent.py) owns the loop and the spans. This keeps the node
alphabet and the execution structure cleanly separated.

Two tiers:
  * CORE  -- search / read_document / finish : the canonical retrieval loop.
  * EXT   -- opt-in tools that INDUCE STRUCTURAL DIVERSITY without changing the loop:
      sub_agent    -> agent fan-out (width)                       (RQ-N2, already present)
      calculator   -> a computation node                         (numeric/temporal tasks)
      compare      -> a fan-out -> aggregate join node (width)    (superlative/count tasks)
      verify_claim -> a verifier node (validity, retry pressure)  (constraint tasks)
      decompose    -> an explicit planner node                    (deeper graphs)
  Each EXT tool is still a pure function of (args, corpus); the *task design* decides
  whether the model needs it, so different task families realize different graph shapes.
  EXT tools are OFF unless requested, so the canonical 3-tool experiments are unchanged.
"""
from __future__ import annotations

import ast
import json
import operator as _op
from typing import Any

from .corpus import Corpus

# ---- CORE tool names (node types) --------------------------------------------
SEARCH = "search"
READ_DOCUMENT = "read_document"
FINISH = "finish"
# ---- EXT tool names ----------------------------------------------------------
SUB_AGENT = "sub_agent"
CALCULATOR = "calculator"
COMPARE = "compare"
VERIFY = "verify_claim"
DECOMPOSE = "decompose"

# The core alphabet used by the canonical experiments.
TOOL_NAMES = (SEARCH, READ_DOCUMENT, FINISH)
# The extended alphabet including the opt-in branch tool (RQ-N2).
TOOL_NAMES_EXT = (SEARCH, READ_DOCUMENT, FINISH, SUB_AGENT)
# Opt-in EXT tools selectable via config.extra_tools / ACG_EXTRA_TOOLS.
EXTRA_TOOL_NAMES = (SUB_AGENT, CALCULATOR, COMPARE, VERIFY, DECOMPOSE)


# ------------------------------------------------------------------------------
# Tool schemas advertised to the model
# ------------------------------------------------------------------------------
def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _ext_schema(name: str) -> dict:
    if name == SUB_AGENT:
        return _schema(
            SUB_AGENT,
            ("Delegate a self-contained sub-question to a fresh research assistant that has the "
             "same search/read tools and returns a short answer. Use it to investigate several "
             "entities independently: emit ONE sub_agent call per entity IN THE SAME TURN and they "
             "are researched in parallel, then combine the answers."),
            {"question": {"type": "string", "description": "A single, self-contained sub-question to research."}},
            ["question"],
        )
    if name == CALCULATOR:
        return _schema(
            CALCULATOR,
            ("Evaluate an arithmetic expression (e.g. '2020 - 1962' or '(3+5)/2'). Use it for any "
             "numeric or date computation instead of doing mental math. Returns the numeric result."),
            {"expression": {"type": "string",
                            "description": "An arithmetic expression using numbers and + - * / ( ) . % ** //"}},
            ["expression"],
        )
    if name == COMPARE:
        return _schema(
            COMPARE,
            ("Aggregate over a list of gathered items to pick or count them. Provide `items` as a "
             "list of {label, value} and an `op`. op='max'/'min' returns the label with the "
             "largest/smallest numeric value; op='count' returns how many items; op='sum' returns "
             "the total. Call this AFTER you have gathered the per-entity values (e.g. via several "
             "searches or sub_agents)."),
            {
                "items": {
                    "type": "array",
                    "description": "The gathered items to aggregate.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "value": {"type": ["number", "string"]},
                        },
                    },
                },
                "op": {"type": "string", "enum": ["max", "min", "count", "sum"],
                       "description": "The aggregation operation."},
            },
            ["items", "op"],
        )
    if name == VERIFY:
        return _schema(
            VERIFY,
            ("Check whether a factual claim is supported by the document store before you rely on "
             "it. Returns supported=true/false with the best supporting doc_id and snippet. Use it "
             "to validate an intermediate conclusion before you finish."),
            {"claim": {"type": "string", "description": "A single factual claim to verify against the documents."}},
            ["claim"],
        )
    if name == DECOMPOSE:
        return _schema(
            DECOMPOSE,
            ("Record your plan: break the question into an ordered list of sub-questions you will "
             "answer. This does not fetch anything; it makes your plan explicit before you act."),
            {"subquestions": {"type": "array", "items": {"type": "string"},
                              "description": "The ordered sub-questions you will resolve."}},
            ["subquestions"],
        )
    raise ValueError(f"unknown extra tool '{name}'")


def tool_schemas(search_top_k: int = 3, elicit_reasoning: bool = False,
                 include_sub_agent: bool = False,
                 extra_tools: tuple[str, ...] = ()) -> list[dict]:
    """OpenAI-style function/tool schemas advertised to the model.

    If `elicit_reasoning` is set, every tool gains a required `thought` argument so the
    model must verbalize WHY it takes each step (RQ Q3), making per-step reasoning
    observable in the trace without changing the graph structure.

    `include_sub_agent` (kept for backwards compatibility) adds the sub_agent branch tool.
    `extra_tools` is the general opt-in mechanism: any subset of EXTRA_TOOL_NAMES to
    advertise alongside the core alphabet (used by the richer task families). Order is
    stable; duplicates and sub_agent double-adds are de-duplicated.
    """
    schemas = [
        _schema(SEARCH,
                (f"Search the document store for documents relevant to a query. Returns up to "
                 f"{search_top_k} results, each with a doc_id, title, and snippet."),
                {"query": {"type": "string", "description": "Keywords or a short question."}},
                ["query"]),
        _schema(READ_DOCUMENT,
                "Read the full text of one document by its doc_id (e.g. 'D03').",
                {"doc_id": {"type": "string", "description": "A doc_id returned by search."}},
                ["doc_id"]),
        _schema(FINISH,
                ("Provide the final answer once you have gathered enough information. This ends the "
                 "task. If the documents do not contain the answer, finish with 'insufficient information'."),
                {"answer": {"type": "string", "description": "The final, concise answer."}},
                ["answer"]),
    ]

    # Resolve the opt-in EXT set (include_sub_agent is folded in for compatibility).
    wanted: list[str] = []
    if include_sub_agent:
        wanted.append(SUB_AGENT)
    for t in extra_tools:
        if t in EXTRA_TOOL_NAMES and t not in wanted:
            wanted.append(t)
    for name in wanted:
        schemas.append(_ext_schema(name))

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


# ------------------------------------------------------------------------------
# Safe arithmetic for the calculator tool (no eval())
# ------------------------------------------------------------------------------
_ALLOWED_BINOPS = {
    ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul, ast.Div: _op.truediv,
    ast.FloorDiv: _op.floordiv, ast.Mod: _op.mod, ast.Pow: _op.pow,
}
_ALLOWED_UNARY = {ast.UAdd: _op.pos, ast.USub: _op.neg}


def _safe_eval(expr: str):
    """Evaluate a pure arithmetic expression via AST whitelist. Raises on anything else."""
    def _e(node):
        if isinstance(node, ast.Expression):
            return _e(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](_e(node.left), _e(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](_e(node.operand))
        raise ValueError("unsupported expression")
    return _e(ast.parse(expr, mode="eval"))


def _to_number(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


# ------------------------------------------------------------------------------
# Tool execution
# ------------------------------------------------------------------------------
def execute(name: str, args: dict, corpus: Corpus, *, search_top_k: int = 3) -> dict:
    """Run one tool. Returns a JSON-serializable result dict.

    `finish` is handled by the agent loop (it terminates the run); we still return a
    structured echo so the result can be traced uniformly.
    """
    # ---- CORE ----
    if name == SEARCH:
        query = str(args.get("query", "")).strip()
        if not query:
            return {"error": "search requires a non-empty 'query'."}
        return {"results": corpus.search(query, top_k=search_top_k)}
    if name == READ_DOCUMENT:
        return corpus.read(str(args.get("doc_id", "")))
    if name == FINISH:
        return {"answer": str(args.get("answer", "")).strip()}

    # ---- EXT ----
    if name == CALCULATOR:
        expr = str(args.get("expression", "")).strip()
        try:
            return {"expression": expr, "result": _safe_eval(expr)}
        except Exception:
            return {"error": f"could not evaluate expression '{expr}'"}

    if name == COMPARE:
        op = str(args.get("op", "")).strip().lower()
        items = args.get("items") or []
        if not isinstance(items, list) or not items:
            return {"error": "compare requires a non-empty 'items' list."}
        if op == "count":
            return {"op": op, "result": len(items)}
        pairs = []
        for it in items:
            if isinstance(it, dict):
                pairs.append((str(it.get("label", "")), _to_number(it.get("value"))))
        numeric = [(lbl, val) for lbl, val in pairs if val is not None]
        if op in ("max", "min"):
            if not numeric:
                return {"error": "compare max/min needs numeric 'value' fields."}
            chosen = (max if op == "max" else min)(numeric, key=lambda p: p[1])
            return {"op": op, "result": chosen[0], "value": chosen[1]}
        if op == "sum":
            return {"op": op, "result": sum(v for _, v in numeric)}
        return {"error": f"unknown op '{op}'. Use max|min|count|sum."}

    if name == VERIFY:
        claim = str(args.get("claim", "")).strip()
        if not claim:
            return {"error": "verify_claim requires a non-empty 'claim'."}
        hits = corpus.search(claim, top_k=1)
        if not hits:
            return {"claim": claim, "supported": False, "evidence": None}
        top = hits[0]
        return {"claim": claim, "supported": True,
                "evidence": {"doc_id": top["doc_id"], "title": top["title"], "snippet": top["snippet"]}}

    if name == DECOMPOSE:
        subs = args.get("subquestions") or []
        subs = [str(s) for s in subs] if isinstance(subs, list) else [str(subs)]
        return {"plan": subs, "n": len(subs)}

    return {"error": f"Unknown tool '{name}'. Valid tools: {', '.join(TOOL_NAMES)}."}
