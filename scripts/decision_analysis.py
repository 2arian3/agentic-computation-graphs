#!/usr/bin/env python3
"""RQ-C2: what inputs make the agent FINISH (short-circuit) vs CONTINUE vs re-read (loop)?

For every decision point (each tool call in each run) we reconstruct what the model knew at
that moment and correlate it with the decision it made. Central test: does the model finish
BECAUSE the answer is already in its context — or does it finish prematurely (short-circuit)?

Feature at each decision: `context_has_answer` = whether a gold-answer alias has appeared in
ANY tool result seen so far in this run (a proxy for "the answer is available").
Decision: finish vs continue (search/read); plus re-read (loop) detection.

  ./.venv/bin/python scripts/decision_analysis.py --trace traces/rqc2_elicited.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acg import graph as G
from acg import tracing as T
from acg.tasks import load_tasks, _normalize
from acg.config import load_config


def _gold_in(text: str, golds: list[str]) -> bool:
    p = _normalize(text)
    return any(_normalize(g) in p for g in golds if g)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="traces/rqc2_elicited.jsonl")
    ap.add_argument("--examples", type=int, default=4)
    args = ap.parse_args()

    golds = {t.task_id: t.answers for t in load_tasks(load_config().tasks_path)}
    by_trace = G.group_by_trace(G.load_spans(args.trace))

    decisions = []          # one record per decision point
    premature_examples, goodfinish_examples = [], []

    for trace_id, spans in by_trace.items():
        root = next((s for s in spans if (s.get("attributes") or {}).get(T.ACG_NODE_TYPE) == T.NODE_TYPE_AGENT_RUN), None)
        if not root:
            continue
        ra = root["attributes"]
        task_id = ra.get(T.ACG_TASK_ID); outcome = ra.get(T.ACG_OUTCOME)
        gold = golds.get(task_id, [])
        tool_spans = sorted(
            [s for s in spans if (s.get("attributes") or {}).get(T.ACG_NODE_TYPE) == T.NODE_TYPE_TOOL],
            key=lambda s: s["attributes"].get(T.ACG_STEP, 0))

        context_has_answer = False
        read_docs = set()
        for ts in tool_spans:
            a = ts["attributes"]
            tool = a.get(T.GEN_AI_TOOL_NAME)
            try:
                targs = json.loads(a.get(T.ACG_TOOL_ARGS) or "{}")
            except Exception:
                targs = {}
            thought = targs.get("thought", "")
            doc_id = str(targs.get("doc_id", "")).strip().upper()
            decision = "finish" if tool == "finish" else "continue"
            is_reread = tool == "read_document" and doc_id in read_docs

            rec = {"task": task_id, "step": a.get(T.ACG_STEP), "tool": tool,
                   "decision": decision, "ctx_has_answer": context_has_answer,
                   "is_reread": is_reread, "outcome": outcome, "thought": thought}
            decisions.append(rec)
            if decision == "finish":
                (goodfinish_examples if context_has_answer else premature_examples).append(rec)

            # update context with this step's RESULT (search/read only)
            result_text = ""
            for e in ts.get("events", []):
                if e["name"] == "acg.tool.result":
                    result_text = e.get("attributes", {}).get("content", "")
            if tool in ("search", "read_document"):
                if _gold_in(result_text, gold):
                    context_has_answer = True
                if tool == "read_document" and doc_id:
                    read_docs.add(doc_id)

    # ---- aggregate ----
    finishes = [d for d in decisions if d["decision"] == "finish"]
    conts = [d for d in decisions if d["decision"] == "continue"]

    def rate(sub, cond):
        s = [d for d in sub if cond(d)]
        return len(s)

    n_fin_has = rate(finishes, lambda d: d["ctx_has_answer"])
    n_fin_no = rate(finishes, lambda d: not d["ctx_has_answer"])
    n_cont_has = rate(conts, lambda d: d["ctx_has_answer"])
    n_cont_no = rate(conts, lambda d: not d["ctx_has_answer"])

    print(f"analyzed {len(by_trace)} runs, {len(decisions)} decision points "
          f"({len(finishes)} finish, {len(conts)} continue)\n")
    print("== Decision vs whether the answer was already in context ==")
    from tabulate import tabulate
    print(tabulate([
        ["answer IN context", n_fin_has, n_cont_has],
        ["answer NOT in context", n_fin_no, n_cont_no],
    ], headers=["", "FINISH", "CONTINUE"], tablefmt="github"))

    p_fin_has = n_fin_has / max(n_fin_has + n_cont_has, 1)
    p_fin_no = n_fin_no / max(n_fin_no + n_cont_no, 1)
    print(f"\nP(finish | answer in context)     = {p_fin_has:.2f}")
    print(f"P(finish | answer NOT in context) = {p_fin_no:.2f}   <- short-circuit tendency")

    # premature finishes = finish without the answer in context
    prem = [d for d in finishes if not d["ctx_has_answer"]]
    prem_wrong = sum(1 for d in prem if d["outcome"] != "correct")
    print(f"\nPremature finishes (finish w/o answer in context): {len(prem)}/{len(finishes)} finishes"
          f"  -> {prem_wrong}/{len(prem)} in runs that ended INCORRECT" if prem else
          "\nNo premature finishes.")
    over = [d for d in conts if d["ctx_has_answer"]]
    print(f"Over-continuation (kept going after answer was available): {len(over)} decisions (wasted steps)")

    rereads = [d for d in decisions if d["is_reread"]]
    rr_no = sum(1 for d in rereads if not d["ctx_has_answer"])
    print(f"Re-reads (loops): {len(rereads)}; of these {rr_no} happened while the answer was NOT yet in context")

    print("\n== Example model `thought` at PREMATURE finishes (why it short-circuited) ==")
    for d in premature_examples[:args.examples]:
        print(f"  [{d['task']} step{d['step']} -> {d['outcome']}] {d['thought'][:180]}")
    print("\n== Example `thought` at CORRECT finishes (answer was in context) ==")
    for d in goodfinish_examples[:2]:
        print(f"  [{d['task']} step{d['step']} -> {d['outcome']}] {d['thought'][:180]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
