"""The thin, emergent agent loop.

Decision 2: we do NOT use LangGraph or any pre-built static workflow. We give the
model a fixed tool set and a minimal loop -- ask the model what to do, run the tool it
asks for, feed the result back, repeat until it stops -- and the graph STRUCTURE
emerges from the model's own decisions. The whole loop is a few dozen lines and is
fully instrumented, so the parent/child span tree it emits is exactly the Agentic
Computation Graph we want to measure.

Span structure emitted per run (one OTel trace):

    agent.run                      (root; node type = agent_run)
      └─ chat (step 0)             (node type = llm_call)
           ├─ execute_tool search  (node type = tool_call)
           └─ execute_tool read    (node type = tool_call)
      └─ chat (step 1)
           └─ execute_tool finish
      └─ ...

Tool spans are parented under the LLM call that emitted them (llm -> tool edges).
Each LLM call records `acg.depends_on` = the tool nodes from the previous step whose
results it consumes (tool -> llm edges). acg/graph.py turns this into the ACG DAG.
"""
from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from opentelemetry.trace import SpanKind

from . import tools as TOOLS
from . import tracing as T
from .config import Config
from .corpus import Corpus
from .llm_client import TracedLLMClient
from .tasks import Task, check_answer

SYSTEM_PROMPT = (
    "You are a research assistant that answers multi-hop questions using ONLY a private "
    "document store, accessed through the provided tools. The world in these documents is "
    "fictional, so you must NOT rely on prior knowledge -- every fact in your answer must "
    "come from a document you read.\n\n"
    "Work step by step:\n"
    "1. Use `search` to find documents relevant to the current sub-question.\n"
    "2. Use `read_document` to read a document's full text by its doc_id.\n"
    "3. Chain across documents: the answer to one hop tells you what to look up next.\n"
    "4. When you have gathered enough facts, call `finish` with a short, concise answer.\n\n"
    "Prefer reading the specific documents you need over guessing. Do not call `finish` "
    "until you can justify the answer from documents you have read."
)

# Appended to the system prompt only when the sub_agent branch tool is enabled (RQ-N2).
# It tells the model the fan-out capability exists; whether it uses it is the measurement.
SUB_AGENT_HINT = (
    "\n\nYou also have a `sub_agent` tool that delegates a self-contained sub-question to a "
    "fresh assistant (with the same search/read tools) and returns its answer. When the "
    "question compares or aggregates over several entities, emit ONE `sub_agent` call per "
    "entity IN THE SAME TURN so they are researched in parallel, then combine their answers "
    "and `finish`."
)

# A nested sub_agent researches ONE sub-question and returns a short answer.
SUB_AGENT_SYSTEM_PROMPT = (
    "You are a research assistant answering ONE focused sub-question using ONLY a private, "
    "fictional document store via the provided tools. Use `search` then `read_document` to "
    "gather the facts, then call `finish` with a short, concise answer. Do not rely on prior "
    "knowledge."
)


@dataclass
class RunResult:
    run_id: str
    task_id: str
    trace_id: str | None
    answer: str | None
    correct: bool
    outcome: str                  # "correct" | "incorrect" | "no_answer"
    num_steps: int
    num_llm_calls: int
    num_tool_calls: int
    total_input_tokens: int
    total_output_tokens: int
    wall_clock_s: float
    tool_call_names: list = field(default_factory=list)


class Agent:
    def __init__(self, config: Config, corpus: Corpus):
        self.cfg = config
        self.corpus = corpus
        self.client = TracedLLMClient(config)
        self.tracer = T.get_tracer()

    def run(self, task: Task, run_id: str | None = None) -> RunResult:
        run_id = run_id or uuid.uuid4().hex[:12]
        root_node_id = f"run:{run_id}"
        tool_schemas = TOOLS.tool_schemas(
            self.cfg.search_top_k, self.cfg.elicit_reasoning,
            include_sub_agent=self.cfg.enable_sub_agent,
        )
        system_content = SYSTEM_PROMPT + (SUB_AGENT_HINT if self.cfg.enable_sub_agent else "")

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Question: {task.question}"},
        ]

        num_llm = num_tool = 0
        in_tok = out_tok = 0
        tool_call_names: list[str] = []
        answer: str | None = None
        t0 = time.time()

        with self.tracer.start_as_current_span("agent.run", kind=SpanKind.INTERNAL) as root:
            trace_id = f"{root.get_span_context().trace_id:032x}"
            root.set_attribute(T.ACG_NODE_TYPE, T.NODE_TYPE_AGENT_RUN)
            root.set_attribute(T.ACG_NODE_ID, root_node_id)
            root.set_attribute(T.ACG_RUN_ID, run_id)
            root.set_attribute(T.ACG_TASK_ID, task.task_id)
            root.set_attribute("acg.question", task.question)

            depends_on = [root_node_id]   # what the next LLM call consumes
            step = 0
            for step in range(self.cfg.max_steps):
                result = self.client.chat(
                    messages, tool_schemas,
                    step=step, depends_on=depends_on, run_id=run_id, task_id=task.task_id,
                )
                num_llm += 1
                in_tok += result.input_tokens
                out_tok += result.output_tokens

                if not result.tool_calls:
                    # No tool call -> the model's content is its final answer.
                    answer = (result.content or "").strip() or None
                    break

                # Append the assistant turn (with tool calls) to the transcript.
                messages.append({
                    "role": "assistant",
                    "content": result.content or "",
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                        for tc in result.tool_calls
                    ],
                })

                # Execute this step's emitted tool calls. The model may issue several in
                # one turn; we run them CONCURRENTLY (up to cfg.max_tool_workers) so that
                # span wall-clock times reflect real parallelism -- the precondition for
                # measuring *executed* width honestly (see acg/graph.py). Records come back
                # in emit order, so the transcript (and thus the run) is byte-reproducible
                # regardless of which tool finished first.
                parent_ctx = self.client.context_for(result.span_context)
                records = self._execute_tool_calls(
                    list(result.tool_calls), step=step, parent_ctx=parent_ctx,
                    llm_node_id=result.node_id, run_id=run_id, task_id=task.task_id,
                )

                step_tool_nodes: list[str] = []
                finished = False
                for r in records:
                    num_tool += 1
                    tool_call_names.append(r["name"])
                    step_tool_nodes.append(r["tool_node_id"])
                    # Feed the tool result back into the transcript (emit order).
                    messages.append({
                        "role": "tool",
                        "tool_call_id": r["tc_id"],
                        "content": r["result_json"],
                    })
                    if r["is_finish"]:
                        answer = r["answer"]
                        finished = True

                if finished:
                    break
                # Next LLM call depends on this step's tool results.
                depends_on = step_tool_nodes or [result.node_id]

            num_steps = step + 1
            correct = check_answer(answer, task.answers)
            outcome = "correct" if correct else ("incorrect" if answer else "no_answer")
            root.set_attribute(T.ACG_OUTCOME, outcome)
            root.set_attribute("acg.answer", answer or "")
            root.set_attribute("acg.num_llm_calls", num_llm)
            root.set_attribute("acg.num_tool_calls", num_tool)

        wall = time.time() - t0
        return RunResult(
            run_id=run_id, task_id=task.task_id, trace_id=trace_id,
            answer=answer, correct=correct, outcome=outcome,
            num_steps=num_steps, num_llm_calls=num_llm, num_tool_calls=num_tool,
            total_input_tokens=in_tok, total_output_tokens=out_tok,
            wall_clock_s=wall, tool_call_names=tool_call_names,
        )

    # ------------------------------------------------------------------ #
    # Tool execution
    # ------------------------------------------------------------------ #
    def _execute_tool_calls(
        self, calls: list, *, step: int, parent_ctx, llm_node_id: str,
        run_id: str, task_id: str,
    ) -> list[dict]:
        """Run one step's emitted tool calls, returning per-call records in EMIT order.

        With cfg.max_tool_workers > 1 and more than one call, the calls run on a thread
        pool so their spans overlap in wall-clock time when the tools do concurrent work;
        a single call (or max_tool_workers == 1) runs inline. Records are re-sorted by emit
        index so the transcript -- and therefore the whole run -- stays reproducible no
        matter which tool finishes first.
        """
        indexed = list(enumerate(calls))
        workers = min(len(indexed), max(1, self.cfg.max_tool_workers))
        if workers > 1 and len(indexed) > 1:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [
                    ex.submit(
                        self._execute_one_tool, tc, step=step, idx=idx,
                        parent_ctx=parent_ctx, llm_node_id=llm_node_id,
                        run_id=run_id, task_id=task_id,
                    )
                    for idx, tc in indexed
                ]
                records = [f.result() for f in futures]
        else:
            records = [
                self._execute_one_tool(
                    tc, step=step, idx=idx, parent_ctx=parent_ctx,
                    llm_node_id=llm_node_id, run_id=run_id, task_id=task_id,
                )
                for idx, tc in indexed
            ]
        records.sort(key=lambda r: r["idx"])
        return records

    def _execute_one_tool(
        self, tc, *, step: int, idx: int, parent_ctx, llm_node_id: str,
        run_id: str, task_id: str, node_prefix: str = "", allow_sub_agent: bool = True,
    ) -> dict:
        """Execute a single tool call inside its own span and return a plain record.

        Pure with respect to agent state: it touches no shared counters or transcript, so
        it is safe to call from worker threads. The caller applies the returned records in
        emit order. Each tool span is parented under the LLM call (llm_node_id) through the
        explicit parent context, which is what keeps cross-thread parenting correct.

        `node_prefix` namespaces the node id (used by nested sub_agents so their subtree
        does not collide with the parent). `sub_agent` (when allowed) runs a nested assistant
        whose spans hang under this tool span, turning the ACG into a real tree.
        """
        name = tc.function.name
        args = TOOLS.parse_arguments(tc.function.arguments)
        tool_node_id = f"{node_prefix}tool:{step}:{idx}"
        with self.tracer.start_as_current_span(
            f"execute_tool {name}", context=parent_ctx, kind=SpanKind.INTERNAL
        ) as tspan:
            tspan.set_attribute(T.ACG_NODE_TYPE, T.NODE_TYPE_TOOL)
            tspan.set_attribute(T.ACG_NODE_ID, tool_node_id)
            tspan.set_attribute(T.GEN_AI_OPERATION_NAME, "execute_tool")
            tspan.set_attribute(T.GEN_AI_TOOL_NAME, name)
            tspan.set_attribute(T.GEN_AI_TOOL_CALL_ID, tc.id or "")
            tspan.set_attribute(T.ACG_RUN_ID, run_id)
            tspan.set_attribute(T.ACG_TASK_ID, task_id)
            tspan.set_attribute(T.ACG_STEP, step)
            tspan.set_attribute(T.ACG_DEPENDS_ON, json.dumps([llm_node_id]))
            tspan.set_attribute(T.ACG_TOOL_ARGS, json.dumps(args)[:2000])

            if name == TOOLS.SUB_AGENT and allow_sub_agent:
                result_obj = self._run_sub_agent(
                    str(args.get("question", "")).strip(),
                    prefix=f"{tool_node_id}/", parent_node_id=tool_node_id,
                    run_id=run_id, task_id=task_id,
                )
            else:
                result_obj = TOOLS.execute(name, args, self.corpus, search_top_k=self.cfg.search_top_k)
            result_json = json.dumps(result_obj)
            tspan.set_attribute(T.ACG_TOOL_RESULT_CHARS, len(result_json))
            tspan.add_event("acg.tool.result", {"content": result_json[:8000]})

        is_finish = name == TOOLS.FINISH
        answer = (str(args.get("answer", "")).strip() or None) if is_finish else None
        return {
            "idx": idx,
            "tool_node_id": tool_node_id,
            "tc_id": tc.id,
            "name": name,
            "result_json": result_json,
            "is_finish": is_finish,
            "answer": answer,
        }

    def _run_sub_agent(self, question: str, *, prefix: str, parent_node_id: str,
                       run_id: str, task_id: str) -> dict:
        """Run a nested, corpus-grounded assistant on ONE sub-question.

        Emits a namespaced subtree (node ids prefixed by the sub_agent tool node) in the
        SAME trace, so acg/graph.py reconstructs the ACG as a real tree/DAG. Sub-agents are
        not offered `sub_agent` themselves, so nesting is exactly one level deep. Tools here
        run serially; the parallelism we measure is across sibling sub_agents at the parent
        level (their tool spans overlap in wall-clock time -> width_executed > 1).
        """
        sub_tools = TOOLS.tool_schemas(
            self.cfg.search_top_k, self.cfg.elicit_reasoning, include_sub_agent=False,
        )
        messages = [
            {"role": "system", "content": SUB_AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {question}"},
        ]
        depends_on = [parent_node_id]
        answer: str | None = None
        for s in range(self.cfg.sub_agent_max_steps):
            result = self.client.chat(
                messages, sub_tools, step=s, depends_on=depends_on,
                run_id=run_id, task_id=task_id, node_id=f"{prefix}llm:{s}",
            )
            if not result.tool_calls:
                answer = (result.content or "").strip() or None
                break
            messages.append({
                "role": "assistant",
                "content": result.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in result.tool_calls
                ],
            })
            parent_ctx = self.client.context_for(result.span_context)
            step_nodes: list[str] = []
            finished = False
            for idx, tc in enumerate(result.tool_calls):
                rec = self._execute_one_tool(
                    tc, step=s, idx=idx, parent_ctx=parent_ctx, llm_node_id=result.node_id,
                    run_id=run_id, task_id=task_id, node_prefix=prefix, allow_sub_agent=False,
                )
                step_nodes.append(rec["tool_node_id"])
                messages.append({"role": "tool", "tool_call_id": rec["tc_id"], "content": rec["result_json"]})
                if rec["is_finish"]:
                    answer = rec["answer"]
                    finished = True
            if finished:
                break
            depends_on = step_nodes or [result.node_id]
        return {"sub_question": question, "answer": answer or ""}
