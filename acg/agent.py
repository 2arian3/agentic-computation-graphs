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
        tool_schemas = TOOLS.tool_schemas(self.cfg.search_top_k, self.cfg.elicit_reasoning)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
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

                step_tool_nodes: list[str] = []
                finished = False
                parent_ctx = self.client.context_for(result.span_context)
                for idx, tc in enumerate(result.tool_calls):
                    name = tc.function.name
                    args = TOOLS.parse_arguments(tc.function.arguments)
                    tool_node_id = f"tool:{step}:{idx}"
                    with self.tracer.start_as_current_span(
                        f"execute_tool {name}", context=parent_ctx, kind=SpanKind.INTERNAL
                    ) as tspan:
                        tspan.set_attribute(T.ACG_NODE_TYPE, T.NODE_TYPE_TOOL)
                        tspan.set_attribute(T.ACG_NODE_ID, tool_node_id)
                        tspan.set_attribute(T.GEN_AI_OPERATION_NAME, "execute_tool")
                        tspan.set_attribute(T.GEN_AI_TOOL_NAME, name)
                        tspan.set_attribute(T.GEN_AI_TOOL_CALL_ID, tc.id or "")
                        tspan.set_attribute(T.ACG_RUN_ID, run_id)
                        tspan.set_attribute(T.ACG_TASK_ID, task.task_id)
                        tspan.set_attribute(T.ACG_STEP, step)
                        tspan.set_attribute(T.ACG_DEPENDS_ON, json.dumps([result.node_id]))
                        tspan.set_attribute(T.ACG_TOOL_ARGS, json.dumps(args)[:2000])

                        result_obj = TOOLS.execute(name, args, self.corpus, search_top_k=self.cfg.search_top_k)
                        result_json = json.dumps(result_obj)
                        tspan.set_attribute(T.ACG_TOOL_RESULT_CHARS, len(result_json))
                        tspan.add_event("acg.tool.result", {"content": result_json[:8000]})

                    num_tool += 1
                    tool_call_names.append(name)
                    step_tool_nodes.append(tool_node_id)

                    # Feed the tool result back into the transcript.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_json,
                    })

                    if name == TOOLS.FINISH:
                        answer = str(args.get("answer", "")).strip() or None
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
