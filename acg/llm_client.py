"""Traced, OpenAI-compatible LLM client.

This is the layer that "measures and traces the prompts we give the model". Every
chat completion is wrapped in an OpenTelemetry GenAI span that records:
  * the pinned request parameters (model, temperature, top_p, seed, max_tokens),
  * the exact prompt sent to the model (as a span event, so prompts are auditable),
  * the response (content + any tool calls),
  * token usage (input/output) and wall-clock latency.

Decode parameters and the seed are pinned from acg.config so the only thing that
changes run-to-run is sampling (Decision 1). We talk to a *local* vLLM/SGLang
OpenAI-compatible endpoint, never a closed product, during data collection.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode, NonRecordingSpan, SpanContext

from . import tracing as T
from .config import Config

# Keep trace files readable: cap how much prompt/response text we store per event.
_MAX_TEXT = 20000


def _truncate(s: str, limit: int = _MAX_TEXT) -> str:
    if s is None:
        return ""
    return s if len(s) <= limit else s[:limit] + f"...<+{len(s) - limit} chars>"


@dataclass
class LLMResult:
    message: Any                 # the OpenAI message object (may carry tool_calls)
    content: str | None
    tool_calls: list             # list of tool_call objects (possibly empty)
    finish_reason: str | None
    input_tokens: int
    output_tokens: int
    node_id: str                 # acg node id for this LLM call
    span_context: SpanContext    # so callers can parent tool spans under this call


class TracedLLMClient:
    def __init__(self, config: Config):
        self.cfg = config
        self.client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.tracer = T.get_tracer()

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        *,
        step: int,
        depends_on: list[str],
        run_id: str,
        task_id: str,
    ) -> LLMResult:
        node_id = f"llm:{step}"
        d = self.cfg.decode
        with self.tracer.start_as_current_span(
            f"chat {self.cfg.model}", kind=SpanKind.CLIENT
        ) as span:
            # --- request attributes (GenAI semantic conventions) ---
            span.set_attribute(T.GEN_AI_SYSTEM, self.cfg.gen_ai_system)
            span.set_attribute(T.GEN_AI_OPERATION_NAME, "chat")
            span.set_attribute(T.GEN_AI_REQUEST_MODEL, self.cfg.model)
            span.set_attribute(T.GEN_AI_REQUEST_TEMPERATURE, d.temperature)
            span.set_attribute(T.GEN_AI_REQUEST_TOP_P, d.top_p)
            span.set_attribute(T.GEN_AI_REQUEST_MAX_TOKENS, d.max_tokens)
            if d.seed is not None:
                span.set_attribute(T.GEN_AI_REQUEST_SEED, d.seed)
            # --- acg structural attributes ---
            span.set_attribute(T.ACG_NODE_TYPE, T.NODE_TYPE_LLM)
            span.set_attribute(T.ACG_NODE_ID, node_id)
            span.set_attribute(T.ACG_RUN_ID, run_id)
            span.set_attribute(T.ACG_TASK_ID, task_id)
            span.set_attribute(T.ACG_STEP, step)
            span.set_attribute(T.ACG_DEPENDS_ON, json.dumps(depends_on))

            # The prompt itself, traced for auditability.
            span.add_event("gen_ai.prompt", {"content": _truncate(json.dumps(messages))})

            req_kwargs = dict(
                model=self.cfg.model,
                messages=messages,
                **d.as_request_kwargs(),
            )
            if tools:
                req_kwargs["tools"] = tools
                req_kwargs["tool_choice"] = "auto"

            try:
                resp = self.client.chat.completions.create(**req_kwargs)
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise

            choice = resp.choices[0]
            msg = choice.message
            tool_calls = list(msg.tool_calls or [])
            finish_reason = choice.finish_reason
            usage = resp.usage
            in_tok = getattr(usage, "prompt_tokens", 0) or 0
            out_tok = getattr(usage, "completion_tokens", 0) or 0

            # --- response attributes ---
            span.set_attribute(T.GEN_AI_RESPONSE_MODEL, resp.model or self.cfg.model)
            span.set_attribute(T.GEN_AI_RESPONSE_FINISH_REASONS, [finish_reason or ""])
            span.set_attribute(T.GEN_AI_USAGE_INPUT_TOKENS, in_tok)
            span.set_attribute(T.GEN_AI_USAGE_OUTPUT_TOKENS, out_tok)
            span.set_attribute("acg.response.tool_call_count", len(tool_calls))

            completion_record = {
                "content": _truncate(msg.content or ""),
                "tool_calls": [
                    {
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": _truncate(tc.function.arguments or "", 4000),
                    }
                    for tc in tool_calls
                ],
            }
            span.add_event("gen_ai.completion", {"content": _truncate(json.dumps(completion_record))})

            span_ctx = span.get_span_context()

        return LLMResult(
            message=msg,
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            input_tokens=in_tok,
            output_tokens=out_tok,
            node_id=node_id,
            span_context=span_ctx,
        )

    def context_for(self, span_context: SpanContext):
        """Build an OTel context whose current span is `span_context`, so a child
        span (e.g. a tool call) can be parented under a *finished* LLM span."""
        return trace.set_span_in_context(NonRecordingSpan(span_context))

    # convenience: a plain completion for the determinism smoke test
    def raw_complete(self, messages: list[dict], **overrides) -> Any:
        kwargs = dict(model=self.cfg.model, messages=messages, **self.cfg.decode.as_request_kwargs())
        kwargs.update(overrides)
        return self.client.chat.completions.create(**kwargs)
