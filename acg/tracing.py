"""OpenTelemetry tracing for the agent.

We capture each graph using OpenTelemetry GenAI spans -- the emerging standard for
tracing LLM and agent calls. Every LLM call and every tool call becomes a span with
parent/child links, token counts, and timing. The agent's parent/child span tree is
exactly the graph we want, so we get structure AND metrics from one instrumentation
layer in a standard format others can read.

Measurement is kept SEPARATE from execution: spans are exported to a local JSONL
store, and the graph is reconstructed offline (acg/graph.py). That way analysis bugs
never affect the runs.

We deliberately avoid a network OTLP collector -- a custom SpanExporter that appends
one JSON object per finished span to a file is dependency-free, fully owned, and
trivially replayable. The schema mirrors the OTLP span model (trace_id, span_id,
parent_span_id, name, kind, start/end ns, attributes, status, events), so the same
traces could be pushed to a real collector later without changing the agent.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Sequence

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, SimpleSpanProcessor

# ---------------------------------------------------------------------------
# GenAI semantic-convention attribute keys (a subset of the OTel GenAI spec)
# plus a small `acg.*` namespace for the things the spec does not yet cover
# (explicit data-dependency edges, loop step index, node type).
# ---------------------------------------------------------------------------
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
GEN_AI_REQUEST_TOP_P = "gen_ai.request.top_p"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_REQUEST_SEED = "gen_ai.request.seed"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_TOOL_CALL_ID = "gen_ai.tool.call.id"

# acg.* extensions ----------------------------------------------------------
ACG_NODE_TYPE = "acg.node.type"            # "agent_run" | "llm_call" | "tool_call"
ACG_RUN_ID = "acg.run_id"
ACG_TASK_ID = "acg.task_id"
ACG_STEP = "acg.step"                      # loop iteration index
ACG_DEPENDS_ON = "acg.depends_on"          # JSON list of node-ids this node consumes
ACG_NODE_ID = "acg.node_id"                # stable, human-readable node id
ACG_TOOL_ARGS = "acg.tool.args"            # JSON of tool arguments (truncated)
ACG_TOOL_RESULT_CHARS = "acg.tool.result_chars"
ACG_OUTCOME = "acg.outcome"                # on the root: "correct" | "incorrect" | "no_answer"

NODE_TYPE_AGENT_RUN = "agent_run"
NODE_TYPE_LLM = "llm_call"
NODE_TYPE_TOOL = "tool_call"

_TRACER_NAME = "acg.agent"
_lock = threading.Lock()


class JSONLFileSpanExporter(SpanExporter):
    """Appends each finished span as a single JSON line to a file."""

    def __init__(self, path: str | Path):
        self._fh = None
        self.set_path(path)

    def set_path(self, path: str | Path) -> None:
        """(Re)direct output to a new file. Lets one provider serve many runs/tests,
        which matters because OTel only honors set_tracer_provider once per process."""
        with _lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
            self.path = Path(path)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a", encoding="utf-8")

    @staticmethod
    def _span_to_dict(span: ReadableSpan) -> dict:
        ctx = span.get_span_context()
        parent = span.parent
        return {
            "trace_id": f"{ctx.trace_id:032x}",
            "span_id": f"{ctx.span_id:016x}",
            "parent_span_id": (f"{parent.span_id:016x}" if parent else None),
            "name": span.name,
            "kind": str(span.kind),
            "start_time_ns": span.start_time,
            "end_time_ns": span.end_time,
            "duration_ns": (span.end_time - span.start_time)
            if (span.end_time and span.start_time) else None,
            "status": str(span.status.status_code),
            "attributes": dict(span.attributes or {}),
            "events": [
                {"name": e.name, "timestamp_ns": e.timestamp, "attributes": dict(e.attributes or {})}
                for e in (span.events or [])
            ],
            "resource": dict(span.resource.attributes or {}),
        }

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            with _lock:
                for span in spans:
                    self._fh.write(json.dumps(self._span_to_dict(span), default=str) + "\n")
                self._fh.flush()
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


_provider: TracerProvider | None = None
_exporter: JSONLFileSpanExporter | None = None


def configure_tracing(trace_file: str | Path, service_name: str = "acg-agent"):
    """Install (once) a TracerProvider that writes GenAI spans to `trace_file` (JSONL).

    Idempotent: the first call creates the provider + exporter; later calls just
    redirect the exporter to a new file. This is required because OTel only honors
    set_tracer_provider once per process, yet we want each run/test to land in its
    own trace file. Uses a SimpleSpanProcessor (synchronous) so spans flush in-process
    and in causal order -- important for a clean, replayable trace store.
    """
    global _provider, _exporter
    if _provider is None:
        resource = Resource.create({"service.name": service_name})
        _provider = TracerProvider(resource=resource)
        _exporter = JSONLFileSpanExporter(trace_file)
        _provider.add_span_processor(SimpleSpanProcessor(_exporter))
        trace.set_tracer_provider(_provider)
    else:
        _exporter.set_path(trace_file)
    return trace.get_tracer(_TRACER_NAME)


def get_tracer():
    return trace.get_tracer(_TRACER_NAME)


def flush_tracing():
    """Force-flush spans without tearing down the provider (safe between runs/tests)."""
    if _provider is not None:
        try:
            _provider.force_flush()
        except Exception:
            pass


def shutdown_tracing():
    """Flush and close the exporter. Call once at process end."""
    global _provider
    if _provider is not None:
        try:
            _provider.shutdown()
        finally:
            _provider = None
