"""Live streaming of the agent's execution, built entirely on the existing tracing.

The instrument's `configure_tracing()` installs a SimpleSpanProcessor, which exports
each span the instant it ends. We attach ONE extra span processor to that same global
provider; its exporter normalizes every finished span and drops it onto the queue for
the run it belongs to (routed by the `acg.run_id` attribute). The agent loop is never
touched — remove this file and runs are byte-identical.

The same `normalize_span()` is reused for replay of archived traces, so the frontend has
a single event schema for both live and replayed runs.
"""
from __future__ import annotations

import json
import queue
import threading
from typing import Any, Optional, Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, SimpleSpanProcessor

from acg import tracing as T
from acg.tracing import JSONLFileSpanExporter


# --------------------------------------------------------------------------- #
# Per-run event queues
# --------------------------------------------------------------------------- #
class RunManager:
    """Holds a bounded event queue per active run, keyed by acg.run_id."""

    def __init__(self) -> None:
        self._queues: dict[str, "queue.Queue[dict]"] = {}
        self._lock = threading.Lock()

    def register(self, run_id: str) -> "queue.Queue[dict]":
        q: "queue.Queue[dict]" = queue.Queue(maxsize=4096)
        with self._lock:
            self._queues[run_id] = q
        return q

    def get(self, run_id: str) -> Optional["queue.Queue[dict]"]:
        with self._lock:
            return self._queues.get(run_id)

    def emit(self, run_id: str, event: dict) -> None:
        q = self.get(run_id)
        if q is None:
            return
        try:
            q.put_nowait(event)
        except queue.Full:
            pass  # a slow/absent consumer must never stall the agent thread

    def unregister(self, run_id: str) -> None:
        with self._lock:
            self._queues.pop(run_id, None)


RUNS = RunManager()


# --------------------------------------------------------------------------- #
# Span -> event normalization (shared by live + replay)
# --------------------------------------------------------------------------- #
def _load_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _event_content(raw_span: dict, name: str) -> Optional[str]:
    for ev in raw_span.get("events") or []:
        if ev.get("name") == name:
            return (ev.get("attributes") or {}).get("content")
    return None


def normalize_span(raw: dict) -> dict:
    """Turn one raw span dict (the traces/*.jsonl schema) into a UI event.

    Works identically for a live span (converted from a ReadableSpan) and a span read
    back from an archived trace file.
    """
    a = raw.get("attributes") or {}
    node_type = a.get(T.ACG_NODE_TYPE)
    node_id = a.get(T.ACG_NODE_ID)

    prompt = _event_content(raw, "gen_ai.prompt")
    completion_raw = _event_content(raw, "gen_ai.completion")
    tool_result = _event_content(raw, "acg.tool.result")

    ev: dict[str, Any] = {
        "kind": "span",
        "node_id": node_id,
        "node_type": node_type,
        "name": raw.get("name"),
        "step": a.get(T.ACG_STEP),
        "depends_on": _load_json(a.get(T.ACG_DEPENDS_ON), []),
        "start_time_ns": raw.get("start_time_ns"),
        "end_time_ns": raw.get("end_time_ns"),
        "duration_ns": raw.get("duration_ns"),
        "is_nested": "/" in str(node_id) if node_id else False,
        # llm-only
        "model": a.get(T.GEN_AI_REQUEST_MODEL),
        "temperature": a.get(T.GEN_AI_REQUEST_TEMPERATURE),
        "top_p": a.get(T.GEN_AI_REQUEST_TOP_P),
        "seed": a.get(T.GEN_AI_REQUEST_SEED),
        "input_tokens": a.get(T.GEN_AI_USAGE_INPUT_TOKENS, 0),
        "output_tokens": a.get(T.GEN_AI_USAGE_OUTPUT_TOKENS, 0),
        "finish_reasons": a.get(T.GEN_AI_RESPONSE_FINISH_REASONS),
        "tool_call_count": a.get("acg.response.tool_call_count"),
        "prompt": prompt,
        "completion": _load_json(completion_raw, None),
        # tool-only
        "tool_name": a.get(T.GEN_AI_TOOL_NAME),
        "tool_args": _load_json(a.get(T.ACG_TOOL_ARGS), {}),
        "tool_result": _load_json(tool_result, tool_result),
        # root-only
        "outcome": a.get(T.ACG_OUTCOME),
        "answer": a.get("acg.answer"),
        "question": a.get("acg.question"),
        "error": a.get("acg.error"),
    }
    return ev


# --------------------------------------------------------------------------- #
# The extra span processor (attached once to the global provider)
# --------------------------------------------------------------------------- #
class _StreamingExporter(SpanExporter):
    """Converts each finished span and routes it to its run's queue by acg.run_id."""

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            for span in spans:
                raw = JSONLFileSpanExporter._span_to_dict(span)
                run_id = (raw.get("attributes") or {}).get(T.ACG_RUN_ID)
                if not run_id:
                    continue
                RUNS.emit(run_id, normalize_span(raw))
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:  # pragma: no cover - nothing to close
        pass


_attached = False
_attach_lock = threading.Lock()


def attach_streaming_processor() -> None:
    """Attach the streaming processor to the (already configured) global provider once.

    `configure_tracing()` must have been called first so a real SDK TracerProvider
    exists. Idempotent: safe to call before every run.
    """
    global _attached
    with _attach_lock:
        if _attached:
            return
        provider = trace.get_tracer_provider()
        add = getattr(provider, "add_span_processor", None)
        if add is None:
            return  # no-op provider (tracing not configured yet)
        add(SimpleSpanProcessor(_StreamingExporter()))
        _attached = True
