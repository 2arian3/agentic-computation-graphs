"""acg — Agentic Computation Graph instrument.

A small, fully-owned toolkit to MEASURE the size and structure of the graphs that
LLM agents generate, for one narrow domain (tool-using multi-hop QA).

Design commitments (from the 4-month RA proposal):
  * Decision 1 — a *local* open model served by vLLM/SGLang, so every decode
    parameter and the seed are pinned and held constant except sampling.
  * Decision 2 — a *thin emergent* agent loop we own (see acg/agent.py), NOT a
    pre-built static workflow. The MODEL defines the graph; we only fix the tool
    alphabet (acg/tools.py).
  * Instrumentation — OpenTelemetry GenAI spans (acg/tracing.py). The agent's
    parent/child span tree IS the graph; acg/graph.py reconstructs the ACG and
    its size/structure metrics offline from the captured trace.
"""

__version__ = "0.1.0"
