# Cross-dataset summary

## Findings

_(regenerated after each dataset; see per-dataset reports for detail)_

**1 of 16 datasets processed so far.** TraceLab is the reference: it is a real
LLM *serving* trace rather than a text trajectory dump, and it is the only source
in the registry expected to ship tokens, latency, timestamps and prefix-cache
splits together. Every one of its 357,161 LLM nodes carries full token and
KV accounting (42.6% of all nodes; the rest are tool/user nodes that have no
tokens by nature) and 100% of nodes carry real timestamps.

Early structural signal: coding-agent graphs are **deep, not wide** (median depth
33 vs median fan-out 3) and **heavy-tailed** (median 46 nodes, p99 2,722, max
18,482). Where multiple tools are emitted in one round, 88.2% show tool intervals
that genuinely overlap in wall-clock time — real measured concurrency, available
because this source ships timestamps. And Claude Code vs Codex differ by ~2.5× in
median nodes, depth and measured parallel width, so "coding agent" is not one
workload.


**Content vs cost are two different axes** (see `CONTENT_AVAILABILITY.md`).
TraceLab scores 100% on cost fields and **0% on semantics** — it is sanitized,
so tool inputs, assistant text and tool outputs are replaced by character
counts. It can show the shape of a computation graph but never why a node was
called. Open-SWE-Traces is the mirror image: 100% reasoning / args / outputs,
zero tokens or timestamps. Only **tau2-bench** carries both — full message
content plus 100% timestamps, 65% token usage and a graded reward — and
**ToolBench** alone preserves pruned branches, making it the only source that
records what an agent tried and rejected.


**First semantic extraction landed (2026-07-29).** `swe_rebench_openhands`:
67,074 graphs / 8.7M nodes from real GitHub repos, with reasoning on 71.2% of
llm nodes and tool input/output on 100% of action nodes. It is the exact
complement of TraceLab — 0% on every cost field, where TraceLab is 0% on every
semantic field. Structurally the two disagree sharply: TraceLab's production
agents reach fan-out 29 and 16.8% of rounds issue multiple tools, while this
benchmark harness never exceeds fan-out 2 and only 4 of 67,074 graphs show any
parallelism at all. Its size distribution is also clipped by the scaffold — 8.9%
of runs sit exactly at the 201-node / 100-iteration ceiling. **Parallelism and
graph size are properties of the harness, not just the model**, so a corpus
built only from benchmark rollouts would wrongly conclude agent graphs are
inherently serial and bounded.


**Three semantic sources now extracted (2026-07-29)** —
`swe_rebench_openhands` (67,074 graphs / 8.7M nodes, OpenHands),
`swe_agent_traj` (80,036 / 4.3M, SWE-agent) and `osworld_gelato`
(2,111 / 64.9K, GUI computer-use). All three are real substrate: real GitHub
repositories and real desktop applications.

The parallelism result is now decisive. Across **13.1 million nodes**, three
independent harnesses, two scaffolds and two domains, **max fan-out is 1** —
the sole exception is 4 graphs out of 67,074 that reach 2. TraceLab's
production sessions, by contrast, have median fan-out 3 and reach 29, with
16.8% of rounds issuing multiple tools. Parallelism in agent computation graphs
is a property of what the harness permits, not of the model, and a corpus drawn
only from benchmark rollouts would conclude — wrongly — that agent graphs are
inherently serial.

Graph size is bounded by configuration too: OpenHands piles 8.9% of runs at
exactly 201 nodes (100 iterations) and OSWorld caps at 100 nodes (50 steps),
while SWE-agent has no ceiling and decays smoothly to 817. Reported size
distributions describe scaffold settings as much as workloads.

Cost and semantics remain disjoint: all three new sources are 0% on every cost
field and 71-100% on reasoning, the exact complement of TraceLab's 100% cost /
0% semantics.

## Comparison table

| # | dataset | status | domain | #graphs | med nodes | med depth | med max-fan-out | %nodes tokens | %nodes timestamps |
|---|---|---|---|---|---|---|---|---|---|
| 1 | tracelab | `extracted` | coding | 4,265 | 46 | 33 | 3 | 42.6% | 100.0% |
| 2 | open_swe_traces | `pending` | coding | — | — | — | — | — | — |
| 3 | agentbank | `pending` | tool_use | — | — | — | — | — | — |
| 4 | toolbench | `pending` | tool_use | — | — | — | — | — | — |
| 5 | tau2_bench | `pending` | tool_use | — | — | — | — | — | — |
| 6 | agentboard | `pending` | tool_use | — | — | — | — | — | — |
| 7 | xlam_apigen | `pending` | tool_use | — | — | — | — | — | — |
| 8 | appworld | `pending` | tool_use | — | — | — | — | — | — |
| 9 | mind2web | `pending` | web | — | — | — | — | — | — |
| 10 | weblinx | `pending` | web | — | — | — | — | — | — |
| 11 | toolsandbox | `pending` | tool_use | — | — | — | — | — | — |
| 12 | gaia | `pending` | qa | — | — | — | — | — | — |
| 13 | crag | `pending` | rag | — | — | — | — | — | — |
| 14 | fanoutqa | `pending` | qa | — | — | — | — | — | — |
| 15 | musique | `pending` | qa | — | — | — | — | — | — |
| 16 | hotpotqa | `pending` | qa | — | — | — | — | — | — |
| 17 | swe_rebench_openhands | `extracted` | coding | 67,074 | 123 | 122 | 1 | 0.0% | 0.0% |
| 18 | swe_agent_traj | `extracted` | coding | 80,036 | 35 | 34 | 1 | 0.0% | 0.0% |
| 19 | osworld_gelato | `extracted` | web | 2,111 | 20 | 19 | 1 | 0.0% | 0.0% |

## Coverage of cost fields (node-weighted)

| dataset | tokens | latency | timestamps | KV/prefix |
|---|---|---|---|---|
| tracelab | 42.6% | 85.9% | 100.0% | 42.6% |
| swe_rebench_openhands | 0.0% | 0.0% | 0.0% | 0.0% |
| swe_agent_traj | 0.0% | 0.0% | 0.0% | 0.0% |
| osworld_gelato | 0.0% | 0.0% | 0.0% | 0.0% |

## Coverage of semantic fields

Reasoning is measured against `llm` nodes and tool i/o against
action (`tool`/`agent`/`retrieval`) nodes — the nodes that could
carry them. A source that strips text reports 0% here even when it
scores 100% on cost fields.

| dataset | reasoning (of llm nodes) | tool input/output (of action nodes) |
|---|---|---|
| tracelab | 0.0% | 0.0% |
| swe_rebench_openhands | 71.2% | 100.0% |
| swe_agent_traj | 100.0% | 100.0% |
| osworld_gelato | 89.6% | 100.0% |

## Registry status

| # | dataset | mode | status | source verified |
|---|---|---|---|---|
| 1 | tracelab | `extract` | `extracted` | yes |
| 2 | open_swe_traces | `extract` | `pending` | no |
| 3 | agentbank | `extract` | `pending` | no |
| 4 | toolbench | `extract` | `pending` | no |
| 5 | tau2_bench | `extract` | `pending` | no |
| 6 | agentboard | `extract` | `pending` | no |
| 7 | xlam_apigen | `extract` | `pending` | no |
| 8 | appworld | `partial` | `pending` | no |
| 9 | mind2web | `extract` | `pending` | no |
| 10 | weblinx | `extract` | `pending` | no |
| 11 | toolsandbox | `partial` | `pending` | no |
| 12 | gaia | `workload_only` | `pending` | no |
| 13 | crag | `workload_only` | `pending` | no |
| 14 | fanoutqa | `workload_only` | `pending` | no |
| 15 | musique | `workload_only` | `pending` | no |
| 16 | hotpotqa | `workload_only` | `pending` | no |
| 17 | swe_rebench_openhands | `extract` | `extracted` | yes |
| 18 | swe_agent_traj | `extract` | `extracted` | yes |
| 19 | osworld_gelato | `extract` | `extracted` | yes |
