# tracelab

**Status:** `extracted` · **Domain:** coding · **Graphs:** 4,265 · **Nodes:** 837,967 · **Edges:** 946,245

## Source

| | |
|---|---|
| Repo | https://github.com/uw-syfi/TraceLab |
| Site | https://tracelab.cs.washington.edu |
| Paper | arXiv:2606.30560 — *TraceLab: Characterizing Coding Agent Workloads for LLM Serving* |
| Asset | `syfi_coding_trace.jsonl.gz`, release **v0.0.1** (53.6 MB gz / 0.65 GB raw) |
| SHA256 | `9d265eae…da0b4e6b` — **matches the checksum published in the README** |
| License | Code Apache-2.0 · **Dataset CC BY 4.0** |

Real Claude Code and Codex sessions from 43 developers, pseudonymised by the
project's own sanitizer. Contents match the documented totals exactly:
357,161 LLM rounds, 432,510 tool records, 4,265 sessions
(claude 140,338 rounds / codex 216,823).

> A newer release **v0.0.2** (2026-07-24, ~2× the size) exists. It is *not*
> covered by the README's published checksums, so v0.0.1 is pinned here for
> reproducibility. Switching is a one-line change in `registry.yaml`.

## Record structure

One JSONL row = one LLM invocation ("round"). Rows group into sessions by
`session_id` and order by `round_index`. Each row carries token accounting
(`input_tokens_total` = `prefix_tokens` + `newly_append_tokens`), an ordered
`timing_events[]` list (`user_message`, `tool_result`, `reasoning`, `text`,
`tool_call`, `usage_report`), and nested `tools[]` with per-call timestamps and
latency. This is **not** a ReAct-text trajectory — it is a serving trace.

## Mapping to the canonical schema

One graph = one session. `llm` node per round, `tool` node per `tools[]` entry,
`user` node per `user_message` event.

**Sub-agent calls become `agent` nodes, not `tool` nodes.** Tools that cross an
agent boundary — Codex `spawn_agent` / `wait_agent` / `close_agent` /
`resume_agent`, Claude `Agent` / `Task` / `Explore` / `SendMessage` — are
delegation, not work, and the corpus should be able to tell them apart.
Classification is by tool name only. **1,867 agent nodes across 300 sessions**
(300 of 4,265, so delegation is rare: 7% of sessions). Note the Claude `Task*`
family (`TaskCreate`/`TaskUpdate`/`TaskList`/`TaskGet`) is the todo list, *not*
sub-agents, and is deliberately excluded.

| canonical field | source | note |
|---|---|---|
| `input_tokens` | `input_tokens_total` | |
| `output_tokens` / `decode_tokens` | `output_tokens` | one decode step per output token |
| `prefill_tokens` | `newly_append_tokens` | tokens *not* served from the prefix cache, i.e. actually prefilled |
| `cache_hit_tokens` | `prefix_tokens` | = Claude `cache_read_input_tokens` / Codex `cached_input_tokens` |
| `prompt_prefix_id` | — | **null**: the trace carries no prefix identity |
| `wall_latency_ms` (llm) | derived | the repo's documented observable-generation-time proxy: first `tool_call` ts − latest `user_message`/`tool_result` ts before it. Null when the round emits no tool call |
| `tool_latency_ms` | `tool_internal_latency_ms` → `tool_wall_latency_ms` | the precedence TraceLab's own analyses use |
| `tool_status` | `is_error` | `true`→error, `false`→ok, **null→`unknown`** (14.9% of tools) |
| `retrieval_k`, `retrieved_ids`, `retries` | — | **null**: absent from source |

Traces are linear — no explored-but-abandoned branches — so every node is
`branch_id "b0"`, `committed=true`. All 4,265 graphs are DAGs.

### The one non-obvious thing: providers place tool results differently

Claude splits the loop across rounds (round *r* emits the call, round *r+1*
carries the `tool_result`). **Codex bundles both into round *r***, because
TraceLab closes a round on the `token_count` event, which fires *after* the
results land (`scripts/extract_codex_rounds.py` ~line 690).

Attributing a Codex result to its own round would be wrong: its timestamp is
strictly after that round's own generation, so that round's LLM cannot have
consumed it. Token accounting confirms round *r+1* is the consumer — for the
worked example, round 1 input (23,463) ≈ round 0 input (22,825) + round 0 output
(500) + result tokens (~138).

So the rule used is provider-independent and timestamp-correct: **a tool result
flows into the next LLM invocation that could have seen it.** Getting this wrong
initially cost 15% of all data edges (they degraded to `order` edges) and
understated depth.

Edge linkage is *exact*, not inferred: every `tool_result` event names the
`tool_call_id` of the node that produced it. **431,684 of 431,818 (99.97%)**
resolve. The 134 that do not: 88 produced by a session's final round (nothing
consumes them), 33 whose id is absent from the entire corpus, 12 emitted in a
different session (the repo documents a Codex subagent-replay caveat), 1 duplicate.

## Structure

Edges: `control` 432,510 · `data` 431,684 · `agent_msg` 48,296 · `order` 33,755.
Nodes: `tool` 432,510 · `llm` 357,161 · `user` 48,296.

| metric | p25 | **p50** | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| nodes | 16 | **46** | 121 | 350 | 2,722 | 18,482 |
| edges | 19 | **53** | 139 | 383 | 3,077 | 21,662 |
| depth | 11 | **33** | 95 | 291 | 2,248 | 14,497 |
| max fan-out | 1 | **3** | 4 | 6 | 10 | 29 |
| loop iterations | 4 | **15** | 44 | 136 | 1,113 | 6,891 |
| candidate parallel width | 1 | **3** | 4 | 6 | 10 | 29 |
| measured parallel width | 1 | **3** | 4 | 6 | 10 | 22 |

**Size is heavy-tailed.** The median session is 46 nodes, but p99 is 2,722 and the
largest is 18,482 — a ~400× spread. Any serving system sized to the median will
meet sessions two orders of magnitude larger.

**Graphs are deep, not wide.** Median depth 33 against median fan-out 3: the
critical path is the dominant cost, and depth tracks node count almost linearly
(mean depth 158 vs mean nodes 196). This is the sequential act→observe loop.

**Parallelism is real and measurable here.** 16.8% of rounds emit more than one
tool call, and **88.2% of those show tool intervals that genuinely overlap in
wall-clock time** — this is observed concurrency, not structural inference.
`measured_parallel_width` (median 3, max 22) is computed from real timestamps by
sweeping simultaneously-open `[start_ts, end_ts]` intervals.

### Claude Code vs Codex are structurally different agents

| | graphs | med nodes | med depth | med fan-out | med measured width | med loops |
|---|---|---|---|---|---|---|
| claude | 2,676 | 33 | 23 | 2 | 2 | 11 |
| codex | 1,589 | 81 | 57 | 4 | 5 | 26 |

Codex builds graphs ~2.5× larger and deeper with ~2.5× the measured parallelism.
Aggregate statistics over "coding agents" hide a factor-of-2.5 split; the corpus
should be sliced by agent, not pooled.

Tools are dominated by shell and file ops across 85 distinct names:
`exec_command` 187,481 · `Bash` 66,983 · `write_stdin` 62,848 · `Read` 32,402 ·
`apply_patch` 24,134 · `Edit` 18,458. Tool status: 345,905 ok / 22,184 error /
64,421 unknown.

## Cost-field coverage (node-weighted)

| field group | coverage | why |
|---|---|---|
| timestamps | **100%** | every node carries real start/end |
| latency | **85.9%** | all tool nodes + llm nodes that emitted a tool call |
| tokens | **42.6%** | exactly the llm nodes — all 357,161 of them |
| KV / prefix | **42.6%** | `cache_hit_tokens` on every llm node; `prompt_prefix_id` null throughout |

42.6% is not a gap — it is 357,161/837,967, i.e. **every** LLM node has full token
and prefix-cache accounting. Tool and user nodes have no tokens *by nature*.
This is the ceiling the rest of the registry will be measured against, and
TraceLab is expected to be the only source that reaches it.

## Visualising the graphs

`src/visualize.py` renders any session as a three-panel figure: a stats header,
the DAG, and a wall-clock execution timeline. Everything drawn comes from
`graphs.jsonl`; a null field renders as `n/a` rather than being filled in.

- **DAG panel** — columns are topological levels, so nodes that *could* run
  concurrently share a column. Colour is node type, edge style is dependency
  kind (data solid, control dashed, order dotted, agent_msg green). LLM nodes are
  annotated `input-k/output` tokens; failed tools get a red ring.
- **Timeline panel** — one row per round. The pale bar is the round span
  (min..max of its event timestamps); the solid inner bar is the generation
  window from `wall_latency_ms`. Tool bars that overlap *on the same row* are
  measured concurrency. Bars below ~0.4% of the span are drawn at a minimum
  width (a session mixes second-scale `exec_command` with 40 ms `write_stdin`),
  and the axis label states how many were clamped.

Two layout decisions worth noting. Rows are per round rather than greedily
packed lanes — packing collapses a session onto one row and hides exactly the
overlap worth seeing. And `user` nodes are seated just left of the round they
feed; their true longest-path level is 0 (they have no predecessor), which would
pin every user turn to column 0 and arc its edge across the whole figure. That
adjustment is layout-only and does not affect the reported depth.

```bash
python -m src.visualize tracelab --pick most-parallel --rounds 0:12
python -m src.visualize tracelab --graph-id tracelab:codex:c2f937ac-... --rounds 0:8
python scripts/render_gallery.py tracelab        # representative set, ~6 s
```

`reports/figures/tracelab/` holds the gallery, each PNG paired with a JSON
summary of the whole session:

| figure | what it shows |
|---|---|
| `subagents_wide_deep.png` | **53 nodes, depth 36, fan-out 8** — L2 spawns 8 parallel `Agent` sub-agents that converge into L3 |
| `subagents_codex_spawn.png` | Codex delegation lifecycle: 10 `spawn_agent` over two rounds, then `wait_agent` joins |
| `wide_deep_busiest.png` | 118 nodes, depth 67, fan-out 9 — 6 user turns, 3 `Agent`s, 9-wide `TaskCreate` and 7-wide `Write` fan-outs |
| `subagents_compact.png` | 22 nodes — the smallest readable sub-agent example |
| `median_session.png` | the 46-node median: a plain serial user→llm→tool→llm chain |
| `typical_claude.png` / `typical_codex.png` | the same size class in each agent |
| `codex_parallel_walkthrough.png` | 3-wide fan-out/fan-in diamonds, rounds 0–8 |
| `widest_fan_out.png` | fan-out 29, 976 rounds, 93.2% cache-hit, 203 tool errors |
| `most_parallel.png` / `deepest.png` | the 18,482-node and 16,291-node tails |

Deep sessions are drawn **serpentine**: levels wrap every ~13–15 columns and
alternate direction (marked ▸/◂), so a 36-level chain reads as three rows
instead of 3,000 px of unreadable single-line text, and the wrap hop stays a
short step down rather than a long diagonal back across the figure.

The pictures make two things immediate that the tables only imply: the median
session is a **thin serial chain**, and where parallelism exists it is a narrow
fan-out/fan-in diamond around a single LLM round — never a wide independent
frontier.

## Reproduce

```bash
.venv/bin/python -m src.extractors.tracelab      # ~28 s -> data/graphs/tracelab/graphs.jsonl
.venv/bin/python -m src.characterize tracelab    # ~16 s -> metrics.json + per_graph_metrics.jsonl
.venv/bin/python scripts/render_gallery.py tracelab   # ~6 s -> reports/figures/tracelab/
```

Both are idempotent; pass `--force` to recompute. Raw artifact and checksum are
recorded in `data/raw/tracelab/_manifest.json`.
