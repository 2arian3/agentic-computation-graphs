# Development history

This directory began as a standalone repo (`~/agentic-graph-corpus`) and was
folded into `agentic-computation-graphs` on 2026-07-29 as `characterization/`.
Its own `.git` was removed so the parent repo tracks these files directly.
The original commit history is preserved below.

```
cde49c1 2026-07-29 compare datasets on real-inference vs simulation
e101655 2026-07-29 add reports/DATASET_COMPARISON.md
2110df8 2026-07-29 extract swe_agent_traj and osworld_gelato
62732fe 2026-07-29 extract swe_rebench_openhands with semantic content
2475af3 2026-07-29 registry: record measured content coverage for osworld_gelato
ceb835c 2026-07-29 acquire 3 real-substrate datasets with full reasoning content
13c7517 2026-07-29 audit semantic content availability across all 16 datasets
f7a2bad 2026-07-27 tracelab: promote sub-agent calls to agent nodes; serpentine DAG layout
3418f7d 2026-07-26 visualize tracelab ACGs
fd39f12 2026-07-26 extract tracelab
```

## Full commit messages

### cde49c1 — compare datasets on real-inference vs simulation

DATASET_COMPARISON.md gains a section separating four independent layers --
inference, environment, user, task -- because a dataset can be real at one and
synthetic at another, which is how a mock-database benchmark gets described as
"real agent data".

All five acquired datasets are real inference; none is scripted or synthesised.
The load-bearing split is production traffic vs benchmark run, and it is
measurable: tracelab averages 11.3 user turns per trace (max 3,680) with 4.2% of
sessions spanning >24h and the longest 1,726h, while the three benchmarks have
exactly one injected task turn (OSWorld zero) and end at a step cap.

swe_rebench_openhands had no model recorded because the parquet carries no model
column. The HF card names Qwen3-Coder-480B-A35B-Instruct under OpenHands
v0.54.0, so it is now set per node and flagged in provenance as dataset-card
level rather than per-row. Re-extracted: 8.7M nodes, model on 100% of llm nodes.

registry.yaml gains provenance_layers per acquired dataset.

Co-Authored-By: Claude <noreply@anthropic.com>

---
### e101655 — add reports/DATASET_COMPARISON.md

Field-by-field comparison of the five acquired datasets: trace counts, what each
trace has, what it explicitly does not have, structure distributions, and which
dataset answers which question. All figures measured from extracted graphs and
manifests rather than taken from documentation.

153,486 graphs / 13.9M nodes extracted across four datasets, zero extraction
losses (OSWorld's 55 non-graphs are INFEASIBLE/TERMINATE outcomes, not failures).

Co-Authored-By: Claude <noreply@anthropic.com>

---
### 2110df8 — extract swe_agent_traj and osworld_gelato

swe_agent_traj: 80,036 graphs / 4.3M nodes / 0 skipped. SWE-agent emits prose +
one fenced command rather than structured tool calls, so reasoning is the text
before the fence and tool_name the command's first token. Reasoning coverage is
100% -- the prompt forces narration before every command. 397 tool names,
because arbitrary bash is permitted; names like `find.` are recorded as the
model actually typed them rather than repaired.

osworld_gelato: 2,111 graphs / 64.9K nodes from real desktop apps. The 55
episodes without a graph are 53 INFEASIBLE (agent refused the task) and 2
TERMINATE -- real outcomes, now counted by reason in the manifest instead of a
generic "skipped". 0 unparseable, 0 empty.

The parallelism result is now decisive: across 13.1M nodes, three harnesses,
two scaffolds and two domains, max fan-out is 1 (4 graphs of 67,074 reach 2).
TraceLab's production sessions reach 29. Size is bounded by configuration too --
OpenHands piles 8.9% of runs at 201 nodes, OSWorld caps at 100, SWE-agent has no
ceiling and decays to 817.

Co-Authored-By: Claude <noreply@anthropic.com>

---
### 62732fe — extract swe_rebench_openhands with semantic content

Schema gains nullable reasoning_text / tool_input / tool_output so "why did this
node run" is a first-class, measurable field rather than an untyped extra. Empty
means empty: TraceLab now reports 0.0% semantic coverage explicitly.

swe_rebench_openhands: 67,074 graphs / 8.7M nodes / 8.6M edges from 1,823 real
GitHub repos, 0 skipped. Reasoning on 71.2% of llm nodes, tool input/output on
100% of action nodes, 0% on every cost field -- the exact complement of TraceLab.

Two findings:
- Fan-out is 1 in 67,070 of 67,074 graphs (max 2 across 8.7M nodes), against
  TraceLab's median 3 / max 29. Parallelism is a property of the harness.
- 8.9% of runs sit exactly at the 201-node ceiling, so the size distribution is
  clipped by OpenHands' 100-iteration cap, not by the tasks.

System prompts are deduplicated to prompts.json (all 67,074 trajectories share
2 distinct prompts) and joined by sha256 rather than copied per graph.

Co-Authored-By: Claude <noreply@anthropic.com>

---
### 2475af3 — registry: record measured content coverage for osworld_gelato

Also documents the two acquisition traps: the HF repo is 13.2 GB because of
34,598 screenshots (irrelevant to graphs; fetch only jsonl/txt/json for 46 MB),
and 2,166 unauthenticated small-file requests trip HTTP 429.

Co-Authored-By: Claude <noreply@anthropic.com>

---
### ceb835c — acquire 3 real-substrate datasets with full reasoning content

Selected against a hard "real, not simulated or synthetic" constraint, which
disqualifies several higher-content options: tau2-bench's user is a GPT-4.1
simulator over mock databases, APIGen-MT and xLAM are synthetic-generation
pipelines, and SWE-smith synthesises its bugs.

  swe_rebench_openhands  67,074 trajectories / 2.0 GB / CC BY 4.0
                         structured tool_calls + tools.json prompt schemas
  swe_agent_traj         80,036 trajectories / 1.1 GB / CC BY 4.0
                         full system_prompt, thought+action, different scaffold
  osworld_gelato         2,166 episodes / MIT / real desktop apps in a VM

scripts/audit_content.py measures coverage rather than trusting docs.

Co-Authored-By: Claude <noreply@anthropic.com>

---
### 13c7517 — audit semantic content availability across all 16 datasets

Adds reports/CONTENT_AVAILABILITY.md and a machine-readable `content` block per
registry entry, auditing a different axis from the cost-field coverage: can a
dataset show WHY a node was called (reasoning text, tool args, tool output,
outcome), not just its shape.

Key result: TraceLab is 100% on cost fields and 0% on semantics -- sanitization
replaces tool inputs, assistant text and tool outputs with character counts.
Open-SWE-Traces is the mirror image (100% reasoning/args/outputs, no cost).
tau2-bench is the only source carrying both; ToolBench alone keeps pruned
branches. 12 of 16 verified by direct inspection; xLAM and GAIA return 401.

Co-Authored-By: Claude <noreply@anthropic.com>

---
### f7a2bad — tracelab: promote sub-agent calls to agent nodes; serpentine DAG layout

Sub-agent boundaries were being flattened into ordinary tool nodes. Codex
spawn_agent/wait_agent/close_agent/resume_agent and Claude Agent/Task/Explore/
SendMessage now emit node_type "agent" (1,867 nodes over 300 sessions).
Classification is by tool name only; the Claude Task* todo family is excluded.
characterize.py counts agent alongside tool for loop iterations.

visualize.py wraps deep graphs serpentine-style (alternating row direction,
marked with a direction glyph) so a 36-level chain reads as three rows instead
of one unreadable 3,000px line, and the wrap edge stays a short hop. Adds
--wrap and --no-timeline.

Co-Authored-By: Claude <noreply@anthropic.com>

---
### 3418f7d — visualize tracelab ACGs

src/visualize.py renders a session as stats header + DAG + wall-clock timeline;
scripts/render_gallery.py picks a representative distinct set off the
precomputed per-graph metrics.

Everything drawn comes from graphs.jsonl and a null field renders as "n/a".
Timeline rows are per round rather than greedily packed lanes, so tool overlap
stays visible; sub-pixel bars are clamped to a visible floor and the axis says
how many. User nodes are seated next to the round they feed (layout only -- the
depth metric is untouched).

Co-Authored-By: Claude <noreply@anthropic.com>

---
### fd39f12 — extract tracelab

Scaffold the corpus pipeline and process dataset 1 end to end.

- src/schema.py: canonical Node/Edge/Graph with every cost field defaulting to
  null, plus a validator that fails loudly on dangling edges or bad enums
- src/characterize.py: structural metrics incl. candidate vs measured parallel
  width; topological pruning keeps sibling-dependency search off the O(n^2) path
- src/extractors/tracelab.py: 4,265 sessions -> 837,967 nodes / 946,245 edges
- registry.yaml: 16 datasets, mode + status + source_verified

TraceLab v0.0.1 verified against the README's published SHA256. Data edges are
exact (tool_result names its tool_call_id): 99.97% resolve.

Co-Authored-By: Claude <noreply@anthropic.com>

---