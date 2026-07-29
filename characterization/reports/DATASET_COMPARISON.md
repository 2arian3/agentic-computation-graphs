# Dataset comparison

What we hold, how many traces each has, and — field by field — what a trace
**does** and **does not** contain.

All figures are measured from the extracted graphs and their manifests, not
taken from dataset documentation. Generated 2026-07-29.

Scope: the five datasets acquired to disk. The other fifteen registry entries
were audited but not downloaded — see [`CONTENT_AVAILABILITY.md`](CONTENT_AVAILABILITY.md)
for why (gated, synthetic, simulated, or environment-only).

---

## 1. Number of traces

| dataset | source traces | **graphs extracted** | nodes | edges | raw on disk | graphs on disk |
|---|---:|---:|---:|---:|---:|---:|
| **tracelab** | 4,265 sessions<br><sub>357,161 LLM rounds · 432,510 tool records</sub> | **4,265** | 837,967 | 946,245 | 52 MB | 730 MB |
| **swe_rebench_openhands** | 67,074 trajectories | **67,074** | 8,701,102 | 8,633,847 | 2.0 GB | 21 GB |
| **swe_agent_traj** | 80,036 trajectories | **80,036** | 4,311,143 | 4,231,107 | 1.1 GB | 7.2 GB |
| **osworld_gelato** | 2,166 episodes | **2,111** <sub>(55 zero-action)</sub> | 64,864 | 62,753 | 46 MB | 71 MB |
| **open_swe_traces** | 207,489 trajectories | *not extracted* | — | — | 18 GB | — |
| | | **153,486** | **13,915,076** | **13,873,952** | | |

**Extraction losses are zero** for the three coding sources. OSWorld's 55
missing graphs are not losses either: 53 are `INFEASIBLE` (the agent judged the
task impossible and refused) and 2 are `TERMINATE` (stopped without acting).
Both are real agent outcomes with no action to record — 0 files were unparseable
and 0 were empty.

`open_swe_traces` is downloaded with a smoke-tested parser but has never been
run to completion.

---

## 2. What each trace HAS

**Y** = present and measured · **P** = partial (figure given) · **N** = absent

| | tracelab | swe_rebench_<br>openhands | swe_agent_<br>traj | osworld_<br>gelato | open_swe_<br>traces* |
|---|:---:|:---:|:---:|:---:|:---:|
| **— why it ran —** | | | | | |
| reasoning text | **N** | **P** 71.2% | **Y** 100% | **P** 89.6% | **Y** 100% |
| tool name | **Y** | **Y** | **Y** | **Y** | **Y** |
| tool arguments | **N** | **Y** 100% | **Y** 100% | **Y** 100% | **Y** 100% |
| tool output | **N** | **Y** 100% | **Y** 100% | **Y** 100% | **Y** 100% |
| system prompt | **N** | **Y** 2 distinct | **Y** 1 distinct | **N** | **Y** |
| task statement | **N** | **Y** issue text | **Y** issue text | **N** UUID only | **Y** |
| tool schemas offered | **N** | **Y** `tools.json` | **N** | **N** | **Y** |
| **— what it cost —** | | | | | |
| input/output tokens | **Y** 42.6% | **N** | **N** | **N** | **N** |
| prefill / decode split | **Y** | **N** | **N** | **N** | **N** |
| KV / prefix-cache hits | **Y** 42.6% | **N** | **N** | **N** | **N** |
| latency | **Y** 85.9% | **N** | **N** | **N** | **N** |
| timestamps | **Y** 100% | **N** | **N** | **N** | **N** |
| **— structure & outcome —** | | | | | |
| user turns | **Y** 48,296 | **Y** 67,380 | **Y** 80,035 | **N** | **Y** |
| named sub-agents | **Y** 1,867 | **N** | **N** | **N** | **N** |
| tool error flag | **P** 85.1% | **N** | **N** | **N** | **N** |
| outcome label | **N** | **Y** `resolved` +<br>`gen_tests_correct` | **Y** `exit_status` | **Y** graded score | **Y** `resolved` |
| explored/pruned branches | **N** | **N** | **N** | **N** | **N** |

<sub>\* `open_swe_traces` figures are from a 400-trajectory audit sample, not a full extraction.</sub>

**Token coverage of 42.6% is not a gap.** It is exactly 357,161 / 837,967 — i.e.
**every** LLM node in TraceLab carries full token and prefix-cache accounting.
Tool and user nodes have no tokens by nature.

---

## 3. What each trace does NOT have

### tracelab — no semantics whatsoever
The sanitizer replaces text with counts before release. `tools[].input_chars`
and `result_chars` survive; the arguments and outputs themselves do not.
`reasoning_output_tokens` is a **count**, not the reasoning. There is no
recovery path — not with better parsing, not ever. Also missing: any
success/failure label for the session, and any prompt.

### swe_rebench_openhands — no cost, no prose on 28.8% of steps
Zero tokens, latency, timestamps, KV. Reasoning is 71.2% rather than 100%
because some assistant messages carry only a tool call with neither
`reasoning_content` nor `content` — the model acted without narrating. That is a
property of the trace, not a parsing loss. No sub-agents; no branching.

### swe_agent_traj — no cost, no structured tool calls
Zero tokens, latency, timestamps, KV. Tool calls are **prose plus a fenced
command**, so `tool_name` is parsed (first token) rather than declared — giving
397 distinct "tools" because arbitrary bash is permitted. No tool schemas are
shipped, so the *offered* action space is unknown. No sub-agents; no branching.

### osworld_gelato — no cost, no task text, no user turn
Zero tokens, latency, timestamps, KV. Critically, **the instruction is not
shipped** — only the OSWorld task UUID — so what the agent was asked to do is
not recoverable from this release without joining upstream. Consequently there
are no `user` nodes and no `agent_msg` edges at all. No sub-agents; no branching.

### All four — no explored-but-abandoned branches
Every graph is a single committed path (`branch_id "b0"`, `committed=true`).
None of these sources record what the agent tried and rejected. In the audited
registry only **ToolBench** preserves pruned branches.

---

## 4. Structure

| metric | tracelab | swe_rebench_openhands | swe_agent_traj | osworld_gelato |
|---|---:|---:|---:|---:|
| median nodes | 46 | 123 | 35 | 20 |
| p90 / p99 nodes | 350 / 2,722 | 195 / 201 | 101 / 323 | 82 / 98 |
| **max nodes** | **18,482** | **201** | **817** | **100** |
| median depth | 33 | 122 | 34 | 19 |
| **median fan-out** | **3** | **1** | **1** | **1** |
| **max fan-out** | **29** | **2** | **1** | **1** |
| median loop iterations | 15 | 60 | 16 | 9 |
| measured parallel width | median 3, max 22 | n/a (untimed) | n/a | n/a |
| cyclic graphs | 0 | 0 | 0 | 0 |

Node mix:

| dataset | llm | tool | agent | user |
|---|---:|---:|---:|---:|
| tracelab | 357,161 | 430,643 | **1,867** | 48,296 |
| swe_rebench_openhands | 4,316,962 | 4,316,760 | 0 | 67,380 |
| swe_agent_traj | 2,115,623 | 2,115,484 | 0 | 80,036 |
| osworld_gelato | 32,432 | 32,432 | 0 | **0** |

---

## 5. Two findings the comparison forces

### Cost and semantics are disjoint in public data

TraceLab and the other three are **exact complements**:

| | cost fields | semantic fields |
|---|---|---|
| tracelab | 42.6–100% | **0%** |
| the other three | **0%** | 71–100% |

No single source has both. Relating *why* a node ran to *what it cost* requires
bridging two datasets — or a Phase 2 instrumented run. (In the wider audit only
**tau2-bench** carries both, and it is disqualified here as a simulation: its
user is a GPT-4.1 simulator over mock databases.)

### The harness decides the shape, not the model

Across **13.1 million nodes**, three independent harnesses, two scaffolds and
two domains, **max fan-out is 1** — the single exception is 4 graphs out of
67,074 that reach 2. TraceLab's production sessions reach **29**, with 16.8% of
rounds issuing multiple tools and 88.2% of those showing *measured* wall-clock
overlap.

Size is bounded by configuration too:

- **OpenHands** piles 8.9% of runs at exactly 201 nodes → a 100-iteration cap
- **OSWorld** stops at 100 nodes → the agent's own observations say *"maximum of 50 steps"*
- **SWE-agent** has no ceiling and decays smoothly to 817
- **TraceLab** is unbounded: p99 2,722, max 18,482

A corpus built only from benchmark rollouts would conclude — wrongly — that
agent computation graphs are inherently serial and bounded. Both properties are
artefacts of harness configuration.

---

## 6. Real LLM calls, or simulation?

"Real vs simulated" is not one question — a trace has four independent layers,
and a dataset can be real at one and synthetic at another. Conflating them is
how a mock-database benchmark gets described as "real agent data".

| layer | what it asks |
|---|---|
| **inference** | did an actual model run, or was the assistant turn generated/templated? |
| **environment** | did tools hit real software, or a mock? |
| **user** | was the other party a human, an LLM simulator, or a static string? |
| **task** | did the task come from the world, or from a generator? |

### The four datasets in hand

| | inference | environment | user | task | overall |
|---|---|---|---|---|---|
| **tracelab** | **real** — commercial API calls | **real** — developers' own machines | **real human** | **real work** | **production traffic** |
| **swe_rebench_openhands** | **real** — Qwen3-Coder-480B | **real** — repo in a container | *static issue text* | **real** GitHub issues | benchmark run |
| **swe_agent_traj** | **real** — Llama-70b/8b/405b | **real** — repo in a container | *static issue text* | **real** GitHub issues | benchmark run |
| **osworld_gelato** | **real** — gelato-30b, gta1-32b | **real** — Ubuntu VM, real apps | *none* | human-authored benchmark | benchmark run |
| **open_swe_traces** | **real** — Minimax-M2.5, Qwen3.5-122B | **real** — repo in a container | *static issue text* | **real** GitHub issues | benchmark run |

**Every assistant turn in all five came from a genuine model inference.** None
of them is scripted, templated, or generated by a data-synthesis pipeline. The
models are named and recorded per node:

| dataset | models (measured from the extracted graphs) |
|---|---|
| tracelab | `gpt-5.5` 103,088 · `claude-opus-4-7` 88,575 · `gpt-5.4` 56,480 · `gpt-5.3-codex` 29,715 · `claude-opus-4-6` 16,852 · `claude-haiku-4-5` 11,040 |
| swe_rebench_openhands | `Qwen3-Coder-480B-A35B-Instruct` (all) † |
| swe_agent_traj | `swe-agent-llama-70b` 87% · `-8b` 11% · `-405b` 2% |
| osworld_gelato | `gta1-32b_baseline` 53% · `gelato-30b` 47% |

† The parquet has **no model column** — this is dataset-card metadata applied at
dataset level, flagged in provenance as `model_source: hf dataset card`. The
other three carry the model per row.

### The line that actually matters: production vs benchmark

All four are real inference, but only **tracelab is production traffic**. The
distinction is measurable, not editorial:

| evidence | tracelab | the three benchmarks |
|---|---|---|
| user turns per trace | **mean 11.3**, max **3,680** | exactly **1** (OSWorld: **0**) |
| session wall-clock span | p90 **5.9 h**, p99 **209 h**, max **1,726 h** | not recorded (no timestamps) |
| traces spanning >24 h | **180 (4.2%)** | n/a |
| who ends the run | the human stops | a step cap or `submit`/`finish` |

A benchmark trace is one injected task statement followed by an uninterrupted
agent loop. A TraceLab session is a **conversation**: a human interrupts,
redirects, walks away, and comes back — 4.2% of sessions span more than a day,
and the longest runs 72 days. That is why TraceLab is unbounded (max 18,482
nodes) while the benchmarks stop at their caps.

For a serving project this is the load-bearing difference. Production traffic
has human think-time gaps, session resumption, and prompt-cache reuse across
hours; benchmark rollouts are clean isolated episodes with none of that.

### What was rejected, and why

The "real, not simulated" constraint eliminated several higher-content options:

| rejected | inference | environment | user | verdict |
|---|---|---|---|---|
| **tau2-bench** | real | **mock databases** | **GPT-4.1 simulator** | simulated on two layers, despite the richest content |
| **APIGen-MT** | real | mock | scripted | **synthetic-generation pipeline** by design |
| **xLAM-60k** | — | — | — | **synthetic**, and gated (401) |
| **SWE-smith** | real | real repos | static | tasks are **LLM-synthesised bugs** |
| **AgentBank** | real | **simulators** (ALFWorld, BabyAI, Jericho) | static | environment is a text-game simulator |
| **ToolBench** | real | **real** RapidAPI calls | static | but queries are **ChatGPT-generated** |
| **Mind2Web / WebLINX** | **none — human demos** | real websites | human | no LLM inference at all |

Two of these are worth keeping in view rather than dismissing: **ToolBench** is
real inference against real third-party APIs and is the only source with pruned
branches; **Mind2Web/WebLINX** contain no model inference whatsoever, so they
cannot answer anything about LLM behaviour.

### Caveat on `osworld_gelato`

Registered `real-controlled`, not `real`. The applications and OS are genuine
software driven by real `pyautogui` calls in an Ubuntu VM — not a simulated
world model. But the task set is a benchmark harness and there is no human in
the loop, so it sits one notch below the SWE datasets on the *task* layer and
well below TraceLab overall.

---

## 7. Which dataset for which question

| question | use | why |
|---|---|---|
| How large/deep/parallel are real agent graphs? | **tracelab** | the only unbounded, production, timed source |
| What does a step cost (tokens, KV, latency)? | **tracelab** | the only source with any cost fields at all |
| Why did the agent call this tool? | **swe_agent_traj** | 100% reasoning coverage — the prompt forces narration |
| Reasoning + the tool schemas it chose from | **swe_rebench_openhands** | ships `tools.json`, structured tool calls |
| Non-coding / GUI behaviour | **osworld_gelato** | only non-coding domain; cleanest per-step schema |
| Sub-agent delegation structure | **tracelab** | the only source with named sub-agents (1,867) |
| What the agent tried and rejected | *none of these* | needs ToolBench (pruned DFS branches) |
| Scaffold-vs-model attribution | **swe_rebench_openhands + swe_agent_traj** | same domain, same task type, different harness |

### The gap worth naming

**No dataset in hand has reasoning *and* named sub-agents.** TraceLab has the
sub-agents but no reasoning; the three semantic sources are all single-agent.
"Real substrate + reasoning + multi-agent" does not exist in any public dataset
found so far, and is a Phase 2 target.
