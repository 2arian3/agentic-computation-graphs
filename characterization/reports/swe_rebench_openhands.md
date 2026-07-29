# swe_rebench_openhands

**Status:** `extracted` · **Domain:** coding · **Graphs:** 67,074 · **Nodes:** 8,701,102 · **Edges:** 8,633,847

The first dataset in the corpus extracted **with semantic content** — every node
can say *why* it ran, not just what shape the run had.

## Source

| | |
|---|---|
| HF dataset | `nebius/SWE-rebench-openhands-trajectories` |
| Licence | **CC BY 4.0** |
| Raw | `trajectories.parquet` 2.0 GB + `tools.json` 18 KB |
| Substrate | **1,823 distinct real GitHub repositories** — sqlglot, Pillow, dvc, dask, conan, streamlink, pennylane, tox |
| Scaffold | OpenHands |
| Outcome | `resolved` 32,161 solved / 34,913 unsolved (**48% resolve rate**) |

Real issues, real repositories, real agent executions. Nothing simulated, nothing
synthetically generated — the resolve rate is what an agent actually achieved,
not a curated success set.

## Mapping to the canonical schema

One row = one trajectory = one graph. Record structure is a message list with
structured `tool_calls`.

| canonical field | source |
|---|---|
| `reasoning_text` (llm) | `reasoning_content`, falling back to `content` — OpenHands puts the rationale in the former when the model emits thinking, the latter otherwise |
| `tool_input` | `function.arguments`, **verbatim** |
| `tool_output` | the following `tool` message content, **verbatim** |
| `reasoning_text` (user) | the GitHub issue text — the reason the trajectory exists |
| `tool_name` | `function.name` |
| everything cost-related | **null** — this source ships no tokens, timestamps or latency |

Result linkage is **positional**: `tool` messages carry no `tool_call_id`, so a
result attaches to the call it follows. Safe here because OpenHands emits one
tool call per assistant message (measured 10,000/10,000 in sample); the only
calls without a following result are terminal `finish`.

### System prompts are deduplicated, not copied

All 67,074 trajectories share **2 distinct** system prompts / tool schemas.
Storing a copy per graph would have added gigabytes of identical text, so each
graph records `provenance.system_prompt_sha256` and the distinct texts are
written once to `data/graphs/swe_rebench_openhands/prompts.json`. Nothing is
lost — it is a join instead of a copy.

The prompt's tool schemas are preserved too, so the *offered* action space is
recoverable alongside the *chosen* actions: `execute_bash`, `str_replace_editor`,
`think`, `finish`, `task_tracker`.

## Structure

Nodes: `llm` 4,316,962 · `tool` 4,316,760 · `user` 67,380 · `agent` **0**.
Edges: `control` 4,316,760 · `data` 4,249,707 · `agent_msg` 67,380. All DAGs.

| metric | p25 | **p50** | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| nodes | 101 | **123** | 153 | 195 | 201 | 201 |
| depth | 100 | **122** | 152 | 194 | 200 | 200 |
| max fan-out | 1 | **1** | 1 | 1 | 1 | 2 |
| loop iterations | 49 | **60** | 75 | 96 | 99 | 99 |

**These graphs are perfectly serial.** Only **4 of 67,074** graphs (0.01%) have
any fan-out above 1, and the maximum anywhere in 8.7 million nodes is 2. Depth
tracks node count almost exactly (mean depth 128.6 vs mean nodes 129.7): the
graph *is* the chain.

**The size distribution is truncated by the harness, not by the task.** 8.9% of
trajectories sit exactly at the 201-node ceiling — OpenHands caps the run at 100
agent iterations. p90, p99 and max are all within 201, so the natural tail is
clipped. Any size statistic from this dataset is a statement about the scaffold's
configuration as much as about the workload.

## The comparison that matters

| | tracelab | swe_rebench_openhands |
|---|---|---|
| substrate | real dev sessions | real GitHub repos |
| median nodes | 46 | 123 |
| median depth | 33 | 122 |
| **median fan-out** | **3** | **1** |
| **max fan-out** | **29** | **2** |
| graphs with parallelism | 16.8% of *rounds* multi-tool | **0.01% of graphs** |
| size tail | p99 2,722, max 18,482 | p99 201, max 201 (capped) |
| **tokens / timestamps** | **100% / 100%** | **0% / 0%** |
| **reasoning / tool i/o** | **0% / 0%** | **71.2% / 100%** |

Two findings fall out of this pair.

**1. The scaffold decides the shape, not the model.** Both datasets are real
coding agents on real code. TraceLab's production agents fan out to 29 parallel
tools; this benchmark harness never exceeds 2. Parallelism is a property of what
the harness *permits*, and a corpus drawn only from benchmark rollouts would
conclude — wrongly — that agent graphs are inherently serial.

**2. Cost and semantics are disjoint in public data.** The coverage rows are
exact complements: TraceLab has all the cost and none of the reasoning; this has
all the reasoning and none of the cost. No single source has both, so any model
that needs to relate *why* a node ran to *what it cost* has to bridge two
datasets — or wait for a Phase 2 instrumented run.

## Coverage

| | cost | | | | semantic | |
|---|---|---|---|---|---|---|
| | tokens | latency | timestamps | KV | reasoning (of llm) | tool i/o (of action) |
| value | 0.0% | 0.0% | 0.0% | 0.0% | **71.2%** | **100.0%** |

Reasoning is 71.2% rather than 100% because some assistant messages carry only a
tool call with neither `reasoning_content` nor `content` — the model acted
without narrating. That is a real property of the trace, not a parsing loss.

## A real chain, verbatim

From `adamchainz/ec2-metadata` (`resolved=1`):

> **REASONING** *"The issue involves the `ec2_metadata.availability_zone_id` property returning a 404 HTTP error in certain EC2 environments. **Technical Details:** method `availability_zone_id`, file `ec2_metadata/__init__.py`, lines 87 and 65 (calls `_get_url`)…"*
> **TOOL** `str_replace_editor`
> **INPUT** `{"command": "view", "path": "/workspace/adamchainz__ec2-metadata__2.9"}`
> **OUTPUT** `Here's the files and directories up to 2 levels deep in …`

## Reproduce

```bash
.venv/bin/python -m src.extractors.swe_rebench_openhands   # ~7 min -> 22.1 GB
.venv/bin/python -m src.characterize swe_rebench_openhands # ~6 min
```

`--max-text N` truncates stored text if a compact variant is wanted; the true
length is always recorded in `extra`, so truncation is never silent. The default
(`0`) keeps every field verbatim.
