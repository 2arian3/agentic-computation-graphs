# swe_agent_traj

**Status:** `extracted` · **Domain:** coding · **Graphs:** 80,036 · **Nodes:** 4,311,143 · **Edges:** 4,231,107 · **Skipped:** 0

## Source

| | |
|---|---|
| HF dataset | `nebius/SWE-agent-trajectories` |
| Licence | **CC BY 4.0** |
| Raw | 12 parquet shards, 1.1 GB |
| Substrate | real GitHub repositories — pydantic, dvc, sqlglot, pyupgrade, textual, sqlfluff |
| Scaffold | **SWE-agent** |
| Models | `swe-agent-llama-70b`, `swe-agent-llama-8b` |
| Outcome | `exit_status`, `generated_patch`, `eval_logs` |

Deliberately paired with [`swe_rebench_openhands`](swe_rebench_openhands.md):
same domain, same kind of task, **different scaffold**. Differences between the
two are attributable to the harness rather than the workload.

## Mapping

SWE-agent does **not** emit structured tool calls. Each `ai` step is prose
followed by one fenced block holding the command:

```
The error we received is an HTTPError ... Let's find the file.

```
find_file "memset.py" src
```
```

So `reasoning_text` = text before the first fence, `tool_input` = the last
fenced block verbatim, `tool_name` = the command's first token. Commands are
often multi-line (`edit 20:24` followed by replacement source); only the first
token is the tool and the whole block is kept.

Role sequence is `system, user, ai, user, ai, …`. `user[0]` is the GitHub issue
(the task); `user[k≥1]` is the observation produced by `ai[k-1]`'s command.
The single system prompt is deduplicated into `prompts.json` — all 80,036
trajectories share **1 distinct** prompt.

**397 distinct tool names**, because SWE-agent permits arbitrary bash alongside
its special commands: `edit` 29,998 · `python` 14,205 · `open` 10,179 ·
`search_dir` 8,393 · `search_file` 5,478 · `ls` 5,071 · `goto` 4,018 ·
`create` 3,742 · `find_file` 3,489 · `submit` 2,742.

Names like `find.` and `cd..` appear because the **model literally emitted a
missing space**. The parser records the first token as typed rather than
repairing it — a malformed command is evidence about the model, not noise.

## Structure

Nodes: `llm` 2,115,623 · `tool` 2,115,484 · `user` 80,036. All DAGs.

| metric | p25 | **p50** | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| nodes | 21 | **35** | 67 | 101 | 323 | 817 |
| depth | 20 | **34** | 66 | 100 | 322 | 816 |
| max fan-out | 1 | **1** | 1 | 1 | 1 | **1** |

**Fan-out is 1 everywhere — not once in 4.3 million nodes does this scaffold
issue two tools from one step.** SWE-agent's interface is a single command per
turn, so the serial shape is architectural, not a model choice.

Unlike OpenHands there is **no hard step ceiling**: the tail runs to 817 nodes
and the distribution decays smoothly rather than piling up at a cap.

## Coverage

| cost | | | | semantic | |
|---|---|---|---|---|---|
| tokens | latency | timestamps | KV | reasoning (of llm) | tool i/o (of action) |
| 0.0% | 0.0% | 0.0% | 0.0% | **100.0%** | **100.0%** |

**Reasoning is 100%** — the highest of any source in the corpus. SWE-agent's
prompt forces the model to narrate before every command, so every single one of
2.1 million llm nodes carries a stated rationale. OpenHands reaches only 71.2%
because its models may call a tool without narrating.

## A real chain, verbatim

From `Azure/azure-functions-python-worker` (`exit_status=submitted`):

> **REASONING** *"Let's first start by understanding where the maximum value for `PYTHON_THREADPOOL_THREAD_COUNT` is being enforced. We'll need to find the relevant file where this value is checked and limited."*
> **TOOL** `search_dir` — `search_dir "PYTHON_THREADPOOL_THREAD_COUNT"`
> **OUTPUT** `Found 47 matches for "PYTHON_THREADPOOL_THREAD_COUNT" in /azure-functions-python-worker: …`
>
> **REASONING** *"The search results indicate that the term appears in multiple files, but given the nature of the issue, we should focus on `dispatcher.py` where thread management…"*
> **TOOL** `open` — `open azure_functions_worker/dispatcher.py`

## Reproduce

```bash
.venv/bin/python -m src.extractors.swe_agent_traj    # -> 7.0 GB
.venv/bin/python -m src.characterize swe_agent_traj
```
