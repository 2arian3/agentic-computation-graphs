# osworld_gelato

**Status:** `extracted` · **Domain:** web (GUI / computer-use) · **Graphs:** 2,111 · **Nodes:** 64,864 · **Edges:** 62,753

The only **non-coding** source in the semantic corpus.

## Source

| | |
|---|---|
| HF dataset | `mlfoundations/gelato-osworld-agent-trajectories` |
| Licence | **MIT** |
| Raw | 2,166 episodes, 46 MB (data files only) |
| Substrate | **real desktop applications** in an Ubuntu VM |
| Agents | `gelato-30b`, `gta1-32b_baseline` (2 runs each) |
| Outcome | per-episode graded score in `result.txt` |

Applications: `multi_apps` 558 · `libreoffice_impress` 282 · `libreoffice_calc`
282 · `chrome` 276 · `gimp` 156 · `os` 144 · `vs_code` 138 ·
`libreoffice_writer` 138 · `vlc` 102 · `thunderbird` 90.

**Realness caveat.** The applications and the OS are real software driven by
real `pyautogui` calls — not a simulated world model. But the task set is a
benchmark harness, so this is registered `real-controlled` rather than `real`.

> The HF repo is 13.2 GB because of 34,598 screenshots, which are irrelevant to
> computation graphs. Only `*.jsonl` / `*.txt` are fetched (46 MB). Note also
> that 2,166 unauthenticated small-file requests trip **HTTP 429** — the
> downloader uses few workers and backs off.

## Mapping

The record structure is already almost canonical — each step carries the
decision and its consequence together, so this needs the least parsing of any
source in the corpus:

| step field | canonical |
|---|---|
| `reasoning` | `llm` node `reasoning_text` |
| `name` | `tool` node `tool_name` |
| `arguments` | `tool` node `tool_input` (structured, verbatim) |
| `tool_output` | `tool` node `tool_output` |
| `command` | `extra.command` — the executed `pyautogui` line |
| `call_id` | `extra.call_id` |

**There is no user/prompt node.** The dataset ships only the OSWorld task UUID,
not the instruction text, so the task statement is not recoverable from this
release. The UUID is kept in provenance so it can be joined against OSWorld
upstream.

**12 tool names:** `click` 15,763 · `hotkey` 6,845 · `type` 4,785 ·
`scroll` 1,568 · `drag_and_drop` 942 · `hold_and_press` 646 · `open` 584 ·
`set_cell_values` 558 · `wait` 487 · `highlight_text_span` 198 ·
`fast_open_terminal` 47 · `switch_applications` 9.

### 55 zero-action episodes are an outcome, not a loss

2,111 graphs come from 2,166 episodes. The 55 difference is **53 `INFEASIBLE`**
(the agent judged the task impossible and refused, mostly GIMP) and **2
`TERMINATE`** (stopped without acting). These are real agent decisions with no
action to record, so they are counted by reason in the manifest rather than
lumped into a generic "skipped". Nothing failed to parse — 0 unparseable, 0 empty.

## Structure

Nodes: `llm` 32,432 · `tool` 32,432 — exactly one tool per decision. All DAGs.

| metric | p25 | **p50** | p75 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| nodes | 10 | **20** | 42 | 82 | 98 | **100** |
| depth | 9 | **19** | 41 | 81 | 97 | 99 |
| max fan-out | 1 | **1** | 1 | 1 | 1 | **1** |

Fan-out is 1 in every one of 2,111 graphs. The ceiling at exactly 100 nodes is
the harness's **50-step budget** (2 nodes per step) — the agent's own
observations say so: *"you have maximum of 50 steps, and current step is 1 out
of 50."* As with OpenHands, the size distribution is bounded by configuration.

These are the **smallest graphs in the corpus** (median 20 nodes): a GUI task is
a short burst of clicks and keystrokes, not a long investigation.

## Coverage

| cost | | | | semantic | |
|---|---|---|---|---|---|
| tokens | latency | timestamps | KV | reasoning (of llm) | tool i/o (of action) |
| 0.0% | 0.0% | 0.0% | 0.0% | **89.6%** | **100.0%** |

## A real chain, verbatim

From `chrome`, `gelato-30b`, `result_score=1.0`:

> **REASONING** *"Checklist: — Confirm task feasibility based on the current Chrome window. — Reopen the last closed tab using the standard browser hotkey. — Verify a new tab appears and content matches a recently closed…"*
> **TOOL** `hotkey` — `{"keys": ["ctrl", "shift", "t"]}`
> **COMMAND** `import pyautogui; pyautogui.hotkey('ctrl', 'shift', 't')`
> **OUTPUT** `Action (tool call) is executed. … current step is 1 out of 50.`

## Reproduce

```bash
.venv/bin/python -m src.extractors.osworld_gelato
.venv/bin/python -m src.characterize osworld_gelato
```
