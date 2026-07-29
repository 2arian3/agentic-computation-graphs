# Content availability audit

**Question:** which datasets let you see *why* a node was called — the model's
actual reasoning, the tool it chose, the arguments it passed, and what came back?

This is a different axis from the cost-field coverage in `SUMMARY.md`. That one
asks "does this node carry tokens/latency/KV". This one asks "does this node
carry **semantics**". A dataset can score 100% on one and 0% on the other —
and the reference dataset does exactly that.

## Headline

**TraceLab cannot answer "why" at all.** It is sanitized by design: tool inputs
are dropped, assistant text and tool outputs are stripped, and only *character
counts* survive. It has perfect structure and perfect cost accounting and zero
semantics. Everything built on it so far describes shape, never intent.

Three datasets do answer "why", and they trade off differently:

| if you want | use | why |
|---|---|---|
| the richest reasoning text | **Open-SWE-Traces** | 100% of assistant steps carry `reasoning_content`; 100% carry tool args; 100% of tool results carry full output |
| reasoning **and** cost **and** outcome | **tau2-bench** | full message content + per-message `timestamp` (100%) and `usage` tokens (65%) + a graded `reward` with written justification |
| why a path was **abandoned** | **ToolBench** | ships the whole DFS tree including `pruned` branches — the only source where the not-taken option is recoverable |

## Audit table

Legend: **Y** present and verified · **P** partial / needs parsing · **N** absent
· **–** not applicable · *gated* = needs an accepted licence agreement

| # | dataset | reasoning | tool name | tool args | tool output | agent id | outcome | tokens | timestamps | verified |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **tracelab** | **N** (`reasoning_output_tokens` is a *count*) | **Y** | **N** (`input_chars` only) | **N** (`result_chars` only) | **Y** (sub-agent tool names) | N | **Y** 100% | **Y** 100% | direct |
| 2 | **open_swe_traces** | **Y** 100% | **Y** | **Y** 100% | **Y** 100% | P (scaffold only) | **Y** `resolved` | N | N | direct |
| 3 | **agentbank** | **Y** (`Thought:` text) | P (inside `Action:`) | P | **Y** (`Observation:`) | N | P | N | N | direct |
| 4 | **toolbench** | P (thin in function-call mode) | **Y** | **Y** | **Y** | N | **Y** `win`/`finish_type` | P (per-file total) | N | direct |
| 5 | **tau2_bench** | **Y** (assistant `content`) | **Y** | **Y** | **Y** | **Y** (`requestor`) | **Y** reward + justification | **Y** 65% | **Y** 100% | direct |
| 6 | **agentboard** | **Y** (trajectory + prompt) | **Y** | **Y** | **Y** | N | **Y** progress rate | ? | ? | README + file listing |
| 7 | **xlam / apigen-mt** | P (`think` call in APIGen-MT) | **Y** | **Y** | **Y** (`observation`) | N | N | N | N | direct (APIGen-MT); xLAM *gated* |
| 8 | appworld | – env only | – | – | – | – | – | – | – | no released trajectories |
| 9 | mind2web | **N** (human demos) | **Y** (`action_reprs`) | **Y** (`operation`) | **N** (DOM state) | N | N | N | P (HAR) | schema |
| 10 | weblinx | **N** (human demos) | **Y** (`action`) | **Y** | N | P (instructor/navigator) | N | N | N | schema |
| 11 | toolsandbox | – env only | – | – | – | – | – | – | – | no released trajectories |
| 12 | gaia | N | N | N | N | N | **Y** answer | N | N | ***gated*** (401) |
| 13 | crag | N | N | N | N | N | **Y** answer | N | N | workload only |
| 14 | fanoutqa | N | N | N | N | N | **Y** answer | N | N | workload only |
| 15 | musique | N | N | N | N | N | **Y** answer + gold decomposition | N | N | workload only |
| 16 | hotpotqa | N | N | N | N | N | **Y** answer | N | N | schema |

## What the good ones actually look like

### Open-SWE-Traces — richest reasoning

Measured over 400 trajectories (23,457 assistant steps):

- `reasoning_content` on **100%** of assistant steps — median 129 chars, max 9,760
- tool `arguments` on **100%** — median 116 chars
- tool result content on **100%** of 23,057 observations — median 1,012 chars, max 30,047
- a `tools` column carrying the full JSON tool *schemas* offered to the model
- `resolved` ∈ {1, 0, −1} per trajectory, plus reference and model patches

A real step, verbatim:

> **REASONING:** *"Let me start by exploring the repository structure… The issue is about `attr.make_class` not working with Python>=3.7 due to a change in how `type()` handles MRO entry resolution. The fix is to replace `type(name, bases, body)` with `types.new_class(name, bases, body)`."*
> **TOOL CALL:** `str_replace_editor`
> **ARGS:** `{"command": "view", "path": "/workspace/python-attrs__attrs__1.0"}`
> **OBSERVED:** `Here's the files and directories up to 2 levels deep in …`

That is exactly the reasoning → decision → consequence chain, and it is complete.

### tau2-bench — reasoning *and* cost *and* a graded outcome

Ships 27 result files (8–41 MB) directly in the repo under
`data/tau2/results/final/`, each holding **200 simulations × 4 trials**.
Measured on the gpt-4.1 airline file (4,982 messages):

- `timestamp` on **100%** of messages
- `usage` (`prompt_tokens` / `completion_tokens`) on **65.3%**
- `cost` on **69.3%**
- `reward_info` on **100%** of simulations — including `db_check`, and
  `nl_assertions` each with a written **justification** of why the agent passed
  or failed
- `requestor` distinguishes the agent from the simulated user

This is the only dataset in the registry with content **and** tokens **and**
timestamps together. Its trade-off is domain narrowness (airline / retail /
telecom customer service) and scale (~5K messages per file, not 357K rounds).

### ToolBench — the only source of counterfactuals

Answer files carry `tree` alongside `answer_generation`. Tree nodes have
`node_type` (`Action` / `Action Input`), `description` (tool name / arguments),
`observation` (tool output), plus `is_terminal`, **`pruned`**, `finished`,
`depth` and `Elo`.

Because pruned branches survive, this is the one dataset that can answer
*"what did the agent try and reject?"* — not just what it did. Reasoning text is
thinner than the others: in function-calling mode the assistant `content` is
often `None`, with intent carried by the tool choice rather than prose.

## Consequences for this project

1. **TraceLab stays the reference for structure and cost, not for intent.** Its
   value is that it is a real serving trace with real tokens, real timestamps
   and real prefix-cache splits. It can never explain a decision. Both facts
   should be stated together whenever it is cited.

2. **No single dataset has everything.** The corpus needs at least two pillars:
   TraceLab for cost-accurate structure, and Open-SWE-Traces or tau2-bench for
   semantics. tau2-bench is the closest thing to a bridge.

3. **This changes the extraction priority.** Datasets 5 (tau2-bench) and 3
   (agentbank) are worth more than their registry order suggests, because they
   carry the reasoning that the "why was this node called" question needs.
   ToolBench matters disproportionately for branch structure.

4. **The canonical schema needs somewhere to put this.** Reasoning text, tool
   arguments and tool outputs have no home in the current `Node` — they would
   land in the untyped `extra` bag. If semantics become a first-class goal,
   the schema should gain explicit (and explicitly nullable) `reasoning_text`,
   `tool_input`, `tool_output` fields so their absence stays as measurable as
   the cost fields' absence is now.

## Caveats

- AgentBoard rows are classified from its README and HF file listing
  (`data.tar.gz` → `data/baseline_results`), not from opening a rollout.
- xLAM-60k and GAIA both return **401** without an accepted licence agreement.
  APIGen-MT-5k is open and was inspected directly; the xLAM row is inferred from
  its documentation and should be confirmed once access is granted.
- appworld and ToolSandbox publish no trajectory assets in any GitHub release —
  they ship environments, so trajectories require a Phase 2 instrumented run.
