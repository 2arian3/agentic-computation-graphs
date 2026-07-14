# 09 — Web dashboard: architecture analysis & UI mapping

This document (a) analyzes the existing ACG instrument as it stands, and (b) explains how
the web dashboard in [`webapp/`](../webapp/) maps onto it **without changing any
experimental logic**. Read it before touching either side.

---

## 1. The existing pipeline (what we are wrapping, not rewriting)

The instrument is a small, layered Python package. The web app treats it as a library and
an on-disk trace store; it adds no new experiment logic.

```
Task ─▶ Agent.run() ─▶ TracedLLMClient.chat() ─▶ local vLLM (OpenAI API, :8000)
 │         │  emergent loop: ask model → run tool → feed back → repeat
 │         │  fixed tools: search / read_document / finish (+ opt-in sub_agent)
 │         ▼
 │   OpenTelemetry GenAI spans (acg/tracing.py)
 │         │  SimpleSpanProcessor → JSONLFileSpanExporter → traces/*.jsonl
 │         │  ONE span per LLM call & per tool call; parent/child + acg.depends_on
 │         ▼
 └▶ offline reconstruction (acg/graph.py) ─▶ ACG DiGraph + ACGMetrics
           │
           ▼
   aggregation (scripts/analyze.py) ─▶ per-task distributions + structural variance
```

### Modules and the exact surface the dashboard uses

| Module | Role | Used by the dashboard for |
|---|---|---|
| `acg/config.py` | `Config` + `DecodeParams`, all env-overridable | Model + parameter controls map 1:1 onto these fields |
| `acg/corpus.py` | Owned mini-wiki + deterministic keyword retrieval; `.noise` knob | Document manager reads/writes `data/corpus.json`; retrieval view shows `search()` scores |
| `acg/tasks.py` | `Task`, `load_tasks`, `check_answer` | Preset prompts come from `tasks.jsonl` / `tasks_branch.jsonl`; custom prompts become an ad-hoc `Task` |
| `acg/tools.py` | Fixed tool alphabet (`search`/`read_document`/`finish`/`sub_agent`), `tool_schemas` | Reasoning capture (`elicit_reasoning`) and branch tool (`enable_sub_agent`) are toggles |
| `acg/llm_client.py` | `TracedLLMClient` — every chat wrapped in a GenAI span (prompt/completion events, tokens) | Prompt preprocessing + reasoning + token metrics are read from these spans |
| `acg/agent.py` | `Agent.run()` — the thin emergent loop; emits the span tree | The dashboard calls this **unchanged** to run an experiment |
| `acg/tracing.py` | OTel provider + JSONL exporter; `configure_tracing` is idempotent | The streaming hook attaches an **extra** span processor to this same provider |
| `acg/graph.py` | `reconstruct_runs`, `compute_metrics`, `ACGMetrics`, `detect_behavioral_repeats`, `_levels_from_root` | ACG visualization + report metrics |
| `acg/provenance.py` | `capture()` — config + git + serving stack | History entries store provenance |
| `scripts/analyze.py` | `summarize`, `structural_variance`, `graph_signature` | Cross-run report (distributions, distinct shapes, GED) |

### The node/edge model (what the graph viz renders)

Every span carries an `acg.*` attribute namespace (see `acg/tracing.py`):

- **Node id scheme** — root `run:<run_id>` (`agent_run`); LLM `llm:<step>` (`llm_call`);
  tool `tool:<step>:<idx>` (`tool_call`). Sub-agent subtrees are **namespaced** with a
  `/` prefix (e.g. `tool:0:1/llm:0`), which is how `graph.py` distinguishes nested from
  top-level nodes.
- **Edges** come from `acg.depends_on` (a JSON list of node ids whose output a node
  consumes): `tool → llm` (an LLM step consumes the previous step's tool results) and
  `llm → tool` (a tool was emitted by that LLM call).
- **Payloads live in span events**, not attributes: `gen_ai.prompt` (the exact messages),
  `gen_ai.completion` (content + tool calls), `acg.tool.result` (the tool's JSON output).

### The one fact that makes live streaming free

`configure_tracing()` installs a **`SimpleSpanProcessor`**, which exports each span
**synchronously the instant it ends**. So the backend can attach a *second* span
processor to the same global provider and receive every finished span in real time —
**no change to `agent.py` or any experiment code**. That is the whole streaming
mechanism (`webapp/backend/streaming.py`).

---

## 2. How the UI maps onto the codebase

| UI area | Backed by | Notes |
|---|---|---|
| **Model select** | `Config.model` / `base_url` / `gen_ai_system` | Live list from `GET /v1/models` when the server is up, plus known presets |
| **Parameters** | `DecodeParams` (`temperature`, `top_p`, `max_tokens`, `seed`) + `max_steps`, `search_top_k`, `max_tool_workers`, `elicit_reasoning`, `enable_sub_agent` | Every control is an existing `Config` field; nothing new is invented |
| **Custom prompt** | ad-hoc `Task(task_id="CUSTOM", question=…, answers=[])` | Ungraded (no gold answer) — outcome shown as *ungraded* |
| **Preset prompts** | `data/tasks.jsonl` + `data/tasks_branch.jsonl` via `load_tasks` | Includes hops + gold answers so outcome is graded |
| **Final response** | `RunResult.answer` | |
| **Document manager** | `data/corpus.json` (+ `data/distractors.json`) | CRUD + "reindex" (a `Corpus.load` round-trip; the index rebuilds on load) |
| **Execution timeline / progressive stages** | SSE stream of finished spans | prompt → retrieval (search results + scores) → tool calls → sub-agents → final answer |
| **ACG visualization** | `graph.reconstruct_runs` → JSON (`graph_export.py`) | Interactive SVG: level layout from `_levels_from_root`, node click → span metadata |
| **Reasoning** | `elicit_reasoning` `thought` args, or LLM completion content | Shown per step when present |
| **Report / metrics** | `ACGMetrics` + `analyze.summarize` | Timing split (LLM vs tool vs wall), tokens, depth/width, distinct shapes, GED; cost is an explicit, labeled token-price estimate |
| **History** | JSON store under `webapp/data/history/` | Stores request + result + metrics + graph + provenance; "rerun" re-submits the stored config |

### Two execution paths, one event schema

1. **Live run** — `POST /api/runs` builds a `Config` from the request, runs
   `Agent.run()` in a worker thread, and streams normalized span events over SSE while
   the run proceeds. Requires the vLLM server on `:8000`.
2. **Replay** — `POST /api/replay` streams an existing `traces/*.jsonl` (spans re-emitted
   in finish order with capped inter-event delays), then reconstructs the ACG. This makes
   the entire visualization/debugging experience usable **with the GPU server down**, and
   turns the ~30 archived traces into a browsable gallery.

Both paths emit the **same** event schema (`span` … `run_finished`), so the frontend has
one code path for rendering.

---

## 3. Ports, processes, and boundaries

- **vLLM model server**: `:8000` (unchanged; owned by the instrument).
- **Dashboard backend** (`uvicorn`): `:8100` — deliberately *not* 8000, to avoid clashing
  with the model server. Serves the JSON/SSE API and (in production) the built frontend.
- **Frontend dev server** (Vite): `:5173`, proxying `/api` → `:8100`.

The UI never imports experiment internals directly; it speaks only to the backend API.
The backend is the *only* place that imports `acg`, and it imports it read-only (plus the
document-manager writes to `data/corpus.json`, which is the app's stated job). No file in
`acg/` or `scripts/` is modified.

---

## 4. What is intentionally *not* changed

- The agent loop, tool alphabet, tracing schema, graph reconstruction, and metrics are
  untouched. The dashboard is an observer + a thin config/prompt front-end.
- Streaming is additive (an extra span processor). If it were ever removed, runs would be
  byte-identical.
- Determinism guarantees still hold: a live run with a fixed seed + `temperature=0`
  produces the same trace whether launched from the CLI or the dashboard.

See [`webapp/README.md`](../webapp/README.md) for install/run/extend instructions.
