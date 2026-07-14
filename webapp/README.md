# ACG Experiment Dashboard

A web app for **running and inspecting** ACG experiments. It runs the existing
instrument unchanged and makes every stage of a run observable: prompt → retrieval →
ACG construction → reasoning → tools → sub-agents → final answer, plus metrics and a
searchable history.

- **Backend** — FastAPI (`webapp/backend/`) that imports the `acg` package read-only and
  streams a run's spans over SSE. No experiment logic is added or changed.
- **Frontend** — Vite + React + TypeScript (`webapp/frontend/`).

> Architecture + UI↔code mapping: [`docs/09-webapp-architecture.md`](../docs/09-webapp-architecture.md).

---

## What you can do

| Area | Feature |
|---|---|
| **Run** | Pick a model, set decode + agent parameters (temperature, top_p, max_tokens, seed, max_steps, search_top_k, max_tool_workers, `elicit_reasoning`, `enable_sub_agent`), and submit a **preset** or **custom** prompt. |
| **Watch it think** | The ACG builds **incrementally** as spans stream in; the timeline shows each LLM/tool step with inputs, outputs, and reasoning as it happens. |
| **ACG graph** | Interactive SVG — zoom, pan, click a node to inspect its metadata; step-through scrubber; sub-agent subtrees and re-reasoning loops highlighted. |
| **Retrieval** | Every `search` with its results **and relevance scores**, plus documents read. |
| **Report** | Nodes/edges/depth/width/width_executed, LLM vs tool vs overhead timing, token usage, a labeled cost estimate, tool breakdown. |
| **Documents** | View / read / add / edit / delete corpus (and distractor) documents; **rebuild index**; preview the real `corpus.search()`. |
| **History** | Every run stored with config + metrics + answer; open details or **rerun**. |
| **Replay** | Stream any archived `traces/*.jsonl` through the same UI — **works with the model server down**. |

---

## Install & run

### Prerequisites
- The instrument's client deps installed in `.venv` (repo root: `make venv`).
- **Backend deps**: `./.venv/bin/pip install -r webapp/requirements.txt`.
- **Node ≥ 18** (only to build the frontend). If you don't have it, install a local copy
  once, e.g. download the Node 20 binary tarball, or use `nvm`.

### Quickest path (production: one server serves API + UI)
```bash
# from the repo root
./.venv/bin/pip install -r webapp/requirements.txt
webapp/run.sh                      # builds the frontend if needed, serves on :8100
# open http://localhost:8100
```
`run.sh` builds `webapp/frontend/dist` (if missing and Node is present) and starts
`uvicorn`, which serves both the JSON/SSE API and the built SPA.

### Manual build + serve
```bash
cd webapp/frontend && npm install && npm run build && cd -
./.venv/bin/python -m uvicorn webapp.backend.main:app --host 0.0.0.0 --port 8100
```

### Dev mode (hot-reload frontend)
```bash
# terminal 1 — backend
./.venv/bin/python -m uvicorn webapp.backend.main:app --reload --port 8100
# terminal 2 — frontend (proxies /api -> :8100)
cd webapp/frontend && npm run dev        # http://localhost:5173
```

### Live runs vs replay
- **Live runs** call the OpenAI-compatible model server. The dashboard reads the endpoint
  from the instrument's `Config`, so point it at your server with env vars before starting
  `uvicorn`/`run.sh`:
  ```bash
  export ACG_BASE_URL=http://localhost:8001/v1        # the vLLM endpoint
  export ACG_SERVED_MODEL_NAME=llama3.1-8b-instruct   # its --served-model-name
  export ACG_GENAI_SYSTEM=llama                        # or "qwen"
  ```
  > On this machine the vLLM container serves on **`:8001`** (host `:8000` is the OptiRag
  > co-tenant). `config.py`'s default is `:8000`, so the override above is needed for live
  > runs. The header shows a green **“model server up”** pill once it connects.
- The dashboard itself runs on **`:8100`** to avoid clashing with either server.
- **No GPU / server down?** Use the **Replay** panel to stream any file in `traces/` — the
  full visualization (graph, timeline, retrieval, reasoning, report) works identically.

### Swapping the served model from the UI
vLLM serves **one model at a time** on the MIG slice, so switching models means restarting
the container. The dashboard can do this for you: pick a model whose weights are already in
`hf-cache/` and click **“⚡ Serve … on vLLM”**. The backend re-runs `docker/serve_vllm.sh`
with that model's flags (from `backend/serving.py::SERVE_SPECS` — parser, chat template,
`--enforce-eager` for FP8, etc.), streams warmup progress, and polls `/v1/models` until the
new model answers (1–2 min for 7–8B, longer for 14B). It holds the run lock during the swap,
so no experiment is interrupted.

Only the four cached models are offered (Qwen2.5-7B, Qwen2.5-14B AWQ/FP8, Llama-3.1-8B); to
add another, drop its weights in `hf-cache/` and add a `SERVE_SPECS` entry. The swap targets
the vLLM endpoint the **backend** is configured with (`ACG_BASE_URL`, default port parsed
from it), not any per-browser override.

---

## API (backend)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health`, `/api/models`, `/api/defaults`, `/api/prompts` | metadata + presets |
| POST | `/api/runs` | **SSE**: start a live run, stream span events → `run_finished` |
| POST | `/api/replay` | **SSE**: stream an archived trace `{file, trace_id?, speed}` |
| GET | `/api/traces`, `/api/traces/runs?file=` | replay gallery |
| GET/POST/PUT/DELETE | `/api/corpus…` | document CRUD; `/api/corpus/reindex`, `/api/corpus/search` |
| GET/DELETE | `/api/history`, `/api/history/{id}` | experiment history |

The SSE event schema (`kind: "run_started" | "span" | "run_finished" | "error"`) is
identical for live and replay, so the frontend has one rendering path.

---

## How streaming works (no change to the agent)

`acg.tracing.configure_tracing()` installs a **SimpleSpanProcessor**, which exports each
span the instant it ends. `backend/streaming.py` attaches **one extra** span processor to
that same global provider and routes each finished span to its run's queue (by
`acg.run_id`). Remove the dashboard and runs are byte-identical. See
[`docs/09-webapp-architecture.md`](../docs/09-webapp-architecture.md) §1.

---

## Extending

- **New metric** — add it to `ACGMetrics` in `acg/graph.py`; it flows through
  `graph_export.metrics_to_json` into the report automatically. Add a `<Metric>` in
  `frontend/src/components/ReportPanel.tsx`.
- **New execution stage to visualize** — the raw span (prompt, completion, tool args/
  result, timings) is already in each `span` event; add a panel that reads `useRun()`
  and derives from `state.spans` (see `RetrievalPanel.tsx` / `ReasoningPanel.tsx`).
- **New tool / node type** — once it appears as a tool span with `gen_ai.tool.name`, the
  timeline, graph, and retrieval views pick it up; add a color in `AcgGraph.tsx` if you
  want a distinct look.
- **Real-time transport** — SSE is already live. To swap in WebSockets, replace
  `streamRun` in `frontend/src/api/client.ts` and the `_sse_from_worker` generator in
  `backend/main.py`; the event schema stays the same.
- **Model pricing** — edit `PRICE_TABLE` in `backend/pipeline.py` (local models are $0).

---

## Layout

```
webapp/
  backend/
    main.py           FastAPI app: routes + SSE + static SPA
    pipeline.py       build Config from a request, run Agent, reconstruct  (the only runner)
    streaming.py      extra span processor -> per-run event queues (live streaming)
    replay.py         stream archived traces/*.jsonl through the same schema
    graph_export.py   nx.DiGraph -> UI JSON (reuses acg.graph)
    corpus_store.py   document CRUD + reindex + search preview
    history_store.py  experiment history (JSON store)
    presets.py        model/prompt presets + server health
    paths.py          resolved filesystem locations
  frontend/
    src/
      api/            client + SSE reader + types
      state/          run store (reducer) + theme
      lib/            formatting + graph layout
      components/     ExperimentPanel, AcgGraph, ExecutionTimeline, NodeInspector,
                      RetrievalPanel, ReasoningPanel, ReportPanel, ReplayGallery, Card
      views/          Dashboard, Documents, History
  run.sh              build (if needed) + serve
  requirements.txt    backend-only deps
```
