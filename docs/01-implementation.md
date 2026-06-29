# 01 — Implementation

What has been built, how it works, and why it is built this way.

---

## 1. Goal recap

Measure the **size and structure of the graph an LLM agent generates** while solving a
task. Each node in the graph is one LLM call or one tool call; each edge is a data
dependency (one step's output feeds another step's input). For one narrow domain —
**tool-using multi-hop QA** — we run the same task many times and characterize how the
graph's size and shape vary run-to-run, and what stays stable.

The whole system is a **measurement instrument**: a thin agent we fully own, wrapped in
standard tracing, plus an offline reconstruction/analysis layer. Nothing about the
measurement changes how the agent runs.

---

## 2. End-to-end architecture

```
 QA task ─▶ Agent loop (acg/agent.py) ─────────────▶ local model server (vLLM, OpenAI API, :8000)
              │  thin & emergent: ask model → run the tool it asks for → feed result back → repeat
              │  fixed tool alphabet: search / read_document / finish      (acg/tools.py)
              │  retrieval over an owned fictional corpus                  (acg/corpus.py + data/)
              ▼
       OpenTelemetry GenAI spans (acg/tracing.py) ──▶ traces/<run>.jsonl   (local JSONL store)
              │  one span per LLM call and per tool call; parent/child links + acg.depends_on edges
              ▼
       Offline reconstruction (acg/graph.py) ───────▶ ACG as a DAG + per-run metrics
              │  node count by type · edges · depth · width · tokens · latency · outcome
              ▼
       Aggregation (scripts/analyze.py) ────────────▶ per-task distributions + structural variance
                                                       (traces/metrics.csv, summary.json, figures/)
```

The key idea: **the agent's parent/child span tree _is_ the graph.** We never build a
graph object during execution — we just emit standard spans, then rebuild the graph from
them offline. Measurement bugs therefore can never affect the runs.

---

## 3. Components (every file, what it does)

### The instrument — `acg/` package

| File | Responsibility | Key points |
|------|----------------|------------|
| [`config.py`](../acg/config.py) | All pinned settings | A run is fully described by `(model, decode params, seed)`. Dataclasses with env-var overrides. `DecodeParams` (temperature/top_p/max_tokens/seed) is the only thing allowed to vary across the variance study. |
| [`tracing.py`](../acg/tracing.py) | OpenTelemetry setup + exporter | Custom `JSONLFileSpanExporter` appends one OTLP-shaped JSON object per finished span. Uses a **SimpleSpanProcessor** (synchronous) so spans flush in causal order. GenAI semantic-convention attribute keys + an `acg.*` namespace for things the spec doesn't cover (node type, step, `depends_on`). `configure_tracing` is **idempotent** (redirects the output file) because OTel only honors `set_tracer_provider` once per process. |
| [`llm_client.py`](../acg/llm_client.py) | Traced model call | Wraps each `chat.completions.create` in a `chat` span. Records the **pinned request params**, the **exact prompt** (as a `gen_ai.prompt` event), the response/tool-calls, **token usage**, and latency. Returns the span context so the caller can parent tool spans under the LLM call that emitted them. |
| [`tools.py`](../acg/tools.py) | The fixed tool alphabet | `search`, `read_document`, `finish` — advertised to the model as OpenAI function schemas. Pure functions of `(args, corpus)`; **no tracing, no control flow** here, so the node alphabet stays cleanly separated from execution. |
| [`corpus.py`](../acg/corpus.py) | Owned mini-wiki + retrieval | Loads `data/corpus.json`; deterministic keyword-overlap `search` (ties broken by id) and `read`. Determinism here guarantees the **only** stochastic part of the system is the model's sampling. |
| [`tasks.py`](../acg/tasks.py) | Task loading + grading | Loads `data/tasks.jsonl`; `check_answer` is a normalized substring match (drops case/punctuation/articles) against gold aliases. |
| [`agent.py`](../acg/agent.py) | The thin emergent loop | ~120 lines. Builds the `agent.run` root span, then loops: call model → if it returns tool calls, run them (each in a child span) and feed results back → repeat until `finish` / a plain answer / `max_steps`. Records the `depends_on` edges that make offline reconstruction exact. Returns a `RunResult` summary. |
| [`graph.py`](../acg/graph.py) | Reconstruction + metrics + drawing | Reads the JSONL trace, rebuilds each run's ACG as a `networkx.DiGraph` from `acg.depends_on`, computes `ACGMetrics`, and renders the graph as ASCII or PNG. |

### Data — `data/`
- [`corpus.json`](../data/corpus.json) — **16 documents**, a self-contained **fictional**
  world (people, cities, countries, a company, a turbine, an alloy, currencies…). Fictional
  on purpose: the model can't answer from memory, so it must actually chain search→read
  across documents — which is where the branching, multi-hop structure we measure lives.
- [`tasks.jsonl`](../data/tasks.jsonl) — **12 multi-hop QA programs** (2–4 hops), each with
  the question, gold answers (+aliases), hop count, and supporting doc ids.

### Serving — `docker/`
- [`serve_vllm.sh`](../docker/serve_vllm.sh) — **primary.** Runs `vllm/vllm-openai` on the
  MIG slice (auto-detects the CDI device), pins `--seed`, enables tool-calling
  (`--enable-auto-tool-choice --tool-call-parser hermes`), `--gpu-memory-utilization 0.85`.
- [`serve_sglang.sh`](../docker/serve_sglang.sh) — **alternative engine** (same OpenAI API on
  :8000). Provided because SGLang's RadixAttention/serving controls matter for the §7 study.
- [`docker-compose.vllm.yml`](../docker/docker-compose.vllm.yml) — compose form (needs the CDI
  device name in `ACG_GPU_DEVICE`).

### Drivers + analysis — `scripts/`
- [`smoke_test.py`](../scripts/smoke_test.py) — validates server up, plain completion,
  **determinism** (temp 0 + seed → identical twice), **tool-calling**.
- [`run_single.py`](../scripts/run_single.py) — Month-1 milestone: one task end-to-end, print
  the reconstructed ACG (ASCII), metrics, and save a PNG.
- [`run_experiment.py`](../scripts/run_experiment.py) — Month-2: many tasks × N reps → trace +
  per-task distributions + structural variance + figures.
- [`determinism_check.py`](../scripts/determinism_check.py) — §7 bonus: separate sampling
  variance from serving-batch noise.
- [`analyze.py`](../scripts/analyze.py) — importable aggregation library + standalone CLI to
  re-analyze any trace.

### Tests — `tests/`
- [`test_acgs.py`](../tests/test_acgs.py) — unit tests (corpus, grading, tool alphabet) + live
  tests that **run multiple QA programs and assert a valid AGC for each** (skips if the server
  is down).

### Config / meta
- [`config/pinned_settings.yaml`](../config/pinned_settings.yaml) — human-readable
  reproducibility manifest (documentation; runtime config is via env — see `.env.example`).
- `Makefile`, `requirements.txt`, `.env.example`, `.gitignore`.

---

## 4. How a graph is captured and rebuilt (the important detail)

**During a run**, the agent emits exactly these spans (one OTel trace per run):

```
agent.run                        node type = agent_run   (the synthetic START / root)
  ├─ chat (step 0)               node type = llm_call    depends_on = [run]
  │    └─ execute_tool search    node type = tool_call   depends_on = [llm:0]
  ├─ chat (step 1)               node type = llm_call    depends_on = [tool:0:0]
  │    └─ execute_tool read_…    node type = tool_call   depends_on = [llm:1]
  └─ chat (step 2) … finish
```

Each span carries `acg.node_id`, `acg.node.type`, `acg.step`, token counts, duration, and
**`acg.depends_on`** — the list of node ids whose output this node consumed. Tool spans are
parented under the LLM call that emitted them (the natural `llm → tool` edge); each LLM
call's `depends_on` points back to the previous step's tool results (the `tool → llm`
edge). This is the proposal's edge definition ("one step's output feeds another's input")
made explicit.

**Offline**, `graph.py` groups spans by `trace_id`, adds a node per span, and adds an edge
`dep → node` for every `depends_on` entry → the ACG as a DAG. Metrics computed per run:

- **size:** node count (LLM + tool), split by type and per tool; edge count
- **structure:** **depth** = longest dependency chain (edges); **width** = max nodes sharing
  one dependency level (parallelism potential)
- **cost:** input/output/total tokens summed over LLM calls
- **latency:** whole-run wall clock (root span), plus per-node and total LLM/tool time
- **outcome:** correct / incorrect / no_answer

Across runs of a task, `analyze.py` adds: the **distribution** of each metric
(mean/median/p95/max/std), a **graph signature** per run (a compact structural
fingerprint), the **count of distinct signatures** and **modal-signature fraction** (a
proxy for a "stable core"), and a sampled **graph-edit-distance** summary.

---

## 5. Design decisions honored (from the proposal)

1. **Local model, not a closed product.** A variance study is only meaningful if everything
   except sampling is held constant and known. We pin decode params + seed on a local vLLM
   server (validated: temp 0 + seed → byte-identical output). Closed products are reserved
   for the Month-4 realism check.
2. **A thin emergent loop, not LangGraph.** In a framework *you* declare the nodes/edges, so
   you'd measure your own structure. Here the **model** decides each step from a fixed tool
   set, so the structure emerges from the model — which is the entire research question.
3. **Fixed tool alphabet.** Exactly three tools, so every tool node in every graph is one of
   a known, countable set of types — a precondition for comparing graphs across runs.
4. **OpenTelemetry GenAI spans.** The emerging standard for tracing LLM/agent calls; the
   parent/child tree is the graph, and the format is one others can read.

---

## 6. Hardware/serving notes baked into the implementation

- **Model:** `Qwen/Qwen2.5-7B-Instruct` (7.6B, BF16). The 16-bit "white-box" ceiling that
  fits a 24 GB MIG slice with KV-cache headroom; ungated; strong native tool-calling. Swap
  via `ACG_MODEL` (set `ACG_TOOL_PARSER` to match the family).
- **MIG slice:** reports ~20.9 / 23.8 GiB free at startup → `--gpu-memory-utilization` must
  be ≤ ~0.85 (the default) or the engine fails to allocate its KV cache.
- **Docker storage:** the containerd image store must live on a large disk (the vLLM image
  is ~30 GB). See [02-usage.md](02-usage.md#prerequisites) and the top-level README.
