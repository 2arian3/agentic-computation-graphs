# Agentic Computation Graphs — measurement instrument

> Measuring the **size and structure of the graphs that LLM agents generate**, for one
> narrow domain (tool-using multi-hop QA), on a controlled setup we fully own.
> This repo implements **Month 1 ("build & validate the instrument")** and the first
> **Month 2** measurements of the [4-month RA proposal](https://app.notion.com/p/38e5b71a74dc8174a76cc46609171d6f).

When an LLM application performs a task it makes *many* calls — it reasons, calls
tools, feeds results back. The full trace of those calls and their dependencies is the
**Agentic Computation Graph (ACG)**: each node is one LLM call or one tool call, each
edge is a data dependency. Graph size drives cost and latency directly. This project
measures that graph carefully and asks: *for the same task given to the same model, how
much does the graph change from run to run, and what stays stable?*

> **📚 Full documentation** is in [`docs/`](docs/):
> [Implementation](docs/01-implementation.md) ·
> [Usage](docs/02-usage.md) ·
> [Results so far](docs/03-results.md) ·
> [Next steps](docs/04-next-steps.md)

---

## What's here (and the first results)

Everything below was produced on this machine: a single **NVIDIA H100 MIG `1g.24gb`
slice (24 GB)** serving **Qwen2.5-7B-Instruct (BF16)** under **vLLM** in Docker.

### Instrument is validated (Month-1 milestone ✅)
- **Determinism is pinnable.** Same prompt, `temperature=0`, `seed=1234` → byte-identical
  output twice (`scripts/smoke_test.py`). This is the white-box control the variance
  study depends on (Decision 1).
- **Native tool-calling works**, so the loop is genuinely emergent (Decision 2).
- **One task runs end-to-end and its graph is reconstructed from the trace**
  (`scripts/run_single.py`). Example (task T02, 3-hop):

  ```
  START -> LLM#0 -> tool:search -> LLM#1 -> tool:search -> LLM#2 -> tool:read_document -> LLM#3 -> tool:finish
  node_count=8  depth=8  width=1  total_tokens=3623  outcome=correct  wall=2.98s
  ```

### First measurements: graph size & structural variance (Month-2 milestone ✅)
`scripts/run_experiment.py` — **12 QA programs × 8 reps = 96 runs**, `temperature=0.7`,
per-run seeds (so the only thing changing is sampling). Accuracy **0.77**.

| task | hops | acc  | nodes mean±sd | nodes med/p95/max | tok mean | tok p95 | #distinct shapes | modal frac |
|------|------|------|---------------|-------------------|----------|---------|------------------|------------|
| T01  | 2    | 1.00 | 5.9±0.3       | 6 / 6 / 6         | 2190     | 2203    | 2                | 0.88       |
| T04  | 2    | 1.00 | 6.0±0.0       | 6 / 6 / 6         | 2201     | 2216    | **1**            | **1.00**   |
| T08  | 2    | 1.00 | 6.0±0.0       | 6 / 6 / 6         | 2316     | 2332    | **1**            | **1.00**   |
| T02  | 3    | 0.75 | 7.8±0.8       | 8 / 9 / 9         | 3814     | 4535    | 3                | 0.50       |
| T05  | 3    | 0.50 | 9.1±1.2       | 10 / 10 / 10      | 4231     | 4844    | 4                | 0.50       |
| T06  | 4    | 1.00 | 11.1±2.6      | 11 / 14 / 14      | 5607     | 7822    | **6**            | **0.38**   |
| T12  | 4    | 1.00 | 9.5±3.0       | 10 / 14 / 16      | 4414     | 7631    | 5                | 0.38       |

(Full table for all 12 tasks is printed by the script and saved to `traces/summary.json`.)

**First honest findings:**
1. **Graph size is a distribution, not a number.** For the same task+model, total tokens
   vary widely — e.g. T06's p95 (7822) is **~1.4× its mean** (5607). Cost planning must
   reason about the tail, exactly as the proposal argues.
2. **Variance scales with task difficulty.** 2-hop tasks are structurally *stable*
   (T04/T08: one single graph shape across all 8 runs). 4-hop tasks are *not* (T06:
   6 distinct shapes, node counts 11–14). A clean, reportable trend.
3. **Width is consistently 1.** Qwen2.5-7B decomposes these questions *serially* — it
   never issues parallel tool calls. The variation lives in **depth / node-count across
   runs**, not in within-run branching. An honest observation about this model+domain.

### Bonus (§7): the two sources of variance, separated
`scripts/determinism_check.py` on T06 (12 reps per regime):

| regime                  | runs | distinct ACG structures | node range |
|-------------------------|------|-------------------------|------------|
| fixed-seed @ temp = 0.0 | 12   | 1                       | 12–12      |
| fixed-seed @ temp = 0.7 | 12   | 1                       | 9–9        |
| varied-seed @ temp = 0.7| 12   | **6**                   | 8–12       |

With a fixed seed the graph is **perfectly reproducible even at temperature 0.7** — so
in these runs *all* run-to-run structural variance comes from **sampling**, and
serving-batch noise contributed none. Exactly the decomposition §7 promises.

Figures: `traces/figures/dist_total_tokens.png`, `dist_node_count.png`, `acg_T02.png`.

---

## How the instrument works

```
 your QA task ──▶ Agent loop (acg/agent.py) ──▶ local model (vLLM, OpenAI API)
                      │  thin, emergent: ask model → run tool → feed back → repeat
                      │  fixed tools: search / read_document / finish  (acg/tools.py)
                      ▼
              OpenTelemetry GenAI spans (acg/tracing.py)  ──▶  traces/*.jsonl
                      │  one span per LLM call & per tool call; parent/child + acg.depends_on
                      ▼
              offline reconstruction (acg/graph.py)  ──▶  ACG (a DAG) + metrics
                      │  node count by type · edges · depth · width · tokens · latency · outcome
                      ▼
              aggregation (scripts/analyze.py)  ──▶  per-task distributions + structural variance
```

The **agent's parent/child span tree is the graph**. Each LLM-call span records
`acg.depends_on` = the tool nodes whose results it consumed; `acg/graph.py` turns those
explicit data dependencies into the ACG DAG offline, so measurement never affects runs.

### The two settled design decisions
1. **Local model, not a closed product.** A variance study is only meaningful if
   everything except sampling is held constant and known. We pin decode params + seed on
   a local vLLM server. Closed products are reserved for the Month-4 realism check.
2. **A thin emergent loop, not LangGraph.** In a framework *you* draw the graph; then
   you'd be measuring your own structure. Here the **model** decides each step from a
   fixed tool set, so the structure emerges from the model — which is the whole question.

---

## Directory layout

```
acg/                  the instrument (a small Python package)
  config.py           pinned model/decode/seed settings (env-overridable)
  tracing.py          OpenTelemetry GenAI spans + local JSONL exporter
  llm_client.py       traced OpenAI-compatible client (records prompts + tokens)
  tools.py            the fixed tool alphabet: search / read_document / finish
  corpus.py           owned fictional mini-wiki + deterministic retrieval
  agent.py            the thin emergent loop (NOT LangGraph)
  tasks.py            task loading + graded answer checking
  graph.py            reconstruct the ACG from a trace + compute metrics + draw
data/
  corpus.json         16-doc owned, fictional, multi-hop knowledge base
  tasks.jsonl         12 multi-hop QA programs (2–4 hops) with gold answers
docker/
  serve_vllm.sh       deploy the model on the MIG slice via vLLM (primary)
  serve_sglang.sh     same, via SGLang (alternative engine for the §7 study)
  docker-compose.vllm.yml
scripts/
  smoke_test.py       validate determinism + tool-calling
  run_single.py       Month-1: one task end-to-end, draw its ACG
  run_experiment.py   Month-2: many tasks × N reps → distributions + variance
  determinism_check.py §7: separate sampling variance from serving noise
  analyze.py          aggregate any trace into distributions + figures
tests/test_acgs.py    unit tests + live tests that produce AGCs for many QA programs
config/pinned_settings.yaml   human-readable reproducibility manifest
traces/               output: span JSONL, metrics.csv, summary.json, figures/
```

## Quickstart

```bash
# 0. client deps
make venv                       # python3 -m venv .venv + pip install -r requirements.txt

# 1. deploy the model on the 24 GB MIG slice (auto-detects the MIG CDI device)
make serve                      # docker/serve_vllm.sh; wait ~1–2 min for warmup
curl http://localhost:8000/v1/models

# 2. validate the instrument
make smoke

# 3. Month-1 milestone: one task end-to-end + its graph
make single TASK=T06

# 4. Month-2: the multi-QA variance study
make experiment REPS=8          # writes traces/metrics.csv, summary.json, figures/

# 5. bonus: sampling vs serving-batch noise
make determinism TASK=T06

# tests (live ACG tests skip automatically if the server is down)
make test
```

## Reproducing on a fresh 24 GB MIG node

The default model is **Qwen2.5-7B-Instruct** (BF16) — the FP16-class ceiling that fits a
24 GB slice with room for KV cache, ungated, with strong native tool-calling. To go
larger you would leave the FP16 white-box regime (e.g. an AWQ-quantized 14B); the model
is a one-line swap via `ACG_MODEL` (set `ACG_TOOL_PARSER` to match, e.g. `llama3_json`
for Llama-3.x).

**Storage gotcha.** Docker's *containerd image store* defaults to `/var/lib/containerd`.
The vLLM image is ~30 GB on disk, so if that path is on a small root partition the pull
fails with `no space left on device`. Relocate it to a big disk once:

```bash
sudo systemctl stop docker docker.socket containerd
sudo mv /var/lib/containerd /big-disk/containerd && sudo ln -s /big-disk/containerd /var/lib/containerd
sudo systemctl start containerd docker
```

**MIG memory.** The slice reports ~20.9 / 23.8 GiB *free* at startup, so
`--gpu-memory-utilization` must be ≤ ~0.85 (the default here) or the engine fails to
allocate its KV cache.

## What is measured (per the proposal)
- **Per run:** node count split by type (LLM vs tool, and per tool), edge count + the
  dependency structure, **depth** (longest chain → latency), **width** (max branching →
  parallelism), **total tokens** (input+output → cost), per-node + whole-run latency,
  and task outcome.
- **Across runs of a task:** the *distribution* of each size metric (mean/median/p95/max),
  **structural variance** (count of distinct graph shapes + a graph-edit-distance summary),
  and whether a **stable core** exists (the modal-shape fraction).

## Next steps (gated, per the proposal)
- Scale reps where variance is high (T06/T12) for tighter tail estimates.
- Sweep temperature to quantify its effect on graph size.
- Month-4 realism check: run a handful of tasks through a closed product and compare the
  rough graph shape. Then decide whether the next contract targets optimization.
```
