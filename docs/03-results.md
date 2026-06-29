# 03 — Results so far

Everything below was produced on this machine on **2026-06-29**: a single **NVIDIA H100
MIG `1g.24gb` slice** serving **Qwen2.5-7B-Instruct (BF16)** under **vLLM** in Docker.
These are the runs executed to date and the numbers they produced.

---

## A. Deployment & instrument validation (Month-1 milestone ✅)

**Deployed:** `vllm/vllm-openai` container `acg-vllm`, model `qwen2.5-7b-instruct`,
OpenAI-compatible API on `:8000`, `--gpu-memory-utilization 0.85`, `--max-model-len 8192`,
engine seed 1234, tool-calling via the `hermes` parser.

**`scripts/smoke_test.py` — all checks passed:**

| Check | Result |
|-------|--------|
| Server up, model served | `['qwen2.5-7b-instruct']` |
| Plain completion | `'pong'` (in=34, out=2) |
| **Determinism** @ temp 0, seed 1234 | `'Red, Blue, Yellow'` returned **byte-identical twice** → decode is pinnable |
| **Tool-calling** | model emitted `search({"query": "Pirelle Institute"})` |

**`scripts/run_single.py --task T02` — one task end-to-end, graph reconstructed from the trace:**

```
START -> LLM#0 -> tool:search -> LLM#1 -> tool:search -> LLM#2 -> tool:read_document -> LLM#3 -> tool:finish
answer: "…the drell."   outcome: correct
node_count=8  depth=8  width=1  total_tokens=3623  llm_calls=4  tool_calls=4  wall=2.98s
```
Saved `traces/figures/acg_T02.png` and `traces/single_T02.jsonl`. This closes the Month-1
milestone: *one task runs end to end, and we can draw its graph from the captured trace.*

---

## B. Multi-QA variance study (first Month-2 measurements ✅)

`scripts/run_experiment.py --tasks all --reps 8 --temperature 0.7 --vary-seed`

- **96 runs** = 12 QA programs × 8 reps; **846 spans** captured; **342 s** total (3.57 s/run).
- **Overall accuracy: 0.771.**

**Overall per-run distributions (n=96):**

| metric | mean | sd | median | p95 | min | max |
|--------|------|----|--------|-----|-----|-----|
| node_count | 7.81 | 2.19 | 8.0 | 10.5 | 5 | 16 |
| num_llm_calls | 3.98 | 1.07 | 4.0 | 5.2 | 3 | 8 |
| num_tool_calls | 3.83 | 1.15 | 4.0 | 5.2 | 2 | 8 |
| depth | 7.81 | 2.19 | 8.0 | 10.5 | 5 | 16 |
| width | 1.00 | 0.00 | 1.0 | 1.0 | 1 | 1 |
| total_tokens | 3450 | 1371 | 3424 | 5198 | 2170 | 9085 |
| wall_clock_s | 3.57 | 1.78 | 3.3 | 7.3 | 1.2 | 9.3 |

**Per-task ACG size & structural variance (all 12 tasks):**

| task | hops | acc | nodes mean±sd | nodes med/p95/max | depth | tok mean | tok p95 | #distinct shapes | modal frac |
|------|------|-----|---------------|-------------------|-------|----------|---------|------------------|------------|
| T01 | 2 | 1.00 | 5.9±0.3 | 6 / 6 / 6 | 5.9 | 2190 | 2203 | 2 | 0.88 |
| T02 | 3 | 0.75 | 7.8±0.8 | 8 / 9 / 9 | 7.8 | 3814 | 4535 | 3 | 0.50 |
| T03 | 3 | 0.50 | 7.0±1.0 | 7 / 8 / 8 | 7.0 | 2949 | 3599 | 2 | 0.50 |
| T04 | 2 | 1.00 | 6.0±0.0 | 6 / 6 / 6 | 6.0 | 2201 | 2216 | **1** | **1.00** |
| T05 | 3 | 0.50 | 9.1±1.2 | 10 / 10 / 10 | 9.1 | 4231 | 4844 | 4 | 0.50 |
| T06 | 4 | 1.00 | 11.1±2.6 | 11 / 14 / 14 | 11.1 | 5607 | 7822 | **6** | **0.38** |
| T07 | 3 | 0.75 | 7.4±1.1 | 8 / 8 / 8 | 7.4 | 3268 | 3596 | 3 | 0.75 |
| T08 | 2 | 1.00 | 6.0±0.0 | 6 / 6 / 6 | 6.0 | 2316 | 2332 | **1** | **1.00** |
| T09 | 3 | 0.75 | 9.4±1.1 | 10 / 10 / 10 | 9.4 | 4183 | 4606 | 3 | 0.75 |
| T10 | 2 | 1.00 | 5.6±0.5 | 6 / 6 / 6 | 5.6 | 2218 | 2236 | 2 | 0.62 |
| T11 | 3 | 0.00 | 9.0±1.0 | 9 / 10 / 10 | 9.0 | 4015 | 4653 | 2 | 0.50 |
| T12 | 4 | 1.00 | 9.5±3.0 | 10 / 14 / 16 | 9.5 | 4414 | 7631 | 5 | 0.38 |

Artifacts: `traces/experiment.jsonl`, `traces/metrics.csv`, `traces/summary.json`,
`traces/figures/dist_node_count.png`, `traces/figures/dist_total_tokens.png`,
`traces/figures/acg_<task>_modal.png`, `traces/figures/acg_<task>_largest.png`.

**Modal graph shape** (most common structure across 8 runs — LLM + tool node counts):

| task | modal shape (tools) |
|------|---------------------|
| T01 | 3 LLM + 3 tool: search×2, read×1, finish |
| T02 | 4 LLM + 3 tool: search×2, read×1, finish |
| T03 | 4 LLM + 4 tool: search×2, read×1, finish |
| T04 | 3 LLM + 3 tool: search×2, read×1, finish |
| T05 | 5 LLM + 5 tool: search×2, read×2, finish |
| T06 | 7 LLM + 7 tool: search×3, read×3, finish |
| T07 | 4 LLM + 4 tool: search×2, read×1, finish |
| T08 | 3 LLM + 3 tool: search×2, read×1, finish |
| T09 | 5 LLM + 5 tool: search×2, read×2, finish |
| T10 | 3 LLM + 3 tool: search×2, read×1, finish |
| T11 | 5 LLM + 5 tool: search×1, read×3, finish |
| T12 | 5 LLM + 5 tool: search×1, read×3, finish |

All reconstructed graphs are **DAGs** (directed acyclic graphs): each node is one LLM or
tool call; solid edges are forward data dependencies only.

### What these numbers say
1. **Graph size is a distribution, not a number.** For a fixed task+model, cost varies
   substantially run-to-run — e.g. T06's p95 tokens (7822) is **≈1.4× its mean** (5607), and
   its node count ranges 11–14. This is precisely the input a cost/latency optimizer needs,
   and the reason "average cost" is misleading.
2. **Variance scales with task difficulty.** 2-hop tasks are structurally *stable* — T04 and
   T08 produce a **single** graph shape across all 8 runs (modal fraction 1.00, sd 0). 4-hop
   tasks are *not* — T06 produces **6 distinct shapes**; T12 ranges 10–16 nodes. A clean,
   reportable trend: more hops → more structural variance.
3. **Width is consistently 1.** Qwen2.5-7B decomposes these questions **serially** — it never
   issues parallel tool calls in one step. So in this model+domain the variation lives in
   **depth / node-count across runs**, not within-run branching. An honest finding to report
   plainly (and a thing to re-check with a stronger model in Month 3).
4. **Accuracy is uneven and correlates with the harder chains.** T11 (0.00) and T03/T05
   (0.50) are the weak spots — useful to know before scaling reps, since outcome interacts
   with graph shape.

---

## C. Complex-task re-run (3–4 hop tasks ✅)

`scripts/run_complex.py --reps 8` → `traces/complex_experiment.jsonl`

- **64 runs** = 8 tasks (T02, T03, T05, T06, T07, T09, T11, T12) × 8 reps @ temp 0.7,
  varied seed.
- **Overall accuracy: 0.641** (282 s total, 4.4 s/run).
- Trace: `traces/complex_experiment.jsonl`. Re-analyze with
  `./.venv/bin/python scripts/analyze.py --trace traces/complex_experiment.jsonl`.

| task | hops | acc | nodes mean±sd | nodes p95/max | depth mean | width mean | tok mean | tok p95 | #distinct shapes | modal frac |
|------|------|-----|---------------|---------------|------------|------------|----------|---------|------------------|------------|
| T02 | 3 | 0.88 | 8.2±1.9 | 11.6 / 13 | 8.2 | 1.0 | 4222 | 6648 | 4 | 0.50 |
| T03 | 3 | 0.50 | 7.0±1.0 | 8 / 8 | 7.0 | 1.0 | 2949 | 3599 | 2 | 0.50 |
| T05 | 3 | 0.38 | 9.1±1.2 | 10 / 10 | 9.1 | 1.0 | 4253 | 4911 | 4 | 0.50 |
| T06 | 4 | 1.00 | 12.2±3.3 | 15.9 / **17** | 12.1 | 1.1 | 6366 | 8590 | 5 | 0.50 |
| T07 | 3 | 0.75 | 7.4±1.1 | 8 / 8 | 7.4 | 1.0 | 3267 | 3596 | 3 | 0.75 |
| T09 | 3 | 0.75 | 9.4±1.1 | 10 / 10 | 9.4 | 1.0 | 4183 | 4605 | 3 | 0.75 |
| T11 | 3 | 0.00 | 8.8±1.0 | 10 / 10 | 8.8 | 1.0 | 3876 | 4653 | 2 | 0.62 |
| T12 | 4 | 0.88 | 9.5±3.0 | 13.9 / 16 | 9.5 | 1.0 | 4392 | 7579 | 4 | 0.50 |

**Modal graph shapes (complex re-run):**

| task | modal shape (tools) |
|------|---------------------|
| T02 | 4 LLM + 3 tool: search×2, read×1 |
| T03 | 4 LLM + 4 tool: search×2, read×1, finish |
| T05 | 5 LLM + 5 tool: search×2, read×2, finish |
| T06 | 7 LLM + 7 tool: search×3, read×3, finish |
| T07 | 4 LLM + 4 tool: search×2, read×1, finish |
| T09 | 5 LLM + 5 tool: search×2, read×2, finish |
| T11 | 4 LLM + 4 tool: search×1, read×2, finish |
| T12 | 5 LLM + 5 tool: search×1, read×3, finish |

Largest graph in this batch: **T06 at 17 nodes / 8,975 tokens**
(`traces/figures/acg_T06_largest.png`). One T06 run had **width = 2** (parallel tool calls).

---

## D. Single-run exemplar graphs

`scripts/run_single.py` — one trace + PNG per complex task (`traces/single_<task>.jsonl`,
`traces/figures/acg_<task>.png`). Header shows task question + final answer; nodes are
variable-width boxes sized to label text.

| task | outcome | nodes | depth | width | tokens | tool pattern (this run) | PNG |
|------|---------|-------|-------|-------|--------|-------------------------|-----|
| T02 | correct | 8 | 8 | 1 | 3623 | search×2, read×1, finish | `acg_T02.png` |
| T03 | incorrect | 6 | 6 | 1 | 2372 | search×1, read×1, finish | `acg_T03.png` |
| T05 | correct | 8 | 8 | 1 | 3514 | search×2, read×1, finish | `acg_T05.png` |
| T06 | correct | 9 | 9 | 1 | 4408 | search×1, read×3 | `acg_T06.png` |
| T07 | incorrect | 6 | 6 | 1 | 2422 | search×1, read×1, finish | `acg_T07.png` |
| T09 | correct | 10 | 10 | 1 | 4453 | search×2, read×2, finish | `acg_T09.png` |
| T11 | correct | 9 | 8 | **2** | 3624 | search×1, read×3 (parallel), finish | `acg_T11.png` |
| T12 | correct | 10 | 10 | 1 | 4552 | search×1, read×3, finish | `acg_T12.png` |

Regenerate any figure from an existing trace (no model required):

```bash
./.venv/bin/python scripts/draw_graphs.py --trace traces/complex_experiment.jsonl
./.venv/bin/python scripts/draw_graphs.py --trace traces/single_T11.jsonl --tasks T11
```

---

## E. Graph figure inventory

| File pattern | Contents |
|--------------|----------|
| `traces/figures/acg_<task>.png` | One exemplar run per complex task (single-run script) |
| `traces/figures/acg_<task>_modal.png` | Most common graph shape from experiment |
| `traces/figures/acg_<task>_largest.png` | Largest graph seen for that task |
| `traces/figures/dist_node_count.png` | Per-task node-count distribution (boxplot) |
| `traces/figures/dist_total_tokens.png` | Per-task token-cost distribution (boxplot) |

---

## F. Variance decomposition — sampling vs serving noise (§7 bonus ✅)

`scripts/determinism_check.py --task T06 --reps 12` (3 regimes × 12 runs):

| regime | runs | distinct ACG structures | node-count range |
|--------|------|-------------------------|------------------|
| fixed-seed @ temp = 0.0 | 12 | 1 | 12–12 |
| fixed-seed @ temp = 0.7 | 12 | 1 | 9–9 |
| varied-seed @ temp = 0.7 | 12 | **6** | 8–12 |

**Reading:** with the seed fixed, the ACG is **perfectly reproducible even at temperature
0.7** (1 structure across 12 runs). Only when the seed varies do we get 6 distinct
structures. So in this batch, **all** run-to-run structural variance is attributable to
**sampling**, and serving-batch noise contributed none that changed graph structure — exactly
the clean separation the proposal's §7 describes. (This is a single-task, 12-run snapshot, not
yet a thorough claim — see [next steps](04-next-steps.md).)

---

## G. Tests

`pytest tests/` → **7 passed** (~26 s with the live server).

| Test | What it checks |
|------|----------------|
| `test_tool_alphabet_is_fixed` | Tool set is exactly search / read_document / finish |
| `test_corpus_search_and_read` | Deterministic retrieval over owned corpus |
| `test_answer_checker` | Normalized substring grading |
| `test_tasks_have_gold` | All tasks have gold answers |
| `test_single_task_produces_acg` | T01 → valid DAG (LLM + tool nodes, connected, acyclic) |
| `test_multiple_qa_programs_produce_acgs` | T01, T02, T04, T08, T06 each → valid DAG; majority correct |
| `test_variance_machinery_over_repeats` | 4× T02 @ temp 0.7 → sane structural-variance stats |

Live tests skip automatically if the model server is not running.

---

## H. Reproducibility

Every run is fully described by `(model, decode params, seed)`, recorded in
[`config/pinned_settings.yaml`](../config/pinned_settings.yaml). Traces are raw OTel spans
in JSONL, so any result above can be rebuilt offline with
`./.venv/bin/python scripts/analyze.py --trace traces/experiment.jsonl` — no model required.

Complex-task numbers: `--trace traces/complex_experiment.jsonl`. The full 12-task study
remains in `traces/experiment.jsonl`.
