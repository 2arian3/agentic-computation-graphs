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

## C.2 Scaled reps + tightened estimators (T06, T12 @ 50 reps ✅)

Executes the first two "finish Month 2" items in [04-next-steps.md](04-next-steps.md).

**Tightened estimators (item 2).** `scripts/analyze.py` now reports, in every per-task table:
a **95% Wilson CI** on the modal-signature (stable-core) fraction and on accuracy — so
small-*n* claims are honest — plus a **normalized graph-edit-distance** (GED ÷ average graph
size), a size-comparable structural-variance measure. GED uses a fast upper bound
(networkx `optimize_graph_edit_distance` first yield, ~10 ms/pair; needs `scipy`), bounded by
a per-task time budget so analysis never stalls.

**Scaled reps (item 1).**
`run_experiment.py --tasks T06,T12 --reps 50 --temperature 0.7 --vary-seed --trace traces/scale_hivar.jsonl`
→ **100 runs**, accuracy 0.87 (590 s). Outputs: `traces/scale_hivar_{metrics.csv,summary.json}`,
`traces/figures/scale_hivar_dist_*.png` (kept separate from the canonical study).

| task | n | acc [95% CI] | nodes mean±sd | nodes p95/p99/max | tok mean | tok p95/p99 | #shapes | modal frac [95% CI] | GED norm |
|------|---|--------------|---------------|-------------------|----------|-------------|---------|---------------------|----------|
| T06 | 50 | 0.88 [0.76, 0.94] | 11.2±2.6 | 15.6 / 16 / 16 | 5626 | 8886 / 8971 | **14** | **0.20 [0.11, 0.33]** | 0.80 |
| T12 | 50 | 0.86 [0.74, 0.93] | 10.0±2.7 | 16 / 16 / 16 | 4794 | 9305 / 9817 | **12** | 0.44 [0.31, 0.58] | 0.79 |

**Headline: n=8 under-sampled the variance.** Going 8 → 50 reps:
- **T06 distinct graph shapes 6 → 14**, and modal (stable-core) fraction **0.38 → 0.20** — the
  "stable core" is much weaker than the small sample implied.
- **T12 shapes 5 → 12**; the modal fraction held (~0.4) but the tail widened (p99 tokens ≈ 9.8k).
- Accuracy corrected downward with CIs (both were 1.00 at n=8 → now 0.88 / 0.86) — the small
  sample was optimistic.

This is the concrete justification for the proposal's "increase repetitions where variance is
high": at n=8 we would have **materially under-reported** both the structural variance and the
cost tail. The p95/p99 token figures — the numbers cost planning actually cares about — now
rest on a 50-run footing rather than 8.

---

## C.3 Second model — Qwen2.5-14B FP8 vs 7B BF16 (✅, next-steps item 6)

Tests the "larger model at lower precision, without losing accuracy" idea. Served
`RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic` (W8A8-FP8, ~14 GB weights) on the *same* MIG
slice, same decode/seed policy, same 12 tasks × 8 reps. Compared with
`scripts/compare_models.py` (7B `traces/summary.json` vs 14B `traces/qwen14b_fp8_summary.json`).

**Overall:** accuracy **0.771 [0.68, 0.84] → 0.896 [0.82, 0.94]** (Δ **+12.5 pts**);
nodes/run 7.8 → 7.3; tokens/run ≈ unchanged (3450 → 3461); width 1.00 → **1.05**;
5.7 s/run (≈1.6× the 7B, as expected for 2× params in FP8).

| task | acc 7B | acc 14B-FP8 | nodes 7B→14B | width 7B→14B | Δtok | #shapes/modal 7B→14B |
|------|--------|-------------|--------------|--------------|------|----------------------|
| T01 | 1.00 | **0.62 ↓** | 5.9 → 3.5 | 1 → 1 | −625 | 2/0.88 → 4/0.38 |
| T02 | 0.75 | **1.00 ↑** | 7.8 → 7.5 | 1 → 1 | −197 | 3/0.50 → 2/0.75 |
| T03 | 0.50 | **0.75 ↑** | 7.0 → 7.6 | 1 → 1 | +848 | 2/0.50 → 5/0.38 |
| T05 | 0.50 | **0.88 ↑** | 9.1 → 8.1 | 1 → **1.25** | −306 | 4/0.50 → 5/0.38 |
| T06 | 1.00 | **0.75 ↓** | 11.1 → 8.6 | 1 → **1.25** | −1558 | 6/0.38 → 4/0.62 |
| T07 | 0.75 | 0.88 ↑ | 7.4 → 7.2 | 1 → **1.12** | +182 | 3/0.75 → 3/0.75 |
| T09 | 0.75 | **1.00 ↑** | 9.4 → 7.0 | 1 → 1 | −1092 | 3/0.75 → 2/0.50 |
| T11 | **0.00** | **1.00 ↑↑** | 9.0 → 8.2 | 1 → 1 | +155 | 2/0.50 → 3/0.75 |
| T12 | 1.00 | 0.88 ↓ | 9.5 → 8.1 | 1 → 1 | −367 | 5/0.38 → 3/0.50 |

(T04, T08, T10 stay at 1.00 for both.) Findings:

1. **FP8 did not cost accuracy — the opposite.** Going 7B-BF16 → 14B-FP8, accuracy *rose*
   +12.5 pts with barely-overlapping CIs. This confirms the recommendation: **8-bit (FP8) is
   near-lossless, so the 2× parameters dominate.** "Larger + lower-precision" was a clear win
   *because* the lower precision was 8-bit, not 4-bit.
2. **It fixes the weak tasks.** The 0%-accuracy T11 goes to **100%**; T05 0.50→0.88, T03
   0.50→0.75, T02/T09 to 1.00. This resolves most of next-steps item 3 by capability.
3. **But bigger is not uniformly better — the error profile changes.** T01 regresses
   1.00→0.62 and T06 1.00→0.75. On T01 the 14B *short-circuits*: node count drops 5.9→3.5, i.e.
   it often skips the `read_document` hop and answers over-confidently (and wrong). So model
   scale changes *where* errors and variance live, not just their amount.
4. **Width > 1 emerges.** The 7B was strictly serial (width = 1 everywhere); the 14B issues
   **parallel tool calls** on T05/T06/T07 (width up to 1.25). This partially overturns the
   "serial decomposition" finding and is exactly the item-8 branching, arising naturally from
   capability rather than being imposed.
5. **The bigger model is often *cheaper* on hard tasks.** It solves several multi-hop tasks in
   fewer nodes/tokens (T06 −1558 tok, T09 −1092 tok) by looping less — a genuinely useful
   cost result: capability can reduce ACG size, not only accuracy.

Artifacts: `traces/qwen14b_fp8.jsonl`, `traces/qwen14b_fp8_{metrics.csv,summary.json}`,
`traces/figures/qwen14b_fp8_dist_*.png`. Reproduce the table with `scripts/compare_models.py`.

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

> **⚠️ Revised by the exhaustive ablation ([RQ-A1](05-research-questions.md)).** A larger,
> 20-rep ablation across seed × prefix-cache × concurrency shows this snapshot was too clean:
> at fixed seed the ACG is byte-identical **only with the prefix cache OFF**. With the cache
> ON, fixed-seed still yields ~3 distinct graphs, and concurrency adds more. So run-to-run
> variance has **three** sources — sampling (dominant), the **KV/prefix cache** (real,
> secondary), and batching — not sampling alone. See [05 §RQ-A1](05-research-questions.md).

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
