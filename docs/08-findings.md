# 08 — Findings & RQ answers (living summary)

**The single-page scoreboard of what we have learned and which research questions are answered.**
This is the quick-reference companion to the chronological record in
[07 — Experiment log](07-experiment-log.md); for the *why* behind the questions see
[05 — Research questions](05-research-questions.md) and [06 — Critical review](06-critical-review-and-directions.md).

> **Living document — keep it updated.** Every time an experiment runs or an RQ is answered, update
> the scoreboard (§2), add/refresh the relevant data table (§3), and re-check the robust/provisional
> split (§4). Update the "Last updated" line below.

**Last updated:** 2026-07-13 · **~1,550 runs** · 4 model/precision configs across 2 families · 1 domain
(tool-using multi-hop QA over a fictional 16-doc corpus). Latest: RQ-N3 (cross-family, Llama-3.1-8B).

---

## 1. Scope — models measured

| # | Model | Precision | Weights | Tool parser |
|---|-------|-----------|---------|-------------|
| M1 | Qwen2.5-7B-Instruct | BF16 (16-bit) | ~15 GB | hermes |
| M2 | Qwen2.5-14B-FP8-dynamic | FP8 (8-bit) | ~14 GB | hermes |
| M3 | Qwen2.5-14B-AWQ | AWQ (4-bit) | ~9 GB | hermes |
| M4 | Llama-3.1-8B-Instruct | BF16 (16-bit) | ~15 GB | llama3_json |

All on one 24 GB H100 MIG slice via vLLM, one model at a time. Decode defaults: temp 0.7, top_p 0.95,
per-run varied seed (fixed 1234 for determinism runs).

---

## 2. RQ scoreboard

### Original supervisor questions
| RQ | Question | Answer |
|----|----------|--------|
| Q1 | Where does run-to-run variance come from? | **Sampling ≫ KV/prefix cache ≫ batching.** Cache-off ⇒ bit-exact; cache-on ~doubles graph size. |
| Q2 | What structure do the graphs have? | **Linear-dominant** (0.84), parallelism ~0, small stable core. Not branchy. |
| Q3 | Can we capture the model's reasoning/decisions? | **Yes.** Per-step reasoning elicited; characteristic failure = penultimate-hop short-circuit. |

### RQ-A/C/D/E
| RQ | Question | Answer |
|----|----------|--------|
| A1 | Sampling vs KV-cache? | Sampling dominates; **KV cache a real 2nd source** (cache-off = byte-identical). |
| A2 | How many reps needed? | **~50/task** (n=8 under-samples: T06 shapes 6→14). |
| A3 | Temperature's effect? | **Master knob** — distinct structures 2→13 as temp 0→1. |
| C1/C1b | Elicit reasoning without distortion? | Yes; **low-distortion** (+0.08 acc within noise, +14% tokens, width unchanged). |
| C2 | Are finish decisions calibrated? | **Yes** — P(finish│answer)=0.46 vs 0.05; premature finishes **100% wrong**. |
| D1 | Does 8-bit hurt? | **No — near-lossless** (0.77→0.90); width>1 emerges. |
| D2 | Does 4-bit hurt? | **Yes — collapses** (0.41, below 7B) via tool-protocol breakdown. |
| E1 | Cost predictable pre-run? | **Yes, R²=0.93** from hop count (caveat: `hops` partly circular). |
| E2 | Is there a stable core? | **Yes, small** (T06=3 steps, T12=1); core depth ⊥ hops. |

### RQ-N (critical-review validity gates)
| RQ | Question | Status | Answer |
|----|----------|--------|--------|
| N1 | Structure from model or corpus? | ✅ | **Model** — survives retrieval noise; cause = non-adaptivity. |
| N2 | Concurrent executor + branch tools change the graph? | ✅ | Executed width **can** exceed 1 (via `sub_agent`), but **emitted ≠ executed**; models still linearize. |
| N8 | Do branch-*requiring* tasks force fan-out? | ✅ | **Mostly no** — models linearize even when invited. |
| N3 | Generalize across families? | ✅ | **Yes** — Llama linearizes *even more strictly* than Qwen. |
| N4 | Causal (not transcript-order) graph? | ⬜ next | — |
| N5 | Quantify optimization waste headroom? | ⬜ next | — |
| N6/N7/N9 | Controller · graph-voting · online cost | ⬜ | — |

---

## 3. Findings with data

### A. Variance & determinism (7B, task T06)
| Regime | Distinct ACG structures |
|--------|------------------------|
| fixed-seed, cache-ON | 3 |
| fixed-seed, cache-OFF | **1 (byte-identical)** |
| concurrent ×8 | 6 |
| varied-seed | 9 |

**KV cache @ temperature 0** (even greedy is non-reproducible with cache on):
| Regime | Structures | Nodes | Tokens |
|--------|-----------|-------|--------|
| cache-ON | 2 | 13.7 | 7364 |
| cache-OFF | 1 | 8.0 | 3303 |

**Temperature sweep** (T06 ×20): distinct structures **2 → 8 → 9 → 13** at temp {0, 0.3, 0.7, 1.0}.

### B. Precision & scaling (canonical 12-task suite)
| Model | Precision | Accuracy | Structure notes |
|-------|-----------|----------|-----------------|
| 7B | BF16 | 0.771 | baseline, width=1 |
| 14B | FP8 (8-bit) | **0.896** | width>1 emerges; weak task T11 0.00→1.00 |
| 14B | AWQ (4-bit) | **0.406** | 45% degenerate short-circuit; linear 0.83→0.31 |

### C. Structure taxonomy (7B, 96 runs)
| Motif | Prevalence |
|-------|-----------|
| linear_chain | **0.84** |
| iterative_multihop | 0.58 |
| redundant_loop | 0.02 |
| parallel_fanout | **0.00** |

**Size is a distribution, not a number** (variance scales with hops):
| Task | Hops | Nodes mean±sd | #distinct shapes | Modal frac |
|------|------|--------------|-----------------|-----------|
| T04 | 2 | 6.0±0.0 | 1 | 1.00 |
| T01 | 2 | 5.9±0.3 | 2 | 0.88 |
| T06 | 4 | 11.1±2.6 | 6 (→14 at n=50) | 0.38 |
| T12 | 4 | 9.5±3.0 | 5 | 0.38 |

### D. Decisions & failure modes (RQ-C2, 180 runs, 697 decision points)
| Metric | Value |
|--------|-------|
| P(finish │ answer in context) | 0.46 |
| P(finish │ answer NOT in context) | 0.05 |
| Premature finishes | 19 → **19/19 wrong** (penultimate-hop short-circuit) |
| Over-continuations | 166 |

### E. Cost & stable core
- **Cost predictor (E1):** hop count → p95 node-count & tokens at **R²=0.93** (LOO-MAE ~0.7 nodes / 440 tokens).
- **Stable core (E2):** T06 has a 3-step core then fans out; T12 core=1. Core-depth↔hops **r=−0.06**; node-count↔hops **r=0.90**.

### F. Corpus-noise sweep (RQ-N1, 7B, 288 runs) — structure barely moves
| Noise | Accuracy | Nodes | Width | Linear |
|-------|----------|-------|-------|--------|
| 0 | 0.77 | 7.7 | 1.01 | 0.82 |
| 1 | 0.73 | 8.3 | 1.03 | 0.84 |
| 2 | **0.56** | 8.0 | 1.01 | **0.91** |

Accuracy falls hard, but the agent plows the same linear chain — it fails by mis-reading a distractor, never re-querying.

### G. Branch-tool matrix (RQ-N2/N8/N3) — emitted vs executed parallelism
6 compare-over-3 branch tasks × 8 reps. `emit/turn` = max tool calls issued in one turn (emitted
parallelism, nesting-robust); `exec_w` = tool spans actually overlapping in wall-clock (executed).

| Model | cfg | acc | %err | emit/turn (mn/mx) | %emit≥2 | exec_w (mn/mx) | %fanout | %used_sub |
|-------|-----|-----|------|-------------------|---------|----------------|---------|-----------|
| Qwen-7B (16b) | plain | 0.60 | 0 | 1.06 / 2 | 0.06 | 1.00 / 1 | 0 | – |
| Qwen-7B (16b) | +sub | 0.71 | 0 | 1.17 / 3 | 0.12 | 1.10 / 3 | 0.08 | 0.50 |
| Qwen-14B FP8 (8b) | plain | 0.81 | 0 | 0.96 / 2 | 0.08 | 0.88 / 1 | 0 | – |
| Qwen-14B FP8 (8b) | +sub | 0.56\* | 0 | 0.85 / 3 | 0.10 | 0.75 / 3 | 0.02 | 0.06 |
| Qwen-14B AWQ (4b) | plain | 0.42 | 0 | 2.06 / **8** | 0.42 | 0.73 / 1 | 0 | – |
| Qwen-14B AWQ (4b) | +sub | **0.90** | 0 | 2.73 / 5 | 0.94 | 1.42 / 3 | **0.31** | 0.52 |
| Llama-3.1-8B (16b) | plain | 0.69 | 0 | **1.00 / 1** | 0 | 1.00 / 1 | 0 | – |
| Llama-3.1-8B (16b) | +sub | 0.52 | **0.10** | 1.29 / 4 | 0.10 | 1.19 / **4** | 0.10 | 0.50 |

\*FP8+sub = tool-protocol breakdown (60% unparsed `<tool_call>`), not a branching result.

1. **Emitted ≠ executed parallelism.** AWQ emits parallel batches in 42% of runs (≤8/turn) yet exec_w=1 — near-instant `search`/`read` never overlap. Real concurrency needs the latency-bearing `sub_agent`.
2. **`sub_agent` unlocks real executed concurrency** (7B 8%, AWQ 31%, Llama 10%; exec_w up to 4) — the only source of `width_executed`>1 in the project.
3. **Models linearize even with parallelism available** — ~50% adopt `sub_agent` but call it serially (family-invariant Qwen-7B ↔ Llama).
4. **`sub_agent` accuracy effect is model-specific/non-monotonic:** rescues AWQ (0.42→0.90), helps 7B (0.60→0.71), hurts FP8 (tool-protocol) and Llama (template).
5. **Emitted parallelism anti-correlates with capability** — the degenerate 4-bit emits the most (42%); careful 16-/8-bit emit ~1/turn.
6. **Cross-family (RQ-N3):** plain Llama is the strictest linearizer of all 8 cells (emit/turn ≡ 1); Llama's fan-out *executes* (exec_w 4) but can't complete on vLLM's `llama3_json` template (a serving limit, not a model one).

---

## 4. Robust vs. provisional

| Finding | Status | Why |
|---------|--------|-----|
| Variance sources (sampling ≫ cache ≫ batch) | **Robust** | Clean mechanism, model-agnostic |
| Precision floor (8-bit safe, 4-bit collapses) | **Robust** | Decision-relevant, cross-config |
| Decision calibration / short-circuit failure | **Robust** | Genuine behavioral property |
| Cost predictability | **Robust** (caveated) | R²=0.93 but `hops` circular |
| Emitted ≠ executed parallelism | **Robust** | Clean measurement distinction |
| Linear-dominant / small stable core | **Now strong** | Survived corpus-noise (N1), concurrent executor + tools (N2/N8), and a 2nd family (N3) |

**Thesis (cross-family):** *ACG structure is a rigid property of the agent's policy, largely decoupled
from task difficulty and stable across families; agents default to linear chains even when a concurrent
executor and a branch tool make fan-out available, and fail by mis-execution rather than adaptive
restructuring.*

---

## 5. Open — not yet answered

| RQ | What's needed |
|----|---------------|
| **N4** | Causal graph via counterfactual ablation (which reads actually influenced the answer) |
| **N5** | Waste-headroom % — the gate for whether optimization is worth a contract |
| N6/N7/N9 | Failure-mode controller · graph-level voting · online (mid-run) cost prediction |
| — | A **reasoning/long-CoT** model (very different token/graph profile) |
| — | A **realistic latency-bearing corpus** (so emitted parallelism can become *productive* executed parallelism) |

Highest-leverage next step: **RQ-N4/N5** — converts the (now solid) characterization into a go/no-go on optimization.
