# 07 — Experiment log (complete record)

A single authoritative record of **every experiment run**, the **model served** for each, the
**settings**, the **results**, and the **takeaway**. For the *why* behind the questions see
[05](05-research-questions.md); for the critical framing see [06](06-critical-review-and-directions.md).

**Status (2026-07-13):** instrument built + validated; one narrow domain (tool-using multi-hop
QA over a fictional 16-doc corpus) characterized across 3 model/precision configs, a retrieval-noise
sweep, and a **branch-tool matrix (RQ-N2/N8)**; **~1,450 recorded runs** in `traces/`. All original
supervisor questions answered; RQ-N1 reframed the open question from *corpus* to *agent capability*,
and **RQ-N2/N8** then showed the linearity is not an executor artifact either — with a concurrent
executor + a `sub_agent` branch tool + branch-requiring tasks, the models still linearize by policy
(executed width does exceed 1, but only in a minority of runs). Next confound: **RQ-N3** (a non-Qwen
family). Nothing has been run on rented GPUs yet.

---

## A. Serving configurations (the models used)

All models served with **vLLM** (`vllm/vllm-openai` Docker) on **one NVIDIA H100 MIG `1g.24gb`
slice** (24 GB), OpenAI-compatible API, **one model at a time** (the slice holds one). Tool
calling via the **`hermes`** parser. Engine seed 1234.

| # | Model served | Precision | Weights | vLLM serving settings | Used for |
|---|---|---|---|---|---|
| M1 | **Qwen2.5-7B-Instruct** | BF16 (16-bit) | ~15 GB | max-model-len 8192→4096*, gpu-util 0.85→0.72*, cache on/off per-exp | **Baseline / canonical** — most experiments |
| M2 | **RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic** | FP8 (8-bit W8A8) | ~14 GB | max-model-len 8192, gpu-util 0.85, cache on | Model scaling + 8-bit precision (RQ-D1) |
| M3 | **Qwen2.5-14B-Instruct-AWQ** | AWQ (4-bit) | ~9 GB | max-model-len 4096, gpu-util 0.72, cache on | 4-bit quantization floor (RQ-D2) |

\* **Config change mid-project:** the slice was later contended by another project (`optirag`)
that took port 8000 and ~3.4 GB, so from RQ-A1 onward M1 was re-served on **port 8001** with
**max-model-len 4096** and **gpu-util 0.72** to fit ~18 GB free. Cross-check: the 96-run 7B study
at 8192 (`experiment.jsonl`) and the 7B noise-0 run at 4096 (`noise0.jsonl`) give identical
accuracy (0.77) and node count (7.8 vs 7.7) — **the context/util change did not materially affect
results.** Decode defaults everywhere: temperature 0.7 (unless noted), top_p 0.95, per-run seed
varied 1000+rep for variance runs / fixed 1234 for determinism runs.

Domain (all experiments unless noted): **12 multi-hop QA tasks** (2–4 hops) over a fictional
16-document corpus (`data/corpus.json`), fixed 3-tool alphabet (search / read_document / finish),
deterministic keyword retrieval.

---

## B. Experiment log (chronological, by phase)

Each row: **model served · script · trace · settings · result · takeaway.**

### Phase 0 — Instrument build & validation (Month 1) · model M1 (7B BF16)
| Exp | What | Result |
|---|---|---|
| 0.1 Smoke test | `smoke_test.py` — server up, determinism, tool-calling | temp 0 + seed 1234 → **byte-identical** output twice; model emits valid tool calls ✅ |
| 0.2 Single run | `run_single.py --task T02` → `single_T02.jsonl` | 1 task end-to-end; **ACG reconstructed & drawn** (8 nodes, correct). Milestone met. |
| 0.3 Per-task single runs | `run_complex.py` → `single_T0*.jsonl` (8 traces) | one drawn ACG per complex task; e.g. T11 shows a **width-2** run (rare parallel tool call). |

### Phase 1 — Characterization of the domain (Month 2) · model M1 (7B BF16)
| Exp | Setup | Result |
|---|---|---|
| **1.1 Main variance study** | `run_experiment.py` all 12 tasks × 8 reps = **96 runs**, temp 0.7, varied seed → `experiment.jsonl` | **accuracy 0.771**; nodes 7.8, tokens 3450, width 1.00. Per-task size distributions + first structural-variance numbers. |
| 1.2 Scaled reps (RQ-A2) | T06,T12 × **50 reps** = 100 runs → `scale_hivar.jsonl` | **n=8 under-samples:** T06 distinct graph shapes **6 → 14**, modal fraction 0.38 → 0.20. Rep budget ⇒ ~50/task. |
| 1.3 Rep-budget @100 | `variance_sources.py` T06 × **100** (varied seed) → `variance_sources.jsonl` | distinct structures **13** (plateau by n≈50); trajectories keep rising (55). |
| 1.4 Structure taxonomy (Q2) | `structure_taxonomy.py` on 96-run 7B trace (analysis-only) | **linear_chain 0.84**, iterative_multihop 0.58, **parallel_fanout 0.00**, redundant_loop 0.02. *Not* branch/parallel — linear dominant. |
| 1.5 Stable core (RQ-E2) | `branch_points.py`, `stable_core_map.py` (analysis) | T06 has a **3-step stable core** then fans out at step 3; T12 core = 1. Core depth **uncorrelated with hops** (r=−0.06); node-count↔hops **r=0.90**. |
| 1.6 Cost predictor (RQ-E1) | `cost_model.py`, leave-one-out on 12 tasks (analysis) | hop count predicts **p95** node-count & tokens at **R²=0.93** (LOO-MAE ~0.7 nodes / 440 tok). Cost predictable pre-run. |

### Phase 2 — Variance mechanism / "is it the KV cache?" (supervisor Q1) · model M1 (7B BF16, :8001)
| Exp | Setup | Result |
|---|---|---|
| 2.1 §7 determinism (initial) | `determinism_check.py` T06, 3 regimes × 12 → `determinism_T06.jsonl` | first pass: fixed seed → 1 structure (later shown incomplete — see 2.2). |
| **2.2 Variance ablation (RQ-A1)** | `variance_sources.py` T06 × 20/arm, temp 0.7 → `_rqA1_cache{ON,OFF}.log` | **fixed-seed cache-ON → 3** structures; **cache-OFF → 1** (byte-identical); **concurrent×8 → 6**; **varied-seed → 9**. Variance = **sampling ≫ KV/prefix cache ≫ batching**. |
| **2.3 KV-cache @ temp 0** | `variance_sources.py` T06 × 20, temp 0, fixed seed → `kv_temp0_cache{ON,OFF}.jsonl` | **cache-ON → 2 structures (13.7 nodes, 7364 tok); cache-OFF → 1 (8.0 nodes, 3303 tok).** Even greedy is non-reproducible with the cache on. **The cache ~doubles graph size** (modal shift, not just jitter). Divergence traced to a single token flip at step 3. |
| 2.4 Temperature sweep (RQ-A3) | T06 × 20, temp {0,0.3,0.7,1.0}, varied seed | distinct structures **2 → 8 → 9 → 13**; distinct answers up to 16/20 at temp 1.0. Temperature = master variance knob. |

### Phase 3 — Reasoning & decisions (supervisor Q3) · model M1 (7B BF16, :8001)
| Exp | Setup | Result |
|---|---|---|
| 3.1 Reasoning capture (RQ-C1) | `ACG_ELICIT_REASONING=1` demo → `elicited.jsonl`; `reasoning_viewer.py` | per-step Saw→Reasoned→Decided viewer. Base 7B verbalizes reasoning only in *synthesis* steps; elicitation adds a `thought` at **every** step. |
| 3.2 Elicitation on vs off (RQ-C1b) | T03,T05,T06 × 12 each → `elicit_off.jsonl` / `elicit_on.jsonl` | accuracy 0.50 → 0.58 (within noise), width unchanged, **+14% tokens**. Elicitation is **low-distortion**. |
| **3.3 Decision analysis (RQ-C2)** | 12 tasks × 15 elicited = **180 runs** → `rqc2_elicited.jsonl`; `decision_analysis.py` | 697 decision points. **P(finish│answer in ctx)=0.46 vs 0.05** (well-calibrated). **19 premature finishes → 19/19 WRONG**; they are **penultimate-hop short-circuits** (stops at the town instead of the country). 166 over-continuations. |

### Phase 4 — Model size & precision (RQ-D) · models M2 (14B FP8), M3 (14B AWQ)
| Exp | Model | Setup | Result |
|---|---|---|---|
| **4.1 14B FP8 (RQ-D1)** | **M2 · Qwen2.5-14B FP8 (8-bit)** | 12 × 8 = 96 → `qwen14b_fp8.jsonl`; `compare_models.py` vs 7B | **accuracy 0.771 → 0.896** (8-bit near-lossless), fewer nodes on hard tasks, **width>1 emerges** (T05/T06/T07), weak task T11 0.00→1.00. But new **short-circuits** (T01 1.00→0.62). |
| **4.2 14B AWQ (RQ-D2)** | **M3 · Qwen2.5-14B AWQ (4-bit)** | 12 × 8 = 96 → `qwen14b_awq.jsonl` | **accuracy collapses to 0.406** — *below the 7B*, −0.49 vs the same 14B at 8-bit. Cause: **degenerate short-circuit 45% of runs** (vs 4% FP8), linear-chain 0.83→0.31. **8-bit safe, 4-bit unsafe for agents.** |

### Phase 5 — Validity gate (RQ-N1) · model M1 (7B BF16, :8001)
| Exp | Setup | Result |
|---|---|---|
| **5.1 Corpus-noise sweep (RQ-N1)** | 16 near-homophone distractors (`data/distractors.json`) + retrieval-noise knob; 12 × 8 at noise 0/1/2 = **288 runs** → `noise{0,1,2}.jsonl`; `run_noise_sweep.py` | accuracy **0.77 → 0.73 → 0.56** (noise bit hard) but **structure ≈ flat** (width ≈1, nodes flat, *more* linear 0.82→0.91). **Agent does NOT restructure under noise** — it fails by mis-reading a distractor. ⇒ linear structure is **not a clean-corpus artifact**, but is a **rigid, non-adaptive model policy**. |

### Phase 6 — Branch tools & executed parallelism (RQ-N2 / RQ-N8) · models M1, M2, M3
**Setup.** The executor was fixed to run a step's emitted tool calls **concurrently**
(`cfg.max_tool_workers`, default 8) and the `width` metric split into **emitted** (structural) vs
**`width_executed`** (top-level tool spans that actually overlap in wall-clock time). Added an opt-in
**`sub_agent`** branch tool — a nested, corpus-grounded assistant whose subtree hangs in the same
trace, so the ACG becomes a real tree — and **6 branch-*requiring* tasks** (`data/tasks_branch.jsonl`:
"which of the three countries on Orrin has a capital that is coastal / inland / a mountain town / …",
each needing a per-country sub-chain). Matrix: **3 models × {plain, +sub_agent} × 6 tasks × 8 reps =
288 runs**, temp 0.7, varied seed, prefix-cache OFF, concurrent executor; `run_experiment.py
--tasks-file …`, one `*_provenance.json` sidecar per cell. FP8 was served with `--enforce-eager` (a
MIG cudagraph NVML-assert workaround; recorded in provenance). `scripts/analyze_branch.py` aggregates.

| model | cfg | acc | emit/turn mean/max | exec_w mean/max | %runs exec≥2 | %used sub_agent |
|---|---|---|---|---|---|---|
| 7B (16-bit)    | plain | 0.60 | 1.06 / 2 | 1.00 / 1 | 0.00 | – |
| 7B (16-bit)    | +sub  | 0.71 | 1.17 / 3 | 1.10 / 3 | 0.08 | 0.50 |
| 14B FP8 (8-bit)| plain | 0.81 | 0.96 / 2 | 0.88 / 1 | 0.00 | – |
| 14B FP8 (8-bit)| +sub  | 0.56\* | 0.85 / 3 | 0.75 / 3 | 0.02 | 0.06 |
| 14B AWQ (4-bit)| plain | 0.42 | 2.06 / **8** | 0.73 / 1 | 0.00 | – |
| 14B AWQ (4-bit)| +sub  | **0.90** | 2.73 / 5 | 1.42 / 3 | **0.31** | 0.52 |

\* **FP8+sub is a tool-protocol breakdown, not a branching result:** 60% of runs emit an unparsed
`<tool_call>{…}` as message *content* (vs 4% plain), so the loop takes it as the final answer and stops
(nodes 9.4→4.1). Adding a 4th tool destabilized the 8-bit model's hermes formatting; 7B-sub (0%) and
AWQ-sub (10%) are unaffected, so the other cells stand.

**Findings.**
1. **Emitted ≠ executed parallelism (the measurement point).** With plain tools, models *emit* parallel
   batches (AWQ **42%** of runs, up to **8** calls/turn) but **`width_executed` stays 1** — in-memory
   `search`/`read` finish too fast to overlap. The concurrent executor is **necessary but not
   sufficient**. Genuine executed concurrency appears only with the latency-bearing `sub_agent`
   (nested LLM calls): **exec_w up to 3, in 8% (7B) / 31% (AWQ) of runs** — the first `width_executed>1`
   in the project. *(A metric bug that keyed "nested" on graph ancestry instead of the `/`-namespaced
   id had hidden this; fixed in `graph._top_level_tool_nodes`.)*
2. **Models linearize even when parallelism is available.** Offered `sub_agent`, ~50% of 7B/AWQ runs
   adopt it, but they mostly invoke sub-agents **one-per-turn** (serial), so real concurrency stays the
   exception. With the harness now *supporting* fan-out, the residual linearity is the model's
   **policy**, not a tooling limit — strengthening RQ-N1's non-adaptivity conclusion.
3. **Emitted parallelism anti-correlates with capability here.** The degenerate 4-bit AWQ emits far more
   parallel batches (42%, ≤8/turn) than the careful 16-bit (6%) or 8-bit (8%) — high emitted parallelism
   is **flailing**, not capability.
4. **`sub_agent` is a non-monotonic accuracy scaffold.** It **rescues** the 4-bit model (0.42→**0.90**),
   **helps** the 16-bit (0.60→0.71), and **breaks** the 8-bit (0.81→0.56, via tool-protocol failure).
   Its value is structured decomposition; its risk is tool-schema fragility — both model-specific.

**Verdict (RQ-N2/N8).** With a truly concurrent executor + a branch tool + branch-requiring tasks,
executed width **does** become non-trivial (>1) — so the earlier `width≈1` was *partly* an executor
artifact — **but** models still predominantly linearize, and accuracy is dominated by decomposition and
tool-protocol robustness, not parallelism. "Agents linearize" survives as a **policy** claim, now on a
harness that no longer forces it.

*(An interim "complex-task re-run" — 8 tasks × 8 = 64 runs, `complex_experiment.jsonl`, 7B,
acc 0.64 — was produced between phases; it is superseded by 1.1/1.2.)*

---

## C. Consolidated findings — what to believe

**Robust (about the model/serving; survive the toy corpus):**
1. **Variance sources** — sampling (seed) dominates; the **KV/prefix cache is a real second
   source** (cache-off ⇒ bit-exact; cache-on ~doubles graph size); batching third.
2. **Precision floor** — 8-bit FP8 near-lossless (accuracy *up* with 2× params); **4-bit AWQ
   collapses** via tool-protocol breakdown.
3. **Decision calibration** — the agent mostly finishes when it has the answer; its
   characteristic failure is the **penultimate-hop short-circuit** (100% wrong).
4. **Cost predictability** — cost (incl. p95 tail) is predictable from cheap task features.
5. **Emitted ≠ executed parallelism** (RQ-N2/N8, Phase 6). With a concurrent executor, models emit
   parallel tool batches but near-instant `search`/`read` never overlap; real `width_executed > 1`
   needs latency-bearing ops (`sub_agent`), and even then appears in a minority of runs. A clean,
   transferable measurement distinction — and a reason "parallelism is rare" must be stated as
   *executed*, not *emitted*.

**Provisional (setup-dependent — the structure chapter):**
6. **Linear-dominant, parallelism-rare, small stable core.** RQ-N1 showed this is **not a clean-
   corpus artifact** (survives retrieval noise); **RQ-N2/N8 showed it is not merely an executor
   artifact** either — given a concurrent executor + a `sub_agent` branch tool + branch-*requiring*
   tasks, the models *still* predominantly linearize (fan-out is a rarely-taken option, adopted
   ~50% but used serially). The cause is **agent non-adaptivity** (a fixed policy). Treat as: *"this
   agent/model-family linearizes by policy,"* not *"LLM agents linearize,"* until **RQ-N3** (non-Qwen).

**The reframed thesis:** *ACG structure is a rigid property of the agent's policy, largely
decoupled from task difficulty; agents fail by mis-execution, not by adaptive restructuring.*

---

## D. Data artifacts index (`traces/`)

| Trace | Model | Runs | Acc | What |
|---|---|---|---|---|
| `experiment.jsonl` | 7B | 96 | 0.77 | main characterization (Phase 1.1) |
| `scale_hivar.jsonl` | 7B | 100 | 0.87 | T06/T12 × 50 (RQ-A2) |
| `variance_sources.jsonl` | 7B | 100 | 0.81 | T06 × 100 rep-budget |
| `determinism_T06.jsonl` | 7B | 12 | 1.00 | §7 determinism regime |
| `kv_temp0_cacheON/OFF.jsonl` | 7B | 20/20 | 1.00 | KV-cache @ temp 0 (2.3) |
| `elicited.jsonl` | 7B | 6 | 0.67 | reasoning demo |
| `elicit_off/on.jsonl` | 7B | 36/36 | 0.50/0.58 | elicitation on-vs-off (RQ-C1b) |
| `rqc2_elicited.jsonl` | 7B | 180 | 0.77 | decision analysis (RQ-C2) |
| `noise0/1/2.jsonl` | 7B | 96 ea | 0.77/0.73/0.56 | corpus-noise sweep (RQ-N1) |
| `qwen14b_fp8.jsonl` | 14B FP8 | 96 | 0.90 | model scaling (RQ-D1) |
| `qwen14b_awq.jsonl` | 14B AWQ | 96 | 0.41 | 4-bit floor (RQ-D2) |
| `branch_{7b,14bfp8,14bawq}_{nosub,sub}.jsonl` | 7B / 14B FP8 / 14B AWQ | 48 ea (288) | Phase 6 | branch-tool matrix (RQ-N2/N8); each has a `*_provenance.json` sidecar pinning model/decode/seed/engine-args |
| `complex_experiment.jsonl` | 7B | 64 | 0.64 | interim complex re-run (superseded) |
| `single_T*.jsonl` | 7B | 1 ea | — | per-task drawn ACGs |

Analysis scripts (17): `analyze, analyze_branch, structure_taxonomy, variance_sources,
branch_points, stable_core_map, cost_model, decision_analysis, reasoning_viewer, compare_models,
run_experiment, run_noise_sweep, run_complex, run_single, determinism_check, draw_graphs,
smoke_test`. Data/config: `data/tasks_branch.jsonl` (branch tasks), `acg/provenance.py` (run-level
provenance module).

---

## E. Next steps

**Direction: the structural chapter is now de-confounded on corpus (RQ-N1), executor, and tools
(RQ-N2/N8); the remaining confound is model family.** Order:

1. ✅ **RQ-N2 + RQ-N8 (done — Phase 6).** Concurrent executor + `width_executed` + a `sub_agent`
   branch tool + branch-requiring tasks. Result: executed width *can* exceed 1 (so `width≈1` was
   partly an executor artifact), but models still linearize by policy; `sub_agent` is a
   non-monotonic accuracy scaffold (rescues 4-bit, helps 16-bit, breaks 8-bit via tool-protocol
   fragility).
2. **RQ-N3 (next) —** replicate the branch matrix on one **non-Qwen** family (Llama-3.1-8B /
   Mistral): is "linearize by policy" Qwen-specific or general? The last big confound; harness ready.
3. **RQ-N4/N5 —** causal graph (counterfactual ablation) + **waste-headroom** quantification →
   decides whether an optimization contract is justified.
4. **Gate → scale:** 50 reps/task on a **realistic, latency-bearing corpus** (HotpotQA-with-
   distractors; slow retrieval so *emitted* parallelism can translate to *executed*). **Rented GPUs
   (gated on N5):** 32B–70B, GPTQ 4-bit recheck, controllers for the named failure modes.

**Open engineering caveats to carry forward:** ~~executor serializes parallel calls~~ (fixed in
Phase 6); executed concurrency is bounded by near-instant corpus tools (needs a slow-tool corpus to
stress); adding a 4th tool broke FP8's hermes formatting (tool-schema fragility — try a different
parser / trimmed schema); controlled runs should be **cache-off** for bit-exactness; the cost model's
`hops` feature is partly circular; task counts are thin (12 canonical / 6 branch — keep CIs); still
Qwen-only until RQ-N3.
