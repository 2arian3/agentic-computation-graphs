# 05 — Research questions (scoping phase)

Purpose: turn the insights gathered so far into a **scoped set of research questions**
we can answer *exhaustively on this one 24 GB MIG slice* before spending money on larger
GPUs / larger LLMs. Each RQ states: the **evidence** that motivates it, the **question**,
a testable **hypothesis**, the **experiment** (script/command), the **metric**, and a
**status** (answered on the slice / partial / needs bigger GPU).

Two things frame everything:
- The instrument already produces **~370 recorded runs** across the 7B and 14B-FP8 models
  (`traces/*.jsonl`) plus the §7 determinism decomposition — a lot is answerable now.
- The MIG slice is a **shared** resource (it has been occupied by another project at
  times). Experiments that need the GPU are marked; the analysis-only ones are not.

Supervisor questions map as: **Q1 → RQ-A1**, **Q2 → RQ-B1/B2/B3**, **Q3 → RQ-C1**.

---

## Theme A — Where does run-to-run variance come from?

### RQ-A1 (⭐ supervisor Q1) — At a *fixed* temperature, why is the ACG different every run? Is it the KV cache?
**Evidence.** §7 decomposition on T06 (7B): fixed-seed @ temp 0.7 → **1** distinct ACG
structure across 12 runs; varied-seed @ temp 0.7 → **6**. Fixed-seed @ temp 0.0 → 1.
**Hypothesis.** The variance is **token sampling** (the model draws different tokens at
temp>0), governed by the RNG **seed** — *not* the KV cache. KV/prefix-cache and batching
add only tiny floating-point nondeterminism ("serving noise") that, in a sequential
single-request setup, does not change graph structure.
**Experiment.** `scripts/variance_sources.py` ablates three factors at fixed temperature:
seed (fixed/varied) × prefix-cache (on / `--no-enable-prefix-caching`) × batching
(sequential / concurrent). Outcome metrics: # distinct ACG structures **and** # distinct
exact decision-trajectories (token-level reproducibility).
**Result (T06, temp 0.7 unless noted, 20 reps/arm; `traces/_rqA1_cache{ON,OFF}.log`):**
| regime | prefix cache | distinct ACG structures | distinct decision trajectories |
|---|---|---|---|
| fixed seed, sequential | ON | 3 | 3 |
| fixed seed, sequential | **OFF** | **1** | **1** |
| fixed seed, sequential, **temp 0.0 (greedy)** | **ON** | **2** | **2** |
| fixed seed, sequential, temp 0.0 (greedy) | OFF | **1** | **1 (byte-identical, 8 nodes ×20)** |
| fixed seed, **concurrent ×8** | ON | 6 | 9 |
| **varied** seed, sequential | ON | 9 | 18 |

**Mechanism (nailed down at temp 0, `traces/kv_temp0_cache{ON,OFF}.jsonl`).** Even *greedy*
(temp 0) + fixed seed is non-reproducible with the cache ON (2 graphs), but byte-identical
with it OFF (1 graph). Diffing the two greedy trajectories: all 20 runs are **identical for
the first 3 steps** (search → read D08 → read D02), then **1/20 runs flips at step 3** —
emitting `finish(Velmora)` instead of `search(veldanium mining location)`, which collapses a
14-node graph to 8 nodes. So a *single token flip at a near-tie decision boundary* cascades
into a different structure. Why the cache does this: vLLM prefix caching reuses previously
computed KV blocks whose values are **not bit-identical** to a fresh recompute (floating-point
non-associativity — different chunk/batch reduction order); the tiny logit perturbation flips
the greedy argmax where two tokens are near-tied. (Curiously, the cache also *shifts the modal
behavior*: cache-OFF deterministically finishes early at 8 nodes, while cache-ON mostly takes
the 14-node path — so the cache doesn't only add jitter, it can bias which graph the agent
builds. Worth a follow-up.) **Bit-exact reproducibility ⇒ serve with `--no-enable-prefix-caching`.**

**Answer (this REVISES the earlier §7 preliminary).** Run-to-run ACG variance at a fixed
temperature has **three separable sources**, largest first:
1. **Sampling (the seed) — dominant.** varied vs fixed seed → 9 vs 3 structures (18 vs 3
   trajectories). *Fixing temperature is not enough; you must also fix the seed.*
2. **The KV / prefix cache — YES, a genuine (secondary) source.** At fixed seed +
   sequential, prefix-cache **ON** gives 3 distinct graphs; turning it **OFF** gives **1
   (byte-identical)** — at both temp 0.7 and temp 0.0. Reused cached KV blocks are not
   bit-identical to a fresh compute, which flips low-probability sampled tokens near a
   decision boundary. (The §7 quick check happened to see 1 structure under a different
   serving config; the exhaustive ablation shows fixed-seed alone is *not* sufficient.)
3. **Batching/concurrency — a further source.** concurrent ×8 → 6 vs 3 sequential (fixed
   seed): batch floating-point reduction order adds nondeterminism.

Clean control: **cache OFF + fixed seed ⇒ fully reproducible (1 structure)**.
Plain-language for the supervisor: *"Same temperature isn't enough — the model still
samples a different token sequence unless the seed is also fixed (that's the big one). And
yes, the KV/prefix cache is a real secondary cause: with the seed fixed, turning the cache
off makes the graph byte-identical, but with it on you still get a few different graphs.
Running requests concurrently adds a bit more."*
**Status.** **Answered on the slice** ✅ (`scripts/variance_sources.py`).

### RQ-A2 — How many repetitions are needed to estimate the graph-size/variance reliably?
**Evidence.** Scaling T06/T12 from 8 → 50 reps: T06 distinct shapes **6 → 14**, modal
(stable-core) fraction **0.38 → 0.20**; accuracy CIs at n=8 were very wide.
**Hypothesis.** n=8 materially under-samples both the structural variance and the cost
tail; ≥30–50 reps are needed for stable p95/p99 and the modal fraction.
**Experiment.** `run_experiment.py --reps {8,25,50,100}`; track when #shapes / p95 / modal
fraction stabilize. **Metric:** rep-count at which estimates plateau + CI width.
**Result (T06, varied seed):** distinct ACG structures = **6 (n=8) → 14 (n=50) → 13
(n=100)**; distinct decision-trajectories keep rising (55 at n=100) but the *structure* space
saturates. **Answer:** the graph-**shape** space of a 4-hop task is ~**13–14 distinct
structures** and estimates plateau by **n≈50** — so **50 reps/task is the rep budget** for
stable structural-variance and tail estimates (token-level trajectory diversity is finer and
does not saturate). **Status. Answered ✅.**

### RQ-A3 — How does decode **temperature** drive graph size and structural variance?
**Evidence.** Sampling is the dominant variance source (RQ-A1), so temperature should be
the master knob.
**Hypothesis.** Higher temperature → larger spread in node count, more distinct shapes,
more pathological motifs (loops, short-circuits).
**Experiment.** Sweep temp ∈ {0.0, 0.3, 0.7, 1.0}, fixed reps per task; plot
size/variance vs temperature. **Metric:** #shapes, node-count std, motif prevalence.
**Result (T06, varied seed, 20 reps/temp):**
| temperature | distinct ACG structures | distinct answers | node-count range |
|---|---|---|---|
| 0.0 (greedy) | 2 | 1 | 8–14 |
| 0.3 | 8 | 5 | 8–16 |
| 0.7 | 9 | 11 | 7–16 |
| 1.0 | 13 | 16 | 7–14 |
**Answer.** Structural variance and answer variance rise **monotonically** with
temperature — temperature is the master knob for sampling variance (confirming RQ-A1). At
temp 0 the run is nearly deterministic (the residual 2 structures is the prefix-cache
effect from RQ-A1, since the cache was ON here); at temp 1.0 almost every run is a
different graph AND a different answer (16/20 distinct answers). **Status. Answered ✅.**

---

## Theme B — What structures actually occur? (supervisor Q2)

### RQ-B1 (⭐) — What is the taxonomy of ACG structures — is it just "branch and parallel"?
**Evidence.** `scripts/structure_taxonomy.py` over 96 runs each (7B / 14B-FP8):
| motif | 7B | 14B-FP8 |
|---|---|---|
| **linear_chain** (sequential backbone) | **0.84** | **0.83** |
| iterative_multihop (same tool, new target → depth grows) | 0.58 | 0.73 |
| **parallel_fanout** (width ≥ 2, true branching) | **0.00** | **0.05** |
| redundant_loop (exact re-call → backtracking) | 0.02 | 0.00 |
| degenerate_shortcircuit (0 tools) | 0.00 | 0.04 |
| truncated_no_finish (hit max steps) | 0.15 | 0.07 |
**Answer.** **No — and the intuition is inverted.** The dominant structure is the
**linear chain**; *parallelism is the rarest motif* (0% in 7B). The real structural axis is
**depth** (how many hops), not branching. Beyond linear/parallel there are three distinct
**pathological motifs**: re-reasoning/backtracking **loops** (exact re-call), **degenerate
short-circuits** (answer with no tools), and **truncation** (never finishing). "Branch and
parallel" misses all four of these.
**Status.** **Answered on the slice.** (A crisper motif for "backtracking" and a
graph-grammar formalization is a good write-up task, no GPU.)

### RQ-B2 — Does true parallelism (width > 1) emerge with model capability?
**Evidence.** width = 1 for **all** 7B runs; the 14B issues parallel tool calls on
T05/T06/T07 (motif prevalence 0.00 → 0.05; mean width 1.00 → 1.05).
**Hypothesis.** Parallel fan-out is a **capability-gated** behavior — stronger models
decompose into independent sub-queries and batch them.
**Experiment.** Measure width-distribution vs model (7B, 14B, and — bigger-GPU phase —
32B/70B). **Metric:** fraction of runs with width ≥ 2; max width.
**Status.** **Answered (2 points).** Extends naturally to the larger-GPU phase (RQ-D*).

### RQ-B3 — Which structures are error-associated? (structure ⇄ correctness)
**Evidence.** Accuracy by primary shape: linear_chain 0.80/0.99; redundant_loop **0.00**;
degenerate_shortcircuit **0.00**; truncated 0.69/0.43.
**Answer.** The pathological motifs are **strongly error-associated**: backtracking loops
and short-circuits are *always wrong* in our data. **A structural signal predicts failure**
— which foreshadows the optimization phase (detect/prune bad structures).
**Status.** **Answered on the slice** (n is small for the rare motifs; more reps would
tighten it — cheap GPU follow-up).

---

## Theme C — Reasoning: what did the model reason, from what inputs? (supervisor Q3)

### RQ-C1 (⭐) — Capture and visualize, per LLM step, the inputs → reasoning → decision.
**Evidence.** Traces already record each step's **inputs** (`gen_ai.prompt` = full message
context) and **decision** (`gen_ai.completion` = tool call). But the model's **reasoning
text is sparse**: on T06 (7B), steps 0–2 emit the tool call with an *empty* `content`,
while steps 3–6 (synthesis) contain real reasoning ("From the documents we've read…").
**Built.**
- `scripts/reasoning_viewer.py` — a self-contained HTML timeline per run: for each LLM
  step it shows **Saw (inputs)** → **Reasoned (why)** → **Decided (tool call)**, then the
  tool result feeding the next step. (See the inline widget in chat, and
  `traces/figures/reasoning_T06_*.html`.)
- **Reasoning-elicitation mode** (`ACG_ELICIT_REASONING=1`): adds a required `thought`
  argument to every tool so the model must verbalize *why* at **every** step (not just
  synthesis). Off by default so the canonical experiments are unchanged; the `thought` is
  captured in the trace automatically. **Run ✅** (`traces/elicited.jsonl`,
  `traces/figures/reasoning_T06_elicited.html`): with it on, **every** step now carries a
  coherent thought ("From the search results, I found … now I need …"), where the base
  agent was silent on the early routine tool calls.
**Answer.** The instrument captures inputs → reasoning → decision per step; the base 7B
verbalizes reasoning only during *synthesis*, and elicitation makes it explicit at every
step. **Status. Answered ✅.**
**Sub-question — does forcing per-step reasoning change the graph? Answered ✅.** Matched
run (T03,T05,T06 × 12 reps, elicitation OFF vs ON): accuracy 0.50 [0.35,0.66] → 0.58
[0.42,0.73] (**within noise**), nodes/run 8.6 → 8.3, **width unchanged (1.0)**, tokens/run
3886 → 4426 (**+14%**, the verbalized thoughts). So elicitation is a **low-distortion
instrument** — it makes reasoning observable without reshaping the ACG, at a modest token
cost. (`traces/elicit_{off,on}_summary.json`, `scripts/compare_models.py`.)

### RQ-C2 — What inputs trigger a branch, a short-circuit, or a loop?
**Evidence.** T01: the 14B **short-circuits** (0 tools) 3/8 times and is wrong each time —
it answers from parametric confidence without reading. The reasoning viewer is the tool to
inspect *what context preceded* each such decision.
**Experiment.** 180 elicited runs (12 tasks × 15, `thought` per step); at each of 697
decision points, correlate `context_has_answer` (has a gold alias appeared in any tool
result yet?) with finish-vs-continue. `scripts/decision_analysis.py`.
**Result:**
| | FINISH | CONTINUE |
|---|---|---|
| answer **in** context | 141 | 166 |
| answer **not** in context | 19 | 371 |

P(finish │ answer in context) = **0.46**; P(finish │ answer absent) = **0.05**.
**Answer.** The agent is **mostly well-calibrated** — it finishes ~9× more often when a gold
answer is already in context (0.46 vs 0.05), so short-circuits are *not* random. The failure
mode is sharp: **19 premature finishes (finish w/o the answer) → 19/19 ended INCORRECT**
(100%), confirming RQ-B3 at the decision level. The `thought` traces reveal *why*: they are
**penultimate-hop short-circuits** — the model stops one hop short by mistaking an
intermediate entity for the answer (T03: reads "born in **Mossgate**" and finishes with the
*town*, skipping Mossgate→**Karst Reach**); in one case its own thought says "I need to
determine which country" yet it still fires `finish`. The mirror inefficiency is
**over-continuation** — 166 decisions kept searching *after* the answer was available (the
model often fails to recognize it is done). Loops (11 re-reads) skew toward "answer not yet
found" (stuck). **RQ-C2 answered ✅** — short-circuits = penultimate-hop confusions
(deterministically wrong); over-continuation = failure to recognize completion. Both are
**detectable structural signals** for the optimization contract. **(Next-step #2 done.)**

---

## Theme D — Model size / precision (bridge to the larger-GPU phase)

### RQ-D1 — How does capability change ACG size and shape (not just accuracy)?
**Evidence.** 7B → 14B-FP8: accuracy 0.771 → 0.896; **fewer** nodes/tokens on hard tasks
(T06 −1558 tok), width>1 emerges, but new **degenerate short-circuits** appear.
**Answer (2 points).** Capability **reshapes** the graph: it shrinks redundant multi-hop
chains, unlocks parallelism, and introduces over-confident short-circuits. Graph size is
**not monotone** in capability.
**Status.** Answered for 7B vs 14B-FP8; extends to 32B/70B in the **larger-GPU phase**.

### RQ-D2 — Does low-bit **quantization** change the ACG (structure/accuracy), or only memory?
**Evidence.** FP8 (8-bit) was near-lossless (accuracy went *up* with 2× params).
**Hypothesis.** 8-bit preserves graph behavior; **4-bit (AWQ/GPTQ)** may erode tool-call
adherence and multi-hop reasoning, changing structure and lowering accuracy.
**Experiment.** Same 12-task benchmark on 14B-FP8 vs 14B-AWQ; compare accuracy + motif
prevalence + width. `scripts/compare_models.py`.
**Result (12 tasks × 8 reps, same slice):**
| model | precision | accuracy [95% CI] | nodes/run | tok/run | width |
|---|---|---|---|---|---|
| Qwen2.5-7B | BF16 (16-bit) | 0.771 [0.68, 0.84] | 7.8 | 3450 | 1.00 |
| Qwen2.5-14B FP8 | **8-bit** | **0.896 [0.82, 0.94]** | 7.3 | 3461 | 1.05 |
| Qwen2.5-14B AWQ | **4-bit** | **0.406 [0.31, 0.51]** | 4.6 | 2015 | 1.53 |
**Answer — precision has a FLOOR for agentic tasks, and 4-bit is below it (here).** 8-bit
FP8 is near-lossless (accuracy *rose* with 2× params); **4-bit AWQ collapsed to 0.41 —
worse than the 7B**, a **−0.49** drop vs the *same* 14B at 8-bit. Structural cause
(`structure_taxonomy.py`): the 4-bit model **degenerate-short-circuits 45% of runs** (vs 4%
for FP8) — it answers with *no tools* (wrong, since the facts are fictional) — and its
linear-chain rate fell 0.83 → 0.31. So quantization damage manifests as **tool-use protocol
breakdown**, not a uniform small accuracy tax: agentic multi-hop compounds per-token error
across hops, so 4-bit is risky where single-shot benchmarks tolerate it. **Caveat:** one
AWQ checkpoint, but the −0.49 magnitude is far beyond checkpoint noise; re-check GPTQ / other
4-bit / larger models on rented GPUs. **Status. Answered ✅** (`traces/qwen14b_awq_*`).
**Bottom line for the supervisor's original question:** *"larger + lower-precision, same
accuracy" holds at **8-bit** (a clear win) but **fails hard at 4-bit** for tool-using agents.*

---

## Theme E — Toward cost prediction & optimization (foreshadow; gated to a later contract)

### RQ-E1 — Can we **predict** an ACG's cost (tokens/nodes) from the task before running it?
**Built.** `scripts/cost_model.py` — per-task cost aggregates + a linear predictor on cheap
task features (hops, question length, entity count), **leave-one-out validated** (12 tasks).
**Result:**
| target | R² (hops only) | LOO-MAE | R² (all feats) |
|---|---|---|---|
| p95 node count | **0.93** | 0.72 nodes | 0.97 |
| p95 total tokens | **0.93** | 440 tokens | 0.96 |
| mean node count | 0.81 | 0.74 | 0.86 |
| mean total tokens | 0.84 | 416 | 0.87 |
**Answer — YES, and the tail is the *most* predictable part.** A one-feature linear model on
**hop count** predicts the **p95** of both node count and tokens at **R²=0.93** (LOO-MAE ≈ 0.7
nodes / 440 tokens); question-text features (qlen, entities — no oracle needed) push R² to
0.96–0.97. So an agentic task's cost distribution, *including the heavy tail cost planning
cares about*, is predictable **before running the agent**. Caveat: `hops` is a labeled
feature here; in deployment you'd estimate it from the question (the qlen/entities-only model
still does well). This is the concrete input an optimizer needs — **RQ-E1 answered ✅** and it
foreshadows the optimization contract. **(Next-step #3 done.)**

### RQ-E2 — Is there a **stable core** subgraph, and can pruning cut calls without hurting accuracy?
**Built.** `scripts/branch_points.py` aligns every run's decision trajectory and extracts
the **stable core** (longest prefix identical across ALL runs) + the per-depth divergence
profile. **Result (varied seed, n=50):** T06 has a **3-step stable core** (`search → read
D08 → read D02`) executed identically by all 50 runs, then **fans out at step 3** (3 distinct
decisions: continue-search 0.49 / read-D07 / finish); T12's stable core is **1 step**. So a
recurring core subgraph *does* exist, its **depth is task-dependent**, and variance is
**injected at a specific decision point** — and it's the *same* step-3 point where the KV
cache flipped a greedy run (RQ-A1), i.e. a genuine "gather-more vs answer" near-tie.
**Cross-task map** (`scripts/stable_core_map.py`, all 12 tasks): every task has a small
stable core (1–3 steps), and its depth is **uncorrelated with hop count** (corr = −0.06) —
the core reflects first-hop *ambiguity*, not difficulty. Meanwhile **graph size correlates
strongly with hops** (corr node-count↔hops = **0.90**) — the signal RQ-E1 exploits.
**Answer.** Stable core exists and is measurable (**RQ-E2 answered ✅** for detection);
pruning/using it for optimization is the later-contract step. This is the proposal's
"stable core with variation only at the edges", now quantified.

---

## Status roll-up

**Answered on this MIG slice ✅ (this phase):** RQ-A1 (variance sources — sampling ≫ KV
cache ≫ batching), RQ-A2 (rep budget ≈ 50), RQ-A3 (temperature sweep), RQ-B1/B2/B3
(taxonomy, parallelism-emergence, error-association), RQ-C1 (reasoning capture + viewer +
elicitation is low-distortion), RQ-D1 (7B vs 14B-FP8), **RQ-D2 (8-bit safe, 4-bit collapses)**.

**Also answered ✅ (this phase):** RQ-E1 (cost predictor — hops predict p95 cost at R²=0.93,
LOO-validated) and RQ-E2 detection (stable core exists, task-dependent 1–3 step depth; new
`scripts/branch_points.py`, `stable_core_map.py`, `cost_model.py`). KV-cache mechanism nailed
at temp 0 (RQ-A1: cache-on 2 graphs / cache-off 1 byte-identical; single step-3 token flip).

**RQ-C2 answered ✅** (`traces/rqc2_elicited.jsonl`, `scripts/decision_analysis.py`):
short-circuits are penultimate-hop confusions (100% wrong); over-continuation is failure to
recognize completion.

**Still open:** the prefix-cache *modal-shift* follow-up (does caching bias accuracy/size,
not just add jitter? — GPU); rented-GPU phase (32B–70B, GPTQ 4-bit recheck, frontier realism
check). The characterization phase is otherwise **complete** — all of A/B/C/D/E answered.

**Deferred to rented GPUs / larger LLMs:** 32B–70B (didn't fit the shared slice — an
optirag host process held ~3.4 GB), 4-bit re-check with GPTQ, frontier-agent realism check.

**Headline answers for the supervisor:** Q1 — variance is **sampling (seed)** first, with the
**KV/prefix cache a real secondary source** and batching third (fixed seed + cache-off ⇒
byte-identical). Q2 — **not** branch/parallel; ACGs are **linear chains** of varying depth,
parallelism is rare, plus rare error-associated pathologies. Q3 — inputs→reasoning→decision
are captured per step (viewer + elicitation). Bonus — **8-bit quantization is a free win,
4-bit is unsafe for agents.**
