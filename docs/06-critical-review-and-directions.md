# 06 — Critical review: threats to validity & research directions

A deliberately skeptical assessment written *before* committing to scale (more reps, rented
GPUs, larger models). The goal is to separate findings that are **robust** from findings that
are **artifacts of our setup**, quantify whether the phenomenon is **big enough to optimize**,
and pick research questions that produce *true, comprehensive* results rather than more
precise measurements of a toy.

---

## 0. The core problem, stated plainly

Our headline structural findings — ACGs are **linear chains**, **parallelism is rare**, a
small **stable core** exists, size scales with **hops** — are, to an unknown degree, **built
in by our design**:

- We authored **linear tasks** (entity → city → country → currency), then "discovered" linear
  graphs. The graph's shape is upper-bounded by the task's shape.
- The **corpus is a 16-document, unambiguous, distractor-free toy** with **deterministic
  keyword retrieval** where each hop has exactly one correct document. Real retrieval is noisy;
  noise is what forces branching, re-querying, and backtracking.
- The **tool alphabet is 3 tools** (search/read/finish). A graph cannot fan out, recurse, or
  branch if the tools don't allow it.
- **The executor serializes tool calls.** Even when the model emits multiple tool calls in one
  turn, `agent.py` runs them one-by-one. So our **`width` metric cannot measure real
  parallelism** — it measures "how many calls were emitted together," not concurrent execution.

**Implication:** "we characterized the size and structure of agentic computation graphs" is,
right now, closer to "we characterized what *this loop with these tools on this toy corpus*
produces." That is a real result about a scaffold; it is not yet a result about LLM agents.

### What IS robust (transfers beyond the toy)
Not everything is contaminated. These findings are about the **model/serving**, not the
corpus, and should transfer:
- **Variance sources** (sampling ≫ KV/prefix cache ≫ batching; cache-off ⇒ bit-exact). Clean,
  methodologically sound, model-agnostic mechanism.
- **Precision floor** (8-bit lossless, 4-bit collapses via tool-protocol breakdown). Decision-
  relevant and likely general (worth a cross-model check).
- **Decision calibration** (P(finish│answer present)=0.46 vs 0.05; premature finishes 100%
  wrong; penultimate-hop short-circuits). This is a genuine behavioral property, though its
  *rate* will change on harder corpora.

### What is setup-dependent (treat as provisional)
- The taxonomy (linear-dominant, parallel-rare), the stable-core depth, the `width` numbers,
  and "cost = f(hops)". These need the realism/robustness tests below before we believe them.

---

## 1. Threats to validity (the obstacles)

1. **External validity (severity: high).** Toy corpus + 3 tools + hand-designed linear tasks.
   The structure findings may not survive contact with realistic retrieval or tools.
2. **Executor serializes parallel calls (high).** `width` is mismeasured; any "parallelism is
   rare" claim is unsafe until the harness executes emitted tool calls concurrently and we
   distinguish *emitted-parallel* from *executed-parallel*.
3. **Single model family (high).** Everything is Qwen2.5-7B/14B. "LLM agents do X" is
   unjustified from one family. Reasoning models (long-CoT) would have entirely different
   token/graph profiles.
4. **Cost-model circularity (medium-high).** `hops` predicts cost at R²=0.93, but `hops` is a
   task-*design* parameter we also used to construct the tasks, and the corpus makes hop-count
   ≈ node-count almost by construction. Needs features derivable from the question alone, on
   tasks not built around a hop count.
5. **Transcript-order ≠ causal dependency (medium-high).** The ACG edges come from
   `acg.depends_on`, which is the *order* the loop fed things back — not which documents
   actually *influenced* the answer. We cannot currently tell a load-bearing read from a wasted
   one. This undercuts both "structure" and "pruning".
6. **Statistical power (medium).** 12 tasks, 8–50 reps. Per-task and regression claims are
   thin; we added Wilson/LOO CIs, but the task count is the real limit.
7. **Optimization headroom is unquantified (medium).** If graphs are already near-minimal
   linear chains, there is little to prune. We must measure *how many tokens/calls are wasted*
   before claiming optimization is worthwhile.
8. **The main variance study ran cache-ON (medium).** So its "structural variance" conflates
   sampling + cache. Controlled scientific runs should be cache-off (or report both).
9. **Novelty positioning (low-medium).** LLM-inference nondeterminism (reused-KV / batch
   variance) is partially known in the systems literature. Our defensible novelty is *that it
   changes graph **structure** and shifts **modal behavior***, not "we found nondeterminism".
10. **Infra fragility / cost (low-medium).** Shared MIG slice (contended with another project),
    tight disk, one-model-at-a-time. Renting GPUs fixes capacity but adds cost that must be
    justified by a concrete, artifact-free experiment plan.

---

## 2. Research questions worth creating now

Ordered by leverage. The top three attack the confounds directly and are all runnable on the
current slice (cheap) — they gate whether the expensive phase is even worth it.

### RQ-N1 — Does structure emerge from the MODEL or the CORPUS? *(highest priority)*
Hold the model fixed; sweep **retrieval realism**: add distractor documents, multiple valid
docs per hop, near-duplicate/ambiguous entities, and degrade the retriever (top-k noise, BM25
over a larger corpus). **Hypothesis:** branching, re-querying, and backtracking rise sharply —
i.e. "linear/no-parallelism" is a clean-corpus artifact. **Metric:** motif prevalence, width,
re-query rate, stable-core depth vs a "noise level" knob. *This single experiment determines
whether the structure chapter is about agents or about our corpus.*

**RESULT — RAN ✅ (7B, 12 tasks × 8 reps/level; distractor knob 0/1/2; `data/distractors.json`,
`scripts/run_noise_sweep.py`, `traces/noise{0,1,2}.jsonl`).**
| noise | accuracy | nodes | width | linear | loop | #search |
|---|---|---|---|---|---|---|
| 0 | 0.77 | 7.7 | 1.01 | 0.82 | 0.02 | 1.4 |
| 1 | 0.73 | 8.3 | 1.03 | 0.84 | 0.02 | 1.6 |
| 2 | **0.56** | 8.0 | 1.01 | **0.91** | 0.05 | 1.5 |

The manipulation *worked* — accuracy fell **0.77 → 0.56** (non-overlapping CIs) — but the
**graph structure barely moved**: width stayed ≈1, node/search/read counts flat, runs got
*slightly more linear* (0.82→0.91), **not branchier**. **The hypothesis (noise induces branching)
was wrong.** The 7B agent does **not adaptively restructure** under retrieval noise — it plows the
same linear `search→read→finish` chain and **fails by mis-reading a distractor**, never
re-querying or cross-checking. Two honest conclusions:
1. **Linear/low-width structure is NOT merely a clean-corpus artifact** — it *persists* under
   distractor noise. Reassuring for the robustness of the structural chapter.
2. **Deeper cause = agent non-adaptivity:** for this model+loop+tools, ACG structure is a fixed
   model *policy*, largely **decoupled from retrieval difficulty** (structure ≠ f(difficulty)).
**Caveat — what this does NOT test.** The knob degrades retrieval *precision* but tasks stay
linearly solvable (the correct doc is still retrievable), so it does not *force* branching. The
stronger artifact test — *does the model branch when the task **requires** fan-out?* — is **RQ-N8**
(comparison/aggregation) + **RQ-N2** (a truly parallel executor + branch tools).
**⇒ Gate-1 verdict:** structure is **robust to distractor noise**; the open question shifts from
"is it the *corpus*?" to "is it *capability/adaptivity*?" — pursue **RQ-N2 (real executor) + RQ-N8
(branch-requiring tasks) + RQ-N3 (2nd model)** next, not more corpus noise.

### RQ-N2 — Does the tool alphabet + a truly-parallel executor change the graph? *(high)*
Fix the executor to run emitted tool calls **concurrently**; distinguish *emitted-parallel*
from *executed-parallel*. Add branch-enabling tools (parallel/map-search, `sub_agent`,
`verify`, calculator). **Hypothesis:** genuine width>1 appears when tasks require fan-out and
the harness supports it; the "width≈1" result is partly an executor artifact. **Metric:** real
concurrency, depth vs width trade, latency (critical-path vs total).

### RQ-N3 — Do the findings generalize across model families? *(high)*
Replicate taxonomy + variance + pathologies on ≥2 non-Qwen families (e.g. Llama-3.x,
Mistral/Ministral) and one **reasoning model**. **Hypothesis:** the *serving* findings (RQ-A1,
precision) replicate; the *structure/pathology rates* differ; reasoning models produce deep
CoT-heavy nodes and possibly fewer short-circuits. **Metric:** which findings are invariant.

### RQ-N4 — A CAUSAL graph, not a transcript-order graph. *(high, novel)*
Build the true dependency graph by **counterfactual ablation**: remove each read/search from
the context and re-run the final answer step — does the answer change? Load-bearing nodes
change it; wasted nodes don't. **Deliverable:** per-run "useful subgraph" vs "executed graph";
a *waste ratio*. This is a genuine measurement contribution and the foundation for pruning.

### RQ-N5 — Quantify the optimization headroom BEFORE building an optimizer. *(high)*
Using RQ-N4's waste ratio + the pathology taxonomy, measure the fraction of tokens/calls that
are **wasted** (redundant reads, over-continuation, pathological loops). **Decision rule:** if
waste is <~15%, optimization is not worth a contract on this domain; if it's large, it is.
*Answer this before promising an optimization deliverable.*

### RQ-N6 — Can a lightweight controller move the accuracy/cost Pareto? *(medium-high)*
Given the two named failure modes (penultimate-hop short-circuit; failure-to-recognize-
completion), test cheap interventions: a "you may be one hop short" check, or a
"you-already-have-the-answer" completion detector. **Baseline:** the raw loop. **Metric:** the
accuracy–cost Pareto front (not a single number). This is the first *real* optimization test.

### RQ-N7 — Is run-to-run variance EXPLOITABLE (not just a nuisance)? *(medium, interesting)*
Sample K graphs for one task and aggregate/vote (self-consistency **over graphs**). **Question:**
does structural diversity buy accuracy at a known cost multiplier, and is graph-level voting
better than answer-level voting? Turns "variance" from a problem into a method.

### RQ-N8 — Difficulty that REQUIRES branching. *(medium)*
Design tasks that structurally demand fan-out (compare/aggregate over N entities: "which of the
three countries on Orrin has the fewest ...") and set/conjunction queries. **Hypothesis:** the
model branches only when the task cannot be linearized; measures whether width is capability- or
task-gated. Pairs with RQ-N2.

### RQ-N9 — Online (mid-run) cost prediction. *(medium)*
Predict *remaining* cost after step k, not just from the task. Enables budget enforcement /
early-exit and is a cleaner predictor (no `hops` circularity — uses observed partial-graph
features). **Metric:** calibration of remaining-node/token estimates vs step depth.

---

## 3. What else to target — applications/tasks (with the honest trade-off)

Breadth vs depth is a real tension; the proposal deliberately chose depth in one domain.
**Recommendation: make the *current* domain realistic before adding domains** — a broad, shallow
survey across coding/web/RAG would produce weaker results than one domain done rigorously.

- **Now, cheap, high-value:** noisy/large-corpus multi-hop RAG (BM25 over a real Wikipedia
  subset; or 2WikiMultihopQA / HotpotQA *with* distractor paragraphs) + branch-requiring tasks
  (RQ-N8). This directly de-toys the current setup.
- **Rented-GPU phase, higher cost:** coding agents (SWE-bench-lite — genuine branch/loop/verify
  structure), web/computer-use agents (richest graphs, hardest to control), and — only if
  scoped carefully — small **multi-agent** teams (the proposal excluded these, but they are where
  ACGs become non-trivial DAGs; adding them changes the thesis, so treat as a separate study).
- **Deliberately avoid** turning this into a benchmark zoo. Two realistic domains characterized
  well beats six characterized shallowly.

---

## 4. What tools/agents to design

1. **Fix the executor first** (measurement correctness): run emitted tool calls concurrently;
   record wall-clock overlap so `width` reflects *executed* parallelism.
2. **Branch/recursion tools:** `parallel_search` / `map_over(entities)`, `sub_agent(subquestion)`
   (turns the graph into a real tree/DAG), `verify(claim)` / `self_check` (legitimate loops),
   `calculator`/`code` (for aggregation tasks).
3. **Agent-architecture comparison:** ReAct (current) vs **plan-then-execute** vs **reflexion/
   critique**. Hypothesis: explicit planning front-loads branching and changes the stable core;
   critique adds structured loops. This isolates how much structure is *loop-design* vs *model*.
4. **A completion/one-hop-short detector** as an actual controllable node (for RQ-N6).

---

## 5. Recommendation before moving forward (the non-satisfying answer)

The exciting move is "rent GPUs and run 70B across domains." The **correct** move is a cheap
**validity gate on the current slice first**, because it decides whether the expensive phase
measures agents or artifacts:

1. **RQ-N1** (corpus-noise sweep) — is the structure real? *(days, no new GPU)*
2. **RQ-N2** (fix executor + branch tools) — is width real? *(days)*
3. **RQ-N4/N5** (causal graph + waste headroom) — is there anything to optimize? *(analysis-heavy)*
4. **RQ-N3** (one non-Qwen family) — does anything generalize? *(one model download)*

**Only if** structure survives noise, width becomes non-trivial with a real executor, and the
waste headroom is material — **then** rent GPUs for scale (50 reps/task per the RQ-A2 budget),
larger/reasoning models, and the optimization experiments (RQ-N6/N7). Committing to an
optimization deliverable before RQ-N5 quantifies the headroom would repeat the exact
over-commitment the proposal warns against.

**Bottom line:** the instrument is solid and several serving/precision/decision findings are
real and transferable. But the *structural* characterization — the heart of the thesis — is not
yet trustworthy because the corpus, tools, and executor constrain the very thing we measure.
De-toy the setup and quantify the optimization headroom before scaling.
