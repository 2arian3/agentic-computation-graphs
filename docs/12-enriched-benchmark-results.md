# 12 — Enriched-benchmark first results: structure by family × backbone

> **What this is.** The first empirical run on the enriched QA benchmark built in
> [docs/11](11-making-applications-more-complex.md): 5 shape-inducing task families + the extended
> tool alphabet (`calculator`/`compare`/`verify_claim`/`decompose`) + BM25 retrieval, run across
> **two backbones**. It answers a first slice of [docs/10](10-research-plan-agentic-application-graphs.md)
> **RQ1** ("what graph properties differ across application types") and adds a **model axis**.

## Setup
- **Benchmark:** `data/corpus_large.json` (61 docs) + `data/tasks_families.jsonl` (**40 tasks**, 5 families),
  BM25 retrieval, tools `search/read_document/finish` + `calculator/compare/verify_claim/decompose`.
- **Runs:** all 40 tasks × **6 reps**, `temperature=0.7`, varied per-run seed → **240 runs/model, 480 total**.
- **Models (one 24 GB MIG slice, served sequentially):** `qwen2.5-14b-instruct-fp8` (8-bit) and
  `qwen2.5-7b-instruct` (BF16).
- **Traces:** `traces/families_fp8.jsonl`, `traces/families_7b.jsonl` (+ `*_metrics.csv`, `*_summary.json`,
  `*_by_family.json`, `*_provenance.json`). Reproduce with `scripts/run_experiment.py … --tasks all` and
  `scripts/analyze_families.py`.

## Per-family × per-backbone (n = 6 × #tasks in family)

| family (n/model) | acc **7B / FP8** | nodes **7B / FP8** | depth **7B / FP8** | width mean **7B / FP8** | width>1 frac **7B / FP8** | tok **7B / FP8** |
|---|---|---|---|---|---|---|
| `linear_bridge` (72) | 0.89 / 0.92 | 8.8 / 8.8 | 8.8 / 8.8 | 1.03 / 1.00 | 0.03 / 0.00 | 7310 / 7714 |
| `numeric_diff` (72) | **0.96** / **0.50** | 8.9 / 5.6 | 8.9 / 5.5 | 1.01 / 1.04 | 0.01 / 0.04 | 7545 / 5125 |
| `counting` (30) | 0.77 / 0.77 | 9.9 / 8.1 | 9.6 / 7.7 | 1.33 / 1.37 | 0.33 / 0.27 | 7610 / 6547 |
| `fan_out_superlative` (18) | 0.61 / 0.50 | 11.8 / 6.8 | 11.6 / 5.9 | 1.22 / **1.83** | 0.17 / **0.44** | 10353 / 5289 |
| `unanswerable` (48) | **1.00** / 0.92 | 11.8 / 8.5 | 11.8 / 8.5 | 1.02 / 1.02 | 0.02 / 0.02 | 11123 / 7498 |

Tool-node types fired as designed: `calculator`→numeric_diff, `compare`→fan_out/counting,
`verify_claim`→linear/counting/unanswerable, `decompose` broadly.

## Findings

1. **Structure is family-dependent — width is finally a variable.** Fan-out families induce real
   branching (`fan_out` width mean up to 1.83 with **44%** of runs > 1; `counting` ~30%), while
   `linear_bridge` and `unanswerable` stay strictly linear (width 1). Before this build, measured width
   was ≈ 1 everywhere. This is the diversity the characterization program needs.

2. **The backbone reshapes the graph for the *same* task.** 7B (BF16) builds **larger, deeper** graphs
   than 14B FP8 across every family (e.g. `fan_out` 11.8 vs 6.8 nodes; `unanswerable` 11.8 vs 8.5): it runs
   more search/verify steps where FP8 is terser. So graph **size is not a property of the task alone** — it
   is a task × backbone interaction, exactly the confound the survey warns about.

3. **8-bit FP8 degrades sharply on tool-composition, not on plain retrieval.** `numeric_diff` accuracy is
   **0.96 (7B) vs 0.50 (FP8)** and `fan_out` **0.61 vs 0.50**, yet `linear_bridge` is a tie (0.89/0.92).
   FP8 was near-lossless on the original retrieval-only tasks (docs/07), so the loss here is specific to
   **arithmetic + multi-tool composition** — a task-family-resolved capability picture that a single
   accuracy number would hide.

4. **Emitted parallelism anti-correlates with capability (replicated on the new tools).** On `fan_out`,
   the *less* accurate FP8 fans out **wider** (width 1.83, 44% >1) than the more accurate 7B (1.22, 17%).
   This matches the earlier `sub_agent` finding (docs/08): weaker models emit more parallel calls that
   don't pay off. Now it reproduces via the `compare`/parallel-`read` path, not just `sub_agent`.

5. **"Cheaper" model ≠ cheaper run.** 7B (fewer params) costs **more tokens per task** than FP8 because it
   builds bigger graphs (`unanswerable` 11123 vs 7498; `fan_out` 10353 vs 5289). Backbone choice changes
   **graph size**, so cost-per-success must be read structurally, not from price-per-token — directly the
   docs/10 cost-prediction motivation.

6. **Structural stability differs by family and backbone.** 7B `linear_bridge` is the most stable (modal
   shape fraction 0.51, only 9 distinct shapes) while `numeric_diff`/`unanswerable` are more variable
   (17–20 shapes). Predictability is itself a family × model property (docs/10 RQ1/RQ3).

## Why is the "larger" 14B FP8 model *less* accurate? — it short-circuits the tool loop

The counterintuitive result — more parameters, yet lower accuracy, smaller graphs, and fewer tool
calls — has a **single mechanism**, and a trace-level check rules out the obvious confound.

**Per-family tool use, split by outcome (short-circuit = a run that finishes with ≤1 tool call):**

| model | family | acc | mean tool calls | **short-circuit %** | nodes \| correct | nodes \| wrong |
|---|---|---|---|---|---|---|
| 7B (BF16) | numeric_diff | 0.96 | 4.3 | **1%** | 9.1 | 3.7 |
| **FP8 (8-bit)** | numeric_diff | 0.50 | 2.6 | **50%** | 9.9 | **1.2** |
| 7B (BF16) | fan_out | 0.61 | 5.9 | 11% | 13.1 | 9.9 |
| **FP8 (8-bit)** | fan_out | 0.50 | 3.6 | **33%** | 9.6 | 4.0 |
| 7B (BF16) | linear_bridge | 0.89 | 4.4 | 0% | — | — |
| FP8 (8-bit) | linear_bridge | 0.92 | 4.4 | 4% | — | — |

**The three questions, answered from the data:**

1. **Why fewer tool calls / smaller graphs / lower accuracy?** They are the *same event*. On
   `numeric_diff`, **half of FP8's runs finish with ≤1 tool call** — a **1-node graph** (an immediate
   `finish`) carrying a guessed answer. On a fictional corpus a guess is always wrong, so
   `accuracy ≈ 1 − short-circuit-rate`. The wrong runs average **1.2 nodes**, the correct runs **9.9**:
   it is *bimodal* (full graph → right, or short-circuit → wrong), not a uniformly-smaller graph.

2. **Is it a tool-parser bug?** **No.** The trace has **0 `acg.error` spans on either model** — no
   malformed tool calls, no serving failures. FP8 is not *failing* to call tools; it is *choosing* not to.
   This resolves the docs/07 "FP8 tool fragility" confound for these runs: the failure is **behavioral,
   not a parsing artifact**.

3. **Why does a 14B model behave worse than a 7B?** Because it is not "larger" in the way that matters
   here: it is 14B parameters at **8-bit** precision vs the 7B at full **16-bit**. When FP8 *does* run the
   loop, its graphs match 7B's (nodes|correct ≈ 9.9 for both) and it answers correctly — so its
   **knowledge is intact**. What quantization degrades is **tool-use discipline**: the adherence to "keep
   retrieving/computing until grounded." That is exactly why the collapse is *task-selective* —
   retrieval-only `linear_bridge` is a tie (0.92/0.89), while the families that demand multi-step tool
   *composition* (`numeric_diff` → calculator; `fan_out` → gather-then-compare) are where FP8 bails early.

**This is the same failure family as 4-bit AWQ** (docs/07: 45% degenerate short-circuit), now graded by
precision: **BF16 ≈ 1% → FP8 8-bit up to 50% → AWQ 4-bit 45%.** The docs/07 rule "8-bit safe, 4-bit
unsafe for agents" should be refined: **8-bit is safe for retrieval but not for tool-composition** —
parameter count does not rescue agentic control once precision is cut. Structurally, the **1-node
short-circuit graph is a clean, predictable failure signature** — directly the failure-prediction target
in [docs/10](10-research-plan-agentic-application-graphs.md) §4.

## Caveats
First cut: `n` varies by family (3–12 tasks × 6 reps), single `temperature=0.7` + varied-seed policy, two
backbones. The parser-fragility confound is **ruled out** for these runs (0 `acg.error` spans; see the
short-circuit analysis), so the FP8 gap is behavioral rather than a serving artifact. Not yet covered:
the two new families (`constraint_satisfaction`, `conditional`) on both backbones, and the AWQ (4-bit) /
Llama-3.1-8B backbones needed to complete the **precision → short-circuit curve**.

## Next steps (analysis-driven)

The short-circuit finding reprioritizes the roadmap ([docs/11](11-making-applications-more-complex.md)):

1. **Complete the precision → short-circuit curve.** Run the enriched benchmark on **AWQ (4-bit)** and
   **Llama-3.1-8B (BF16)** and plot short-circuit-rate vs. precision per family. Prediction from the data
   so far: 4-bit AWQ should short-circuit *more* than FP8 (docs/07 saw 45%), and a second BF16 family
   (Llama) should behave like 7B (~low). If Llama-BF16 also short-circuits little, precision — not
   family or vendor — is the driver.
2. **Make short-circuit rate a first-class metric.** Add `short_circuit_frac` (runs with ≤1 tool call, i.e.
   a ≤1-node ACG) to `scripts/analyze_families.py` and the summary — it is the single best predictor of
   family accuracy here and a clean **failure signature**.
3. **Failure prediction is nearly trivial structurally** and worth stating as a docs/10 RQ result: a
   1-node realized graph predicts a wrong answer with near-certainty. The interesting learned task is
   predicting short-circuit *propensity* from (model, precision, family) **before** running.
4. **Sweep temperature.** Does higher temperature raise or lower the short-circuit rate? (Determinism
   study machinery already exists in `scripts/determinism_check.py`.)
5. **Finish coverage:** run the two new families (`constraint_satisfaction`, `conditional`) on both
   existing backbones so docs/12 covers all 7 families × all backbones.

These 480 labeled realized graphs (structure + cost + outcome, with per-family variance **and a clean
failure label**) are the first concrete slice of **ACG-Bench**
([docs/10](10-research-plan-agentic-application-graphs.md) §3) and training data for the graph predictors (§4).
