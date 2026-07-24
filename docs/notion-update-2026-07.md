# Progress Update — Agentic Computation Graphs (2026-07)

## Summary
We rebuilt the tool-using QA benchmark so that the agent's **computation graphs actually vary in shape** (previously they were near-identical linear chains), then ran a controlled **7B vs 14B** comparison (**480 runs**). The headline result is counterintuitive: the "larger" **14B (8-bit) model is *less* accurate than the 7B (16-bit)** because it **short-circuits the tool loop** — and *fewer tool calls, smaller graphs, and lower accuracy turn out to be the same phenomenon.*

## What was built
- **Corpus:** a procedural, fully-**fictional** knowledge-base generator (knowledge-graph backbone) → **61 docs + 6 distractors**, scalable to thousands. Retrieval upgraded to **BM25**. Fictional so the model can't answer from memory — verified with a **closed-book baseline = 0/8** (retrieval genuinely does the work).
- **Tools (the graph's node types):** added four to the base `search` / `read_document` / `finish`, each chosen to force a structure:
  - `calculator` → computation node
  - `compare` → aggregation / **width** (fan-out then merge)
  - `verify_claim` → verifier node
  - `decompose` → planner node
- **Task families (7 families, 54 tasks):** `linear_bridge`, `numeric_diff`, `counting`, `fan_out_superlative`, `unanswerable`, `constraint_satisfaction`, `conditional` — each engineered to force a distinct graph shape.
- **Agent:** the thin, **emergent** ReAct-style loop (model chooses each step; nothing hand-wired). A **multi-agent variant** via `sub_agent` emergently forms a `planner (decompose) → workers (sub_agents) → aggregator (compare)` tree (one run: 38 nodes, depth 22, 3 sub-agents, real parallelism = 3).

## Experiment: families × two backbones
All 40 tasks (5 families that existed at run time) × 6 reps × {Qwen2.5-7B BF16, Qwen2.5-14B FP8} = **480 runs**, `temperature 0.7`, one 24 GB GPU slice.

| Family | acc 7B / FP8 | nodes 7B / FP8 | width>1 runs 7B / FP8 | tokens 7B / FP8 |
|---|---|---|---|---|
| linear_bridge | 0.89 / 0.92 | 8.8 / 8.8 | 3% / 0% | 7310 / 7714 |
| numeric_diff | **0.96 / 0.50** | 8.9 / 5.6 | 1% / 4% | 7545 / 5125 |
| counting | 0.77 / 0.77 | 9.9 / 8.1 | 33% / 27% | 7610 / 6547 |
| fan_out_superlative | 0.61 / 0.50 | 11.8 / 6.8 | 17% / 44% | 10353 / 5289 |
| unanswerable | 1.00 / 0.92 | 11.8 / 8.5 | 2% / 2% | 11123 / 7498 |

## Findings
1. **Width is finally a measurable variable.** The enriched tools/tasks produce real branching (`fan_out` reaches **44%** of runs with width>1), where the old benchmark was width-1 everywhere.
2. **The backbone reshapes the graph for the *same* task.** 7B builds bigger, deeper graphs than 14B FP8 across every family → graph size is a **task × model interaction**, not a task property.
3. **The 14B FP8 is less accurate because it short-circuits the tool loop.** On `numeric_diff`, **50% of FP8 runs finish with ≤1 tool call** (a 1-node graph carrying a guessed answer → wrong), vs **1%** for 7B. FP8's *wrong* runs average **1.2 nodes**; its *correct* runs **9.9** (same as 7B) — bimodal, not uniformly smaller.
4. **It's behavioral, not a bug.** **0 tool-parse errors** on either model — FP8 *chooses* not to use tools. 8-bit quantization degrades **tool-use discipline**, not knowledge: the collapse is task-selective (retrieval-only `linear_bridge` is a tie 0.92/0.89; tool-composition collapses).
5. **Same failure family as 4-bit AWQ, graded by precision:** BF16 ≈ 1% → FP8 8-bit up to 50% → AWQ 4-bit 45% short-circuit. Refines "8-bit safe, 4-bit unsafe" into **"8-bit is safe for retrieval, not for tool-composition."**
6. **"Cheaper" model ≠ cheaper run** — 7B uses *more* tokens per task because it builds larger graphs. Cost must be read structurally.
7. **The 1-node short-circuit graph is a clean, predictable failure signature** — a structural predictor of failure.

## Tooling / infrastructure
- All results are browsable in the **web dashboard**: a **Families** tab (per-family × backbone table incl. short-circuit rate), and **Replay** (all 240 runs per model, filterable by family, reconstructing each run's ACG).
- Deterministic + reproducible: seeded generator, seeded BM25, pinned decode params.

## Next steps
- **Complete the precision → short-circuit curve:** run the enriched benchmark on **AWQ (4-bit)** and **Llama-3.1-8B (BF16)**. Prediction: AWQ short-circuits more; Llama-BF16 stays low → confirms *precision*, not family/vendor, is the driver.
- Make **`short_circuit_frac`** a first-class metric (already surfaced in the dashboard).
- Run the two new families (`constraint_satisfaction`, `conditional`) on both backbones; sweep temperature.
- Feed the **480 labeled realized graphs** (structure + cost + outcome + failure label) into the ACG-Bench graph predictors.
