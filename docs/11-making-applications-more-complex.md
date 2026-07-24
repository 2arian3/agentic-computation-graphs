# 11 — Making the applications more complex: a build roadmap

> **Goal.** Move the QA benchmark from "one graph shape + sampling jitter" to a **controlled
> distribution over genuinely different realized-graph shapes**, so the graphs are actually worth
> characterizing ([docs/10](10-research-plan-agentic-application-graphs.md)). This doc records the
> concrete build: **Steps 1–4 are implemented and verified** (below), followed by the **next steps
> after step 4**.

## Why the current setup produces "small, generic" graphs

Grounded in the artifacts, not opinion:

- **Corpus:** 16 docs / **3.5 KB** (`data/corpus.json`). Retrieval is `top-3 of 16` — never wrong, no stress.
- **Tasks:** 12 linear multi-hop + 6 fan-out. Almost everything is a serial chain → measured **width ≈ 1**;
  structural variance comes *only* from sampling.
- **Tools:** `search / read_document / finish (/sub_agent)` — this alphabet can only express
  `retrieve → read → … → finish` chains. No verifier, computation, or aggregation nodes.

The root cause is the **tool alphabet + task structure**, not just corpus size. So the design objective is:

> **Maximize the controlled structural diversity of realized graphs** — task families that *require*
> different topologies, and tools that *create* structural node types — while preserving the control
> property (pinnable seed, no memorization) that is the whole methodological edge.

---

## What's implemented now (Steps 1–4 — DONE ✅)

### Step 1 — extended tool alphabet · `acg/tools.py`, `acg/config.py`, `acg/agent.py`
Four opt-in tools that **induce structural diversity without changing the emergent loop** (they are pure
functions of `(args, corpus)`; the *task* decides whether the model needs them):

| Tool | Node type it creates | Induces |
|---|---|---|
| `calculator` | computation | numeric/temporal reasoning |
| `compare` (`max`/`min`/`count`/`sum`) | fan-out → aggregate (join) | **width** + aggregation |
| `verify_claim` | verifier | validity checks, retry pressure |
| `decompose` | planner | explicit plan, deeper graphs |

Enabled per-experiment via `ACG_EXTRA_TOOLS=calculator,compare,verify_claim,decompose` (and the existing
`ACG_ENABLE_SUB_AGENT=1`). **Off by default → the canonical 3-tool experiments are byte-for-byte unchanged**
(verified). The `calculator` uses an AST whitelist, not `eval()`.

### Step 2 — procedural KB + shape-diverse task generator · `scripts/gen_corpus.py`
A seeded generator builds a fictional **knowledge-graph backbone** (continents, countries, cities,
institutes, companies, materials, currencies) and renders one prose document per entity + near-duplicate
distractors, then emits tasks across **five families each engineered to force a distinct graph shape**:

```
./.venv/bin/python scripts/gen_corpus.py --scale 12 --seed 1234
# -> data/corpus_large.json (61 docs), distractors_large.json (6), tasks_families.jsonl (40 tasks)
```
Families: `linear_bridge` (depth), `fan_out_superlative` (width+compare), `counting` (width+compare),
`numeric_diff` (calculator), `unanswerable` (early-stop/failure). Fully deterministic; `--scale` dials
size up to thousands of docs.

### Step 3 — BM25 retrieval · `acg/corpus.py`
Okapi BM25 (no dependency) as an opt-in scoring mode so ranking stays meaningful on the larger corpus.
Select with `ACG_RETRIEVAL=bm25`; the default stays the deterministic keyword-overlap so **prior results
don't move**. Verified: BM25 retrieves the correct supporting docs for every task family.

### Step 4 — external-dataset loader + closed-book baseline · `acg/external.py`, `scripts/closed_book.py`
- `load_context_qa()` reads a canonical `context-QA` JSONL (matches HotpotQA/2Wiki `context`) into
  `(tasks, Corpus)` so the **same tools work unchanged**; `hotpot_to_context()` / `musique_to_context()`
  normalize raw dataset formats. Fixture: `data/external_sample.jsonl`.
- `closed_book.py` runs the **memorization control**: answer with no tools/retrieval. Report
  **retrieval-necessity gap = open-book − closed-book**. Verified: **0/8** closed-book on the fictional
  families (the model says "insufficient information") → the corpus/graph is doing the work.

### Verified live (FP8 14B, `temp=0`, extended tools + BM25) — the payoff
Different task families now realize **structurally different, correct** graphs:

| Family | Realized ACG (from the trace) | New structure |
|---|---|---|
| `numeric_diff` | `search → search → calculator → finish` | **computation node**; answer 18 ✓ |
| `fan_out_superlative` | `search → [read × 3 in parallel] → compare` | **width = 3** + **aggregate node**; answer Quinomont ✓ |
| `counting` | `search → read → verify_claim → finish` | **verifier node**; answer 1 ✓ |

This is the first non-trivial width in the project without `sub_agent`, and the first appearance of
computation/aggregation/verifier node types — exactly the diversity a characterization study needs.

### How to run the enriched benchmark
```bash
# 1. (re)generate the larger, shape-diverse benchmark
./.venv/bin/python scripts/gen_corpus.py --scale 12 --seed 1234

# 2. a single task, drawn graph  (point the client at whatever model is currently served)
ACG_SERVED_MODEL_NAME=<served-name> ACG_EXTRA_TOOLS=calculator,compare,verify_claim,decompose \
ACG_RETRIEVAL=bm25 ACG_MAX_STEPS=12 \
  ./.venv/bin/python - <<'PY'
# (drive acg.agent.Agent over data/corpus_large.json + data/tasks_families.jsonl; see docs example)
PY

# 3. memorization control
ACG_SERVED_MODEL_NAME=<served-name> ./.venv/bin/python scripts/closed_book.py --tasks data/tasks_families.jsonl
```

---

## Next steps after Step 4

> **Progress (2026-07-22): Steps 5, 6, 8 done.**
> **5 ✅** — variance study run on the enriched benchmark across two backbones (480 runs); first
> structure-by-family × backbone results in [docs/12](12-enriched-benchmark-results.md). `run_experiment.py`
> now takes `--corpus/--distractors`; `scripts/analyze_families.py` produces the per-family rollup.
> **6 ✅** — added `constraint_satisfaction` (conjunction → verify-loops) and `conditional` (boolean branch →
> conditional routing) families; benchmark is now **7 families / 54 tasks** (existing 5 unchanged). Scale via
> `gen_corpus.py --scale N`.
> **8 ✅** — MAS arm verified: with `ACG_ENABLE_SUB_AGENT=1` the aggregation families emergently form a
> `planner(decompose) → workers(sub_agent) → aggregator(compare)` tree (one run: 38 nodes, depth 22, 3
> sub-agents, width_executed up to 3) — a distinct graph family vs. the single-agent ~6–11-node runs.
> Remaining: **7** (memory tools — the agent.py scratchpad refactor), **9** (real datasets), **10** (ACG-Bench predictors).
>
> **Analysis reprioritization (2026-07-22):** the docs/12 finding — 14B FP8 is *less* accurate than 7B BF16
> because it **short-circuits the tool loop** (finishes with a ≤1-node graph ~50% of the time on
> tool-composition tasks; 0 parse errors, so it's behavioral) — makes the **precision→short-circuit sweep
> (AWQ 4-bit + Llama-8B)** and a first-class **`short_circuit_frac` metric** the top near-term items. Full
> analysis-driven next steps are in [docs/12](12-enriched-benchmark-results.md#next-steps-analysis-driven).

### Step 5 — re-run the variance study on the enriched benchmark *(immediate)*
Run `run_experiment.py` / `determinism_check.py` over `corpus_large.json` + `tasks_families.jsonl` with the
extended tools, **N reps × varied seed**, per family. Now that shapes differ *by design*, measure
**structural variance within vs. across families** and the modal-shape fraction per family. Needs a small
runner tweak so the scripts accept `--corpus/--tasks` paths (currently hardcoded to `data/corpus.json`).

### Step 6 — scale the KB and add two more families
Raise `--scale` toward thousands of docs (retrieval is CPU-side, so the MIG slice is unaffected; index lives
on `/data`). Add `constraint_satisfaction` (find the entity satisfying A∧B∧C → verify-loops) and
`ambiguous/underspecified` (forces `decompose` + conditional routing) families to broaden the shape space.

### Step 7 — memory tools (`note_write` / `note_read`) *(needs a small refactor)*
Memory nodes require **per-run mutable state** threaded through `execute()` and made **thread-safe** for the
concurrent tool executor (`max_tool_workers>1`). Deferred deliberately: it touches `agent.py`'s tool
dispatch and the `(args, corpus)` purity contract, so it deserves its own change rather than being folded in
here. Design: a per-run `Scratchpad` object passed alongside `corpus`; `note_write(key,value)` /
`note_read(key)` create `memory` nodes and state edges in the ACG.

### Step 8 — a controlled multi-agent topology family
Add a `planner → N workers → aggregator` variant (still **emergent** fan-out via `sub_agent`, not
hand-wired) so single-agent vs. multi-agent **graph families** can be compared. Keep it emergent — a
hand-drawn graph would violate Decision 2 (you'd be measuring your own scaffold).

### Step 9 — wire in real datasets + report the gap *(external validity)*
Fetch **MuSiQue** (gold reasoning graphs), **2WikiMultiHopQA** (typed question shapes), **HotpotQA**
(distractor/fullwiki), **FanOutQA** (real width). Normalize via `acg/external.py`, run with the same tools,
and **always report the closed-book gap**. Fetch is a follow-on (network + license acceptance); the loader
and control are already in place and tested on a fixture.

### Step 10 — feed the enriched traces into ACG-Bench
The enriched runs are the QA slice of **ACG-Bench** ([docs/10](10-research-plan-agentic-application-graphs.md)
§3): canonicalize the realized graphs, attach cost/outcome labels + per-family structural-variance profiles,
and train the graph predictors (cost/latency/failure) from §4.

### Sweep axis (applies throughout)
Run the above across the four served models (7B BF16, 14B FP8/AWQ, Llama-8B) — precision already changes
structure (AWQ collapses, FP8 near-lossless), so **model × tool-set × task-family** is a rich design space.

---

## Files added / changed in Steps 1–4
| File | Change |
|---|---|
| `acg/tools.py` | + `calculator`, `compare`, `verify_claim`, `decompose`; `extra_tools` param; AST-safe calc |
| `acg/config.py` | + `extra_tools` (`ACG_EXTRA_TOOLS`) |
| `acg/agent.py` | pass `extra_tools`; `EXTRA_TOOLS_HINT` |
| `acg/corpus.py` | + BM25 scoring mode (`ACG_RETRIEVAL=bm25`), default unchanged |
| `acg/external.py` | **new** — external context-QA loader + HotpotQA/MuSiQue normalizers |
| `scripts/gen_corpus.py` | **new** — procedural KB + shape-diverse task generator |
| `scripts/closed_book.py` | **new** — memorization-control baseline |
| `data/external_sample.jsonl` | **new** — schema fixture |
| `data/corpus_large.json`, `distractors_large.json`, `tasks_families.jsonl` | **new** — generated benchmark |

**Control preserved throughout:** every addition is deterministic (seeded generator, seeded BM25, pinned
decode) and defaults off, so the canonical experiments remain reproducible and unchanged.
