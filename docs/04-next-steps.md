# 04 — Next steps

The instrument is built and validated, and the first measurements are in
([03-results.md](03-results.md)). This is the gated roadmap for the rest of the contract,
plus concrete near-term TODOs. Order matters: each phase de-risks the next, and we pivot
cheaply if an early phase shows the question is uninteresting.

---

## Immediate (finish Month 2)

1. **Scale reps where variance is high.** T06 and T12 (4-hop) show the widest spread (6 and
   5 distinct shapes at n=8). Run **30–50 reps** on at least these to get stable tail
   estimates (the p95/p99 is what cost planning cares about). Cheap:
   `make experiment REPS=40` (≈8 min for all tasks at the current 3.6 s/run; or restrict to
   `--tasks T06,T12`).
2. **Tighten the variance estimators.** With more reps, report a real **graph-edit-distance**
   distribution per task (the machinery exists in `analyze.py`; currently sampled). Add a
   confidence interval on the modal-signature fraction (the "stable core" proxy).
3. **Fix or quarantine the weak tasks.** T11 is 0% correct and T03/T05 are 50%. Check whether
   it's a retrieval/answer-grading issue or genuine model failure; decide whether to keep
   them (failures are still valid graphs) or revise the gold/aliases.
4. **Persist run-level provenance.** Write the resolved config (model, decode, seed policy,
   git SHA, image digest) alongside each `experiment.jsonl` so a trace is self-describing.

## Month 3 — characterize properly & stress the findings

5. **Temperature sweep.** Quantify how graph size/variance scales with decode temperature
   (the main driver). Run the grid `temp ∈ {0.0, 0.3, 0.7, 1.0}`, fixed reps, and plot
   node-count/​token distributions vs temperature. Expect: higher temp → larger spread, more
   distinct shapes.
6. **Second model if it fits 24 GB.** Compare against another open model (e.g. a different 7–8B
   instruct, or an AWQ-quantized 14B) to test whether the "serial decomposition / width = 1"
   observation is model-specific. One-line swap via `ACG_MODEL` + matching `ACG_TOOL_PARSER`.
7. **Stable-core analysis.** Formalize the recurring subgraph: across runs of a task, extract
   the subgraph present in ≥90% of runs and report its size vs the full-graph size. This turns
   the "modal fraction" proxy into the actual stable-core result the proposal asks for.
8. **Provoke branching (width > 1).** Optionally allow/encourage parallel tool calls (the API
   already returns lists of tool calls; the loop already handles them — the model just never
   chose to). A prompt variant or a harder task family could surface real width, making the
   width metric non-trivial. Keep this *measured*, not *imposed*.

## Month 4 — consolidate, realism check, decide

9. **External-validity check (the one closed-product touch).** Run a handful of the same tasks
   through Claude Code / Codex and compare the *rough* graph shape to the local model's, to see
   whether our findings sit in the same regime as a frontier agent. This is a sanity check, not
   a controlled measurement (closed products can't be pinned).
10. **Write-up + release.** Short report / workshop-length paper; release the **trace dataset**
    (the JSONL store) and the **harness** (this repo) as the concrete artifacts.
11. **The gated decision.** With the supervisor, decide whether the next contract targets
    **optimization** — e.g. predicting an ACG's cost ahead of time from the task, or pruning
    the graph to cut calls without hurting accuracy. Optimization is deliberately *out of scope*
    until this characterization exists.

---

## Engineering backlog (nice-to-have, not blocking)

- **Concurrency for throughput.** Runs are currently sequential (one run at a time). vLLM
  batches happily; a small async/threadpool driver would cut wall-clock for large `REPS`
  sweeps. (Note: batching is also a *variable* in the §7 study — keep a single-request mode
  for clean sampling-vs-serving isolation.)
- **A real OTLP path.** The JSONL exporter is OTLP-shaped; optionally also push to a collector
  / Jaeger for interactive trace viewing, without changing the agent.
- **HotpotQA bridge.** The proposal mentions standard benchmarks. Add a loader that maps a
  slice of HotpotQA (question + its gold paragraphs as the corpus) into our `Task`/`Corpus`
  format, so results are comparable to other work alongside the owned fictional set.
- **Per-run answer/justification logging** for error analysis on the weak tasks.
- **GED performance guard.** Cap graph size / add a hard timeout so the exact graph-edit-distance
  never stalls analysis on unexpectedly large graphs.

---

## Open questions for the supervisor (from the proposal)

- **Domain confirmation.** Tool-using multi-hop QA is the proposed domain; swapping benchmarks
  is cheap and leaves the rest of the plan unchanged.
- **Characterization vs optimization balance.** Optimization is intentionally kept out of the
  4-month commitment and made the explicit decision point at the end. If a guaranteed
  optimization deliverable is expected, the timeline needs rebalancing.
- **What "enough reps" means.** Define the target precision for the tail (e.g. p95 within ±X%),
  which sets the rep budget per task.
