# Documentation index

Project: **Agentic Computation Graphs (ACG) — measurement instrument**.
Implements Month 1 + first Month 2 of the [RA proposal](https://app.notion.com/p/38e5b71a74dc8174a76cc46609171d6f).

| Doc | What it covers |
|-----|----------------|
| [01 — Implementation](01-implementation.md) | What is built: architecture, every component, how the ACG is captured & reconstructed, design decisions, how it maps to the proposal |
| [02 — Usage](02-usage.md) | How to use the code: setup, deploy the model, run each script, configuration, extending the corpus/tools |
| [03 — Results so far](03-results.md) | What has been run to date: 96-run full study, 64-run complex-task re-run, graph figures, tests |
| [04 — Next steps](04-next-steps.md) | The gated roadmap (rest of Month 2 → Month 4) and concrete TODOs |
| [05 — Research questions](05-research-questions.md) | Scoped RQs (incl. the 3 supervisor questions) with hypotheses, experiments, metrics, preliminary answers, and what needs bigger GPUs |
| [06 — Critical review & directions](06-critical-review-and-directions.md) | Skeptical assessment: threats to validity (are we measuring the scaffold?), robust-vs-provisional findings, new RQs (N1–N9), obstacles, and the validity-gate recommendation before scaling |
| [**07 — Experiment log (complete record)**](07-experiment-log.md) | **The clear observation of everything done:** every experiment in detail — model served, settings, results, takeaway — plus the serving-config table, consolidated findings, data-artifact index, and next steps |

**Read [07 — Experiment log](07-experiment-log.md) for the full "what we've done till now" picture.**
Start at the top-level [../README.md](../README.md) for a one-page overview. All docs are kept in
sync with the code at `/mnt/agentic-computation-graphs`.

**Status at a glance (2026-07-12):** instrument built & validated; **~1,150 runs** across
**3 serving configs** — Qwen2.5-7B (BF16), Qwen2.5-14B (FP8, 8-bit), Qwen2.5-14B (AWQ, 4-bit) —
all on one 24 GB MIG slice via vLLM. All 3 supervisor questions answered (variance = sampling≫KV
cache≫batching; structure = linear-dominant not branch/parallel; reasoning captured). Precision:
8-bit lossless, 4-bit collapses. Validity gate RQ-N1 done → pivot to **agent capability**
(RQ-N2/N8/N3) next. Not yet on rented GPUs.
