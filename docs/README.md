# Documentation index

Project: **Agentic Computation Graphs (ACG) — measurement instrument**.
Implements Month 1 + first Month 2 of the [RA proposal](https://app.notion.com/p/38e5b71a74dc8174a76cc46609171d6f).

| Doc | What it covers |
|-----|----------------|
| [01 — Implementation](01-implementation.md) | What is built: architecture, every component, how the ACG is captured & reconstructed, design decisions, how it maps to the proposal |
| [02 — Usage](02-usage.md) | How to use the code: setup, deploy the model, run each script, configuration, extending the corpus/tools |
| [03 — Results so far](03-results.md) | What has been run to date: 96-run full study, 64-run complex-task re-run, graph figures, tests |
| [04 — Next steps](04-next-steps.md) | The gated roadmap (rest of Month 2 → Month 4) and concrete TODOs |

Start at the top-level [../README.md](../README.md) for a one-page overview, then read
these in order. All four are kept in sync with the code at
`/mnt/agentic-computation-graphs`.

**Status at a glance (2026-06-29):** instrument built and validated (7/7 tests pass);
Qwen2.5-7B-Instruct served live on the 24 GB MIG slice via vLLM; 96-run full study,
64-run complex-task re-run, readable ACG graph figures, and the §7 variance decomposition
completed.
