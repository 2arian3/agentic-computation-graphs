# characterization — public-dataset ACG corpus

Extracts **agentic computation graphs** from public agent trace datasets into one
canonical schema, then measures their size, structure and field coverage.

This is the empirical counterpart to the instrumented runs in `acg/`. Where the
main project *generates* graphs from a controlled agent on vLLM, this directory
*harvests* them from traces that already exist, to answer: how large and how
parallel are the graphs real agents actually produce, and what do public datasets
record about them?

Corresponds to **Step 9 (real datasets)** on the roadmap in `docs/11`.

## Headline results

**153,486 graphs / 13.9M nodes** extracted from four datasets, zero extraction
losses.

| dataset | graphs | nodes | med nodes | med depth | **med fan-out** | reasoning | tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| tracelab | 4,265 | 838K | 46 | 33 | **3** (max 29) | 0% | 42.6% |
| swe_rebench_openhands | 67,074 | 8.70M | 123 | 122 | **1** (max 2) | 71.2% | 0% |
| swe_agent_traj | 80,036 | 4.31M | 35 | 34 | **1** (max 1) | 100% | 0% |
| osworld_gelato | 2,111 | 64.9K | 20 | 19 | **1** (max 1) | 89.6% | 0% |

Three findings, all measured:

1. **Parallelism is a property of the harness, not the model.** Across 13.1M
   nodes from three benchmark harnesses, two scaffolds and two domains, max
   fan-out is **1** (4 graphs of 67,074 reach 2). TraceLab's *production*
   sessions reach **29**, with 16.8% of rounds issuing multiple tools and 88.2%
   of those showing measured wall-clock overlap. This independently corroborates
   the main project's "agents linearize by policy" result — and shows the
   linearization is partly imposed by the scaffold.

2. **Graph size is bounded by configuration.** OpenHands piles 8.9% of runs at
   exactly 201 nodes (a 100-iteration cap); OSWorld stops at 100 (a 50-step
   budget); SWE-agent has no ceiling and decays to 817; TraceLab is unbounded to
   18,482. Published size distributions describe scaffold settings as much as
   workloads.

3. **Cost and semantics are disjoint in public data.** TraceLab has 100%
   timestamps / 42.6% tokens / 42.6% KV and **0% reasoning**. The other three
   have 71–100% reasoning and **0% of every cost field**. No public source has
   both — which is the concrete argument for Phase 2 instrumented runs.

## Reports

| file | contents |
|---|---|
| `reports/SUMMARY.md` | cross-dataset comparison table (regenerated, never hand-edited) |
| `reports/DATASET_COMPARISON.md` | trace counts, field-by-field has/hasn't, and **real-inference vs simulation** across four layers |
| `reports/CONTENT_AVAILABILITY.md` | audit of all 16 originally-surveyed datasets: can each show *why* a node ran |
| `reports/<dataset>.md` | per-dataset source, mapping, structure, caveats |
| `reports/figures/tracelab/` | rendered ACGs incl. an 8-wide sub-agent fan-out and the Codex `spawn_agent` lifecycle |

## Layout

```
registry.yaml            19 datasets: source, licence, status, content + provenance audit
src/schema.py            canonical Node/Edge/Graph + strict validation
src/characterize.py      structural metrics and coverage
src/visualize.py         renders an ACG (stats header + DAG + wall-clock timeline)
src/extractors/          one parser per dataset
scripts/                 make_summary.py, render_gallery.py, audit_content.py
data -> /data/agentic-graph-corpus/data     (symlink; NOT in git)
```

## Data is not in git

`data/` is a symlink to `/data/agentic-graph-corpus/data` — **131 GB** of raw
downloads and derived graphs, deliberately untracked and reproducible from
`registry.yaml` plus the extractors. The root disk is only 20 GB, which is why it
lives on the `/data` volume.

## Reproducing

```bash
cd characterization
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python networkx pandas orjson pyyaml pyarrow matplotlib huggingface_hub

.venv/bin/python -m src.extractors.tracelab        # ~28 s
.venv/bin/python -m src.characterize tracelab      # ~16 s
.venv/bin/python scripts/render_gallery.py tracelab
.venv/bin/python scripts/make_summary.py
```

Every extractor is idempotent (`--force` to redo) and takes `--limit N` for a
smoke run. Raw downloads must be re-fetched first; see each dataset's report for
its source URL and checksum.

## Design rule

**Absent means null, never a guess.** Every cost and semantic field on `Node`
defaults to `None`, and `validate_graph` rejects dangling edges and bad enums.
Measuring honestly which fields a dataset *lacks* is a primary result, not a
defect — that is how findings (3) above is even statable.

See `DEVELOPMENT_HISTORY.md` for the original standalone commit history.
