# 02 — Usage

How to set up, deploy the model, and run everything. All paths are relative to the
project root `/mnt/agentic-computation-graphs`. There is a `Makefile` with shortcuts;
each section shows both the `make` target and the raw command.

---

## Prerequisites

- An NVIDIA GPU visible to Docker via CDI. On this machine it's an **H100 MIG `1g.24gb`
  slice**; `serve_vllm.sh` auto-detects the CDI device with
  `nvidia-smi -L | grep -oE 'MIG-[0-9a-f-]+'`.
- Docker with the **containerd image store on a big disk** (the vLLM image is ~30 GB). If
  `docker pull` fails with `no space left on device`, relocate it once:
  ```bash
  sudo systemctl stop docker docker.socket containerd
  sudo mv /var/lib/containerd /big-disk/containerd && sudo ln -s /big-disk/containerd /var/lib/containerd
  sudo systemctl start containerd docker
  ```
  (On this machine that's already done — it points to `/mnt/containerd`.)
- Python 3.10+ for the client venv.
- The model is **ungated** (Qwen2.5), so no HuggingFace token is required.

---

## 1. Install client dependencies

```bash
make venv
# equivalent to:
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
```

> Note: only *client* libraries are installed here (openai, opentelemetry, networkx,
> numpy, matplotlib, pytest…). The model itself runs in the Docker container, not in this
> venv. On a small-root-disk machine, point pip's temp/cache at the big disk:
> `TMPDIR=/mnt/.../.tmp PIP_CACHE_DIR=/mnt/.../.pipcache`.

## 2. (Optional) pre-download the model weights

Speeds up first container start; otherwise the container downloads them itself.

```bash
HF_HOME=$PWD/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 \
  ./.venv/bin/python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen2.5-7B-Instruct', ignore_patterns=['*.pth','original/*'])"
```

## 3. Deploy the model

```bash
make serve            # bash docker/serve_vllm.sh
docker logs -f acg-vllm                 # watch warmup (~1–2 min)
curl http://localhost:8000/v1/models    # -> qwen2.5-7b-instruct when ready
```

Alternative engine (same API on :8000):
```bash
make serve-sglang     # bash docker/serve_sglang.sh
```
Stop the server: `make stop`.

## 4. Validate the instrument

```bash
make smoke            # ./.venv/bin/python scripts/smoke_test.py
```
Checks: server up · plain completion · **determinism** (temp 0 + seed → identical twice) ·
**tool-calling** emits a structured call. All four must pass before collecting data.

---

## 5. Run things

### One task end-to-end (Month-1 milestone) — draw its graph
```bash
make single TASK=T06
# ./.venv/bin/python scripts/run_single.py --task T06
```
Prints the reconstructed ACG (ASCII) + metrics; saves `traces/figures/acg_T06.png` and
`traces/single_T06.jsonl`. Options: `--temperature`, `--seed`.

### The multi-QA variance study (Month-2)
```bash
make experiment REPS=8
# ./.venv/bin/python scripts/run_experiment.py --tasks all --reps 8 --temperature 0.7 --vary-seed
```
Runs every task `REPS` times, writes one trace per run into `traces/experiment.jsonl`,
then prints the per-task table and writes `traces/metrics.csv`, `traces/summary.json`,
and the boxplot figures.

- `--tasks all` or e.g. `--tasks T02,T06`.
- `--vary-seed` gives each rep a distinct (reproducible) seed → **captures sampling
  variance**. Without it the fixed seed makes every rep identical (zero variance).
- `--temperature 0.7` is the study default; use `0.0` for a determinism baseline.

### Separate the two variance sources (§7 bonus)
```bash
make determinism TASK=T06
# ./.venv/bin/python scripts/determinism_check.py --task T06 --reps 12
```
Runs three regimes (fixed-seed temp 0, fixed-seed temp 0.7, varied-seed temp 0.7) and
reports how many distinct ACG structures appear in each.

### Re-analyze an existing trace (no model needed)
```bash
make analyze          # ./.venv/bin/python scripts/analyze.py --trace traces/experiment.jsonl
```

### Tests
```bash
make test             # ./.venv/bin/python -m pytest tests/ -v -s
```
Unit tests always run; the **live** tests that produce AGCs for multiple QA programs skip
automatically if the server is unreachable.

---

## 6. Configuration

Everything is overridable by environment variable (see [`.env.example`](../.env.example)).
Copy it and source it, or export individual vars:

```bash
cp .env.example .env && set -a && source .env && set +a
```

| Variable | Default | Meaning |
|----------|---------|---------|
| `ACG_BASE_URL` | `http://localhost:8000/v1` | model endpoint |
| `ACG_SERVED_MODEL_NAME` | `qwen2.5-7b-instruct` | model id used in requests |
| `ACG_TEMPERATURE` | `0.7` | decode temperature (`0.0` for determinism) |
| `ACG_TOP_P` / `ACG_MAX_TOKENS` | `0.95` / `1024` | decode knobs |
| `ACG_REQUEST_SEED` | `1234` | per-request seed |
| `ACG_MAX_STEPS` | `8` | agent loop cap |
| `ACG_SEARCH_TOP_K` | `3` | results returned by `search` |
| `ACG_MODEL` | `Qwen/Qwen2.5-7B-Instruct` | served model (for `serve_*.sh`) |
| `ACG_MAX_MODEL_LEN` | `8192` | context length |
| `ACG_GPU_MEM_UTIL` | `0.85` | vLLM mem fraction (≤0.85 on this MIG slice) |
| `ACG_TOOL_PARSER` | `hermes` | tool-call parser (`llama3_json` for Llama-3.x) |
| `ACG_GPU_DEVICE` | auto | pin a specific MIG CDI device |

---

## 7. Outputs (where results land)

```
traces/
  experiment.jsonl     one OTel trace (many spans) per run — the raw, replayable record
  single_<task>.jsonl  a single-run trace
  metrics.csv          one row per run: all per-run ACG metrics (open in any tool)
  summary.json         overall + per-task distributions and structural variance
  figures/
    acg_<task>.png       a drawn ACG
    dist_node_count.png  per-task graph-size distribution (boxplot)
    dist_total_tokens.png per-task cost distribution (boxplot)
```

`make clean` removes generated traces/figures (keeps code + `data/`).

---

## 8. Extending the benchmark

- **Add tasks:** append lines to [`data/tasks.jsonl`](../data/tasks.jsonl)
  (`{"task_id","hops","question","answers":[...],"supporting":[...]}`). Make sure the answer
  is derivable by chaining documents in `corpus.json`.
- **Add documents:** append to [`data/corpus.json`](../data/corpus.json)
  (`{"id","title","text"}`). Keep the world self-consistent so multi-hop chains resolve.
- **Change the tool set:** edit [`acg/tools.py`](../acg/tools.py) (`tool_schemas` + `execute`).
  Remember: the tool names are the tool-node *types*, so changing them changes the node
  alphabet of every graph.
- **Swap the model:** set `ACG_MODEL` and a matching `ACG_TOOL_PARSER`, re-run `make serve`.
  Going beyond ~8B means leaving FP16 (e.g. an AWQ-quantized 14B); note that in the manifest.
