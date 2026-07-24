#!/usr/bin/env bash
# Pre-download all 4 model configs into the HF cache on /data (the big volume),
# so vLLM can serve them offline (serve_vllm.sh mounts this cache, HF_HUB_OFFLINE=1).
export HF_HOME=/data/hf-cache
export HF_XET_HIGH_PERFORMANCE=1
HF=/home/ubuntu/agentic-computation-graphs/.venv/bin/hf
mkdir -p /data/hf-cache

models=(
  "Qwen/Qwen2.5-7B-Instruct"                       # M1 BF16  ~15GB
  "RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic"      # M2 FP8   ~14GB
  "Qwen/Qwen2.5-14B-Instruct-AWQ"                  # M3 AWQ   ~9GB
  "NousResearch/Meta-Llama-3.1-8B-Instruct"        # M4 BF16  ~15GB
)

rc=0
for m in "${models[@]}"; do
  echo "===== $(date -Is) START $m ====="
  # Exclude the redundant PyTorch consolidated checkpoints (vLLM uses safetensors).
  # NOTE: --exclude takes ONE glob per flag; extra bare globs become a file allowlist.
  if "$HF" download "$m" --exclude "original/*"; then
    echo "===== $(date -Is) OK   $m ====="
  else
    echo "===== $(date -Is) FAIL $m ====="
    rc=1
  fi
done
echo "===== $(date -Is) ALL DONE (rc=$rc) ====="
exit $rc
