#!/usr/bin/env bash
# Serve the local open model with vLLM on a single 24GB MIG slice.
#
# Why vLLM (Decision 1 in the proposal): we need a white-box serving stack where
# we can PIN the decode parameters and the seed, hold everything constant except
# sampling, and run unlimited trials at no cost. vLLM's OpenAI-compatible server
# gives us native tool-calling plus an engine `--seed`, which is exactly what a
# variance study requires.
#
# The MIG 1g.24gb slice fits an 8B-class model in 16-bit with room for KV cache,
# which is the regime that gives full control. MIG blocks peer-to-peer / sharding,
# but every agent node is a single independent inference that fits on one slice,
# so that limitation does not affect this project.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

# ---- Config (override via env) ------------------------------------------------
MODEL="${ACG_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_NAME="${ACG_SERVED_MODEL_NAME:-qwen2.5-7b-instruct}"
PORT="${ACG_PORT:-8000}"
CONTAINER_NAME="${ACG_CONTAINER:-acg-vllm}"
IMAGE="${ACG_VLLM_IMAGE:-vllm/vllm-openai:latest}"
HF_CACHE="${ACG_HF_CACHE:-$PROJECT_ROOT/hf-cache}"
MAX_MODEL_LEN="${ACG_MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${ACG_GPU_MEM_UTIL:-0.85}"   # MIG slice reports ~20.9/23.8 GiB free at startup
SEED="${ACG_ENGINE_SEED:-1234}"
TOOL_PARSER="${ACG_TOOL_PARSER:-hermes}"   # hermes works for Qwen2.5; use llama3_json for Llama-3.x

# Pick the MIG slice. Override with ACG_GPU_DEVICE to pin a specific UUID.
GPU_DEVICE="${ACG_GPU_DEVICE:-$(nvidia-smi -L | grep -oE 'MIG-[0-9a-f-]+' | head -n1)}"
if [[ -z "$GPU_DEVICE" ]]; then
  # No MIG slice found; fall back to the whole GPU 0.
  GPU_DEVICE="0"
fi
echo ">> Serving $MODEL on CDI device nvidia.com/gpu=$GPU_DEVICE (port $PORT)"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --device "nvidia.com/gpu=${GPU_DEVICE}" \
  --ipc=host \
  --shm-size=2g \
  -p "${PORT}:8000" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE="${ACG_HF_OFFLINE:-1}" \
  -e VLLM_LOGGING_LEVEL=INFO \
  "$IMAGE" \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --seed "$SEED" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_PARSER"

echo ">> Container '$CONTAINER_NAME' started. Follow logs with:"
echo "   docker logs -f $CONTAINER_NAME"
echo ">> Health (once warmed up): curl http://localhost:${PORT}/v1/models"
