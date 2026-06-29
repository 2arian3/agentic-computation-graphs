#!/usr/bin/env bash
# Alternative serving engine: SGLang (the proposal allows vLLM OR SGLang).
#
# SGLang is interesting for this project specifically because of RadixAttention
# (prefix KV-cache reuse) and its serving-layer controls -- relevant to the Phase-7
# "bonus result" that separates intended sampling variance from incidental
# serving-batch noise. vLLM (serve_vllm.sh) is the default for Month 1 because its
# tool-call parsers and `--seed` are the most battle-tested; this script is provided
# so the engine can be swapped without touching any client code (both expose the same
# OpenAI-compatible API on :8000).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"

MODEL="${ACG_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
SERVED_NAME="${ACG_SERVED_MODEL_NAME:-qwen2.5-7b-instruct}"
PORT="${ACG_PORT:-8000}"
CONTAINER_NAME="${ACG_CONTAINER:-acg-sglang}"
IMAGE="${ACG_SGLANG_IMAGE:-lmsysorg/sglang:latest}"
HF_CACHE="${ACG_HF_CACHE:-$PROJECT_ROOT/hf-cache}"
MAX_MODEL_LEN="${ACG_MAX_MODEL_LEN:-8192}"
MEM_FRACTION="${ACG_GPU_MEM_UTIL:-0.85}"
SEED="${ACG_ENGINE_SEED:-1234}"
TOOL_PARSER="${ACG_SGLANG_TOOL_PARSER:-qwen25}"   # SGLang's parser name for Qwen2.5

GPU_DEVICE="${ACG_GPU_DEVICE:-$(nvidia-smi -L | grep -oE 'MIG-[0-9a-f-]+' | head -n1)}"
[[ -z "$GPU_DEVICE" ]] && GPU_DEVICE="0"
echo ">> Serving $MODEL on CDI device nvidia.com/gpu=$GPU_DEVICE via SGLang (port $PORT)"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --device "nvidia.com/gpu=${GPU_DEVICE}" \
  --ipc=host \
  --shm-size=2g \
  -p "${PORT}:30000" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e HF_HUB_OFFLINE="${ACG_HF_OFFLINE:-1}" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port 30000 \
  --context-length "$MAX_MODEL_LEN" \
  --mem-fraction-static "$MEM_FRACTION" \
  --random-seed "$SEED" \
  --tool-call-parser "$TOOL_PARSER"

echo ">> Container '$CONTAINER_NAME' started. Logs: docker logs -f $CONTAINER_NAME"
echo ">> Health (once warm): curl http://localhost:${PORT}/v1/models"
