"""Hot-swap the model vLLM serves, by re-running docker/serve_vllm.sh.

The MIG slice serves ONE model at a time, so switching means restarting the `acg-vllm`
container and waiting for warmup. This module drives that from the backend and streams
progress, but changes nothing about how a model is served — it just invokes the repo's
existing serve script with the right env for the chosen model, then polls /v1/models
until the new model answers.

Only models whose weights are already in hf-cache/ are offered (served offline), so a swap
never triggers a multi-GB download. Serving is serialized and holds the live-run lock, so a
swap never races an in-flight experiment.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from urllib.parse import urlparse

from acg.config import Config

from . import paths, presets, pipeline

CONTAINER = os.environ.get("ACG_CONTAINER", "acg-vllm")

# served-model-name -> how to serve it. Only the four models cached in hf-cache/.
# Flags mirror docs/ + the notes: llama needs its chat template + llama3_json parser;
# FP8 on the MIG slice needs --enforce-eager; controlled serving uses prefix cache off.
SERVE_SPECS: dict[str, dict] = {
    "qwen2.5-7b-instruct": {
        "label": "Qwen2.5-7B-Instruct (BF16)", "model": "Qwen/Qwen2.5-7B-Instruct",
        "parser": "hermes", "system": "qwen", "max_len": 8192, "gpu": 0.85,
        "extra": "--no-enable-prefix-caching", "warmup_s": 200,
    },
    "qwen2.5-14b-instruct-awq": {
        "label": "Qwen2.5-14B-Instruct (AWQ, 4-bit)", "model": "Qwen/Qwen2.5-14B-Instruct-AWQ",
        "parser": "hermes", "system": "qwen", "max_len": 8192, "gpu": 0.85,
        "extra": "--quantization awq_marlin --no-enable-prefix-caching", "warmup_s": 380,
    },
    "qwen2.5-14b-instruct-fp8": {
        "label": "Qwen2.5-14B-Instruct (FP8, 8-bit)", "model": "RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic",
        "parser": "hermes", "system": "qwen", "max_len": 8192, "gpu": 0.85,
        "extra": "--enforce-eager --no-enable-prefix-caching", "warmup_s": 380,
    },
    "llama3.1-8b-instruct": {
        "label": "Llama-3.1-8B-Instruct (BF16)", "model": "NousResearch/Meta-Llama-3.1-8B-Instruct",
        "parser": "llama3_json", "system": "llama", "max_len": 4096, "gpu": 0.80,
        "extra": "--no-enable-prefix-caching --chat-template "
                 "/vllm-workspace/examples/tool_chat_template_llama3.1_json.jinja",
        "warmup_s": 220,
    },
}

_swap_lock = threading.Lock()
_busy = threading.Event()


def _sh(*args: str) -> str:
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True, timeout=15).strip()
    except Exception as e:
        return f"({type(e).__name__}: {e})"


def _mig_device() -> str | None:
    m = re.search(r"MIG-[0-9a-f-]+", _sh("nvidia-smi", "-L"))
    return m.group(0) if m else None


def _port(base_url: str) -> int:
    p = urlparse(base_url).port
    return p or 8000


def _container_running() -> bool:
    return _sh("docker", "inspect", "-f", "{{.State.Running}}", CONTAINER).strip() == "true"


def _logs_tail(n: int = 24) -> str:
    return _sh("docker", "logs", "--tail", str(n), CONTAINER)


def state() -> dict:
    cfg = Config()
    served = presets.server_models(cfg.base_url)
    current = served[0] if served else None
    return {
        "base_url": cfg.base_url,
        "server_up": bool(served),
        "current": current,
        "busy": _busy.is_set(),
        "servable": [
            {"served": k, "label": v["label"], "system": v["system"],
             "current": k == current}
            for k, v in SERVE_SPECS.items()
        ],
    }


def swap(model: str, on_event) -> None:
    """Restart vLLM to serve `model`, streaming progress via on_event(dict)."""
    if model not in SERVE_SPECS:
        on_event({"kind": "serve_error", "error": f"unknown/unservable model '{model}'"})
        return
    spec = SERVE_SPECS[model]
    cfg = Config()

    if not _swap_lock.acquire(blocking=False):
        on_event({"kind": "serve_error", "error": "another model swap is already in progress"})
        return
    _busy.set()
    try:
        # Block runs during the swap (and wait for any in-flight run to finish first).
        on_event({"kind": "serve_progress", "msg": "acquiring run lock (waiting for any active run)…"})
        with pipeline.LIVE_RUN_LOCK:
            if model in presets.server_models(cfg.base_url):
                on_event({"kind": "serve_done", "model": model, "system": spec["system"], "already": True})
                return

            env = os.environ.copy()
            env.update({
                "ACG_MODEL": spec["model"],
                "ACG_SERVED_MODEL_NAME": model,
                "ACG_TOOL_PARSER": spec["parser"],
                "ACG_MAX_MODEL_LEN": str(spec["max_len"]),
                "ACG_GPU_MEM_UTIL": str(spec["gpu"]),
                "ACG_PORT": str(_port(cfg.base_url)),
                "ACG_CONTAINER": CONTAINER,
                "ACG_HF_OFFLINE": "1",
                "ACG_EXTRA_ARGS": spec["extra"],
            })
            mig = _mig_device()
            if mig:
                env["ACG_GPU_DEVICE"] = mig

            on_event({"kind": "serve_progress", "msg": f"restarting container for {spec['model']} …"})
            proc = subprocess.run(
                ["bash", str(paths.REPO_ROOT / "docker" / "serve_vllm.sh")],
                env=env, capture_output=True, text=True, timeout=150,
            )
            for line in (proc.stdout + proc.stderr).splitlines()[-6:]:
                if line.strip():
                    on_event({"kind": "serve_progress", "msg": line.strip()})
            if proc.returncode != 0:
                on_event({"kind": "serve_error", "error": (proc.stderr or proc.stdout)[-1000:]})
                return

            # Poll for readiness while the engine loads weights + warms up.
            deadline = time.time() + spec["warmup_s"]
            t0 = time.time()
            while time.time() < deadline:
                served = presets.server_models(cfg.base_url)
                if model in served:
                    on_event({"kind": "serve_done", "model": model, "system": spec["system"]})
                    return
                if not _container_running():
                    on_event({"kind": "serve_error",
                              "error": "container exited during startup:\n" + _logs_tail()})
                    return
                on_event({"kind": "serve_progress", "msg": f"loading weights / warming up… {int(time.time()-t0)}s"})
                time.sleep(3)

            on_event({"kind": "serve_error",
                      "error": f"timed out after {spec['warmup_s']}s. Last logs:\n" + _logs_tail()})
    except subprocess.TimeoutExpired:
        on_event({"kind": "serve_error", "error": "serve script timed out"})
    except Exception as e:
        on_event({"kind": "serve_error", "error": f"{type(e).__name__}: {e}"})
    finally:
        _busy.clear()
        _swap_lock.release()
