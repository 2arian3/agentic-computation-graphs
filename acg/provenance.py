"""Run-level provenance: capture everything needed to reproduce a batch of runs.

A variance study is only meaningful if a trace is self-describing. For every batch of
runs we write a sibling `traces/<name>.provenance.json` that pins:

  * the resolved config -- model, base_url, and the full decode block (temperature,
    top_p, max_tokens, seed) plus max_steps / max_tool_workers / search_top_k;
  * the seed POLICY (fixed vs per-run varied) and rep/task counts, passed by the runner;
  * the environment -- git SHA + dirty flag, host, python;
  * the SERVING stack -- the served model id (from /v1/models) and the vLLM container's
    image digest + engine args (quantization, max-model-len, gpu-util, engine seed, tool
    parser, prefix-caching on/off).

So `(provenance.json, trace.jsonl)` together fully describe how a result was produced.
Nothing here talks to the model; it only reads local git/docker state and the server's
already-public /v1/models endpoint.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

from .config import Config, PROJECT_ROOT


def _sh(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _git_info(root: Path) -> dict:
    g = ("git", "-C", str(root))
    return {
        "sha": _sh(*g, "rev-parse", "HEAD"),
        "short_sha": _sh(*g, "rev-parse", "--short", "HEAD"),
        "branch": _sh(*g, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_sh(*g, "status", "--porcelain")),
    }


def _server_info(base_url: str) -> dict:
    """Served model id (from the OpenAI-compatible /models) + the vLLM container's
    image + engine args, so the exact serving config is recorded, not assumed."""
    info: dict = {"base_url": base_url, "models": None, "docker": None}
    try:
        with urlopen(base_url.rstrip("/") + "/models", timeout=5) as r:
            info["models"] = [m.get("id") for m in json.load(r).get("data", [])]
    except Exception:
        pass
    container = os.environ.get("ACG_CONTAINER", "acg-vllm")
    args = _sh("docker", "inspect", container, "--format", "{{json .Args}}")
    image = _sh("docker", "inspect", container, "--format", "{{.Image}}")
    if args or image:
        info["docker"] = {
            "container": container,
            "image": image,
            "engine_args": json.loads(args) if args else None,
        }
    return info


def capture(cfg: Config, *, experiment: str, extra: dict | None = None) -> dict:
    """Snapshot config + environment + serving stack for one batch of runs.

    `extra` is where the runner records what config alone cannot know: the seed policy
    (fixed vs varied and the base value), rep count, and the task set.
    """
    return {
        "experiment": experiment,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git": _git_info(PROJECT_ROOT),
        "config": cfg.to_dict(),   # includes the full decode block: temperature/top_p/max_tokens/seed
        "server": _server_info(cfg.base_url),
        "run": extra or {},
    }


def write(prov: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    return path


def capture_and_write(cfg: Config, path: str | Path, *, experiment: str,
                      extra: dict | None = None) -> dict:
    prov = capture(cfg, experiment=experiment, extra=extra)
    write(prov, path)
    return prov
