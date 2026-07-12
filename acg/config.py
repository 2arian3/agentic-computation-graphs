"""Central configuration.

Everything that must be *held constant* for a variance study lives here, so a run
is fully described by (model, decode params, seed). Values are overridable by
environment variables (see .env.example) but the defaults are the canonical
experiment settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACES_DIR = PROJECT_ROOT / "traces"
DATA_DIR = PROJECT_ROOT / "data"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass
class DecodeParams:
    """The sampling knobs. Pinned so the ONLY source of run-to-run variance is the
    intended sampling randomness (plus, as a bonus result, serving-batch noise)."""
    temperature: float = field(default_factory=lambda: _env_f("ACG_TEMPERATURE", 0.7))
    top_p: float = field(default_factory=lambda: _env_f("ACG_TOP_P", 0.95))
    max_tokens: int = field(default_factory=lambda: _env_i("ACG_MAX_TOKENS", 1024))
    # Per-request seed. With a fixed seed + temperature=0 the decode is as
    # deterministic as the serving layer allows (see scripts/smoke_test.py).
    seed: int | None = field(default_factory=lambda: _env_i("ACG_REQUEST_SEED", 1234))

    def as_request_kwargs(self) -> dict:
        kw = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            kw["seed"] = self.seed
        return kw


@dataclass
class Config:
    # --- serving endpoint (the local vLLM/SGLang OpenAI-compatible server) ---
    base_url: str = field(default_factory=lambda: _env("ACG_BASE_URL", "http://localhost:8000/v1"))
    api_key: str = field(default_factory=lambda: _env("ACG_API_KEY", "EMPTY"))
    model: str = field(default_factory=lambda: _env("ACG_SERVED_MODEL_NAME", "qwen2.5-7b-instruct"))
    gen_ai_system: str = field(default_factory=lambda: _env("ACG_GENAI_SYSTEM", "qwen"))

    # --- agent loop ---
    max_steps: int = field(default_factory=lambda: _env_i("ACG_MAX_STEPS", 8))
    # Max tool calls run CONCURRENTLY within one step. The model can emit several tool
    # calls in a single turn; a truthful width/parallelism measurement requires the
    # executor to actually run them at once rather than serialize them. Set to 1 to force
    # serial execution -- required for the clean sampling-vs-serving isolation study (§7),
    # where keeping a single request in flight makes the trace bit-reproducible.
    max_tool_workers: int = field(default_factory=lambda: _env_i("ACG_MAX_TOOL_WORKERS", 8))

    # --- decode ---
    decode: DecodeParams = field(default_factory=DecodeParams)

    # --- data / corpus ---
    corpus_path: Path = field(default_factory=lambda: DATA_DIR / "corpus.json")
    tasks_path: Path = field(default_factory=lambda: DATA_DIR / "tasks.jsonl")
    search_top_k: int = field(default_factory=lambda: _env_i("ACG_SEARCH_TOP_K", 3))

    # --- reasoning capture (RQ Q3) ---
    # When true, each tool gains a required `thought` argument so the model must state
    # WHY it takes each step. Off by default so the canonical experiments are unchanged.
    elicit_reasoning: bool = field(
        default_factory=lambda: os.environ.get("ACG_ELICIT_REASONING", "0") == "1")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["corpus_path"] = str(self.corpus_path)
        d["tasks_path"] = str(self.tasks_path)
        return d


def load_config() -> Config:
    return Config()
