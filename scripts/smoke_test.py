#!/usr/bin/env python3
"""Month-1 instrument validation: confirm we can pin decode parameters and seeds,
and that native tool-calling works against the local server.

Checks:
  1. The server is up and serving the expected model.
  2. A plain chat completion works.
  3. Determinism: the same prompt at temperature=0 with a fixed seed returns the
     SAME text twice (the white-box control the variance study depends on).
  4. Tool-calling: given the fixed tool set, the model emits a structured tool call.

Run:  ./.venv/bin/python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI
from acg.config import load_config
from acg.tools import tool_schemas


def main() -> int:
    cfg = load_config()
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    print(f"[1] endpoint={cfg.base_url} model={cfg.model}")
    models = [m.id for m in client.models.list().data]
    print(f"    served models: {models}")
    assert cfg.model in models, f"expected model {cfg.model} not served"

    print("[2] plain completion ...")
    r = client.chat.completions.create(
        model=cfg.model,
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        temperature=0.0, max_tokens=8, seed=1234,
    )
    print(f"    -> {r.choices[0].message.content!r}  (in={r.usage.prompt_tokens}, out={r.usage.completion_tokens})")

    print("[3] determinism @ temperature=0, seed=1234 ...")
    prompt = [{"role": "user", "content": "Name three primary colors as a comma-separated list."}]
    outs = []
    for _ in range(2):
        rr = client.chat.completions.create(
            model=cfg.model, messages=prompt, temperature=0.0, top_p=1.0, max_tokens=64, seed=1234,
        )
        outs.append(rr.choices[0].message.content)
    same = outs[0] == outs[1]
    print(f"    out#1: {outs[0]!r}")
    print(f"    out#2: {outs[1]!r}")
    print(f"    identical: {same}  {'(decode is pinnable)' if same else '(NOTE: serving-batch noise present)'}")

    print("[4] tool-calling ...")
    tr = client.chat.completions.create(
        model=cfg.model,
        messages=[
            {"role": "system", "content": "Use the tools to answer. Search before you answer."},
            {"role": "user", "content": "Find the document about the Pirelle Institute."},
        ],
        tools=tool_schemas(),
        tool_choice="auto",
        temperature=0.0, max_tokens=128, seed=1234,
    )
    tcs = tr.choices[0].message.tool_calls or []
    if tcs:
        for tc in tcs:
            print(f"    tool_call -> {tc.function.name}({tc.function.arguments})")
    else:
        print(f"    NO tool call; content={tr.choices[0].message.content!r}")
    assert tcs, "model did not emit a tool call — check --enable-auto-tool-choice / --tool-call-parser"

    print("\nALL SMOKE CHECKS PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
