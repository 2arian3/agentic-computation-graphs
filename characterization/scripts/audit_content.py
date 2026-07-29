"""Audit a downloaded dataset for semantic content coverage.

Answers, per dataset, the question the cost-coverage metrics cannot: does a
record carry the model's reasoning, the tool it chose, the arguments it passed,
what came back, and the prompt it was working under?

Reports measured percentages -- never a claim from documentation.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.common import RAW  # noqa: E402


def _pct(a: int, b: int) -> str:
    return f"{a:,} ({100 * a / b:.1f}%)" if b else "0 (n/a)"


def _nonempty(v) -> bool:
    return bool(v) and str(v).strip() not in ("", "None", "null")


def audit_swe_rebench_openhands() -> None:
    """OpenHands scaffold: message list with structured tool_calls."""
    print("=" * 74)
    print("swe_rebench_openhands  (nebius/SWE-rebench-openhands-trajectories)")
    print("=" * 74)
    root = RAW / "swe_rebench_openhands"
    pf = next(root.rglob("*.parquet"))
    f = pq.ParquetFile(pf)
    print(f"file {pf.name}  rows={f.metadata.num_rows:,}  cols={f.schema_arrow.names}")
    batch = next(f.iter_batches(batch_size=150))
    rows = batch.to_pylist()

    asst = reas = args = tools_msg = out = sysmsg = 0
    roles: Counter[str] = Counter()
    tool_names: Counter[str] = Counter()
    for r in rows:
        tr = r.get("trajectory")
        if isinstance(tr, str):
            tr = json.loads(tr)
        for m in tr or []:
            roles[m.get("role")] += 1
            if m.get("role") == "system":
                if _nonempty(m.get("content")):
                    sysmsg += 1
            elif m.get("role") == "assistant":
                asst += 1
                if _nonempty(m.get("reasoning_content")) or _nonempty(m.get("content")):
                    reas += 1
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or tc
                    tool_names[fn.get("name")] += 1
                    if _nonempty(fn.get("arguments")):
                        args += 1
            elif m.get("role") == "tool":
                tools_msg += 1
                if _nonempty(m.get("content")):
                    out += 1
    print(f"sampled {len(rows)} trajectories; roles={dict(roles)}")
    print(f"  assistant steps with reasoning/content : {_pct(reas, asst)}")
    print(f"  tool calls with arguments             : {args:,}")
    print(f"  tool results non-empty                : {_pct(out, tools_msg)}")
    print(f"  system prompt present                 : {sysmsg:,} trajectories")
    print(f"  tool names                            : {dict(tool_names.most_common(8))}")
    print(f"  outcome labels                        : resolved / exit_status / gen_tests_correct")
    print(f"  prompt tool schemas                   : tools.json "
          f"({'present' if (root / 'tools.json').exists() else 'MISSING'})")


def audit_swe_agent() -> None:
    """SWE-agent scaffold: thought prose + a fenced command per step."""
    print()
    print("=" * 74)
    print("swe_agent_traj  (nebius/SWE-agent-trajectories)")
    print("=" * 74)
    root = RAW / "swe_agent_traj"
    shards = sorted(root.rglob("*.parquet"))
    print(f"{len(shards)} shards")
    t = pq.read_table(shards[0], columns=["trajectory", "instance_id", "model_name",
                                          "exit_status", "target"]).slice(0, 150).to_pylist()
    roles: Counter[str] = Counter()
    steps = withtext = sysp = ai = ai_text = 0
    models: Counter[str] = Counter()
    for r in t:
        models[r.get("model_name")] += 1
        tr = r["trajectory"]
        if isinstance(tr, str):
            tr = json.loads(tr)
        for s in tr:
            steps += 1
            roles[s.get("role")] += 1
            if _nonempty(s.get("text")):
                withtext += 1
            if s.get("role") == "system" and _nonempty(s.get("system_prompt")):
                sysp += 1
            if s.get("role") == "ai":
                ai += 1
                if _nonempty(s.get("text")):
                    ai_text += 1
    print(f"sampled {len(t)} trajectories, {steps:,} steps; roles={dict(roles)}")
    print(f"  steps with text                       : {_pct(withtext, steps)}")
    print(f"  agent steps with reasoning + action   : {_pct(ai_text, ai)}")
    print(f"  full system_prompt present            : {sysp:,} trajectories")
    print(f"  models                                : {dict(models.most_common(5))}")
    print(f"  outcome labels                        : exit_status / eval_logs / target (gold patch)")
    print(f"  NOTE: tool name is inside the action text (fenced command) -> needs parsing")


def audit_osworld() -> None:
    """Computer-use agent: one JSONL per episode, one record per step."""
    print()
    print("=" * 74)
    print("osworld_gelato  (mlfoundations/gelato-osworld-agent-trajectories)")
    print("=" * 74)
    root = RAW / "osworld_gelato"
    files = sorted(root.rglob("traj.jsonl"))
    print(f"{len(files):,} episode files")
    steps = reas = args = out = cmd = 0
    names: Counter[str] = Counter()
    apps: Counter[str] = Counter()
    for p in files[:400]:
        apps[p.parent.parent.name] += 1
        try:
            recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        except json.JSONDecodeError:
            continue
        for rec in recs:
            for s in (rec if isinstance(rec, list) else [rec]):
                if not isinstance(s, dict):
                    continue
                steps += 1
                names[s.get("name")] += 1
                if _nonempty(s.get("reasoning")):
                    reas += 1
                if _nonempty(s.get("arguments")):
                    args += 1
                if _nonempty(s.get("tool_output")):
                    out += 1
                if _nonempty(s.get("command")):
                    cmd += 1
    print(f"sampled 400 episodes, {steps:,} steps")
    print(f"  steps with reasoning                  : {_pct(reas, steps)}")
    print(f"  steps with structured arguments       : {_pct(args, steps)}")
    print(f"  steps with executed command           : {_pct(cmd, steps)}")
    print(f"  steps with tool_output                : {_pct(out, steps)}")
    print(f"  tool names                            : {dict(names.most_common(10))}")
    print(f"  applications                          : {dict(apps.most_common(10))}")


if __name__ == "__main__":
    which = sys.argv[1:] or ["openhands", "sweagent", "osworld"]
    if "openhands" in which:
        audit_swe_rebench_openhands()
    if "sweagent" in which:
        audit_swe_agent()
    if "osworld" in which:
        audit_osworld()
