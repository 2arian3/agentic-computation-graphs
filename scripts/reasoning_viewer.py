#!/usr/bin/env python3
"""RQ (supervisor Q3): SEE, per LLM step, what the model saw, what it reasoned, and
what it decided.

Builds a self-contained HTML timeline for one run: each LLM call becomes a card with
three panels — SAW (the inputs to that step: the question, or the tool result it just
received), REASONED (the model's verbalized reasoning — its `content`, plus the
`thought` arg when reasoning-elicitation is on), and DECIDED (the tool call it emitted,
with arguments) — followed by the tool's result feeding the next step.

  ./.venv/bin/python scripts/reasoning_viewer.py --trace traces/experiment.jsonl --task T06
  ACG_ELICIT_REASONING=1 ...run a fresh trace first to populate REASONED...
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from acg import graph as G
from acg import tracing as T


def _events(span):
    return {e["name"]: e.get("attributes", {}).get("content", "") for e in span.get("events", [])}


def _trailing_tool_msgs(messages):
    """Tool-result messages at the tail (the inputs this LLM step consumes)."""
    out = []
    for m in reversed(messages):
        if m.get("role") == "tool":
            out.append(m.get("content", ""))
        elif m.get("role") == "assistant":
            break
    return list(reversed(out))


def extract_steps(spans):
    llm = sorted([s for s in spans if (s.get("attributes") or {}).get(T.ACG_NODE_TYPE) == T.NODE_TYPE_LLM],
                 key=lambda s: (s["attributes"].get(T.ACG_STEP, 0)))
    steps = []
    for s in llm:
        a = s["attributes"]
        ev = _events(s)
        try:
            messages = json.loads(ev.get("gen_ai.prompt", "[]"))
        except Exception:
            messages = []
        try:
            completion = json.loads(ev.get("gen_ai.completion", "{}"))
        except Exception:
            completion = {}
        step = a.get(T.ACG_STEP, 0)
        if step == 0:
            q = next((m["content"] for m in messages if m.get("role") == "user"), "")
            saw = [("question", q)]
        else:
            saw = [("tool result", c) for c in _trailing_tool_msgs(messages)]

        tcs = completion.get("tool_calls", [])
        thoughts = []
        decisions = []
        for tc in tcs:
            try:
                args = json.loads(tc.get("arguments", "{}"))
            except Exception:
                args = {"_raw": tc.get("arguments", "")}
            if isinstance(args, dict) and args.get("thought"):
                thoughts.append(args.pop("thought"))
            decisions.append((tc.get("name", "?"), args))
        reasoning = (completion.get("content") or "").strip()
        steps.append({
            "step": step,
            "in_tok": a.get(T.GEN_AI_USAGE_INPUT_TOKENS, 0),
            "out_tok": a.get(T.GEN_AI_USAGE_OUTPUT_TOKENS, 0),
            "saw": saw,
            "reasoning": reasoning,
            "thoughts": thoughts,
            "decisions": decisions,
            "final": None if tcs else reasoning,
        })
    return steps


_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0f1117;color:#e6e6e6;margin:0;padding:24px;}
h1{font-size:18px;margin:0 0 4px} .sub{color:#9aa4b2;font-size:13px;margin-bottom:20px}
.step{border:1px solid #2a2f3a;border-radius:10px;margin:0 0 14px;overflow:hidden}
.hd{background:#171a22;padding:8px 12px;font-weight:600;font-size:13px;display:flex;justify-content:space-between}
.hd .tok{color:#7d8797;font-weight:400}
.cols{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1px;background:#2a2f3a}
.cell{background:#12151c;padding:10px 12px}
.lbl{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#7d8797;margin-bottom:6px}
.saw .lbl{color:#5ea9ff} .reason .lbl{color:#c99bff} .decide .lbl{color:#ffb454}
.mono{font-family:ui-monospace,Menlo,monospace;font-size:12px;white-space:pre-wrap;word-break:break-word;line-height:1.45}
.pill{display:inline-block;background:#243044;color:#8cc2ff;border-radius:5px;padding:1px 7px;font-size:12px;margin-bottom:6px}
.empty{color:#5b6472;font-style:italic}
.final{background:#132018;border:1px solid #274b34;border-radius:10px;padding:12px;margin-top:6px}
.ok{color:#57d98a}.bad{color:#ff7a7a}
.args{color:#cfe3ff}
"""


def _esc(x):
    return html.escape(str(x))


def render_html(meta, steps) -> str:
    out = [f"<style>{_CSS}</style>",
           f"<h1>Reasoning trace — {_esc(meta['task_id'])} "
           f"<span class='{ 'ok' if meta['outcome']=='correct' else 'bad'}'>[{_esc(meta['outcome'])}]</span></h1>",
           f"<div class='sub'>{_esc(meta['question'])}</div>"]
    for st in steps:
        saw_html = "".join(
            f"<div class='pill'>{_esc(kind)}</div><div class='mono'>{_esc(txt[:800])}</div>"
            for kind, txt in st["saw"]) or "<div class='empty'>(nothing)</div>"

        reason_bits = []
        if st["reasoning"]:
            reason_bits.append(f"<div class='mono'>{_esc(st['reasoning'][:800])}</div>")
        for th in st["thoughts"]:
            reason_bits.append(f"<div class='mono'>“{_esc(th[:600])}”</div>")
        reason_html = "".join(reason_bits) or "<div class='empty'>(model emitted no reasoning text — went straight to the tool call)</div>"

        if st["decisions"]:
            dec_html = "".join(
                f"<div class='pill'>{_esc(name)}</div><div class='mono args'>{_esc(json.dumps(args, ensure_ascii=False))}</div>"
                for name, args in st["decisions"])
        else:
            dec_html = f"<div class='pill'>final answer</div><div class='mono'>{_esc((st['final'] or '')[:800])}</div>"

        out.append(
            f"<div class='step'><div class='hd'><span>LLM step {st['step']}</span>"
            f"<span class='tok'>{st['in_tok']}→{st['out_tok']} tok</span></div>"
            f"<div class='cols'>"
            f"<div class='cell saw'><div class='lbl'>Saw (inputs)</div>{saw_html}</div>"
            f"<div class='cell reason'><div class='lbl'>Reasoned (why)</div>{reason_html}</div>"
            f"<div class='cell decide'><div class='lbl'>Decided</div>{dec_html}</div>"
            f"</div></div>")
    out.append(
        f"<div class='final'><b>Final answer:</b> {_esc(meta['answer'])} &nbsp; "
        f"<span class='{ 'ok' if meta['outcome']=='correct' else 'bad'}'>({_esc(meta['outcome'])})</span></div>")
    return "\n".join(out)


def render_png(meta, steps, out_path, *, wrap_w: int = 104, fontsize: float = 8.5) -> str:
    """Render the same Saw→Reasoned→Decided timeline to a PNG (matplotlib, no browser needed).

    A single-column dark card stack: one card per LLM step with the step's inputs, the model's
    verbalized reasoning, and the tool call it emitted, then a final-answer banner. Geometry is in
    line-units (y) / char-units (x) with the figure sized so one text line == one unit.
    """
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    C = {"bg": "#0f1117", "card": "#12151c", "border": "#2a2f3a", "text": "#dfe6ee",
         "muted": "#7d8797", "saw": "#5ea9ff", "reason": "#c99bff", "decide": "#ffb454",
         "ok": "#57d98a", "bad": "#ff7a7a"}

    def wrap(s, maxlines):
        out = []
        for para in (str(s).replace("\t", " ").splitlines() or [""]):
            out += (textwrap.wrap(para, wrap_w - 6) or [""])
        return out[:maxlines] + (["   …(truncated)"] if len(out) > maxlines else [])

    blocks = []
    for st in steps:
        saw_lines = []
        for k, t in st["saw"]:
            saw_lines += wrap(f"{k}: {t}", 12)
        r = st["reasoning"] or ""
        for th in st["thoughts"]:
            r += (("\n" if r else "") + f'"{th}"')
        reason_lines = wrap(r or "(no reasoning text — went straight to the tool call)", 18)
        if st["decisions"]:
            dec_lines = []
            for name, args in st["decisions"]:
                dec_lines += [f"{name}"] + wrap("    " + json.dumps(args, ensure_ascii=False), 8)
        else:
            dec_lines = ["final answer:"] + wrap("    " + (st["final"] or ""), 10)
        blocks.append((st, [("SAW (inputs)", "saw", saw_lines or ["(nothing)"]),
                            ("REASONED (why)", "reason", reason_lines),
                            ("DECIDED", "decide", dec_lines)]))

    GAP_SEC, HDR, CARD_PAD, CARD_GAP, TITLE_H = 0.5, 1.6, 0.6, 0.8, 3.2

    def sec_h(lines):
        return 1.1 + len(lines)

    def block_h(secs):
        return HDR + sum(sec_h(l) + GAP_SEC for _, _, l in secs) + CARD_PAD

    oc = meta.get("outcome", "?")
    final_lines = textwrap.wrap(f"Final answer: {meta.get('answer', '')}   [{oc}]", wrap_w) or [""]
    FINAL_H = len(final_lines) + 1.5
    total = TITLE_H + sum(block_h(s) + CARD_GAP for _, s in blocks) + FINAL_H
    unit_in = fontsize * 1.5 / 72.0
    char_in = fontsize * 0.60 / 72.0

    fig = plt.figure(figsize=((wrap_w + 2) * char_in, total * unit_in), dpi=150)
    fig.patch.set_facecolor(C["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, wrap_w + 2); ax.set_ylim(0, total); ax.invert_yaxis(); ax.axis("off")

    def txt(x, y, s, c, weight="normal", size=fontsize, mono=True):
        ax.text(x, y, s, color=c, fontsize=size, fontweight=weight, va="top", ha="left",
                linespacing=1.5, family=("monospace" if mono else "DejaVu Sans"))

    head_c = C["ok"] if oc == "correct" else C["bad"]
    txt(0.8, 0.4, f"Reasoning trace — {meta.get('task_id', '')}   [{oc}]", head_c, "bold", fontsize + 4, mono=False)
    for i, ln in enumerate(textwrap.wrap(meta.get("question") or "", wrap_w)[:2]):
        txt(0.8, 1.9 + i, ln, C["muted"], mono=False)

    y = TITLE_H
    for st, secs in blocks:
        h = block_h(secs)
        ax.add_patch(Rectangle((0.5, y), wrap_w + 1.0, h - 0.2, facecolor=C["card"],
                               edgecolor=C["border"], linewidth=1.0))
        txt(1.0, y + 0.4, f"LLM step {st['step']}", C["text"], "bold")
        txt(wrap_w - 11, y + 0.4, f"{st['in_tok']}→{st['out_tok']} tok", C["muted"])
        yy = y + HDR
        for title, ck, lines in secs:
            txt(1.2, yy, title, C[ck], "bold", fontsize - 0.5)
            txt(2.2, yy + 1.1, "\n".join(lines), C["text"])
            yy += sec_h(lines) + GAP_SEC
        y += h + CARD_GAP

    for i, ln in enumerate(final_lines):
        txt(0.8, y + 0.2 + i, ln, head_c, "bold", fontsize + 1, mono=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=C["bg"])
    plt.close(fig)
    return str(out_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default="traces/experiment.jsonl")
    ap.add_argument("--task", default=None, help="pick the deepest run of this task")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--format", choices=["png", "html", "both"], default="both",
                    help="output format (default both)")
    args = ap.parse_args()

    by_trace = G.group_by_trace(G.load_spans(args.trace))
    # choose a run
    runs = G.reconstruct_runs(args.trace)
    chosen = None
    if args.run_id:
        chosen = next((r for r in runs if r.run_id == args.run_id), None)
    elif args.task:
        cand = [r for r in runs if r.task_id == args.task]
        chosen = max(cand, key=lambda r: r.metrics.node_count) if cand else None
    else:
        chosen = max(runs, key=lambda r: r.metrics.node_count)
    if chosen is None:
        print("no matching run"); return 1

    spans = by_trace[chosen.trace_id]
    meta = G._run_metadata(chosen.graph)
    steps = extract_steps(spans)

    base = Path(args.out) if args.out else Path(f"traces/figures/reasoning_{chosen.task_id}_{chosen.run_id}")
    base.parent.mkdir(parents=True, exist_ok=True)
    print(f"run {chosen.run_id} ({chosen.task_id}, {chosen.metrics.node_count} nodes, "
          f"{len(steps)} LLM steps, outcome={meta['outcome']})")
    if args.format in ("html", "both"):
        p = base.with_suffix(".html")
        p.write_text(render_html(meta, steps), encoding="utf-8")
        print(f"wrote {p}")
    if args.format in ("png", "both"):
        p = render_png(meta, steps, base.with_suffix(".png"))
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
