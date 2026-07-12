"""Tests that produce Agentic Computation Graphs for multiple QA programs.

Two kinds of tests:
  * unit tests (no server) — corpus retrieval, answer checking, tool alphabet.
  * live tests (need the local vLLM/SGLang server) — run real multi-hop QA tasks,
    reconstruct each ACG from the trace, and assert it is a well-formed graph over
    the fixed node alphabet. These are skipped automatically if the server is down.

Run:  ./.venv/bin/python -m pytest tests/ -v -s
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from acg import tracing as T
from acg import graph as G
from acg.config import load_config
from acg.corpus import Corpus
from acg.tools import TOOL_NAMES, tool_schemas
from acg.tasks import load_tasks, check_answer
from acg.agent import Agent

# Tasks exercised by the live multi-program test (mix of 2/3/4-hop).
LIVE_TASK_IDS = ["T01", "T02", "T04", "T08", "T06"]


def _server_up(cfg) -> bool:
    try:
        from openai import OpenAI
        OpenAI(base_url=cfg.base_url, api_key=cfg.api_key).models.list()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def corpus(cfg):
    return Corpus.load(cfg.corpus_path)


@pytest.fixture(scope="session")
def live(cfg):
    if not _server_up(cfg):
        pytest.skip(f"local model server not reachable at {cfg.base_url}")
    return True


# ----------------------------- unit tests --------------------------------- #
def test_tool_alphabet_is_fixed():
    schemas = tool_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert names == list(TOOL_NAMES) == ["search", "read_document", "finish"]


def test_corpus_search_and_read(corpus):
    hits = corpus.search("Pirelle Institute", top_k=3)
    assert hits and hits[0]["doc_id"] == "D01"
    assert "drell" in corpus.read("D03")["text"].lower()
    assert "error" in corpus.read("NOPE")


def test_answer_checker():
    assert check_answer("The currency is the drell.", ["drell", "the drell"])
    assert check_answer("Karst Reach", ["Karst Reach"])
    assert not check_answer("Velmora", ["Brandeth"])
    assert not check_answer(None, ["x"])


def test_tasks_have_gold(cfg):
    tasks = load_tasks(cfg.tasks_path)
    assert len(tasks) >= 10
    assert all(t.answers for t in tasks)


# ------------- unit tests: emitted vs executed width (no server) ---------- #
def _mk_span(node_id, node_type, deps, start_ns, end_ns, **attrs):
    """A minimal JSONL-style span dict like acg.tracing exports."""
    a = {T.ACG_NODE_ID: node_id, T.ACG_NODE_TYPE: node_type, T.ACG_DEPENDS_ON: deps}
    a.update(attrs)
    dur = (end_ns - start_ns) if (start_ns is not None and end_ns is not None) else None
    return {
        "trace_id": "trace-a", "span_id": node_id, "name": node_id,
        "start_time_ns": start_ns, "end_time_ns": end_ns, "duration_ns": dur,
        "attributes": a,
    }


def _two_tool_run_spans(t0_start, t0_end, t1_start, t1_end):
    """One run: root -> llm:0 -> {tool:0:0, tool:0:1}, both emitted in the same turn."""
    return [
        _mk_span("run:r", T.NODE_TYPE_AGENT_RUN, [], 0, 1000,
                 **{T.ACG_OUTCOME: "correct", T.ACG_RUN_ID: "r", T.ACG_TASK_ID: "TX"}),
        _mk_span("llm:0", T.NODE_TYPE_LLM, ["run:r"], 10, 20,
                 **{T.GEN_AI_USAGE_INPUT_TOKENS: 100, T.GEN_AI_USAGE_OUTPUT_TOKENS: 20}),
        _mk_span("tool:0:0", T.NODE_TYPE_TOOL, ["llm:0"], t0_start, t0_end,
                 **{T.GEN_AI_TOOL_NAME: "search", T.ACG_STEP: 0}),
        _mk_span("tool:0:1", T.NODE_TYPE_TOOL, ["llm:0"], t1_start, t1_end,
                 **{T.GEN_AI_TOOL_NAME: "search", T.ACG_STEP: 0}),
    ]


def test_max_temporal_overlap():
    f = G._max_temporal_overlap
    assert f([]) == 0
    assert f([(0, 10)]) == 1
    assert f([(0, 10), (10, 20)]) == 1           # touching endpoints are not overlap
    assert f([(0, 10), (5, 15), (6, 7)]) == 3    # triple overlap at t=6..7
    assert f([(0, 10), (None, 5), (2, 8)]) == 2  # invalid interval skipped


def test_width_distinguishes_emitted_from_executed():
    # Identical STRUCTURE (two tool calls at one dependency level => width 2), but
    # different wall-clock: overlapping => executed 2; back-to-back => executed 1.
    m_par = G.compute_metrics(G.build_graph(_two_tool_run_spans(30, 50, 35, 55)))
    assert m_par.width == 2            # emitted / structural
    assert m_par.width_executed == 2   # realized parallelism

    m_seq = G.compute_metrics(G.build_graph(_two_tool_run_spans(30, 50, 50, 70)))
    assert m_seq.width == 2            # same shape...
    assert m_seq.width_executed == 1   # ...but nothing actually ran concurrently
    assert "width_executed" in m_par.to_row()   # surfaced for the metrics CSV


def _sub_agent_run_spans():
    """root -> llm:0 -> {sub_agent A, sub_agent B} (overlapping) -> llm:1 -> finish.
    Each sub_agent tool span envelopes a nested llm + search subtree (namespaced ids)."""
    S = [
        _mk_span("run:r", T.NODE_TYPE_AGENT_RUN, [], 0, 1000,
                 **{T.ACG_OUTCOME: "correct", T.ACG_RUN_ID: "r", T.ACG_TASK_ID: "TX"}),
        _mk_span("llm:0", T.NODE_TYPE_LLM, ["run:r"], 10, 20,
                 **{T.GEN_AI_USAGE_INPUT_TOKENS: 50, T.GEN_AI_USAGE_OUTPUT_TOKENS: 10}),
    ]
    for k, (a, b) in enumerate([(30, 200), (32, 205)]):   # two sub_agents, overlapping
        tid = f"tool:0:{k}"
        S += [
            _mk_span(tid, T.NODE_TYPE_TOOL, ["llm:0"], a, b,
                     **{T.GEN_AI_TOOL_NAME: "sub_agent", T.ACG_STEP: 0}),
            _mk_span(f"{tid}/llm:0", T.NODE_TYPE_LLM, [tid], a + 2, a + 8,
                     **{T.GEN_AI_USAGE_INPUT_TOKENS: 20, T.GEN_AI_USAGE_OUTPUT_TOKENS: 5}),
            _mk_span(f"{tid}/tool:0:0", T.NODE_TYPE_TOOL, [f"{tid}/llm:0"], a + 9, a + 15,
                     **{T.GEN_AI_TOOL_NAME: "search", T.ACG_STEP: 0}),
        ]
    S += [
        _mk_span("llm:1", T.NODE_TYPE_LLM, ["tool:0:0", "tool:0:1"], 220, 230,
                 **{T.GEN_AI_USAGE_INPUT_TOKENS: 80, T.GEN_AI_USAGE_OUTPUT_TOKENS: 12}),
        _mk_span("tool:1:0", T.NODE_TYPE_TOOL, ["llm:1"], 231, 235,
                 **{T.GEN_AI_TOOL_NAME: "finish", T.ACG_STEP: 1}),
    ]
    return S


def test_sub_agent_produces_tree_and_executed_width():
    g = G.build_graph(_sub_agent_run_spans())
    m = G.compute_metrics(g)
    assert nx.is_directed_acyclic_graph(g)
    # nested subtree nodes exist, namespaced under their sub_agent tool node
    assert "tool:0:0/llm:0" in g and "tool:0:1/tool:0:0" in g
    assert m.width >= 2            # two sub_agent calls emitted at one level
    assert m.width_executed == 2   # the two sub_agent containers overlap; nested tools excluded
    assert m.depth >= 4            # a real tree, deeper than a flat chain
    assert m.num_tool_calls == 5   # 2 sub_agents + 2 nested searches + 1 finish


# --------------------------- live ACG tests ------------------------------- #
def _assert_valid_acg(g: nx.DiGraph):
    assert g.number_of_nodes() > 0
    assert nx.is_directed_acyclic_graph(g), "ACG must be a DAG"
    # exactly one synthetic root (agent_run)
    roots = [n for n, d in g.nodes(data=True) if d["type"] == T.NODE_TYPE_AGENT_RUN]
    assert len(roots) == 1
    # every non-root node is one of the fixed types; tools are from the fixed alphabet
    for n, d in g.nodes(data=True):
        assert d["type"] in (T.NODE_TYPE_AGENT_RUN, T.NODE_TYPE_LLM, T.NODE_TYPE_TOOL)
        if d["type"] == T.NODE_TYPE_TOOL:
            assert d["tool_name"] in TOOL_NAMES
    # the graph is connected from the root (no orphan nodes)
    assert len(nx.descendants(g, roots[0])) == g.number_of_nodes() - 1
    # at least one LLM call drives the graph
    assert any(d["type"] == T.NODE_TYPE_LLM for _, d in g.nodes(data=True))


def test_single_task_produces_acg(cfg, corpus, live, tmp_path):
    T.configure_tracing(tmp_path / "single.jsonl")
    agent = Agent(cfg, corpus)
    task = {t.task_id: t for t in load_tasks(cfg.tasks_path)}["T01"]
    res = agent.run(task)
    T.flush_tracing()

    runs = G.reconstruct_runs(tmp_path / "single.jsonl")
    assert len(runs) == 1
    run = runs[0]
    _assert_valid_acg(run.graph)
    m = run.metrics
    assert m.num_llm_calls >= 1
    assert m.total_tokens > 0
    assert m.node_count == m.num_llm_calls + m.num_tool_calls
    # trace's run_id matches the live result
    assert run.run_id == res.run_id


def test_multiple_qa_programs_produce_acgs(cfg, corpus, live, tmp_path):
    """The headline test: run several QA programs and get a valid AGC for each."""
    trace_file = tmp_path / "multi.jsonl"
    T.configure_tracing(trace_file)
    agent = Agent(cfg, corpus)
    tasks = {t.task_id: t for t in load_tasks(cfg.tasks_path)}

    expected_run_ids = {}
    for tid in LIVE_TASK_IDS:
        res = agent.run(tasks[tid])
        expected_run_ids[res.run_id] = tid
    T.flush_tracing()

    runs = G.reconstruct_runs(trace_file)
    # one ACG per QA program
    assert len(runs) == len(LIVE_TASK_IDS)
    seen_tasks = set()
    for run in runs:
        _assert_valid_acg(run.graph)
        assert run.run_id in expected_run_ids
        assert run.task_id == expected_run_ids[run.run_id]
        seen_tasks.add(run.task_id)
        m = run.metrics
        # multi-hop QA must use tools (search/read), not answer in one shot
        assert m.num_tool_calls >= 1
        assert m.depth >= 2
        assert m.outcome in ("correct", "incorrect", "no_answer")
    assert seen_tasks == set(LIVE_TASK_IDS)

    # the instrument should solve a clear majority of these graded tasks
    correct = sum(1 for r in runs if r.metrics.outcome == "correct")
    assert correct >= len(LIVE_TASK_IDS) // 2, f"only {correct}/{len(LIVE_TASK_IDS)} correct"


def test_variance_machinery_over_repeats(cfg, corpus, live, tmp_path):
    """Run the SAME task several times at temperature>0 and confirm the structural-
    variance machinery produces sane numbers (the Month-2 contribution in miniature)."""
    import scripts.analyze as analyze
    cfg.decode.temperature = 0.7
    trace_file = tmp_path / "var.jsonl"
    T.configure_tracing(trace_file)
    agent = Agent(cfg, corpus)
    task = {t.task_id: t for t in load_tasks(cfg.tasks_path)}["T02"]

    import uuid
    for _ in range(4):
        agent.run(task, run_id=uuid.uuid4().hex[:12])
    T.flush_tracing()

    runs = G.reconstruct_runs(trace_file)
    assert len(runs) == 4
    sv = analyze.structural_variance(runs)
    assert sv["distinct_signatures"] >= 1
    assert 0.0 < sv["modal_signature_fraction"] <= 1.0
    assert "num_llm_calls" in sv["modal_signature"]
