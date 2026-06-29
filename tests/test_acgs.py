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
