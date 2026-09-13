"""Pytest configuration for the 4-Agent System test suite.

Forces the LangGraph nodes to use the deterministic StubLLM so that graph-level
unit tests run quickly and do not depend on a live cloud LLM endpoint. Tests that
exercise MCP tools still use the real tool implementations.

The online research phase is switched off here for the same reason and with the
same force. `rpc_run_goal` now searches the web before the Architect opens, so
without this every test that starts a run would fetch a dozen live pages and
embed them: slow, non-deterministic, dependent on someone else's uptime, and
quietly mutating the developer's corpus as a side effect of running the suite.
It took the suite from 28s to 138s and broke a stop test whose five-second wait
had been generous. Tests that mean to exercise the phase turn it back on and
answer it with `MockTransport`.

The corpus bootstrap is switched off here for the same reason -- see
`_no_corpus_bootstrap`.
"""

from __future__ import annotations

import pytest

import langgraph_agent.nodes as _nodes
import langgraph_agent.web_research as _web_research
from langgraph_agent.config import StubLLM
from langgraph_agent.control import RUN_CONTROL

# Patch the LLM lookup used by agent nodes so every test gets deterministic,
# parser-friendly responses without making network calls. This has to be
# `get_agent_llm`: it is what the nodes import, and patching `get_llm` here
# only set an unused attribute on the module.
_nodes.get_agent_llm = lambda agent, temperature=0.1: StubLLM()


@pytest.fixture(autouse=True)
def _clear_run_control():
    """No test may leak a stop into the next one.

    The stop is a process-global flag by design -- the graph compiles without a
    checkpointer, so there is nowhere else for it to live -- which means a test
    that sets it and does not clear it would silently make every later test's
    nodes bail before calling their model.
    """
    RUN_CONTROL.disarm()
    yield
    RUN_CONTROL.disarm()


@pytest.fixture(autouse=True)
def _no_snapshot_clobber(monkeypatch, tmp_path):
    """A test run must not overwrite the developer's last-run snapshot.

    `runs/last_run.json` is the recovery story for a real run -- it is what a
    reloaded console shows and the only copy of a stopped run's partial work.
    Any test calling `rpc_run_goal` writes it, so running the suite replaced it
    with the outcome of a fixture goal like `"g"`, silently, and the run someone
    actually cared about was gone.
    """
    import serve
    monkeypatch.setattr(serve, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(serve, "LAST_RUN_PATH", tmp_path / "runs" / "last_run.json")


@pytest.fixture(autouse=True)
def _no_online_research(monkeypatch):
    """No test reaches the network, and none rewrites the corpus on disk.

    Overridden in `test_web_research.py`, which turns it back on and serves
    every request from a mock transport.
    """
    monkeypatch.setattr(_web_research, "WEB_SEARCH_ENABLED", False)


@pytest.fixture(autouse=True)
def _no_corpus_bootstrap(monkeypatch):
    """No test builds a corpus, and none rebuilds the developer's.

    `rpc_run_goal` indexes the project when there is nothing to search, which
    is what makes a fresh install work. Left on, every test that starts a run
    would index the whole checkout -- once per test, and in CI every time,
    since `knowledge/` is not committed and so every test there starts from
    `absent`. Switched off here for the same reason and with the same force as
    the online phase above; the tests that mean to exercise it turn it back on
    and hand it fakes.
    """
    import serve
    monkeypatch.setattr(serve, "INDEX_PROJECT_BEFORE_RUN", False)


@pytest.fixture(autouse=True)
def _no_planner_project_map(monkeypatch):
    """No planning test searches the developer's corpus.

    `_make_plan` shows the Planner the files the corpus ranks closest to the
    goal. Left on, every test that plans would search `knowledge/` -- the real
    one on a developer's machine and nothing at all in CI -- so a planning test
    would pass or fail by what the checkout it ran in had indexed. The tests
    that exercise the map turn it back on and answer the search themselves.
    """
    monkeypatch.setattr(_nodes, "PLANNER_PROJECT_MAP", False)


@pytest.fixture(autouse=True)
def _embedder_on_the_cpu(monkeypatch):
    """No test puts the embedding model on a card.

    `EMBEDDING_DEVICE` comes from the developer's environment, and on a machine
    that names a card every test that embeds would load the model there --
    taking memory a local seat may be using, and making results depend on
    which machine ran the suite. The tests that exercise a card set the
    setting themselves and answer with a fake model.
    """
    import langgraph_agent.graphrag_server as graphrag_server

    monkeypatch.setattr(graphrag_server, "EMBEDDING_DEVICE", "cpu")


@pytest.fixture(autouse=True)
def _the_default_embedding_model(monkeypatch):
    """Every test indexes and searches with MiniLM, whatever the developer chose.

    `EMBEDDING_MODEL` comes from the environment, and the console's choice is a
    process-global: either one leaking into a test would point its corpus at
    another directory and its searches at a daemon. The tests that exercise
    another model set it themselves and answer with fakes.
    """
    import langgraph_agent.graphrag_server as graphrag_server

    monkeypatch.setattr(graphrag_server, "EMBEDDING_MODEL", graphrag_server.EMBEDDING_MODEL_NAME)
    monkeypatch.setattr(graphrag_server, "_embedding_model_override", None)
