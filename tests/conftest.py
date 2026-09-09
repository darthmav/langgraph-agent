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
