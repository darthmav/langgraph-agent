"""Pytest configuration for the 4-Agent System test suite.

Forces the LangGraph nodes to use the deterministic StubLLM so that graph-level
unit tests run quickly and do not depend on a live cloud LLM endpoint. Tests that
exercise MCP tools still use the real tool implementations.
"""

from __future__ import annotations

import pytest

import langgraph_agent.nodes as _nodes
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
