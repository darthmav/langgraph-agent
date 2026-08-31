"""Pytest configuration for the 3-Agent System test suite.

Forces the LangGraph nodes to use the deterministic StubLLM so that graph-level
unit tests run quickly and do not depend on a live Ollama/OpenAI/Anthropic
endpoint. Tests that exercise MCP tools still use the real tool implementations.
"""

from __future__ import annotations

import langgraph_agent.nodes as _nodes
from langgraph_agent.config import StubLLM

# Patch the LLM lookup used by agent nodes so every test gets deterministic,
# parser-friendly responses without making network calls.
_nodes.get_llm = lambda *args, **kwargs: StubLLM()
