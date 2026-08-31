"""LangGraph Agent - 3-Agent System implementation.

Planner, Researcher, Builder architecture with:
- Strict system prompts
- State injection on every turn
- Tool binding per agent
- Ollama support for local LLMs
"""

from langgraph_agent.graph import create_agent_graph
from langgraph_agent.state import AgentState, ResearchStatus

__version__ = "0.2.0"
__all__ = ["create_agent_graph", "AgentState", "ResearchStatus"]
