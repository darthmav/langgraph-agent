"""LangGraph Agent - 4-Agent System implementation.

Architect, Planner, Researcher, Builder architecture with:
- Strict system prompts
- State injection on every turn
- Tool binding per agent
- Cloud-only inference (Ollama Cloud for every seat; Anthropic/OpenAI optional)
"""

from langgraph_agent.graph import create_agent_graph
from langgraph_agent.state import AgentState, ResearchStatus, Verdict

__version__ = "0.2.0"
__all__ = ["create_agent_graph", "AgentState", "ResearchStatus", "Verdict"]
