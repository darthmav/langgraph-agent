# LangGraph Agent

A LangGraph-based orchestration system with **Planner**, **Researcher**, and **Builder** nodes.

## Architecture

```
Human (optional gates) → Planner → Researcher → Builder → END
                              ^                    |
                              +------ loop --------+
```

**Nodes:**
- **Planner** — Interprets input using LLM, creates actionable plans, routes to next node
- **Researcher** — Gathers information (LLM simulation, MCP GraphRAG scaffolding ready)
- **Builder** — Generates code (LLM simulation, MCP filesystem scaffolding ready)

**Feedback Loop:** Builder → Planner when feedback contains keywords like "fix", "change", "wrong", etc.

## Installation

```bash
pip install -e .
```

Requires `OPENAI_API_KEY` environment variable for real LLM calls. Without it, uses stub responses for testing.

## Usage

```python
from langgraph_agent import create_agent_graph, AgentState

graph = create_agent_graph(max_iterations=3)

state: AgentState = {
    "input": "Create a REST API with FastAPI",
    "plan": [],
    "current_step": 0,
    "research_findings": None,
    "builder_output": None,
    "messages": [],
    "status": "started",
    "next_node": "",
    "feedback": None,
    "iteration": 0,
    "max_iterations": 3,
}

result = graph.invoke(state)
print(result["plan"])
print(result["builder_output"])
```

### With Feedback (Replanning)

```python
state["feedback"] = "Use PostgreSQL instead of SQLite"
result = graph.invoke(state)  # Will loop back to Planner
```

## Running Tests

```bash
# With stub LLM (fast, no API key needed)
OPENAI_API_KEY="" python -m pytest tests/ -v

# With real OpenAI API
python -m pytest tests/ -v
```

## Example

```bash
# Run with stub LLM
OPENAI_API_KEY="" python example_usage.py

# Run with real API
python example_usage.py
```

## MCP Integration (Scaffolding)

The `mcp_client.py` module provides scaffolding for connecting to MCP servers:

- **GraphRAG MCP** — For knowledge base queries and summarization
- **Filesystem MCP** — For reading/writing files
- **Git MCP** — For version control operations

To enable real MCP integration:
1. Set up MCP servers (e.g., `MCP_GRAPHRAG_URL`, `MCP_FILESYSTEM_URL`)
2. Update `MCPClient._discover_tools()` to connect to actual servers
3. The node implementations already call MCP when available, falling back to LLM

## Project Structure

```
├── pyproject.toml              # Dependencies
├── src/langgraph_agent/
│   ├── __init__.py
│   ├── state.py                # AgentState TypedDict
│   ├── config.py               # LLM setup (OpenAI + stub)
│   ├── nodes.py                # Planner, Researcher, Builder
│   ├── graph.py                # StateGraph wiring + feedback loop
│   └── mcp_client.py           # MCP client scaffolding
├── tests/
│   └── test_graph.py           # 5 passing tests
├── example_usage.py            # Demo script
└── README.md
```

## Next Steps

1. **Real LLM**: Set `OPENAI_API_KEY` environment variable
2. **MCP Servers**: Connect to actual GraphRAG and filesystem MCP servers
3. **Human-in-the-loop**: Add approval gates before Builder executes
4. **Persistence**: Add checkpointing for long-running agents
