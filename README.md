# 3-Agent AI System

**Planner · Researcher · Builder**

A multi-agent system for software development experiments, powered by LangGraph + GraphRAG + MCP.
Cloud-only by default: **Anthropic** is the primary LLM provider. **OpenAI** remains available as an optional cloud provider.

## 🎨 Web Console

```bash
# Quick launch
./launch_console.sh

# Or manually
python serve.py
# Open: http://localhost:8080
```

See [`frontend/README.md`](frontend/README.md) for full documentation.

## Architecture

```
Human (optional gates) → Planner → Researcher → Builder → END
                              ^                    |
                              +------ loop --------+
```

### The Three Agents

| Agent        | Responsibility                                      | Tools                          |
|--------------|-----------------------------------------------------|--------------------------------|
| **Planner**  | Turns goals into structured plans, routes next      | None (reasoning only)          |
| **Researcher** | Gathers deep, relationship-aware knowledge        | GraphRAG MCP only              |
| **Builder**  | Implements the plan (writes code, edits files)      | Filesystem, Git, Terminal      |

### The Three Technologies

| Technology   | Role                                                |
|--------------|-----------------------------------------------------|
| **LangGraph** | Orchestration: control flow, shared state, loops   |
| **GraphRAG**  | Intelligent retrieval using knowledge graph        |
| **MCP**       | Universal connector for tools and data sources     |

## Installation

```bash
pip install -e ".[dev]"
```

### Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

The default uses Anthropic:

```bash
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

### Optional OpenAI provider

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

## Usage

### Basic Example

```python
from langgraph_agent import AgentState, create_agent_graph

graph = create_agent_graph()

state: AgentState = {
    "goal": "Create a hello.txt file containing 'Hello World'",
    "messages": [],
    "plan": "",
    "research": "",
    "builder_report": "",
    "next_agent": "Researcher",
    "research_status": "",
    "blockers": "",
    "files_changed": [],
    "step_count": 0,
}

result = graph.invoke(state)
print(result["plan"])
print(result["builder_report"])
```

### With Anthropic

```python
import os

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
os.environ["ANTHROPIC_MODEL"] = "claude-3-5-sonnet-20241022"

result = graph.invoke(state)
```

### With OpenAI

```python
import os

os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["OPENAI_MODEL"] = "gpt-4o-mini"

result = graph.invoke(state)
```

## Running Tests

```bash
# With stub LLM (fast, no API key needed)
python -m pytest tests/ -v

# With test coverage
pytest --cov=langgraph_agent tests/
```

## Development Tools

```bash
# Linting
ruff check src/ tests/

# Auto-fix linting issues
ruff check --fix src/ tests/

# Formatting
ruff format src/ tests/

# Type checking
mypy src/langgraph_agent/

# Test coverage
pytest --cov=langgraph_agent tests/
```

## Example

```bash
# Run with stub LLM (no API key needed)
python example_usage.py

# Run with Anthropic (ensure ANTHROPIC_API_KEY is set)
python example_usage.py
```

## Shared State Schema

Per the 3-Agent System specification:

| Field              | Written By  | Description                                    |
|--------------------|-------------|------------------------------------------------|
| `goal`             | User        | The user's original goal                       |
| `messages`         | All         | Conversation history / log                     |
| `plan`             | Planner     | Structured plan with steps                     |
| `research`         | Researcher  | Findings from GraphRAG queries                 |
| `builder_report`   | Builder     | Implementation report                          |
| `next_agent`       | Planner     | Which agent runs next                          |
| `research_status`  | Researcher  | `ready_for_builder` \| `need_replan` \| `no_relevant_knowledge` |
| `blockers`         | Builder     | What's blocking progress                       |
| `files_changed`    | Builder     | List of modified file paths                    |
| `step_count`       | Builder     | Number of steps (for loop limit, max 8)        |

## MCP Integration

The `mcp_client.py` module exposes the documented agent tool belts through a
unified MCP-style interface:

| Tool | Agent | Purpose |
|------|-------|---------|
| `search_knowledge_graph` | Researcher | Search the knowledge graph + vector store |
| `query_knowledge_graph` | Researcher | Query entity/relationship neighborhoods |
| `filesystem_read` | Builder | Read a file |
| `filesystem_write` | Builder | Write a file |
| `git_status` | Builder | `git status --porcelain` |
| `git_diff` | Builder | `git diff` |
| `terminal_execute` | Builder | Run a safe shell command |
| `run_tests` | Builder | Run `pytest` |

The Researcher and Builder nodes call these tools through `MCPClient`, preserving
the documented specialization:

- Planner → no tools
- Researcher → GraphRAG read-only tools only
- Builder → filesystem / git / terminal / test tools only

To switch from the bundled tool implementations to external MCP servers,
update `MCPClient._discover_tools()` to connect over stdio or HTTP and route each
tool name to the external server.

## Project Structure

```
├── pyproject.toml              # Dependencies
├── src/langgraph_agent/
│   ├── __init__.py
│   ├── state.py                # AgentState, ResearchStatus
│   ├── config.py               # LLM setup (Anthropic primary, OpenAI optional)
│   ├── nodes.py                # Planner, Researcher, Builder
│   ├── graph.py                # StateGraph wiring
│   ├── graphrag_server.py      # GraphRAG MCP server
│   └── mcp_client.py           # MCP client / tool bindings
├── prompts/
│   ├── planner.txt             # Planner system prompt
│   ├── researcher.txt          # Researcher system prompt
│   └── builder.txt             # Builder system prompt
├── tests/
│   └── test_graph.py           # Graph tests
├── example_usage.py            # Demo script
├── test_cloud.py               # Cloud LLM end-to-end test
├── README.md
├── .env.example                # Environment variables template
├── ruff.toml                   # Linter config
└── mypy.ini                    # Type checker config
```

## Hardware Requirements

**Cloud providers:** no local GPU required. The embedding model runs locally for GraphRAG, but no conversational LLM is hosted on-device.

## Next Steps

1. **Configure cloud LLM** — Set `ANTHROPIC_API_KEY` in `.env`
2. **MCP Servers** — Connect to actual GraphRAG and filesystem MCP servers
3. **Human-in-the-loop** — Add approval gates before Builder executes
4. **Persistence** — Add checkpointing for long-running agents

## Documentation

Based on the [3-Agent System Full Guide](../Downloads/3-Agent-System-Full-Consolidated-Guide.md).
