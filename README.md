# 3-Agent Local AI System

**Planner · Researcher · Builder**

A fully local, zero-fee multi-agent system for software development experiments. Built with LangGraph + GraphRAG + MCP.

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

### Local LLM Setup (Recommended)

For fully local execution with zero service fees:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull recommended model (fits 32GB RAM, 3GB GPU)
ollama pull qwen3:8b
# Alternative: ollama pull qwen2.5:7b
```

### Environment Configuration

Copy `.env.example` to `.env` and configure:

```bash
# For local execution (no API keys needed)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b

# For cloud APIs (optional)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
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

### With Local LLM

```python
import os

os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["OLLAMA_MODEL"] = "qwen3:8b"

# Now all LLM calls use Ollama
result = graph.invoke(state)
```

## Running Tests

```bash
# With stub LLM (fast, no setup needed)
OPENAI_API_KEY="" python -m pytest tests/ -v

# With local Ollama
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
# Run with stub LLM
OPENAI_API_KEY="" python example_usage.py

# Run with Ollama (ensure ollama is running)
python example_usage.py

# Run with OpenAI
export OPENAI_API_KEY=sk-...
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

## MCP Integration (Scaffolding)

The `mcp_client.py` module provides scaffolding for:

- **GraphRAG MCP** — Knowledge base queries and summarization
- **Filesystem MCP** — Reading/writing files
- **Git MCP** — Version control operations

To enable real MCP integration:
1. Set up MCP servers (stdio or HTTP)
2. Update `MCPClient._discover_tools()` to connect
3. Nodes call MCP when available, fall back to LLM

## Project Structure

```
├── pyproject.toml              # Dependencies
├── src/langgraph_agent/
│   ├── __init__.py
│   ├── state.py                # AgentState, ResearchStatus
│   ├── config.py               # LLM setup (Ollama, OpenAI, Anthropic)
│   ├── nodes.py                # Planner, Researcher, Builder
│   ├── graph.py                # StateGraph wiring
│   └── mcp_client.py           # MCP client scaffolding
├── tests/
│   └── test_graph.py           # 8 passing tests
├── example_usage.py            # Demo script
├── README.md
├── .env.example                # Environment variables template
├── ruff.toml                   # Linter config
└── mypy.ini                    # Type checker config
```

## Hardware Requirements

**Recommended:** 32 GB RAM, 3 GB GPU (your machine)

- **Qwen3 8B** (Q4/Q5 quant) — fits comfortably, primary recommendation
- **Qwen2.5 7B/8B** — alternative, similar performance
- Avoid 32B+ models (require 75GB+ even at 1-bit quant)

## Next Steps

1. **Run local LLM** — Set up Ollama with `qwen3:8b`
2. **MCP Servers** — Connect to actual GraphRAG and filesystem MCP servers
3. **Human-in-the-loop** — Add approval gates before Builder executes
4. **Persistence** — Add checkpointing for long-running agents

## Documentation

Based on the [3-Agent System Full Guide](../Downloads/3-Agent-System-Full-Consolidated-Guide.md).
