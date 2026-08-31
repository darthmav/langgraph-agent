# Claude Code Project Instructions

## Project Overview

This is **langgraph-agent**, a fully local 3-Agent AI system for software development experiments:

- **Planner** — interprets goals, creates structured plans, routes to next agent.
- **Researcher** — gathers context via the GraphRAG MCP tool (`search_knowledge_graph`).
- **Builder** — implements plans using filesystem, git, terminal, and test MCP tools.

Tech stack: Python 3.10+, LangGraph, Chroma + sentence-transformers + NetworkX, MCP (local stdio-compatible tool binding).

## Quick Reference

```bash
# Install dependencies
pip install -e ".[dev]"

# Run all checks
ruff check src/ tests/
mypy src/langgraph_agent/
OPENAI_API_KEY="" python -m pytest tests/ -v

# Start the web console
./launch_console.sh

# Re-index project files into GraphRAG
python scripts/reindex.py

# Run the example
OPENAI_API_KEY="" python example_usage.py
```

## Project Structure

```
├── src/langgraph_agent/
│   ├── __init__.py            # create_agent_graph, AgentState, ResearchStatus
│   ├── state.py               # AgentState schema + ResearchStatus enum
│   ├── config.py              # LLM setup (Ollama/OpenAI/Anthropic) + StubLLM
│   ├── nodes.py               # Planner, Researcher, Builder nodes + prompt loading
│   ├── graph.py               # StateGraph wiring + conditional edges
│   ├── graphrag_server.py     # GraphRAG MCP server (knowledge graph + vector store)
│   └── mcp_client.py          # MCP client / local tool bindings
├── prompts/
│   ├── planner.txt            # System prompt (loaded by nodes.py)
│   ├── researcher.txt
│   └── builder.txt
├── tests/
│   └── test_graph.py          # Pytest suite
├── scripts/
│   ├── reindex.py             # Re-index files into GraphRAG
│   ├── index_knowledge.py     # First-time indexing
│   ├── verify_and_test.py     # Manual verification runner
│   ├── auto_verify.py         # Silent verification
│   ├── quick_test.sh          # Bash quick check
│   └── full_setup.py          # Automated setup + re-index
├── frontend/
│   ├── index.html             # Web console SPA
│   └── README.md
├── serve.py                   # Python HTTP server + API backend
├── example_usage.py           # Demo script
├── test_ollama.py             # Local Ollama end-to-end test
├── README.md                  # User-facing documentation
└── .env.example               # Environment variables template
```

## Conventions

- **Python formatting/linting:** `ruff` configured in `ruff.toml`.
- **Type checking:** `mypy` configured in `mypy.ini`.
- **Tests:** `pytest` in `tests/`.
- **Default LLM:** `qwen3:8b` via Ollama (local, zero fees). Cloud providers require API keys.
- **Tool specialization (critical):**
  - Planner → no tools.
  - Researcher → GraphRAG read-only tools only.
  - Builder → filesystem, git, terminal, test tools only.

## State Schema

Every node reads/writes `AgentState`:

```python
{
    "goal": str,
    "messages": list[str],
    "plan": str,
    "research": str,
    "builder_report": str,
    "next_agent": "Researcher" | "Builder" | "END",
    "research_status": "ready_for_builder" | "need_replan" | "no_relevant_knowledge",
    "blockers": str,
    "files_changed": list[str],
    "step_count": int,
}
```

## Common Tasks

### Run the test suite
```bash
OPENAI_API_KEY="" python -m pytest tests/ -v
```

### Add a new Builder tool
1. Add the tool method to `src/langgraph_agent/mcp_client.py` in `_discover_tools()`.
2. Name it clearly under the `filesystem_`, `git_`, `terminal_`, or `test_` namespace.
3. Update `src/langgraph_agent/nodes.py` if the Builder should invoke it automatically.
4. Add a test in `tests/test_graph.py` or create a focused unit test.
5. Update `README.md` MCP Integration section.

### Add a new Researcher tool
1. Prefer GraphRAG read-only tools only.
2. Extend `src/langgraph_agent/graphrag_server.py` and expose via `mcp_client.py`.

### Change the default LLM
- Update `src/langgraph_agent/config.py` default model and `.env.example`.
- Update `frontend/index.html` and `serve.py` if the default is shown in the UI.

## Important Notes

- Do not let every agent call every tool; the specialization is the whole point.
- Empty state fields must render as `(empty)` in the state injection block.
- The Builder must actually call a tool to claim it changed a file.
- Knowledge base files under `knowledge/chroma/` are runtime artifacts; avoid committing them unless intentionally versioning an index.
- `__pycache__` directories should never be committed (they are removed from git in this repo).

## Troubleshooting

- **Tests are slow** — The first run loads `sentence-transformers` and Chroma. Subsequent runs use the cached singleton.
- **Mypy errors from upstream stubs** — Prefer `# type: ignore[...]` with a comment over disabling strict mode.
- **GraphRAG returns no results** — Run `python scripts/reindex.py` to rebuild the knowledge base.
