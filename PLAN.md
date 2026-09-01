# Plan: 3-Agent System Implementation

## Goal

Maintain `langgraph-agent` as a cloud-only 3-Agent AI system for software development experiments. The only LLM backends are cloud providers: Anthropic is the primary default; OpenAI remains optional. Ensure the implementation follows the documented architecture: Planner, Researcher, Builder, shared `AgentState`, GraphRAG read-only tools for Researcher, filesystem/git/terminal/test tools for Builder, and LangGraph as the only router.

## Current Status

- The three agents, shared `AgentState`, and strict system prompts are implemented.
- All tests pass with the deterministic `StubLLM`.
- Anthropic is the default LLM backend.
- OpenAI remains an optional cloud provider.

## Defaults

| Component | Default | Notes |
|-----------|---------|-------|
| Planner   | Anthropic `claude-3-5-sonnet-20241022` | Override with `PLANNER_PROVIDER` / `PLANNER_MODEL` |
| Researcher| Anthropic `claude-3-5-haiku-20241022`   | Override with `RESEARCHER_PROVIDER` / `RESEARCHER_MODEL` |
| Builder   | Anthropic `claude-3-5-haiku-20241022`   | Override with `BUILDER_PROVIDER` / `BUILDER_MODEL` |
| OpenAI fallback | `gpt-4o-mini` | Set `OPENAI_API_KEY` and `OPENAI_MODEL` to use |
| Embeddings| `all-MiniLM-L6-v2` | Runs on-device for GraphRAG; no API key required |

## Implementation Notes

1. **Planner routing** — `graph.py` respects `state["next_agent"]` set by the Planner node on every turn.
2. **Builder blockers** — `builder_node` parses `## Next Steps / Blockers`, sets `state["blockers"]` when execution fails or the LLM reports a blocker, and the graph loops back to Planner/Researcher accordingly.
3. **Researcher tool boundary** — `researcher_node` calls GraphRAG through the MCP-style client (`search_knowledge_graph`, `query_knowledge_graph`), not by importing the knowledge base directly.
4. **Builder tool boundary** — `builder_node` uses `filesystem_write` (and other builder tools) through the MCP-style client.
5. **State injection** — empty fields render as `(empty)` on every turn.
6. **GraphRAG read-only** — only `search_knowledge_graph` and `query_knowledge_graph` are exposed; indexing is done via `scripts/reindex.py` / `scripts/index_knowledge.py`.
7. **Step limit** — `MAX_STEPS = 8` prevents infinite loops.

## Running the System

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy environment template and edit
cp .env.example .env
# Default .env uses Anthropic; uncomment OpenAI vars to switch.

# Run tests (use StubLLM, no API key needed)
python -m pytest tests/ -v

# Run example
python example_usage.py

# Start web console
./launch_console.sh
```

## Optional OpenAI Backend

```bash
# In .env, uncomment and set:
# OPENAI_API_KEY=sk-...
# OPENAI_MODEL=gpt-4o-mini
```

## Next Steps

1. Add human-in-the-loop approval gates before Builder executes.
2. Add checkpointing/persistence for long-running agents.
3. Connect to external MCP servers over stdio/HTTP when needed.
