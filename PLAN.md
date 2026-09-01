# Plan: 4-Agent System Implementation

## Goal

Maintain `langgraph-agent` as a cloud-only 4-Agent AI system for software development experiments. Inference is cloud-only: Anthropic is the default for the Architect seat, the remaining seats run Ollama `:cloud` tags proxied to ollama.com by the local daemon, and OpenAI remains optional. The only model that runs on this machine is the `all-MiniLM-L6-v2` embedding model, which belongs to GraphRAG rather than to a seat.

The implementation follows the documented architecture: Architect, Planner, Researcher, Builder, a shared `AgentState`, GraphRAG read-only tools for the Researcher, filesystem/git/terminal/test tools for the Builder, and LangGraph as the only router.

## Current Status

- The four agents, shared `AgentState`, and strict system prompts are implemented.
- The Architect holds the approval gate: the run ends on its `approved` verdict, not on the Builder's say-so.
- All tests pass with the deterministic `StubLLM`.
- Seats without credentials silently fall back to `StubLLM`; `get_agent_status()` is what reports the difference to the console.

## Defaults

| Component  | Default | Notes |
|------------|---------|-------|
| Architect  | Anthropic `claude-opus-5` | Override with `ARCHITECT_PROVIDER` / `ARCHITECT_MODEL` |
| Planner    | Ollama `qwen3.5:397b-cloud` | Override with `PLANNER_PROVIDER` / `PLANNER_MODEL` |
| Researcher | Ollama `gemma4:cloud` | Override with `RESEARCHER_PROVIDER` / `RESEARCHER_MODEL` |
| Builder    | Ollama `kimi-k3:cloud` | Override with `BUILDER_PROVIDER` / `BUILDER_MODEL` |
| OpenAI fallback | `gpt-4o-mini` | Set `OPENAI_API_KEY` and `OPENAI_MODEL` to use |
| Embeddings | `all-MiniLM-L6-v2` | Runs on-device for GraphRAG; no API key required |

## Implementation Notes

1. **Architect gate** — the Architect sets architectural direction before planning, then reviews. Its `verdict` (`plan` / `approved` / `revise` / `need_research`) is what routes the loop and what ends it.
2. **Planner routing** — `graph.py` respects `state["next_agent"]` set by the Planner node on every turn.
3. **Builder blockers** — `builder_node` parses `## Next Steps / Blockers`, sets `state["blockers"]` when execution fails or the LLM reports a blocker, and the graph loops back to Planner/Researcher accordingly.
4. **Researcher tool boundary** — `researcher_node` calls GraphRAG through the MCP-style client (`search_knowledge_graph`, `query_knowledge_graph`), not by importing the knowledge base directly.
5. **Builder tool boundary** — `builder_node` uses `filesystem_write` (and the other builder tools) through the MCP-style client. The Architect and Planner get no tools.
6. **State injection** — empty fields render as `(empty)` on every turn.
7. **GraphRAG read-only** — only `search_knowledge_graph` and `query_knowledge_graph` are exposed; indexing is done via `scripts/reindex.py` / `scripts/index_knowledge.py`.
8. **Step limit** — `MAX_STEPS = 8` prevents infinite loops, counted at the Architect gate rather than at the Builder so that Planner/Researcher cycles cannot run uncounted. `RECURSION_LIMIT` is derived from it.
9. **No `temperature` to modern Anthropic models** — sampling parameters were removed on the Opus 5 / Sonnet 5 / 4.6+ families and are rejected with a 400 that reads like an auth failure. `_accepts_temperature()` gates this.

## Running the System

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy environment template and edit
cp .env.example .env
# Set ANTHROPIC_API_KEY for the Architect seat; `ollama signin` for the :cloud tags.

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

1. Add human-in-the-loop approval gates alongside the Architect gate.
2. Add checkpointing/persistence for long-running agents.
3. Connect to external MCP servers over stdio/HTTP when needed.
