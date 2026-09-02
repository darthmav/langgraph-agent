# Claude Code Project Instructions

## Project Overview

This is **langgraph-agent**, a cloud-only 4-Agent AI system for software development experiments.

- **Architect** — the leading authority. Sets architectural direction before planning, then holds the approval gate: the run ends on its `approved` verdict, not the Builder's say-so. No tools.
- **Planner** — interprets goals, creates structured plans, routes to next agent.
- **Researcher** — gathers context via the GraphRAG MCP tool (`search_knowledge_graph`).
- **Builder** — implements plans using filesystem, git, terminal, and test MCP tools.

Inference is cloud-only. The only thing that runs on this machine is the
embedding model (`all-MiniLM-L6-v2`), which belongs to GraphRAG, not to a seat.

Tech stack: Python 3.10+, LangGraph, Chroma + sentence-transformers + NetworkX, MCP (local stdio-compatible tool binding).

## Quick Reference

```bash
# Install dependencies
pip install -e ".[dev]"

# Run all checks
ruff check src/ tests/ serve.py scripts/ example_usage.py test_cloud.py
mypy src/langgraph_agent/ serve.py
python -m pytest tests/ -v

# Start the web console
./launch_console.sh

# Re-index project files into GraphRAG
python scripts/reindex.py

# Run the example
python example_usage.py
```

## Project Structure

```
├── src/langgraph_agent/
│   ├── __init__.py            # create_agent_graph, AgentState, ResearchStatus, Verdict
│   ├── state.py               # AgentState schema + ResearchStatus / Verdict enums
│   ├── config.py              # Seats, LLM setup (Anthropic + Ollama Cloud) + StubLLM
│   ├── nodes.py               # Architect, Planner, Researcher, Builder nodes + prompt loading
│   ├── graph.py               # StateGraph wiring + conditional edges
│   ├── graphrag_server.py     # GraphRAG MCP server (knowledge graph + vector store)
│   └── mcp_client.py          # MCP client / local tool bindings
├── prompts/
│   ├── architect.txt          # System prompt (loaded by nodes.py)
│   ├── planner.txt
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
├── test_cloud.py              # Cloud LLM end-to-end test
├── README.md                  # User-facing documentation
└── .env.example               # Environment variables template
```

## Conventions

- **Python formatting/linting:** `ruff` configured in `ruff.toml`.
- **Type checking:** `mypy` configured in `mypy.ini`.
- **Tests:** `pytest` in `tests/`.
- **Default seats** (`DEFAULT_SEATS` in `config.py`):

  | Seat | Provider | Model |
  |---|---|---|
  | Architect | ollama | `kimi-k3:cloud` |
  | Planner | ollama | `qwen3.5:397b-cloud` |
  | Researcher | ollama | `gemma4:cloud` |
  | Builder | ollama | `kimi-k3:cloud` |

  Anthropic and OpenAI remain optional cloud providers; no seat uses either by
  default, so a fresh checkout runs without an API key of its own. `:cloud`
  tags are proxied to ollama.com by the local daemon, which holds the
  credentials.
- **Never send `temperature` to a modern Anthropic model.** Sampling parameters
  were removed on the Opus 5 / Sonnet 5 / 4.6+ families and are rejected with a
  400 that reads like an auth failure. `_accepts_temperature()` gates this.
- **Tool specialization (critical):**
  - Architect → no tools.
  - Planner → no tools.
  - Researcher → GraphRAG read-only tools only.
  - Builder → filesystem, git, terminal, test tools only.

## State Schema

Every node reads/writes `AgentState`:

```python
{
    "goal": str,
    "architecture": str,
    "verdict": "plan" | "approved" | "revise" | "need_research",
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

The Architect writes `architecture` and `verdict`; the verdict is what routes
the loop and what ends it. `step_count` is incremented by the Architect gate,
not by the Builder — every cycle passes the gate, but a Planner/Researcher loop
never reaches the Builder and would otherwise run uncounted.

## Common Tasks

### Run the test suite
```bash
python -m pytest tests/ -v
```

### Add a new Builder tool
1. Add the tool method to `src/langgraph_agent/mcp_client.py` in `_discover_tools()`.
2. Name it clearly under the `filesystem_`, `git_`, `terminal_`, or `test_` namespace.
3. Add a JSON schema for it to `BUILDER_TOOLS` in `src/langgraph_agent/nodes.py`.
   The Builder can only call what is offered there — a tool the MCP client
   exposes but `BUILDER_TOOLS` omits is refused by name in `_run_builder_tools`.
   Keep GraphRAG out of that list; retrieval is the Researcher's.
4. Add a test in `tests/test_graph.py` or create a focused unit test.
5. Update `README.md` MCP Integration section.

### Add a new Researcher tool
1. Prefer GraphRAG read-only tools only.
2. Extend `src/langgraph_agent/graphrag_server.py` and expose via `mcp_client.py`.

### Change a seat's model
- Update `DEFAULT_SEATS` (and `_DEFAULT_AGENT_MODELS`) in `src/langgraph_agent/config.py`, plus `.env.example`.
- Add the model to `AGENT_LLM_OPTIONS` so it appears in the console dropdown.
- Nothing in `frontend/index.html` hard-codes a model; the seat cards render whatever `list_seats` reports.

### Add a fifth agent
The seat list is `AGENTS` in `config.py` and is read by everything that iterates
agents. Adding one means: a node in `nodes.py`, a prompt in `prompts/`, wiring
plus a router in `graph.py`, entries in `AGENTS` / `DEFAULT_SEATS` /
`_DEFAULT_AGENT_MODELS`, a `StubLLM` branch, and a `ROLE_COLOR` entry in
`frontend/index.html`.

### The console API
The console drives a single `POST /rpc` taking `{method, params}` and returning
`{result, elapsed_ms}` or `{error: {message}, elapsed_ms}`. Methods live in
`RPC_METHODS` in `serve.py`. The `/api/*` routes are compatibility wrappers over
the same functions — `launch_console.sh` polls `/api/status` as its readiness
check, so it must keep working.

## Important Notes

- Do not let every agent call every tool; the specialization is the whole point.
- Empty state fields must render as `(empty)` in the state injection block.
- The Builder must actually call a tool to claim it changed a file. It drives
  its own tools through `bind_tools` and a turn loop capped at
  `MAX_BUILDER_TOOL_TURNS`; `files_changed` is appended only when a write
  tool reports success, never from the model's prose. A seat whose model
  cannot call tools (`StubLLM`) still reports but changes nothing.
- **The Builder must run what it writes.** Every file it wrote with a
  `RUNNABLE_SUFFIXES` extension is executed by `_verify_written_files` after
  the tool loop, and a file that raises becomes a blocker plus a `FAILED` line
  in the report — the feed says "N did not run" rather than "Implementation
  complete". Enforced in code, not left to the prompt, for the same reason
  `files_changed` is: the Builder's account of its own work is not evidence.
  The Architect still rules on the result; a failed verification informs that
  verdict rather than overriding it.
- Knowledge base files under `knowledge/` (`chroma/`, `knowledge_graph.json`) are runtime artifacts; avoid committing them unless intentionally versioning an index.
- A reindex **rebuilds** rather than accumulates: it clears the graph and prunes Chroma ids that no longer qualify, so excluded or deleted files stop answering searches.
- `PROJECT_INDEX_EXCLUDES` entries are matched as plain substrings, not globs. `"*.egg-info"` matches nothing.
- A seat with no credentials silently becomes `StubLLM`. `get_agent_status()` is the only thing that reports the difference — keep the chip and banner wired to it.
- **Key presence is not liveness.** A key can authenticate and the seat still be unusable (no credits, rate limit, model not on the account). `_SeatLLM` records the real outcome of each call in `_seat_failures`, and `get_agent_status()` reports that over any static check. Never re-add a presence-only check as the sole signal.
- `stubbed` and `live` are different failures: a stubbed seat completes the run with canned text, a failing seat kills it. The console words them differently; keep it that way.
- `__pycache__` directories should never be committed (they are removed from git in this repo).

## Troubleshooting

- **Tests are slow** — The first run loads `sentence-transformers` and Chroma. Subsequent runs use the cached singleton.
- **Mypy errors from upstream stubs** — Prefer `# type: ignore[...]` with a comment over disabling strict mode.
- **GraphRAG returns no results** — Run `python scripts/reindex.py` to rebuild the knowledge base.
- **No LLM output / canned text** — A seat pointed at Anthropic or OpenAI needs that provider's key in `.env`; without one it runs `StubLLM` and the console shows a `NO KEY` chip. No seat uses either by default. The Ollama seats need the daemon running and signed in (`ollama signin`) for `:cloud` tags.
- **A 400 from Anthropic that looks like an auth error** — Check nothing is passing `temperature` to an Opus 5 / Sonnet 5 / 4.6+ model; sampling parameters are rejected on those families.
- **Graph tab is empty** — Run `python scripts/reindex.py` (or press Reindex project on the Corpus tab). A `TypeError` on every insert used to leave the graph empty while the script still reported success; the corpus is only real if `rag_stats` shows non-zero nodes.
