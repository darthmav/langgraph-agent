# 4-Agent AI System

**Architect · Planner · Researcher · Builder**

A multi-agent system for software development experiments, powered by LangGraph + GraphRAG + MCP.

Inference is **cloud only**. All four seats run Ollama Cloud tags, which the
local daemon proxies to ollama.com using credentials it holds itself, so the
crew runs without an API key of your own. Anthropic and OpenAI are available
per seat if you want them. The only thing that runs on this machine is the
embedding model.

## 🎨 Web Console

```bash
# Build the knowledge graph first — the Graph tab is empty without it
python scripts/reindex.py

# Quick launch
./launch_console.sh

# Or manually
python serve.py
# Open: http://localhost:8080
```

Five tabs: **Engineer** (give the Architect a goal, watch the stages),
**Graph** (the knowledge graph as a force-directed map — press *Sweep all*),
**Retrieval** (semantic search plus an RPC telemetry log), **Corpus**
(the indexed documents, and buttons to reindex, export or clear them), and
**State** (the raw `AgentState`).

*Export* downloads the whole corpus as one JSON file — the knowledge graph plus
every chunk with its text and metadata. Embeddings are left out: they are most
of the bytes and the least portable part, and the embedder is local, so a
reindex regenerates them.

*Clear corpus* empties the store in place. The files under `knowledge/` stay,
holding an empty index — the same shape a reindex leaves behind. It arms on the
first click and disarms itself after a few seconds.

Both *Clear corpus* and *Reindex project* are refused while a run is in flight,
and the refusal says which run. The Researcher searches this corpus, and
changing it underneath a run does not fail its search — an emptied corpus
answers "nothing found", and one midway through a rebuild answers from the part
of itself that exists so far. The run would plan around an absence that was
manufactured out from under it, without anything raising.

The left rail is the crew: one card per seat, each with its model, where the
prompt goes (`REMOTE` / `LOCAL`), and a status chip when the seat cannot
actually run — `NO KEY`, `FAILING`, `OFFLINE` or `NOT PULLED`.

### Stopping a run

**Stop** sits next to Run and halts the run in flight. It is cooperative: work
already started — a file write, a staged commit, a test run, a model call —
finishes first, and nothing further is begun, so a stopped run never leaves a
half-written file behind. Expect it to land within one tool call rather than
instantly.

Nothing is thrown away. The run returns its state as it stood, the console shows
what was written, what nobody got to run, and what blocked, and the server keeps
that snapshot in `runs/last_run.json` — so a reload, or a run that died on a
failing seat, still has something to show. Reloading during a run reattaches to
it, Stop included.

The **×** in the top right shuts down the console and the server, the same way
Ctrl+C does. With a run in flight it asks first, then stops the run and waits
for it to write its snapshot before the process goes — so exiting mid-run is as
recoverable as stopping one.

The **expect failures** checkbox is unchanged by this. It excuses a file the run
*meant* to fail, which is still executed and still reported; a file the stop
prevented anyone from running is unproven rather than expected, so it goes on
blocking approval either way.

See [`frontend/README.md`](frontend/README.md) for full documentation.

## Architecture

```
START → Architect → Planner → (Researcher | Builder) → Architect → END
             ^                                             |
             +--------- revise / need_research -------------+
```

The Architect runs twice per cycle: once to set direction before anything is
planned, and again as the approval gate. The Builder does not decide the work is
finished — it reports, and the authority that set the constraints rules on it.

### The Four Agents

| Agent | Responsibility | Default seat | Tools |
|---|---|---|---|
| **Architect** | Sets direction and constraints; rules `approved` / `revise` / `need_research` | `kimi-k3:cloud` (ollama) | None (reasoning only) |
| **Planner** | Turns goals into structured plans, routes next | `qwen3.5:397b-cloud` (ollama) | None (reasoning only) |
| **Researcher** | Gathers deep, relationship-aware knowledge | `nemotron-3-ultra:cloud` (ollama) | GraphRAG MCP only |
| **Builder** | Implements the plan (writes code, edits files) | `kimi-k3:cloud` (ollama) | Filesystem, Git, Terminal |

Every seat is reassignable live from its dropdown in the console; selections last
for the life of the process.

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

No API key is required: the default seats are all Ollama Cloud tags, and the
daemon holds those credentials. Sign it in once with `ollama signin`.

A key is only needed if you move a seat onto Anthropic or OpenAI:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

A seat pointed at a provider whose key is missing falls back to a canned stub.
It does so **visibly** — the seat card shows a `NO KEY` chip and the console
banners it — rather than pretending to run the model.

A key that exists but does not work (no credits, rate limited, model not on the
account) is a different failure: the seat shows `FAILING` with the provider's own
message and runs abort rather than quietly producing stub text. That state is
recorded from the actual outcome of a call, so it appears after the first run.

The three Ollama seats need the daemon running and signed in, since it holds the
ollama.com credentials for `:cloud` tags:

```bash
ollama signin
ollama pull kimi-k3:cloud       # and qwen3.5:397b-cloud, nemotron-3-ultra:cloud
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

## Diagnosing the seats

`scripts/diagnose_seats.py` answers "which model actually works in which seat",
in two phases that are deliberately kept apart because one is far cheaper than
the other.

```bash
# What it knows how to run — costs nothing
python scripts/diagnose_seats.py --list
python scripts/diagnose_seats.py --dry-run

# Phase 1 only: one call per model per role
python scripts/diagnose_seats.py --phase probe

# Phase 2 only: whole runs, one sandbox each
python scripts/diagnose_seats.py --phase teams --configs baseline,legacy --verbose

# Include the paid Anthropic controls
python scripts/diagnose_seats.py --anthropic --exercise all
```

**Phase 1 — role probes.** One bounded call per (model, role) pair, through the
real role prompt and the real parser the node uses. It answers whether a model
can hold a seat at all: did it answer, did the answer parse, and — for the
Builder — can it call a tool. `empty` is the status that matters, because a
seat that returns nothing loops the run rather than failing it.

**Phase 2 — team runs.** Each configuration runs the same short exercise through
the same graph the console drives, instrumented per node. It answers what the
probes cannot: whether four seats that each work alone make progress *together*.
Watch the `cycles` column — a run that passes the gate repeatedly while
producing nothing is the hand-off loop worth acting on.

Team runs let the Builder write files, so each one runs in its own sandbox
directory (a `chdir`), never in the project. Reports land in
`reports/diagnostics/<timestamp>/` as both `report.md` and `results.json`.

> Run `python scripts/reindex.py` first if you care about the results. Against
> an empty corpus every Researcher falls back to its model, which reads exactly
> like a bad Researcher seat — the script warns when it finds one, and records
> the corpus size in the report.

Nothing here is a benchmark. One short exercise per configuration is a data
point against non-deterministic models, not a ranking.

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

Per the 4-Agent System specification:

| Field              | Written By  | Description                                    |
|--------------------|-------------|------------------------------------------------|
| `goal`             | User        | The user's original goal                       |
| `messages`         | All         | Conversation history / log                     |
| `architecture`     | Architect   | Direction and constraints, injected downstream  |
| `verdict`          | Architect   | `plan` \| `approved` \| `revise` \| `need_research` |
| `plan`             | Planner     | Structured plan with steps                     |
| `research`         | Researcher  | Findings from GraphRAG queries                 |
| `builder_report`   | Builder     | Implementation report                          |
| `next_agent`       | Planner     | Which agent runs next                          |
| `research_status`  | Researcher  | `ready_for_builder` \| `need_replan` \| `no_relevant_knowledge` |
| `blockers`         | Builder     | What's blocking progress                       |
| `files_changed`    | Builder     | List of modified file paths                    |
| `step_count`       | Architect   | Cycles through the gate (loop limit, max 8)    |

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

- Architect → no tools
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
│   ├── nodes.py                # Architect, Planner, Researcher, Builder
│   ├── graph.py                # StateGraph wiring
│   ├── graphrag_server.py      # GraphRAG MCP server
│   └── mcp_client.py           # MCP client / tool bindings
├── prompts/
│   ├── planner.txt             # Planner system prompt
│   ├── researcher.txt          # Researcher system prompt
│   └── builder.txt             # Builder system prompt
├── tests/
│   └── test_graph.py           # Graph tests
├── scripts/
│   ├── reindex.py              # Rebuild the GraphRAG corpus
│   ├── index_knowledge.py      # First-time indexing
│   └── diagnose_seats.py       # Which model works in which seat
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
