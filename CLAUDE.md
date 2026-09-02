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

# Find out which model actually works in which seat
python scripts/diagnose_seats.py --list
python scripts/diagnose_seats.py --phase probe

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
│   ├── control.py             # RUN_CONTROL: the emergency stop signal
│   ├── graphrag_server.py     # GraphRAG MCP server (knowledge graph + vector store)
│   └── mcp_client.py          # MCP client / local tool bindings
├── prompts/
│   ├── architect.txt          # System prompt (loaded by nodes.py)
│   ├── planner.txt
│   ├── researcher.txt
│   └── builder.txt
├── tests/
│   ├── test_graph.py          # Pytest suite
│   └── test_diagnose_seats.py # Guards the seat diagnostic's verdicts
├── scripts/
│   ├── reindex.py             # Re-index files into GraphRAG
│   ├── index_knowledge.py     # First-time indexing
│   ├── verify_and_test.py     # Manual verification runner
│   ├── auto_verify.py         # Silent verification
│   ├── quick_test.sh          # Bash quick check
│   ├── full_setup.py          # Automated setup + re-index
│   └── diagnose_seats.py      # Role probes + team runs per seating
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
  | Researcher | ollama | `qwen3.5:397b-cloud` |
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
    "failed_verification": list[str],
    "expect_failures": bool,
    "step_count": int,
}
```

`stopped`, `stop_reason`, `run_id` and `elapsed_s` are added to the payload
`run_goal` returns, not to `AgentState`: that schema describes what the agents
write, not how the run ended.

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
`RPC_METHODS` in `serve.py`. `export_corpus` and `clear_corpus` go through it
like everything else rather than getting a file-download route: a failure then
lands on the console's telemetry path instead of replacing the page with a JSON
error, and the browser builds the file at the other end. The `/api/*` routes are compatibility wrappers over
the same functions — `launch_console.sh` polls `/api/status` as its readiness
check, so it must keep working.

`run_goal` blocks its own HTTP thread for the whole run, which is why the server
is a `ThreadingHTTPServer` — `stop_run` and `run_progress` are served on other
threads while it sits there. Only one run may be in flight: a second `run_goal`
is refused rather than allowed to clobber `_run_progress`, which has always been
a single global.

`shutdown` is the console's exit button. `serve_forever` runs on its own thread
so the main thread can wait on both ways out — Ctrl+C and `_shutdown_requested`.
Calling `server.shutdown()` from the request thread that asked for it would
deadlock: it blocks until the serve loop stops, and that loop is what has to
deliver the reply.

## Important Notes

- Do not let every agent call every tool; the specialization is the whole point.
- Empty state fields must render as `(empty)` in the state injection block.
- The Builder must actually call a tool to claim it changed a file. It drives
  its own tools through `bind_tools` and a turn loop capped at
  `MAX_BUILDER_TOOL_TURNS`; `files_changed` is appended only when a write
  tool reports success, never from the model's prose. A seat whose model
  cannot call tools (`StubLLM`) still reports but changes nothing.
- **`state["files_changed"]` accumulates across passes; the local list does
  not.** Inside `builder_node` the local `files_changed` is what *this* pass
  wrote, and the verification logic depends on that: `carried` uses it to tell
  a path this pass rewrote from one only an earlier pass touched, and the
  "deleted file clears its failure" exemption applies to carried paths alone.
  The state field answers a different question — what the whole run produced —
  so `all_files_changed` merges the previous record in before it is written
  back. Overwriting it per pass meant a run that wrote a file on one cycle and
  nothing on the next ended reporting it had changed nothing while the file sat
  on disk, and the Architect ruled on that empty record: a build with a file to
  its name was approved as having produced none. That is the same false account
  as claiming a file that was never written, pointing the other way. The feed
  line still counts this pass, and names the running total only when the two
  differ, so a quiet pass never reads as though the run lost its work.
- **The Builder must run what it writes.** Every file it wrote with a
  `RUNNABLE_SUFFIXES` extension is executed by `_verify_written_files` after
  the tool loop, and a file that raises becomes a blocker plus a `FAILED` line
  in the report — the feed says "N did not run" rather than "Implementation
  complete". Enforced in code, not left to the prompt, for the same reason
  `files_changed` is: the Builder's account of its own work is not evidence.
  A file clears only by running clean: files that failed on an earlier pass are
  re-verified even when the current pass did not touch them, because otherwise
  the Builder retires a failure by doing nothing. Two paths are exempt. The
  first is a carried path that no longer exists on disk — deleting the file is
  a real fix,
  and re-running a missing path fails forever, which pinned
  `failed_verification` open and made the gate rewrite every `approved` to
  `revise` until the step ceiling. That exception applies only to carried
  paths; a path in `files_changed` was just written by a tool that reported
  success. The second is a **module inside a package** (`_is_package_module`:
  its directory has an `__init__.py`). `python pkg/mod.py` puts `pkg/` on
  `sys.path` instead of the project root, so a module importing its own package
  absolutely — the normal way to write one — raises `ModuleNotFoundError`
  however correct it is; executing it proves nothing and produces a failure no
  edit to the file can clear. Such a file is reported `SKIPPED`, not passed:
  the report says how many were skipped and that a package module only proves
  itself through a root-level script that imports it. Those scripts are
  ordinary files and still get executed.
- **A silent Researcher is not research.** `_parse_researcher_output` defaults
  the status to `ready_for_builder`, so a seat that answered with nothing was
  announced as "Research complete" while `research` reached the Builder empty.
  The cost was not one bad cycle but a loop: the Builder reports an empty
  store, the gate rules `need_research`, and the run goes back round to the
  same silent seat, burning a step per cycle to the ceiling. `_said_nothing`
  catches both shapes — an empty response, and one that filled in the headings
  and left every section blank — and `_as_text` flattens content blocks first,
  because a list reads as truthy and non-empty to `len()` and to the section
  regexes alike. It routes to the **Builder**, not back to the Planner: the
  plan is not what failed, and re-planning would aim the run at the same seat
  again. The feed message names the seat's model, since changing it is the
  only thing that actually fixes this. Which model to name is a measurement,
  not a guess: `scripts/diagnose_seats.py --phase probe` asks each candidate
  the Researcher's own question with retrieval stubbed out, and on this
  machine `gemma4:cloud`, `nemotron-3-ultra:cloud` and `kimi-k2.7-code:cloud`
  all answer with nothing, while `qwen3.5:397b-cloud` and `kimi-k3:cloud`
  answer in full. Re-run it before changing the seat rather than trusting that
  list, which is one machine on one day.
- **The Researcher's model is only consulted when retrieval is thin.**
  `_gather_research` calls GraphRAG first and, whenever the top hit scores
  above `0.3`, formats those chunks straight into the findings and returns
  without invoking the seat at all. So on a question the corpus answers well,
  the Researcher's model is not a variable: two different models produce
  byte-identical `research`, in ~0.0s. This is worth knowing before blaming or
  crediting a Researcher seat for a run's quality, and it is why the seat's
  model matters most in precisely the case that is hardest to notice -- a
  question the corpus cannot answer, which is also when a silent seat sends the
  run round the loop. `scripts/diagnose_seats.py` keeps the two apart: the
  `research` exercise measures retrieval, `offcorpus` is the one that reaches
  the model, and the phase 1 probes stub retrieval out entirely.
- **A failed verification blocks approval.** `failed_verification` carries the
  paths, and the Architect rewrites its own `approved` to `revise` while that
  list is non-empty — the one place the gate's ruling is overridden. The step
  ceiling and `RUN_BUDGET_SECONDS` still end the run, so the block cannot hang
  it. The cost is real: a goal that legitimately wants a failing file (a
  deliberate fixture, an expected-to-fail test) can no longer be approved and
  will run to one of those limits — unless the run opts out.
- **`expect_failures` is the per-run opt-out**, set by the caller (the console
  checkbox, or `run_goal({goal, expect_failures: true})`) and never by an
  agent. It suppresses the block, not the check: the file is still executed,
  still reported as `FAILED`, and still listed in `failed_verification`. It
  just stops overruling the gate and sets no blocker. It is per-run rather
  than per-file because the Builder chooses the filenames, so a run with it on
  will not block on an unintended failure either — which is why the failure
  stays visible in the report instead of being dropped. Defaults off.
- **The emergency stop is cooperative, and the recovery is the point.**
  `RUN_CONTROL` (`control.py`) is a process-global flag, not a state field: the
  graph compiles without a checkpointer, so nothing outside a node can write
  into a state the graph is already streaming. It is checked at the top of the
  Builder's tool turn loop, before each file in `_verify_written_files`, at the
  top of the three tool-free nodes, and between supersteps in `serve.py` — never
  inside a tool batch, because skipping calls there leaves `ToolMessage` replies
  missing for `tool_call_id`s the model was already told about. Nothing in
  flight is abandoned, for the same reason the Builder's deadline never abandons
  a tool call. It is kept apart from `_Deadline` deliberately: a run the operator
  stopped must not be reported as one that overran its own budget, and the report
  is the only place anyone finds out which happened.
- **A stopped run must come back with what it produced.** The final state used
  to be returned to the caller and dropped, so a stop that lost the partial
  plan, research and report would be no better than killing the server —
  which is the thing it replaces. `rpc_run_goal` writes a snapshot
  (`runs/last_run.json`, atomically) in every exit path, *including the
  exception path*, and `last_run` serves it to a console that reloaded. The
  bookkeeping is in a `finally`; before that, a run that raised left
  `_run_progress["running"]` True forever and the console polled a run that no
  longer existed.
- **The stop does not touch `expect_failures`, and must not.** A file the stop
  prevented anyone from executing is `unverified`, and `unverified` blocks
  approval whatever the opt-out says — that opt-out is for a file the run meant
  to fail, still executed and still reported, not for a gap in the evidence. So
  ticking the box cannot make a stopped run look finished. The console says so
  explicitly in the recovery block rather than leaving it to be inferred.
- **The exit button never kills a run by surprise.** `shutdown` without
  `stop_first` refuses while a run is in flight and says what is running; with
  it, the run is stopped through the ordinary emergency stop and the exit is
  *deferred to the run's own `finally`*. That deferral is the whole point:
  exiting the moment the stop is requested would race the snapshot, and a run
  stopped only so the console could close is exactly the one worth keeping.
- **Three nested timeouts, and none of them is redundant.**
  `LLM_TIMEOUT_SECONDS` (config.py) bounds one provider call at the socket —
  each provider spells it differently (`client_kwargs={"timeout":…}` for
  Ollama, `default_request_timeout` for Anthropic, `request_timeout` for
  OpenAI), so a missed keyword silently restores an unbounded wait on that
  provider alone. `NODE_DEADLINE_SECONDS` (nodes.py) bounds a whole node turn,
  and covers the Architect, Planner and Researcher — all single-call, tool-free
  nodes that can be wrapped whole. It is what catches a model that streams
  slowly but never stops, and a node making several calls that each finish just
  inside their own limit. `BUILDER_DEADLINE_SECONDS` is the Builder's larger
  equivalent — that node legitimately makes up to `MAX_BUILDER_TOOL_TURNS`
  round trips, some of which run tests. `RUN_BUDGET_SECONDS` (serve.py) bounds
  the run, but is checked **between
  graph supersteps** — a node in flight never reaches a superstep boundary, so
  it cannot end a hung node. That gap is the whole reason for the other two:
  before them a stalled seat hung a run indefinitely while the console still
  named the *previous* node as current, and `_SeatLLM` recorded nothing because
  a hang raises nothing.
- **The Builder's deadline may never abandon a tool call.** Only the model's
  own call is wrapped in `_with_deadline` — discarding a half-received response
  costs a turn and nothing else. The tool calls underneath it write files,
  stage commits and run commands, so they always run to completion and the
  budget is re-checked at the top of the next turn instead. A worker abandoned
  mid-`filesystem_write` would go on writing into the project after the node
  returned, which is worse than the hang the deadline exists to stop.
- **A file the deadline stopped us running is `unverified`, not `ok`.** It
  reads `NOT RUN` in the report, joins `failed_verification` so the next cycle
  re-runs it, and blocks approval even under `expect_failures` — that opt-out
  is for a file the run *meant* to fail, which is still executed and still
  reported, not for one nobody executed. `VERIFY_RESERVE_SECONDS` is held back
  from the tool loop (plus whatever the loop leaves unspent) so the pass
  normally gets to run at all. `MIN_VERIFY_SLICE_SECONDS` is the floor below
  which a file is left unrun rather than started: `min(VERIFY_TIMEOUT_SECONDS,
  remaining)` rounded down to a zero-second timeout, and a working file came
  back `FAILED` with "timed out after 0 seconds" — a false accusation that also
  set the "files that do not run" blocker.
- **A timed-out Architect can never rule `approved`.** The gate is what ends
  the run, so that fallback is the one place a hang could become a false
  success. It falls back to `revise` on the gate pass and `plan` on the opening
  pass; both route to the Planner, and neither ends anything. The architecture
  already in state is kept, since losing it would strip the constraints out of
  every later prompt.
- **A timed-out Planner must leave a non-empty `plan`.** This is load-bearing,
  not cosmetic: the Architect increments `step_count` only while a plan exists,
  so an empty fallback would send the run round Planner → Builder → Architect
  uncounted until LangGraph's recursion limit killed it by exception,
  discarding every message it had produced — the exact failure the counter was
  moved to the gate to prevent. It keeps a real plan from a previous cycle when
  there is one, and otherwise writes `_PLANNER_TIMED_OUT`.
- Work run under `_with_deadline` **must not write to state.** The abandoned
  worker cannot be cancelled — Python cannot interrupt a thread blocked on a
  socket — so it may finish long after the node returned and would land its
  result in a state the graph had moved past. Every bounded node follows the
  same shape: `_rule_on_state`, `_make_plan` and `_gather_research` read state
  and return parsed output, and their nodes apply it. Keep that split for any
  node put under a deadline. The worker is a bare daemon thread on purpose:
  `ThreadPoolExecutor`'s atexit hook joins its non-daemon threads, so one
  abandoned worker would hold up interpreter shutdown.
- Knowledge base files under `knowledge/` (`chroma/`, `knowledge_graph.json`) are runtime artifacts; avoid committing them unless intentionally versioning an index.
- A reindex **rebuilds** rather than accumulates: it clears the graph and prunes Chroma ids that no longer qualify, so excluded or deleted files stop answering searches.
- **A corpus clear empties in place and must reach disk.** `clear()` deletes
  every Chroma id, *then* clears the graph — never the other way round, and it
  raises rather than report a half-wipe as success, because the two halves
  answer different questions and a corpus that disagrees with itself is worse
  than one that is merely stale. The files under `knowledge/` are kept, holding
  an empty store: Chroma has that directory open, and pulling it out from under
  a live client is the worse failure. The trailing `_save_graph()` is
  load-bearing — `index_project_files` has exactly that hole today, where
  `graph.clear()` is persisted only as a side effect of indexing something
  afterwards, so a reindex matching zero files leaves the old graph on disk.
- **Changing the corpus is refused while a run is in flight.** Both writers —
  `clear_corpus` and `reindex` — go through
  `_refuse_while_a_run_is_in_flight()`. Not for consistency: because the
  failure would be silent. An emptied corpus does not break the Researcher's
  search, it returns no hits; a corpus midway through a rebuild returns
  whatever fraction of itself has been re-added. Either reads as
  `no_relevant_knowledge` and routes the run as though the knowledge base had
  simply had nothing useful to say. Nothing raises and nothing is logged, so
  the seat can never find out it was cut off and the operator is told instead.
  The refusal names the goal, the way `rpc_shutdown` does. `export_corpus` has
  no such guard: reading the corpus takes nothing away from the run using it.
- **The export omits embeddings, and says so in the file.** They are most of
  the bytes and the least portable part — a reader without the same model
  cannot use them — and the embedder runs locally, so a reindex regenerates
  them. The `note` field carries that so the omission is not left to be
  discovered. The graph half is `node_link_data`, the same format
  `_save_graph` writes, so it compares directly against
  `knowledge/knowledge_graph.json`.
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
