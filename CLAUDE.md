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
│   ├── mcp_client.py          # MCP client / local tool bindings
│   ├── exceptions.py          # Public error surface (re-exports _internal/)
│   └── _internal/
│       └── exceptions.py      # LangGraphAgentError and its five subclasses
├── prompts/
│   ├── architect.txt          # System prompt (loaded by nodes.py)
│   ├── planner.txt
│   ├── researcher.txt
│   └── builder.txt
├── tests/
│   ├── test_graph.py          # Pytest suite
│   ├── test_diagnose_seats.py # Guards the seat diagnostic's verdicts
│   ├── test_console_stop.py   # Emergency stop, deferred exit, snapshot
│   ├── test_corpus_admin.py   # Corpus clear / export / reindex guards
│   ├── test_mcp_tools.py      # Builder tool belt
│   ├── test_imports.py        # Pins the package's public surface
│   └── test_spectral_graph.py # The spectral_graph package
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
  | Architect | ollama | `qwen3.5:397b-cloud` |
  | Planner | ollama | `qwen3.5:397b-cloud` |
  | Researcher | ollama | `qwen3.5:397b-cloud` |
  | Builder | ollama | `qwen3.5:397b-cloud` |

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
- **"Described but not written" compares two spellings of a path, so both go
  through `_report_path_key`.** The check is the report's harshest claim --
  the mirror of a Builder inventing a file, and the Architect rules on it --
  which makes a false one expensive. One side is prose from a `## Files
  Modified` bullet, the other is the raw argument `filesystem_write` recorded,
  and they disagree over decoration rather than substance: backticks, bold,
  `*`/`1.` markers, `./` or an absolute prefix, and above all an annotation
  (`- test_spectral_graph.py (new file)`), which is what filed a written,
  executed, passing file under "not written". The normalizer errs one way on
  purpose -- over-stripping only hides a real accusation, under-stripping
  invents one -- so it also drops "None"/"N/A" answers and prose sentences,
  and its trailing-parenthesis strip is a whitelist of annotation words
  because real filenames here carry parentheses
  (`examples/filter_band_pass_(40_60_hz).png`).
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
- **Verification runs headless, and never with a keyboard.** The subprocess
  gets `MPLBACKEND=Agg` with `DISPLAY`/`WAYLAND_DISPLAY` removed
  (`HEADLESS_VERIFY_ENV`), and `stdin` closed. Both are about the same
  failure: a file that waits for a human burns its whole
  `VERIFY_TIMEOUT_SECONDS` and is reported `FAILED` for it. A plotting
  example ending in `plt.show()` — the ordinary way to write one — is
  correct code that hung forever, and the first one to do it ate most of the
  Builder's budget, so the rest of the pass came back `NOT RUN` and blocked
  approval. The display variables are unset as well as `MPLBACKEND` set,
  because a library that probes for a display itself never consults
  `MPLBACKEND`. `stdin=DEVNULL` is in `_terminal_execute` and `_run_tests`
  rather than the verify path alone: no Builder tool call ever has someone at
  the keyboard behind it.
- **A timeout must carry what the file printed.** `_terminal_execute` returns
  `TimeoutExpired`'s captured `stdout`/`stderr` and sets `timed_out`;
  `_timeout_detail` appends the *tail* to the report. The message alone cannot
  distinguish a file that blocked on its first line from one that did all its
  work and then waited at the end, and the Builder does not leave that blank —
  it read a bare timeout as a missing dependency, installed a package that was
  already there, and spent a second full timeout on an identical retry. The
  tail is what says how far it got, so truncation trims the front.
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
- **The seat lights answer "who is working", and `run_progress["node"]`
  cannot.** `graph.stream` yields an update when a superstep *completes*, so
  `node` names the seat that just finished -- right for the feed, which lists
  what happened, and backwards for a light meaning "this seat is working": it
  lights the previous seat for the whole of the next seat's turn, so the
  slowest node in the run is the one node whose light never comes on and a
  stalled Architect shows as a busy Builder. `ACTIVITY` (`control.py`, a
  process-global beside `RUN_CONTROL` for the same reason) is entered and left
  by `_tracked` in `graph.py`, so a seat is lit for exactly as long as its node
  is on the stack, and `run_progress` reports it as `active`. The marking is a
  wrapper rather than lines in the node bodies because every node has several
  exits -- the stop, a deadline fallback, the ordinary return -- and only a
  `finally` covers the one nobody thought about. `leave` checks the name it was
  given: a worker `_with_deadline` abandoned finishes late, and must not darken
  the seat working now. `_finish_run` clears it as a backstop, since a light
  left on outlives the run in every console still open.
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
- **The deferred exit is a handshake, and both halves decide under
  `_run_lock`.** `rpc_shutdown` reads `running` and sets `_exit_after_run` in
  one hold of the lock, *before* `RUN_CONTROL.stop()`; `_finish_run` (the run's
  `finally`) clears `running` and reads `_exit_after_run` in one hold of the
  same lock. So whoever holds it either sees a live run and hands it the exit,
  or sees none and takes the exit itself — exactly one, never neither. Claiming
  the exit after the stop lost it outright: the stop is what sends the run to
  its teardown, a run at a superstep boundary gets there in microseconds, it
  read the flag unset and declined to request the shutdown — correctly, on what
  it could see — and by the time the flag was set there was no run left to
  honour it. Nothing set `_shutdown_requested` at all, so the X did not shut
  the server down. `rpc_stop_run` has the same shape and now guards the same
  way, or it pins `stopping` True on a run that has already ended.
- **The console confirms the exit by the server's silence, never by the
  reply.** The reply to `shutdown` says the exit was *asked for*, and an exit
  can be lost after it is asked for, so `waitForExit` polls until the socket
  is dead — on both branches, including the one with nothing in flight, which
  used to print "The server has stopped" the instant the reply landed and read
  identically whether the process went or stayed. The probe is a bare `fetch`
  rather than `rpc()`, since only the fetch failing is evidence; `rpc()` throws
  on an ordinary error reply too. When the wait runs out it says the server is
  still running instead of leaving "Closing…" over a live process.
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
- **No corpus exists until someone indexes one, and reading is not indexing.**
  `GraphRAGKnowledgeBase.__init__` *creates* the store -- `mkdir`, plus
  Chroma's files -- so the two doors are kept apart: `get_knowledge_base()`
  builds and is reserved for `rpc_reindex` and the index scripts, while
  `open_knowledge_base()` returns `None` when there is nothing on disk and is
  what every read goes through (`corpus_exists` / `corpus_state` answer without
  opening anything at all). Wiring a read to the creating door is not a
  cosmetic mistake: `serve.py` used to start a preload thread at import and the
  console polls `rag_stats` every five seconds, so starting the server was
  enough to leave a store behind within seconds -- one that had never been
  indexed and reported itself as a knowledge base to everything that looked
  afterwards. The console reports `absent` / `empty` / `indexed` rather than a
  bare count, because four zeros read as a knowledge base that happens to be
  empty, and only one of those states means "press Reindex". *Export* and
  *Clear* refuse when it is absent: creating a store in order to empty it
  leaves behind exactly what was asked to be removed.
- **A search with no corpus returns nothing, and says so.** It used to come
  back with one fabricated row -- `[GraphRAG not indexed]`, score 0.0, in
  `results` -- which is a made-up retrieval hit in the field real ones arrive
  in, and the Builder reads that field. The empty answer carries
  `source: "no_corpus"` and `NO_CORPUS_NOTE` instead, worded once in
  `graphrag_server` so the MCP tools, the Builder's belt and the console cannot
  describe the same absence differently.
- **`stats()` carries a structural health check, and it counts components
  with networkx rather than with the spectrum.** `connectivity()` reports
  `components`, `largest_component`, `isolated_nodes` and `lambda_2` -- the A1
  application from `reports/spectral_applicability.md`. It exists because an
  entity-extraction regression in `add_document` changes no counter and raises
  nothing: it fragments the graph, and that is visible here and nowhere else.
  Three decisions that are not interchangeable with the obvious alternatives.
  *Components come from `nx.number_connected_components`, not from a
  zero-eigenvalue count.* The identity "multiplicity of 0 = number of
  components" holds for `L = D - A`; on the **normalized** Laplacian an
  isolated node has `D^-1/2 = 0`, so the `I` term leaves a bare 1 on its
  diagonal and it contributes eigenvalue **1, not 0**. On this project's graph
  shape that returns 1 where the truth is 30, because 29 of the 30 components
  are orphans -- and it costs 42x the linear-time answer to get there.
  *`lambda_2` is measured on the largest component*, because on the whole graph
  it is identically 0 whenever the corpus is disconnected, which a real one is;
  a signal that reads 0.0 every time is not a signal. *And it is normalized*,
  so it stays in [0, 2] and does not scale with degree -- an unnormalized
  `lambda_2` grows as documents mention more entities, which makes this
  reindex's value incomparable with last week's, and comparing across reindexes
  is the whole point. `lambda_2` is `None`, never 0.0, when there is nothing to
  measure: 0.0 is a real reading meaning "about to split in two", and the empty
  corpus must not report the alarming one. A failure to compute it is named in
  `lambda_2_unavailable` rather than dropped, so "could not measure" never
  reads as "measured 0". `spectral_graph` is imported **inside** the method: it
  lives at the project root and is not part of the installed distribution, so a
  top-level import would turn a missing diagnostic into a module that will not
  load at all from anywhere but the root. The result is cached against
  `(nodes, edges)` because the console polls `stats()` every five seconds and
  the eigendecomposition is ~44ms on a 920-node graph; that key is sound
  because every mutation path here only adds (`add_document`) or zeroes
  (`clear`), and a re-add that moves neither count moves no structure either.
- **`bottleneck()` has three verdicts, and the middle one is why it is worth
  having.** The A3 application from `reports/spectral_applicability.md`: sweep
  the normalized Fiedler vector for the narrowest cut, then name the nodes
  whose edges cross it -- the entities two topic areas connect through, which
  are the terms a search should expand on for a query straddling both. Degree
  does not find them: a bridge entity mentioned by two documents has degree 2.
  The trap is that **a minimisation always returns something**. Ask for the
  narrowest cut in a well-knit corpus and you get one, and calling it a bridge
  is a fabricated finding of the kind `search` was fixed for. Cheeger's *lower*
  bound is what refuses it, because `mu_2 / 2` is a proof that no cut anywhere
  in the graph is narrower: `certified_none` when that bound is itself above
  `BOTTLENECK_CONDUCTANCE` (a theorem about the whole graph, not a failed
  search), `found` when the sweep cut came in under the line, and
  `inconclusive` when the bound permits a bottleneck the sweep did not find.
  That third state is real, not hedging -- the Cheeger bracket runs from 4x to
  546x wide across the shapes in
  `reports/spectral_architecture_benchmark.md`, so the sweep genuinely can
  miss, and collapsing it into "no bottleneck" reports a gap in the evidence as
  a finding. Like `connectivity()` it runs on the largest component, because on
  a disconnected graph the Fiedler vector is a component indicator and an
  orphan is a conductance-0 cut that would win every time -- "your corpus has
  an orphan" is what `connectivity()` already says. `tied_cuts` reports
  `mu_2 ~= mu_3`, meaning several equally narrow cuts and an arbitrary choice
  between them: measured on a three-topic corpus, the *split* alternates
  between runs (99/198 and 97/200) while the conductance and the bridge
  entities are identical across all 12. Without the flag a working diagnostic
  reads as a broken one, and the tie is itself a finding -- three or more topic
  areas, not two. It is a separate RPC rather than part of `stats()`: an
  eigenvector plus a sweep over every edge is not something to put on a
  five-second poll, and it takes no run guard, for the reason `export_corpus`
  takes none.
- **`topics()` chooses `k` from the eigengap, but only when the eigengap is
  decisive.** The A2 application from `reports/spectral_applicability.md`:
  Ng-Jordan-Weiss clustering over the normalized Laplacian, which on a
  bipartite document/entity graph groups documents with the entities that
  define them -- so a cluster reads as a topic and `top_entities` names it.
  This is the whole-corpus map `neighborhood()` cannot give.
  The proposal's weak point was `k`, and it is not wired straight through.
  `reports/spectral_architecture_benchmark.md` measured the eigengap heuristic
  wrong on 3 of 8 architectures, k = 10 for a barbell whose answer is 2, and it
  always returns *some* k -- so on a corpus with no topics it invents one, and
  clusters shown without that caveat are a fabricated map. What rescues it is
  that the failures are **undecided**, not merely wrong: the winning gap barely
  beats the runner-up. Measured across 18 corpora with a planted topic count the
  eigengap was correct every time at a decisiveness of 5.1-23.4, while a grid, a
  small-world ring, an expander and one dense topic all landed at 1.0-1.8 --
  nothing observed between 1.8 and 4.5, so `EIGENGAP_DECISIVENESS = 3.0` sits in
  open space rather than on a boundary. Under it the verdict is
  `no_clear_structure` and **no clusters are returned**, because a map of a
  corpus with no topics is worse than no map. An explicit `k` skips the gate --
  the caller has decided -- but the decisiveness is still reported.
  **Per-cluster conductance is the second, independent check** and is why
  clusters are never returned bare: it exposes a `k` that split a real
  community even when the eigengap looked decisive, and it is what makes a
  forced `k` visibly wrong (0.36-0.45 on a corpus with one topic, against
  0.009-0.018 when the topics are real). A bad `k` from the caller raises
  `ValueError` rather than returning `unavailable`: that verdict means the
  measurement could not be taken, and filing a malformed request under it would
  blame the solver for the caller. Runs on the largest component, like
  `connectivity()` and `bottleneck()` -- components are already clusters, so on
  a disconnected graph the eigenvectors would spend themselves rediscovering
  the orphans `connectivity()` has already counted.
- **The embedding model loads on first use, not on construction.**
  `GraphRAGKnowledgeBase.embedder` is a lazy property and
  `sentence_transformers` is imported inside it. Only `add_document` and
  `search` embed; counting the corpus, listing its documents, drawing its graph
  and exporting it do not, and those are what the console does on a timer.
  Loading it in `__init__` meant every header poll paid for the model and
  importing the module pulled in torch behind it.
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
- **GraphRAG returns no results** — Check whether there is a corpus at all: the console header reads `no corpus — nothing indexed` when none has been built, and nothing builds one for you. Run `python scripts/reindex.py`, or press *Reindex project*.
- **No LLM output / canned text** — A seat pointed at Anthropic or OpenAI needs that provider's key in `.env`; without one it runs `StubLLM` and the console shows a `NO KEY` chip. No seat uses either by default. The Ollama seats need the daemon running and signed in (`ollama signin`) for `:cloud` tags.
- **A 400 from Anthropic that looks like an auth error** — Check nothing is passing `temperature` to an Opus 5 / Sonnet 5 / 4.6+ model; sampling parameters are rejected on those families.
- **Graph tab is empty** — Run `python scripts/reindex.py` (or press Reindex project on the Corpus tab). A `TypeError` on every insert used to leave the graph empty while the script still reported success; the corpus is only real if `rag_stats` shows non-zero nodes.
