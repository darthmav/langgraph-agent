# Claude Code Project Instructions

## Project Overview

This is **langgraph-agent**, a cloud-only 4-Agent AI system for software development experiments.

- **Architect** — the leading authority. Sets architectural direction before planning, then holds the approval gate: the run ends on its `approved` verdict, not the Builder's say-so. No tools.
- **Planner** — interprets goals, creates structured plans, routes to next agent.
- **Researcher** — gathers context via the GraphRAG MCP tool (`search_knowledge_graph`).
- **Builder** — implements plans using filesystem, git, terminal, and test MCP tools.

Inference is cloud-only. The only thing that runs on this machine is the
embedding model (`all-MiniLM-L6-v2`), which belongs to GraphRAG, not to a seat.

Tech stack: Python 3.12+, LangGraph, Chroma + sentence-transformers + NetworkX, MCP (local stdio-compatible tool binding).

## Quick Reference

```bash
# Install everything on Arch / Omarchy (system packages, venv, Ollama, corpus)
./install.sh

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
│   ├── lexical.py             # BM25 + rank fusion: the lexical half of search
│   ├── web_research.py        # Online research: keyless search, our own selection gate
│   ├── html_text.py           # HTML → text by link density. No library, no dependency
│   ├── corpus_health.py       # Is the corpus still the project? missing / extra / oversized
│   ├── corpus_spectral.py     # connectivity / topics / bottleneck / duplicate_entities (mixin)
│   ├── exceptions.py          # Public error surface (re-exports _internal/)
│   ├── self_healing/          # Opt-in retry / circuit-breaker decorators (see note below)
│   │   ├── logger.py          # SelfHealingLogger: structured, severity-leveled healing log
│   │   └── decorators.py      # retry_with_backoff, circuit_breaker, self_healing_wrapper
│   └── _internal/
│       └── exceptions.py      # LangGraphAgentError and its five subclasses
├── src/quisce/                # QuICSE engine prototype. Ships in the
│   ├── quisce_engine.py       #   distribution, imported by nothing here --
│   ├── baseline.py            #   see the note below the tree
│   └── test_harness.py
├── prompts/
│   ├── architect.txt          # System prompt (loaded by nodes.py)
│   ├── planner.txt
│   ├── researcher.txt
│   └── builder.txt
├── tests/
│   ├── conftest.py            # Forces StubLLM; switches the web phase off
│   ├── test_claims.py         # Documentation claims, made executable
│   ├── test_corpus_absent.py  # The two doors: reading never creates a corpus
│   ├── test_graph.py          # Pytest suite
│   ├── test_diagnose_seats.py # Guards the seat diagnostic's verdicts
│   ├── test_thinking.py       # The per-seat thinking switch: support, wire, parsing
│   ├── test_console_stop.py   # Emergency stop, deferred exit, snapshot
│   ├── test_chunking.py       # Document chunking, chunk ids, search collapse
│   ├── test_corpus_admin.py   # Corpus clear / export / reindex guards
│   ├── test_mcp_tools.py      # Builder tool belt
│   ├── test_imports.py        # Pins the package's public surface
│   ├── test_lexical.py        # BM25, rank fusion, the relevance floor
│   ├── test_self_healing.py   # self_healing: logger, retry, circuit breaker
│   ├── test_uploads.py        # Uploading a document into the corpus
│   ├── test_web_research.py   # Fan-out, the selection gate, storage, the empty answers
│   ├── test_html_text.py      # The from-scratch HTML reader
│   ├── test_web_entities.py   # Fetched pages stay out of the entity graph
│   ├── test_run_research_phase.py # Where the research phase sits in a run
│   ├── test_corpus_staleness.py   # Corpus vs disk, and not crying wolf
│   ├── test_graph_queries.py  # Undirected traversal of the knowledge graph
│   ├── test_research_length.py # How much retrieved evidence reaches the Builder
│   └── test_spectral_graph.py # The spectral_graph package
├── scripts/
│   ├── reindex.py             # Re-index files into GraphRAG
│   ├── index_knowledge.py     # First-time indexing
│   ├── verify_and_test.py     # Manual verification runner
│   ├── auto_verify.py         # Silent verification
│   ├── quick_test.sh          # Bash quick check
│   ├── full_setup.py          # Automated setup + re-index
│   ├── spectral_benchmark.py  # Graph-architecture sweep behind the A-numbers
│   └── diagnose_seats.py      # Role probes + team runs per seating
├── frontend/
│   ├── index.html             # Web console SPA
│   └── README.md
├── .github/workflows/
│   └── ci.yml                 # ruff, mypy, pytest, root scripts, on every push
├── install.sh                 # Arch / Omarchy: everything, from nothing to a running console
├── serve.py                   # Python HTTP server + API backend
├── example_usage.py           # Demo script
├── test_cloud.py              # Cloud LLM end-to-end test
├── README.md                  # User-facing documentation
└── .env.example               # Environment variables template
```

### `src/quisce/` is a prototype, and nothing here depends on it

Roughly 960 lines: `quisce_engine.py`, a `baseline.py` to measure it against,
and a `test_harness.py`. Three facts about it are worth knowing before anyone
edits or removes it, because none is visible from the code.

It **ships**. `[tool.setuptools.packages.find]` says `where = ["src"]`, so
`pip install` puts `quisce` on the path beside `langgraph_agent` -- unlike
`spectral_graph/`, which lives at the root precisely so it does not. Importing
it pulls in torch, which is why nothing imports it at module scope.

Nothing in this project imports it at all, and no test covers it. The pytest
suite does not touch it, and CI reaches it only through `ruff`, which lints
`src/` whole. `mypy` names `src/langgraph_agent/` and `serve.py`, so this
package is not type-checked by anything.

Its specification is gone. `from quisce import QuICSEModule` was documented in
the behavioural-system spec, deleted on 2026-09-09 with the other spectral
write-ups, so the package's own `__init__.py` is now the only description of
its entry point. That is the thing to fix first if it stays.

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
- **`self_healing` is a standalone utility, not wired into any node.** It gives
  `retry_with_backoff`, `circuit_breaker` and `self_healing_wrapper` decorators
  (tenacity + pybreaker) plus a structured `SelfHealingLogger`, for a caller
  that wants in-process retry/circuit-breaker resilience around its own
  function calls. It is deliberately **not** applied to the LLM seat calls or
  MCP tool calls: those already have a resilience design of their own —
  `LLM_TIMEOUT_SECONDS` / `NODE_DEADLINE_SECONDS` / `BUILDER_DEADLINE_SECONDS`,
  and the rule that work run under a deadline must never write to state (see
  Important Notes below). Adding a retry loop underneath an abandoned,
  still-running worker would violate that rule outright, and a circuit breaker
  keyed on a seat's exceptions would interact with `_seat_failures`
  (`get_agent_status()`) in ways nobody has designed for. Reach for it for a
  *new*, independent integration point (e.g. a future external API call a
  Builder tool makes) rather than retrofitting it onto the existing node/seat
  call paths.

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

### The Engineer panel is one column

The Action Feed that sat to the right of the Director is gone. It was a third
telling of one story: the stage cards repeated what the Director reports and
what the Crew lights already say, and `run_progress.active` drives those lights
from the seat *currently* executing, which is strictly better than the feed's
`node` -- that names the seat which just finished.

Two things it held were not repeated anywhere, so they moved rather than went.
The **clock** and the last finished stage now live in a single `#run-live` line
inside the Director, rewritten in place each second and replaced by the verdict
when the run ends. That line has to keep naming the last stage, not just the
elapsed seconds: showing only a counter made a working run and a wedged one
look identical, which is the whole reason the feed listed stages at all. The
**"changed this machine"** notice is now a `.touched` block on the verdict
message itself -- it is a safety notice about files on disk, so it belongs with
the result someone reads rather than in a pane beside it.

`stageCard` and the `.stage` CSS went with the column; the Clear button's
tooltip no longer promises to clear a pane that does not exist.

### The console API
The console drives a single `POST /rpc` taking `{method, params}` and returning
`{result, elapsed_ms}` or `{error: {message}, elapsed_ms}`. Methods live in
`RPC_METHODS` in `serve.py`. `export_corpus` and `clear_corpus` go through it
like everything else rather than getting a file-download route: a failure then
lands on the console's telemetry path instead of replacing the page with a JSON
error, and the browser builds the file at the other end. `upload_document` goes
the other way through the same door for the same reason -- the console reads
the file and posts its text, rather than a multipart route being added -- and
takes one document per call, so one PDF among the markdown fails on its own
instead of taking the batch down. The `/api/*` routes are compatibility wrappers over
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

## Documentation claims are tests

`tests/test_claims.py` exists because prose goes stale silently. On 2026-09-09
this project found four claims that had been true when written and had stopped
being true -- a comment saying `graphrag_server.py` sat ~250 characters under
the indexing limit when it had 28,000 to spare, a seat diagnostic quoting a
relevance floor that no longer matched the constant, this file asserting the
twenty best-connected entities were all real terms when three were
sentence-openers, and twelve citations pointing at reports deleted that
morning. Every one was found by someone reading carefully, which is why each
had survived being read many times. The rule now: **a claim worth writing down
is a claim worth failing a build over.**

Most of those guards *recompute* the claim from the repository, so they need no
maintenance and catch drift nobody predicted: every path cited in prose or a
docstring resolves, the Project Structure tree above matches the tree on disk
in both directions, the seat table matches `DEFAULT_SEATS`, the documented
Python floor matches `pyproject.toml`, no walked file exceeds
`MAX_INDEXABLE_BYTES`, and CI runs the same commands the Quick Reference does.

Two are different and the difference is the interesting part. A figure quoted
in prose is checked against its constant through a small registry, because the
real fix for a figure is not to restate it -- `scripts/diagnose_seats.py` reads
`RETRIEVAL_RELEVANCE_FLOOR` rather than quoting it, and markdown that cannot do
that gets a registry entry instead. And the entity audit **cannot** be a rule:
`Measured` sits at 4 position-free capitals against 34 forced, `Spectral` at 2
against 24, and no threshold separates the sentence-opener from the domain
term. That is this file's own rejected-heuristic finding met one level up. So
the top-twenty entity list is *pinned* rather than judged: the test fails when
the graph's answer moves, which is not a bug report but the audit asking to be
redone at the one moment the answer changed.

The guards are proven by breaking each claim and watching the test fail. A
guard that cannot fail pins nothing, and the first version of the entity guard
was asleep -- it fired only at zero free capitals, and `Measured` had acquired
four the moment this file described the problem.

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
  `_gather_research` calls GraphRAG first and, whenever the top hit clears
  `RETRIEVAL_RELEVANCE_FLOOR`, formats those chunks straight into the findings
  and returns without invoking the seat at all. So on a question the corpus answers well,
  the Researcher's model is not a variable: two different models produce
  byte-identical `research`, in ~0.0s. This is worth knowing before blaming or
  crediting a Researcher seat for a run's quality, and it is why the seat's
  model matters most in precisely the case that is hardest to notice -- a
  question the corpus cannot answer, which is also when a silent seat sends the
  run round the loop. `scripts/diagnose_seats.py` keeps the two apart: the
  `research` exercise measures retrieval, `offcorpus` is the one that reaches
  the model, and the phase 1 probes stub retrieval out entirely.
- **`RETRIEVAL_RELEVANCE_FLOOR` is a property of the embedding model, and the
  number it replaced was inside the wrong population.** The floor is what
  `_gather_research` reads off `results[0]["score"]` to decide whether the
  corpus answered at all. It was `0.3`, hard-coded beside the comparison in
  `nodes.py`, and a cosine similarity has no absolute meaning -- so it now
  lives beside `EMBEDDING_MODEL_NAME` in `graphrag_server.py`, where a model
  swap cannot step over it. Measured on this corpus, twelve questions it
  answers against twelve it cannot:

  | population | min | median | max |
  |---|---|---|---|
  | on-corpus | **0.442** | 0.564 | 0.685 |
  | off-corpus | 0.144 | 0.208 | **0.306** |

  Those separate with an empty band from 0.306 to 0.442, and `0.3` sat at the
  top of the wrong one. The single question that crossed it is this project's
  own off-corpus probe -- the `offcorpus` exercise on PostgreSQL vacuum, which
  scored **0.306** and was therefore served to the Builder as though the corpus
  had answered it. That exercise exists because it is *"the only team exercise
  where the Researcher's model is the variable"*, and the gate had been quietly
  denying it that for as long as the number stood: it was measuring retrieval
  and reporting on the seat. `0.37` is the midpoint of the empty band, chosen
  the way `EIGENGAP_DECISIVENESS` was -- a value in open space rather than on
  an observed boundary. The table above is measured through `search` **as it
  now ships**, hybrid re-rank included, and that matters: the re-rank promotes
  the chunk the lexical half also likes, which is often a better answer
  carrying a slightly lower cosine, so the on-corpus minimum fell from 0.492 to
  0.442 as retrieval got better. A floor left calibrated against dense-only
  ordering would describe code that no longer runs. The test pins it against the two measured populations
  rather than against the literal number, because sliding it back under 0.306
  would show up nowhere else: the run completes and the findings look like
  findings.
- **Search is hybrid, and BM25 re-ranks the dense window rather than
  retrieving beside it.** A dense embedding is a poor instrument for "this
  passage contains this exact rare identifier", which is what a plan naming
  `BUILDER_DEADLINE_SECONDS` is really asking -- the model was trained to put
  *similar* passages together, and every seat-timeout constant here is similar
  to every other one. Measured through `search` against the real store, on 541
  identifiers each defined in exactly one project file: the defining file came
  first **53.4%** of the time on dense alone and **65.1%** with the re-rank
  (McNemar p < 0.001, 85 fixed against 22 broken). Recall at 5 moved 93.0% to
  95.4%, which says where the gain is -- the right file was nearly always
  retrieved and simply not ranked first. Prose improved too, 66.0% to 68.2% on
  400 held-out passages, so this is not a code-search special case bought at
  the expense of ordinary questions.
  Three decisions in `lexical.py` are not interchangeable with the obvious
  alternatives. *It re-ranks rather than retrieving in parallel*: full fusion
  over the whole corpus measured 67.3% against 66.2%, six queries in 541 and
  inside the noise, and it costs the thing that matters -- a document only the
  lexical side found has no dense score, and the relevance floor is read off
  exactly that field. Every result therefore still comes from the dense window
  and still carries the cosine that window gave it, so the floor's calibration
  survives. *Identifiers are indexed whole **and** in pieces*, because emitting
  only the pieces makes `NODE_DEADLINE_SECONDS` and `BUILDER_DEADLINE_SECONDS`
  near-identical -- the confusion the lexical half exists to resolve. The
  acronym seam is part of that: without it `GraphRAGKnowledgeBase` splits to
  `graph` + `ragknowledge` + `base`, and the word `knowledge` is absent from
  the index entirely. *And ranks are fused, not scores*: a BM25 score is
  unbounded and corpus-relative while a cosine sits in [-1, 1], so any weighted
  sum needs a scaling constant that is really a third hyperparameter, tuned on
  one corpus and wrong on the next.
  The index is built from the store on first use and **dropped** by
  `add_document` and `clear` rather than amended -- rebuilding over ~1,200
  chunks takes 0.17s, and an index maintained in parallel with Chroma is a
  second account of the same corpus, free to disagree with it. A store that
  cannot answer `get` falls back to dense-only instead of raising: the lexical
  half improves an ordering that is already correct without it.
- **The Builder gets the whole retrieved passage, from every result.**
  `RESEARCH_RESULTS` feeds both the `top_k` the Researcher asks for and the
  slice it forwards, so the two cannot drift: the search used to request five
  results and hand on three, discarding two passages that had already been
  retrieved, ranked and re-ranked -- and the *diverse* two, since
  `SEARCH_ESCALATION` widens the window precisely to stop one file taking every
  hit. `RESEARCH_SNIPPET_CHARS` then caps each passage at 1,500 characters
  rather than the old 300.
  The 300 was the worse of the two, because of *which* 300 it kept. A chunk is
  the unit retrieval judges -- it is selected because it matched, and the
  matching sentence sits anywhere inside it -- so keeping the opening third
  keeps the part with no particular reason to be relevant. Measured on this
  corpus, chunks run to a mean of 882 characters and a p99 of 1,373, so 300
  carried **34%** of a typical passage. On a `research/web` document the
  opening is the provenance header, and there the loss was total: the
  top-scoring source of the 2026-09-09 run reached the Builder as a title, a
  URL, a retrieval timestamp and the goal it was fetched for, truncated
  mid-word, with not one character of the article attached. The Builder was
  handed a citation and no evidence. 1,500 clears the p99, so a passage
  normally arrives whole and the cap guards against a pathological chunk;
  measured across three questions the Builder now receives ~4,900 characters
  from five sources against 900 from three. A cut that does bite is announced
  in the text (`_research_snippet`), for the reason `_fit_to_index_limit`
  writes its note onto the page: a silent trim is indistinguishable from a
  source that had nothing more to say, and the Builder cannot ask.
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
- **A command's timeout is a budget, not a verdict.** `terminal_execute`
  defaulted to 30 seconds while `run_tests` got 600, and `BUILDER_TOOLS` offered
  only `command` -- so the Builder could not raise it and was never told the
  limit existed. This project's own `scripts/verify_and_test.py` finishes clean
  in ~33s and was reported `FAILED` for the difference, which is
  `MIN_VERIFY_SLICE_SECONDS`' false accusation one level up: the Builder is
  handed a working script labelled broken and spends its next turns repairing
  code that was never wrong. `TERMINAL_TIMEOUT_SECONDS` is 60, matching
  `VERIFY_TIMEOUT_SECONDS` -- both bound a single command a seat is waiting on
  -- and `timeout` is now in the schema with the cap named in the description,
  since a limit a seat cannot see is one it reads as a failure. A requested
  value is clamped by `TERMINAL_TIMEOUT_MAX_SECONDS` rather than trusted: a
  tool call is never abandoned, so an unbounded request does not overrun a
  deadline, it hangs the pass past every deadline there is. Raising the default
  does not make every script runnable inside a pass, and is not meant to --
  `example_usage.py` is a full 4-agent run at ~256s, past
  `BUILDER_DEADLINE_SECONDS` (240) entirely. That is a thing to run outside the
  Builder, not a number to raise.
- **`terminal_execute` has no shell, which is what lets it accept `;` and
  `()`.** It was `shell=True` behind a whitelist of permitted characters, and
  the whitelist refused the ordinary way to write a one-liner: `python -c
  "import x; print(x.y)"` trips on `;` `(` `)`, as does any path containing
  parentheses -- and this project has those
  (`examples/filter_band_pass_(40_60_hz).png`), so a runnable file named that
  way could never be verified and would sit in `failed_verification` forever.
  Meanwhile the filter admitted a bare `rm -rf /` without complaint: it never
  guarded against a destructive command, only against chaining one onto
  another. The command is now `shlex.split` into an argv list and run
  directly, so the shell metacharacters are inert *data* -- `echo hi; rm -rf
  /` runs `echo` with four literal arguments, which is a stronger guarantee
  than refusing the string was, and a test asserts the canary file survives
  rather than asserting the refusal.
  The trade is that shell *features* are gone rather than rejected -- and for a
  while they were also silent. A pipe was accepted and meaningless, handed to
  the program as the literal argument `|`, which is worse than a refusal: the
  Builder reads the strange result as the command failing and repairs the wrong
  thing. `SHELL_OPERATORS` now refuses an operator that stands alone in argv
  and says which one it was (`_shell_operator_error`). The measured case:
  `wc -l notes.md && tail -50 notes.md` gave `wc` the arguments `&&`, `tail`
  and `-50` and came back `wc: invalid option -- '5'`, never having counted the
  file it was handed, with nothing in the message naming the chain.
  **This is not the character whitelist coming back**, and the difference is
  where it runs. The old filter scanned the raw string, so it refused
  `python -c "import x; print(y)"` over a `;` that was never syntax. This runs
  *after* `shlex.split` and matches whole tokens, so an operator inside a
  quoted argument is untouched and a filename containing one is still a
  filename. It is also not a safety mechanism -- the operators were already
  inert -- so it stays a refusal *before* the spawn, and the guarantee remains
  the canary file, not the rejection.
  Two things follow from matching tokens. A *spaced* operator is caught;
  `echo hi; rm -rf /` splits to `['echo', 'hi;', ...]`, so that `;` rides on
  `hi`, stays inert, and is what the canary test still pins -- this catches the
  shapes a caller writes on purpose, not every shape that exists. And a literal
  `&&` can no longer be passed as an argument even quoted, since `shlex` leaves
  the two indistinguishable; nothing here needs that. Redirection is answered
  with `filesystem_write`, the way `cd` is answered with `cwd`, while chaining
  has no replacement and is not given a fake one.
  **A redirect glued to its target is caught too** (`_GLUED_REDIRECT`):
  `shlex` keeps `2>/dev/null` as one token, so on the run of 2026-09-10 it
  reached `find` as a path. Output redirection only -- a glued `<` cannot be
  told from `grep "<div>"`, and `>=` is a version bound -- at the cost `&&`
  already pays: an argument really beginning with `>` cannot be passed. A glob
  or `~` cannot be refused, since `find -name "*.py"` needs the literal, so
  `_unexpanded_hint` names one only when the program quotes it back in its
  own error (`cat: 'dir/*': No such file or directory`).
  Two failures the shell used to fold into a return code are
  now surfaced by name: unbalanced quotes cannot be parsed (`shlex` raises,
  and the message says to check the quoting), and a missing program raises
  `FileNotFoundError` rather than returning 127, so the error names the
  program instead of leaving "not found" to read as a missing file argument.
  The other feature that had to be *replaced* rather than merely lost is
  `cd`: it is a builtin, not a program, so `cd there && python x.py` does not
  run in the wrong directory, it fails with `Command not found: 'cd'` -- and a
  Builder with no other spelling to try burns turns rediscovering absolute
  paths, which is what it did on the run of 2026-09-08. `cwd` is in the schema
  for the reason `timeout` is: the tool honoured the argument while
  `BUILDER_TOOLS` offered no way to ask for it. It is validated by
  `_resolve_cwd` **before** the subprocess rather than left to raise, because
  a missing directory raises the same `FileNotFoundError` a missing program
  does and would be answered `Command not found: 'python'` -- naming the one
  thing that was fine -- while a `cwd` pointing at a file raises
  `NotADirectoryError`, which is not a `FileNotFoundError` at all and falls
  through to a bare errno string. A relative `cwd` resolves against the
  project root, the base the filesystem tools already use, and `~` is not
  expanded: there is no shell here, and a `cwd` that expanded what an argument
  on the same line would not is a worse surprise than a refusal naming the
  path.
  **The schema was not enough on its own, which is measured rather than
  assumed.** Re-running the same goal after `cwd` shipped, the Builder reached
  for `cd there && ...` three times before using the argument that replaces
  it, then used `cwd` on all eleven of its remaining terminal calls and never
  went back. A description is consulted before the turn; an error is read at
  the moment the mistake is made, so `_missing_program_error` names the
  replacement in the `FileNotFoundError` path. Only the `cd` family gets one:
  `export` and `source` are equally builtins with nothing here to replace
  them, so they get the fact that ends the retry instead of an alternative
  that does not exist. And the report line carries `[cwd=...]`, because the
  Architect rules on that report and `find . -type f -> ok` names no
  directory -- once the Builder passes `cwd`, the command alone stopped
  locating anything, since the same relative command means a different thing
  in every directory it could have run in.
  A failed line also says why (`_failure_reason`): eleven of forty-seven
  calls on the 2026-09-10 run read only `failed`, eight of them correct
  refusals of shell syntax. That run is also why `prompts/builder.txt` names
  the shapes outright -- the Builder sent calls in batches that all failed
  before one refusal could be read, and the system prompt is read before the
  batch.
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
- **A document is embedded in chunks, because the model's window is 256
  tokens and silence is how it says so.** `add_document` used to embed a whole
  file in one `encode()` call while `MAX_INDEXABLE_BYTES` allowed 100 KB, so
  everything past roughly the first thousand characters was discarded --
  without an error, a warning, or a counter that moved. Measured before the
  fix: **73 of 77 documents truncated, 224,809 tokens present and 19,147
  embedded, 91.5% of the corpus unreachable by search**, with the vector for
  all 46,094 characters of `CLAUDE.md` bit-identical (cosine 1.000000) to the
  vector for its first 1,000. Two failures followed and neither announces
  itself. Retrieval acquired a **length bias**: a short file is fully
  represented while a long one is represented by its preamble, so the file that
  answers the query loses to a shorter one that merely mentions it -- *"how
  does connectivity report isolated nodes and lambda_2"* returned an examples
  script rather than the `graphrag_server.py` holding that exact docstring. And
  scores sat low enough that plan-shaped queries fell under the relevance gate in
  `_gather_research`, discarding retrieval and sending the run to the
  Researcher's model -- the loop *"A silent Researcher is not research"*
  already describes. After chunking, the same ten queries average **+0.162**
  and none falls under the gate.
  Four decisions in it are not interchangeable with the obvious alternatives.
  *`CHUNK_MAX_TOKENS` is 254, not 256*, because the model adds two special
  tokens. *Boundaries are not snapped to line breaks*: snapping must either
  shorten a chunk, dropping tokens the model could have read, or lengthen it
  past the window, which truncates again -- so the cut stays where the
  arithmetic puts it and `CHUNK_OVERLAP_TOKENS` absorbs the cosmetic cost.
  *Chunks are re-encoded and trimmed by `_fit_chunks`* rather than the constant
  being padded by the drift observed once: slicing on the parent's token
  boundaries does not bound the slice, because a word-piece tokenizer decides
  on context, and 16 of 1,052 full-size chunks came back a token longer
  standalone -- 257 against a 256 window, handed straight back to the bug. Which
  end is trimmed is load-bearing: a chunk's tail is covered by the next chunk's
  overlap and its head by the previous one's, so the **last** chunk is trimmed
  at the head and every other at the tail. Trimming the last one's tail would
  drop the final tokens of the file silently, which is the original bug in
  miniature. *And `search` collapses chunks back onto documents*, keeping each
  document's best chunk, so `id` stays the path callers index the graph with
  and print as a filename while `content` becomes the passage that matched --
  the 300 characters the Researcher forwards to the Builder used to be 300
  characters of whichever file's *header* scored best. It widens the query once
  (`SEARCH_ESCALATION`) when one large file monopolises the first window of
  hits, since answering a request for five sources with one is narrower than
  what chunking replaced. Rows are keyed `path#0007`; `_document_id_of` resolves
  one back to its document and deliberately still resolves a bare, unsuffixed
  id, because that is what a store written before chunking holds and reading it
  as a stranger would prune it.
- **A document's previous chunks are deleted before its new ones land.** Not
  left to `upsert`: a file that shrank between reindexes -- ten chunks down to
  three -- overwrites 0..2 and leaves 3..9 in the store, still matching queries
  with text the file no longer contains. That is the same "a reindex rebuilds
  rather than accumulates" rule `index_project_files` follows for whole
  documents, one level down. For the same reason its stale-pruning compares the
  *document* a row belongs to and never the row's own id: `CLAUDE.md#0007` is
  not in `wanted` and never will be, so an id-to-id comparison finds every
  chunk stale and deletes the corpus on every reindex. It happens to end in the
  same place while everything is re-added in the same pass, which is exactly
  why it would sit there unnoticed until something indexed a subset.
- **A capital is not evidence of an entity, and `ENTITY_STOPWORDS` is a list
  rather than a rule.** `add_document` mints an entity for every capitalised
  token over four characters. Across prose and numpy-style docstrings that
  fires constantly on words whose capital comes from where they sit rather
  than what they mean: a sentence opener (`Every`, `Nothing`), a docstring
  heading (`Returns`, `Parameters`, `Raises`), a report heading (`Files`,
  `Status`), a Python literal (`False`, `None`). Before the list, `False` was
  the **4th** best-connected node in the graph and `Returns` the **6th**, above
  `Fiedler`, `Planner` and `Cheeger` -- 21 documents joined through `Returns`,
  which records only that all 21 contain a docstring. Those edges are read as
  evidence by `neighborhood()`, by `top_entities`, and by
  `duplicate_entities()`. After the list the twenty best-connected entities are
  all real terms and no stopword token remains in the graph.
  **That claim expired once, and the way it expired is the thing to remember.**
  A hand-audited list is audited against a *vocabulary*, and this corpus's
  vocabulary kept growing -- largely through prose written into this very file.
  By 2026-09-09 `Tests`, `Measured` and `System` were the 6th, 8th and 10th
  best-connected entities: the original failure exactly, one vocabulary later.
  `Measured` is the sharpest case, because CLAUDE.md opens sentences with it
  twenty-nine times, so the prose recording these measurements was minting the
  entity. Thirty tokens were added and the claim holds again.
  The audit that chose them is worth repeating rather than re-deriving. For
  every candidate, count the capitals that **position does not explain** --
  not at a line start, not after a full stop, not the first cell of a table
  row. Zero of those means the token is recording where it sits. That is the
  same heuristic this bullet rejects below, used the only way it is sound: to
  *nominate* candidates for a human to rule on, never to filter. Two
  nominations were refused on exactly that reading, and they are why this
  stays a hand audit. `L_dense` scores zero free capitals because an
  assignment starts its line -- it is a real identifier. `Spectral` scores
  two, both marginal, and stays because it is a term the corpus is about.
  `System` was nominated by rank and cleared by the count at 22 free capitals,
  which is an entity that earned itself. Re-run the count before trusting the
  list again; it is a claim about a vocabulary, and vocabularies move.
  **The obvious alternative was built, measured and rejected.** Dropping a
  token that only ever appears where a capital is forced (line start, after a
  full stop) removes the same noise and severs real edges doing it: `Planner`
  fell from 18 documents to 15, `Spectral` 19 to 15, `ValueError` 14 to 11,
  because a term introduced in a bulleted list (`- **Planner** -- interprets
  goals`) appears nowhere else in that document. A hand-audited list costs no
  meaningful term a single edge, and its failures are visible on the page
  rather than buried in a heuristic. The list is matched case-insensitively
  against the **whole** token, never as a prefix, so `Returns` is stopped while
  `Researcher` is not.
  **It does not make `topics()` decisive, and was never going to.** That was
  the first hypothesis and the measurement refused it: removing every one of
  the 97 entities that bridge this corpus's two topic areas -- the most any
  such filter could achieve -- moves the eigengap decisiveness from 1.12x to
  1.09x, and the list itself moves it to 1.22x against a 3.0x threshold. The
  corpus's normalized spectrum is a smooth continuum (0, 0.105, 0.125, 0.145,
  0.175, ...) because it genuinely has no decisive k, not because boilerplate
  is gluing it together; the same extractor recovers planted topics at 14-25x
  on a synthetic corpus that does have them. `no_clear_structure` on this
  corpus is the correct answer, and tuning the extractor until the number rose
  would have manufactured the finding `topics()` exists to refuse.
  What the list does not touch either is the **degree-1 majority**: 60% of
  entities are mentioned by exactly one document, contribute no relation, and
  are what floods `duplicate_entities()` with structurally identical pendants.
  That is a separate defect with a separate fix.
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
- **The A-numbers, and where they went.** `connectivity()`, `topics()`,
  `bottleneck()`, `neighborhood(split=True)` and `duplicate_entities()` are
  referred to throughout as the A1-A5 applications. That numbering came from
  `reports/spectral_applicability.md`, which ranked five uses of the graph
  spectrum by what they would cost and what they would prove, and from
  `reports/spectral_architecture_benchmark.md`, which measured the eigengap
  heuristic and the Cheeger bracket across eight graph architectures. **Both
  were deleted on 2026-09-09, with the rest of the spectral write-ups**, and
  the citations that pointed at them were removed rather than left dangling.
  The labels stay because five methods and their tests are named by them and
  renumbering proves nothing. Nothing consulted at read time was lost with the
  files: every measurement those reports supplied and this file relies on is
  quoted inline where it is used -- the Cheeger bracket's 4x to 546x, the
  eigengap wrong on 3 of 8 architectures, the 5.1-23.4 decisiveness range on
  planted topics against 1.0-1.8 on structureless graphs. The code they
  justify is untouched: `spectral_graph/`, `corpus_spectral.py` and
  `test_spectral_graph.py` all remain.
- **`stats()` carries a structural health check, and it counts components
  with networkx rather than with the spectrum.** `connectivity()` reports
  `components`, `largest_component`, `isolated_nodes` and `lambda_2` -- the A1
  application (see *the A-numbers* below). It exists because an
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
  having.** The A3 application: sweep
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
  546x wide across the eight architectures the benchmark covered, so the sweep
  genuinely can
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
  decisive.** The A2 application:
  Ng-Jordan-Weiss clustering over the normalized Laplacian, which on a
  bipartite document/entity graph groups documents with the entities that
  define them -- so a cluster reads as a topic and `top_entities` names it.
  This is the whole-corpus map `neighborhood()` cannot give.
  The proposal's weak point was `k`, and it is not wired straight through.
  The architecture benchmark measured the eigengap heuristic
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
- **`neighborhood(split=True)` names the sides for the centre, and the flag is
  off by default.** The A4 application: one eigenvector, the cheapest of the
  five, splitting a traced neighbourhood into
  what clusters with the node you looked up and what is peripheral to it. Off
  by default because it costs 1.7-3.3ms against the 1.0ms `neighborhood()`
  itself takes -- it triples a call the console makes on every click, and a
  caller who only wants the node list should not pay for a division it will not
  draw. The sides are `center` / `other` rather than the sign of the
  eigenvector, which is arbitrary: an eigenvector and its negation are equally
  valid, so a raw sign swaps the halves between runs for nothing. `mu_2` is
  returned because the split is always *available* and only sometimes
  *meaningful* -- a sign cut exists for any connected graph. A trace that stays
  inside one topic area reads ~0.2 and has divided it arbitrarily; one spanning
  two reads an order of magnitude lower. The caller gets the number, not a
  verdict, because the line worth drawing depends on the use. Runs on the
  centre's component: `min_degree` pruning **can** disconnect the sweep
  (measured at depth 3, min_degree 3), and on a disconnected graph the Fiedler
  vector is a component indicator, so the "split" would be that stray node
  against everything else. Detached nodes get `side: None`. In the console the
  side rides on opacity, never colour -- colour already carries
  document-vs-entity and that is what the graph is mostly read for.
- **`duplicate_entities()` proposes merges and never makes one, and names
  generate the candidates while structure is only the evidence.** The A5
  application: `add_document` mints an entity per capitalised token, so one
  thing arrives under several names -- `Builder` and `BUILDER` from a heading,
  `Entity` and `Entities` from a plural.
  **This used to rank by distance in a spectral embedding, and the measurement
  retired that.** On this corpus it returned 33,060 candidates of which **67%
  sat at distance exactly 0.0000 with neighbourhood overlap 1.00** -- its own
  "strongest merge evidence there is" -- topped by `['LEGAL', 'Virginia']` and
  `['Consequences', 'PIPEDA']`. Those are pendant collisions: two entities each
  mentioned by one document, the same one, are structurally identical by
  construction, and 60% of this corpus's entities have degree 1. The true
  duplicates -- `Builder`/`Builders` and all thirty case variants -- were **not
  candidates at any rank**. Graded against ground truth, spectral distance
  scored 0% precision and 0% recall; so did Jaccard, and so did containment as
  a ranker. Name similarity scored **100% precision on its top 20**.
  Two separate things were wrong. *A real duplicate is asymmetric*: `Builder`
  has 29 documents to `Builders`' 4, a subset that reads containment 1.00 and
  Jaccard 0.14, so every symmetric measure ranks the true pair below thousands
  of unrelated ones -- which is why the evidence here is **containment**, with
  Jaccard kept beside it precisely so the disagreement stays visible. *And
  structure cannot generate candidates on a real corpus at all*: tightened as
  far as it goes -- Jaccard 1.00 with at least three shared documents -- the
  survivors are `Oppenheim`/`Schafer` (two authors cited in the same three
  papers), `Nyquist`/`Frequency`, and
  `BUILDER_DEADLINE_SECONDS`/`NODE_DEADLINE_SECONDS`. All collocations. The
  synthetic fixture that once justified the structural signal assigns entities
  to documents **at random**, which makes an identical neighbourhood
  astronomically improbable and therefore real evidence; a corpus is the
  opposite, since entities of one topic are mentioned in the same documents --
  that is what a topic *is*. The property the test depended on is the property
  real data lacks. So the old note's "structure beats names here and names
  actively mislead" was true of the fixture and false of the corpus.
  **What this gives up, on purpose: two names for one thing sharing no
  characters.** `LanguageModel` for an entity already called something else is
  not found and cannot be. That is pinned by a test, because it was *measured*
  rather than merely unimplemented -- anyone reinstating a structural generator
  should re-run that measurement first.
  Pairs come in two kinds and they differ in how much they must prove. `case`
  (the same token bar capitalisation) is certain on the name alone and carries
  **no** structural floor -- these are the most asymmetric pairs in the corpus
  (`BUILDER` in one document against `Builder` in 29) and any evidence
  threshold would drop every one of the thirty. `lexical` is a likeness, so it
  must also clear `DUPLICATE_CONTAINMENT`; that is what separates
  `Builder`/`Builders` from two similar names for different things.
  Comparison is blocked on a case-folded four-character prefix: 1,528
  comparisons instead of 948,753, which took the scan from 33,060 pairs in
  1.34s to 95 in 0.016s. It only ever proposes: which entities mean the same
  thing is a decision about meaning that the graph cannot make.
- **The embedding model loads on first use, not on construction.**
  `GraphRAGKnowledgeBase.embedder` is a lazy property and
  `sentence_transformers` is imported inside it. Only `add_document` and
  `search` embed; counting the corpus, listing its documents, drawing its graph
  and exporting it do not, and those are what the console does on a timer.
  Loading it in `__init__` meant every header poll paid for the model and
  importing the module pulled in torch behind it.
- **Online research costs nothing, and that is a constraint rather than a
  happy accident.** `web_research` reaches the internet through DuckDuckGo's
  keyless HTML endpoint, or a SearxNG instance the operator hosts
  (`SEARXNG_URL`). No API key, no account, no per-call charge — every general
  web search API that would be easier charges money, including the ones with a
  free tier, and this project does not buy services. The result list is the
  *only* thing taken from outside: ranking, extraction and selection are all
  in-tree, which is why `html_text` is a from-scratch reader rather than a
  dependency.
  The phase has four stages and the last two are why it is ours.
  `expand_queries` fans a goal into several searches using `lexical.tokenize`,
  so a goal naming `BUILDER_DEADLINE_SECONDS` also searches
  `builder deadline seconds` — a search engine has seen the second and never
  the first. Those orderings merge through `reciprocal_rank_fusion`, the same
  function `search` merges its dense and lexical halves with, so a page several
  queries agree on outranks one that topped a single query. `html_text.extract`
  reads each page by **link density plus block length**: a nav strip is short
  and almost all anchor text, an article is long and almost none, and a block
  is dropped only when *both* tests fail it — headings are exempt from the
  length test, being short by nature. It reports `kept`/`dropped`/`fell_back`
  because an extractor that cannot be graded would be the one unmeasured link
  in a chain where the floor, the chunker and the duplicate scan were all
  tuned against printed numbers.
  Then `select_pages` decides what is *embedded*, which the engine's rank
  deliberately does not. Every fetched page is scored with `BM25Index` against
  the goal and kept only above a **ratio of the best page's score** — never an
  absolute, for the reason `lexical.py` already gives: a BM25 score is
  unbounded and corpus-relative, so a constant is a hyperparameter fitted to
  one set of pages and meaningless on the next. Zero is the single absolute
  floor, because zero means no shared term at all. This gate is the answer to
  the problem the phase creates: a web document competes with the checkout's
  own files at retrieval time, under a `RETRIEVAL_RELEVANCE_FLOOR` calibrated
  on a corpus containing none of them.
  A fetched page is written under `research/web/` **before** it is embedded,
  the same ordering and for the same reason as `store_uploaded_document`, and
  that directory must stay inside `PROJECT_INDEX_PATTERNS` and out of
  `PROJECT_INDEX_EXCLUDES`. Filenames are deterministic
  (`host-path-<8 hex>.md`) so re-researching a topic overwrites in place rather
  than growing the corpus by a near-identical document per run. Provenance —
  URL, title, retrieval time, originating goal — goes in the document *text*
  and never in the metadata: metadata is `_document_metadata`'s two keys,
  shared with `index_project_files`, so a third key set only on this path would
  vanish the moment a reindex re-read the file from the walk.
  `search_web` distinguishes `disabled`, `error` and a genuine empty result,
  and `research_online` reports `considered` beside `documents` — a phase that
  read twelve pages and kept none is working correctly on a goal the web has
  nothing to say about, and must not read as one that failed.
  **Every query fits `WEB_QUERY_MAX_CHARS`.** DuckDuckGo refuses a long query
  with `302` to its `/50x.html` page, which `raise_for_status` reported as a
  bare status: on 2026-09-10 a paragraph-length goal sent 1,227-, 816- and
  840-character queries and every search failed. The measurements behind the
  default are at the constant. **Its bot check is a `202`** with no results,
  which parsed as a genuine empty web; `_search_duckduckgo` recognises it by
  its markup and raises `_SearchBlocked`, and the fan-out stops there, since
  each further request prolongs the block.
  **It runs before the Architect opens, and that ordering is the design.**
  `rpc_run_goal` calls `_research_online_before_the_run` after claiming the run
  lock and before `graph.stream`, so the corpus is whole by the time any seat
  searches it and does not move again for the rest of the run. This is not a
  way around `_refuse_while_a_run_is_in_flight`; it is the only ordering that
  obeys it. Measured end to end on 2026-09-09: 12 pages read, 8 embedded, 174
  passages, 19.6s, and the Researcher then retrieved one of them at 0.74 --
  comfortably over the floor -- with its provenance header intact in the
  matched chunk.
  **It is handed the creating door unopened.** Storing a page is indexing, so
  `_kb_for_indexing` is the right door here -- a goal researched against a
  machine with no corpus should leave one behind holding what it found. But it
  was resolved on the way *in*, as the argument to `research_online`, before
  the phase knew whether it would run at all: so a run left one behind holding
  **nothing**. The phase is off in every test and on any machine without
  `WEB_SEARCH_ENABLED`, and a goal the web cannot answer keeps no pages either,
  so a machine that had never been indexed came out of its first run reporting
  `empty` where the truth was `absent` -- and only one of those two means
  "press Reindex". This is the *"no corpus exists until someone indexes one"*
  rule reaching the one caller that really does index, and failing on timing
  rather than on which door it picked. `research_online` now takes a factory
  and calls it on the first page that earns a place, once for the batch. The
  regression is pinned twice in `test_corpus_absent.py`: through `rpc_run_goal`
  on a fresh machine, and on the factory itself, which must be called never for
  an empty selection and exactly once for a full one.
  The phase reaches the Researcher through **the corpus and nothing else**.
  There is deliberately no path by which a fetched page skips retrieval: it is
  embedded, and the Researcher finds it with the same search, the same hybrid
  re-rank and the same relevance floor as everything else. A page fetched for a
  goal that cannot be retrieved for that goal should not reach the Builder on
  the strength of having been fetched.
  Every failure is swallowed into the report and none of them stops the run --
  a goal the web cannot answer, an engine that is down, a machine with no
  network. The feed line words the empty outcomes apart, because "switched
  off", "we asked and it broke" and "we read twelve pages and none earned a
  place" call for completely different things and an empty count reads
  identically in all three.
  **The suite must never run this.** `tests/conftest.py` switches it off for
  every test, the same way and for the same reason it forces `StubLLM`.
  Without that, every test starting a run fetched a dozen live pages and
  embedded them: it took the suite from 28s to 138s, broke a stop test whose
  five-second wait had been generous, and wrote **31 web pages into the
  developer's own corpus** from goals like `"Do a thing"` and `"a long job"` --
  Bible verses and project-management blogs, indexed as project knowledge.
  `test_web_research.py` turns it back on and answers every request from a
  `MockTransport`.
- **A fetched web page is a retrieval source, not knowledge-graph material.**
  `add_document` skips entity extraction for anything under `WEB_RESEARCH_DIR`
  (`_is_web_document`), decided from the **path** and never from the call site
  -- a reindex re-reads these as ordinary markdown, so a rule applied only
  where a page is first stored would be reversed by the next rebuild, silently.
  The capital rule is far worse on web prose than on this project's files:
  `ENTITY_STOPWORDS` is hand-audited, and it was audited against source code
  and numpy docstrings where the forced capitals are `Returns`, `Every`,
  `False`. Web prose opens sentences with a different vocabulary entirely.
  Measured on 2026-09-09, after a handful of runs: 15 fetched pages had minted
  **551 entities no project document mentions -- 19% of the whole graph, 36.7
  per page** -- `Although`, `Afterward`, `Altogether`, `Again`, `Accessed`.
  Skipping them took the graph from 2,830 entities to 2,285 and duplicate
  candidates from 205 to 171. The page is still chunked, embedded and fully
  retrievable; it simply stops voting on what the entities of this project are.
  The side effect is that such a page has no edges and is therefore an isolated
  node -- which is exactly the symptom `connectivity()` exists to detect, and
  the only place an entity-extraction regression shows at all. So
  `connectivity()` excludes them and reports `web_documents` alongside, because
  15 permanent false isolates would bury the one reading that matters.
- **The staleness verdict is withheld while a run is in flight.** The research
  phase writes each page to disk and embeds it as a separate step, so between
  those two there is a file the walk can see and the store cannot. Observed
  live: the header read `stale: 1 not indexed` mid-phase and cleared itself
  moments later. The counts stay -- they are the truth about that instant --
  and only the accusation is withheld, under `settling`. A verdict that
  flickers teaches the operator to ignore the one that does not.
- **Deleting a file does not delete the document, and git will not tell you
  either.** Two separate gaps, and the 2026-09-09 cleanup hit both. Removing a
  file from disk leaves its chunks in the store answering searches with text
  that is nowhere in the project until something rebuilds -- which is exactly
  what `corpus_staleness`'s `extra` reports, and it did: three entries appeared
  the moment the files were unlinked and cleared on the next reindex. So the
  sequence is delete, reindex, check; not delete and assume.
  The second gap is why this paragraph exists at all. Everything removed that
  day was **untracked**, so the deletion produced no diff, no commit and no
  history -- `git log` will never show it happened. A future reader finding
  these paths referenced in an old report has nothing to consult, which is the
  only reason to write it down here.
  Removed deliberately, and not to be re-added: the bot-detection and
  security-circumvention material (`docs/legal-boundaries.md`,
  `docs/research/bot-detection-techniques.md`,
  `reports/security-circumvention-blocker.md` -- the last being the Architect's
  own refusal of a goal asking for anti-bot circumvention, which is a record of
  the gate working rather than of work done). That line of work is not part of
  this project. Alongside them went the dead run artifacts (`AUDIT_REPORT.md`,
  `CLARIFICATION_NEEDED.md`, `run_exercise.py`, `ai_efficiency_report.md`) and
  the unrelated `files/` deliverable.
  `reports/diagnostics/` went too, and it is the case worth generalising: the
  `.gitignore` already called those sweeps *"per-run measurements against
  non-deterministic models, not history"*, yet five of them were sitting in the
  corpus as indexed documents, because **the walk is a glob and not git**.
  Anything gitignored is still retrievable unless it is also in
  `PROJECT_INDEX_EXCLUDES`, so a transient artifact that outlives its run does
  not merely take up disk -- it answers questions. The corpus went from 122
  documents to 108 on the reindex that followed. `reports/diagnostics/` is now
  in `PROJECT_INDEX_EXCLUDES`, because deleting the sweeps fixed the afternoon
  and not the next `diagnose_seats.py` run; the deliberate `reports/*.md`
  beside them stay indexed, and the test asserts both halves through the walk.
- **The corpus rots in silence, so something has to compare it against the
  project.** `corpus_health.corpus_staleness` is that comparison, and it rides
  on `rag_stats` because the console header is where it will actually be seen.
  It exists because of a failure with no symptom at all: measured here on
  2026-09-09, the store held **8 documents, all of them uploads, against a walk
  offering 103**. Every project query therefore scored under
  `RETRIEVAL_RELEVANCE_FLOOR` -- `BUILDER_DEADLINE_SECONDS` returned an
  unrelated upload at 0.284 -- so `_gather_research` discarded retrieval and
  fell through to the Researcher's seat on *every* run, which is the
  step-burning loop described above. Nothing reported it. `rag_stats` said
  `indexed`, every counter was non-zero and consistent with the others, and the
  full suite passed. The only way to see it was to put two numbers side by side
  that nobody had ever compared.
  Three states, and they are not one request. `missing` is in the walk and not
  in the corpus -- work written since the last reindex. `extra` is in the
  corpus and not in the walk -- a file deleted or renamed, whose text is
  nowhere in the project and still answers searches. `oversized` is in the walk
  and too large to index, so **correctly** absent: it is reported *apart* from
  `stale` because reindexing cannot change it, and a header permanently asking
  for a rebuild that would do nothing is a warning nobody reads.
  Both size tests are exact rather than approximate, and that is the whole
  reliability of the thing: `index_project_files` measures *characters* while
  `stat` counts *bytes*, and a UTF-8 file is never fewer bytes than characters.
  So `st_size <= MAX_INDEXABLE_BYTES` already proves a file indexable and
  settles nearly every file without a read; only the handful above that line
  are ambiguous, and those are read and measured properly. Guessing either way
  invents an accusation -- excluding an indexable file calls a document that
  belongs in the corpus `extra`, including a skipped one calls a document the
  indexer is right to omit `missing` -- and one standing false accusation
  teaches the operator to ignore the whole signal.
  It lives in its own module because writing it inside `graphrag_server` pushed
  that file from 98,920 characters to 104,582, past `MAX_INDEXABLE_BYTES`: the
  module that defines the corpus would have been dropped *from* the corpus at
  the next reindex, silently, as the direct result of adding the check meant to
  catch exactly that. That is why it is a separate module and stays one.
  **The margin is not thin, and the note that said it was never described a
  file that shipped.** "~250 characters under the limit" was written in the
  same commit that split the spectral applications out to `corpus_spectral.py`
  (~35,400 characters), which took `graphrag_server.py` to ~71,000 -- so the
  sentence was wrong the moment it was committed, having measured a state that
  existed only partway through the change. Editing that file is not the live
  hazard it claimed. Both numbers are quoted here to the nearest thousand on
  purpose: a character count in prose goes stale on the next edit, which is
  what happened, and the guard was never meant to be a figure anybody tracks by
  hand. `oversized` is the guard -- it reports a file the walk offers and the
  indexer must skip, apart from `stale` because a reindex cannot fix it.
  The walk is cached for `WALK_CACHE_SECONDS` since the console polls every
  five seconds; `forget_expected_documents` drops it, and
  `rpc_upload_document` calls it because an upload is the one writer that
  changes what the walk would find.
- **`query_graph` traverses undirected, and must.** Every edge in the knowledge
  graph runs **document -> entity**, so an entity has in-edges only and a
  *directed* walk from one reaches nothing. It traversed directed until
  2026-09-09 and returned the entity by itself -- `neighbors` of length 1 --
  for every entity in the corpus: `Planner` has 21 edges and reported none, and
  its real 2-hop neighbourhood is 614 nodes. That is the Researcher's
  `query_knowledge_graph` tool, so one half of retrieval answered every
  relationship question with "this term connects to nothing", which reads as a
  finding rather than as a broken traversal. `neighborhood` already took the
  undirected view and carried the comment explaining why; this is the same fix,
  arrived at late. Both now share `_resolve_node`, so the console and the
  Researcher cannot resolve an id differently.
- Knowledge base files under `knowledge/` (`chroma/`, `knowledge_graph.json`) are runtime artifacts; avoid committing them unless intentionally versioning an index.
- A reindex **rebuilds** rather than accumulates: it clears the graph and prunes Chroma ids that no longer qualify, so excluded or deleted files stop answering searches.
- **A corpus clear empties in place and must reach disk.** `clear()` deletes
  every Chroma id, *then* clears the graph — never the other way round, and it
  raises rather than report a half-wipe as success, because the two halves
  answer different questions and a corpus that disagrees with itself is worse
  than one that is merely stale. The files under `knowledge/` are kept, holding
  an empty store: Chroma has that directory open, and pulling it out from under
  a live client is the worse failure. The trailing `_save_graph()` is
  load-bearing, and `index_project_files` now ends with the same two lines for
  the same reason. It had exactly this hole: its `graph.clear()` and its prune
  of stale Chroma rows were persisted and invalidated only as a side effect of
  `add_document`, so a reindex that added nothing — no matching files, or every
  one of them oversized or unreadable — emptied the graph in memory, reported
  `indexed: 0` and success, and left the old `knowledge_graph.json` for the
  next process start to reload. The corpus it had just rebuilt came back. The
  lexical index went the same way: built before the prune, it went on answering
  with the rows the prune deleted. Both lines are unconditional there, since a
  reindex is the one operation after which that index must be rebuilt anyway.
- **An uploaded document is a file first and a document second, and that
  ordering is the whole design.** `store_uploaded_document` writes the upload
  under `uploads/` and only then calls `add_document`. The corpus is a function
  of what is on disk: `index_project_files` clears the graph and prunes every
  stored row whose document is not in the walk, so a document embedded straight
  into the store and nowhere else survives exactly until the next reindex,
  which then deletes it **silently**, in a pass that reports success and a file
  count that looks right. Writing the file is what puts an upload *inside* the
  rebuild instead of underneath it, which is why `uploads/` must stay out of
  `PROJECT_INDEX_EXCLUDES` — a new entry that merely contains that string would
  end every upload one reindex later, and `test_uploads.py` checks it through
  `iter_project_files` rather than by reading the list, because that is the way
  it breaks. Gitignoring the directory does not hide it: the walk is a glob,
  not git.
  Every refusal is a `ValueError` naming what was wrong, because each
  alternative to refusing is worse than a failed upload and none of them
  announces itself. A `.pdf` has no text extractor here, so accepting one
  embeds whatever its bytes decode to under a real filename and it reads as a
  source in the corpus from then on — the fabricated retrieval hit `search` was
  fixed to stop returning. A file over `MAX_INDEXABLE_BYTES` is embedded now
  and skipped at every rebuild after, so the limit is applied here character
  for character against the walk's own test rather than approximated. A name
  carrying a path writes outside the directory the walk looks at, which is the
  same disappearance by another route, so only the last component is used and
  `..` is refused by name (`PurePosixPath("..").name` is `".."`, not `""`).
  `INDEXABLE_SUFFIXES` is derived from `PROJECT_INDEX_PATTERNS` rather than
  restated beside it, and a suffix is *stored* lower-cased because the walk is
  a glob and a glob is case-sensitive here — `NOTES.MD` written as given is a
  file the reindex cannot see. `_document_metadata` is shared with
  `index_project_files` for the same reason: two spellings of one document is
  two documents, one of them orphaned in the graph at the next rebuild.
  `add_document` returns its chunk count so the console can say what a document
  became; that number is the only thing distinguishing a file that landed whole
  from one the embedder read the header of.
- **Changing the corpus is refused while a run is in flight.** All three
  writers — `clear_corpus`, `reindex` and `upload_document` — go through
  `_refuse_while_a_run_is_in_flight()`. Not for consistency: because the
  failure would be silent. An emptied corpus does not break the Researcher's
  search, it returns no hits; a corpus midway through a rebuild returns
  whatever fraction of itself has been re-added. Either reads as
  `no_relevant_knowledge` and routes the run as though the knowledge base had
  simply had nothing useful to say. Nothing raises and nothing is logged, so
  the seat can never find out it was cut off and the operator is told instead.
  The refusal names the goal, the way `rpc_shutdown` does. `export_corpus` has
  no such guard: reading the corpus takes nothing away from the run using it.
  The upload is guarded for a *different* reason than the other two, and it is
  worth keeping straight: an upload only ever adds, so it could not manufacture
  an absence. What it cannot do is add while a search is reading — it mutates
  the same networkx graph `neighborhood` and the lexical index iterate — and a
  run should be answered by the corpus it started against either way.
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
- **A seat's thinking box says what its next call does.** A switchable model
  is always sent the flag, off included, since omitting it means *on* for
  `qwen3.5:397b-cloud`, Opus 5 and Sonnet 5 -- which is why it defaults on.
  `thinking_support` reads support rather than listing it, and "could not
  ask" stays unknown, never "cannot think". A Claude reply that thought is a
  list of blocks, so nodes read replies through `_as_text`.
- `__pycache__` directories should never be committed (they are removed from git in this repo).

## Troubleshooting

- **Tests are slow** — The first run loads `sentence-transformers` and Chroma. Subsequent runs use the cached singleton.
- **Mypy errors from upstream stubs** — Prefer `# type: ignore[...]` with a comment over disabling strict mode.
- **GraphRAG returns no results** — Check whether there is a corpus at all: the console header reads `no corpus — nothing indexed` when none has been built, and nothing builds one for you. Run `python scripts/reindex.py`, or press *Reindex project*.
- **No LLM output / canned text** — A seat pointed at Anthropic or OpenAI needs that provider's key in `.env`; without one it runs `StubLLM` and the console shows a `NO KEY` chip. No seat uses either by default. The Ollama seats need the daemon running and signed in (`ollama signin`) for `:cloud` tags.
- **A 400 from Anthropic that looks like an auth error** — Check nothing is passing `temperature` to an Opus 5 / Sonnet 5 / 4.6+ model; sampling parameters are rejected on those families.
- **Graph tab is empty** — Run `python scripts/reindex.py` (or press Reindex project on the Corpus tab). A `TypeError` on every insert used to leave the graph empty while the script still reported success; the corpus is only real if `rag_stats` shows non-zero nodes.
