---
name: console
description: Build, run, and drive the Ambiguity 4-agent console. Use when asked to start the console or web UI, run the server, reindex the knowledge graph, take a screenshot of the console, run the tests, or interact with the running app.
---

# Ambiguity 4-agent console

A LangGraph + GraphRAG console: a `ThreadingHTTPServer` on :8080 that serves
the SPA in `frontend/` and answers `POST /rpc` with `{method, params}`. Drive
it with **`.claude/skills/console/driver.py`** — it speaks the same RPC surface
the SPA does, and screenshots the UI with headless chromium (no Playwright, no
xvfb, no browser extension).

All paths below are relative to the project root (`ambiguity2/`).

## Prerequisites

Python ≥ 3.12 (this box runs 3.14.7 via mise; CI runs 3.12 and 3.14) and
`chromium` on PATH for screenshots. Nothing was `apt-get`-installed for this —
chromium was already present:

```bash
python3 --version   # 3.14.7
which chromium      # /usr/bin/chromium
```

**Live agent runs also need the local Ollama daemon**, because every default
seat is an Ollama *Cloud* tag that the local daemon proxies:

```bash
systemctl is-enabled ollama.service   # enabled
systemctl is-active  ollama.service   # active
```

## Build

There is no venv in a fresh checkout. Create one and install **in this order** —
the order is load-bearing:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e ".[dev]"
```

CPU torch **first**, from the CPU index. `sentence-transformers` pulls torch,
and torch from PyPI drags in the whole CUDA stack (19 packages). Installing it
from the CPU index first pins it so dependency resolution cannot reach for the
CUDA build. This ordering is in `.github/workflows/ci.yml`, not the README.

Verify:

```bash
.claude/skills/console/driver.py doctor
```

```
root        /home/darthmav/Work/ambiguity2
interpreter /home/darthmav/Work/ambiguity2/.venv/bin/python
venv        present
import      langgraph_agent OK
chromium    /usr/bin/chromium
server      up
```

## Run (agent path)

```bash
.claude/skills/console/driver.py up                 # start, wait for readiness
.claude/skills/console/driver.py smoke              # end-to-end check, non-zero on failure
.claude/skills/console/driver.py shot /tmp/c.png    # headless screenshot
.claude/skills/console/driver.py down               # clean stop
```

`up` takes ~8s. A passing `smoke` looks like this:

```
  PASS  status responds  embedding=all-MiniLM-L6-v2
  PASS  corpus indexed  corpus=indexed
  PASS  graph has nodes  nodes=1134 edges=2272
  PASS  graph not stale
  PASS  documents listed  n=75
  PASS  semantic search returns hits  top=src/langgraph_agent/graph.py
  PASS  four seats configured  n=4
  note  0/4 seats live -- run_goal will fail until a tag is pulled
  PASS  unknown method returns error envelope, not 500

SMOKE OK
```

Any of the 20 RPC methods directly:

```bash
.claude/skills/console/driver.py rpc rag_stats
.claude/skills/console/driver.py rpc search_documents '{"query":"planner agent","k":3}'
.claude/skills/console/driver.py seats
```

Methods: `rag_stats bottleneck topics duplicate_entities list_documents
query_graph search_documents reindex upload_document export_corpus clear_corpus
list_seats set_seat llm_options status run_goal run_progress stop_run last_run
shutdown`.

### Building the knowledge graph

The corpus is absent in a fresh checkout — the Graph tab draws "No graph yet"
and `rag_stats` reports zeros. **Nothing builds it but a run**, which rebuilds
it before the Architect opens; there is no script, no install step and no
button. Any run does it, so this is only a shortcut:

```bash
.claude/skills/console/driver.py reindex
```

It starts the server if it is down and drives one discussion-only goal. Takes a
minute or two the first time (it downloads `all-MiniLM-L6-v2` from
HuggingFace); after that a rebuild that changes nothing costs ~0.1s, because a
document whose text has not changed keeps the vectors it has. Expect:

```
[Corpus] There was no corpus on this machine, so the project was indexed
         before the run: 77 document(s), 1618 passage(s) in 57.4s.
corpus: indexed -- 77 documents, 1618 chunks, 1190 nodes
```

## Run (human path)

```bash
./launch_console.sh
```

Prints seat status, starts the server, opens a browser at
http://localhost:8080, tails the log, stops on Ctrl+C. It launches
`python serve.py` off PATH, so **it only works if the venv is active** —
otherwise it dies with `ModuleNotFoundError: No module named 'langgraph'`.

Five tabs: Engineer, Graph (default), Retrieval, Corpus, State.

## Test

```bash
.venv/bin/ruff check src/ tests/ serve.py scripts/ example_usage.py test_cloud.py
.venv/bin/mypy src/langgraph_agent/ serve.py
.venv/bin/python -m pytest tests/ -q
```

Observed: ruff clean, `mypy` — no issues in 20 source files, pytest
**501 passed, 1 warning in 32.09s** (the warning is chromadb calling the
deprecated `asyncio.iscoroutinefunction` on 3.14; upstream, ignore it).

Tests use a stub LLM — no seats, no daemon, no keys needed.

## Gotchas

- **Indexing from outside the server used to leave it blind, and that is why
  nothing does it any more.** A script wrote the graph to disk while a live
  `serve.py` kept the NetworkX graph it had built at startup and never re-read
  it. The symptom was a split: `search_documents` returned real hits (chunks
  come from chroma) while `rag_stats` reported `total_nodes: 0, total_edges: 0,
  total_documents: 0` and `staleness.stale: true`, and the Graph tab stayed
  empty *forever*. The rebuild now happens inside the running server, on the
  object the console reads, so the split cannot occur — and a restart is no
  longer part of building a corpus.

- **Never `pkill -f serve.py`.** The pattern matches the shell running the
  pkill, so it kills its own caller — the command dies with exit 144 and the
  server survives. Same trap with `pgrep -f reindex.py`, which reports a
  finished reindex as still RUNNING. Use `driver.py down` (the app's own
  `shutdown` RPC) and read the log for completion.

- **A failed RPC method returns HTTP 200** with an `error` member, by design —
  the console renders errors into its telemetry log. A 200 does not mean the
  call worked; look at the body.

- **All four seats ship dead.** `DEFAULT_SEATS` is four Ollama seats on
  `qwen3.5:397b-cloud`, and a fresh daemon has nothing pulled, so every seat
  badges `NOT PULLED` and `run_goal` fails. Everything read-only — Graph,
  Retrieval, Corpus, State, all of `smoke` — works fine without them. (The
  older `console.md` claimed the default backend was Anthropic and told you to
  set `ANTHROPIC_API_KEY`; both were wrong. `.env` holds no API keys at all,
  only `BUILDER_DEADLINE_SECONDS`, `VERIFY_RESERVE_SECONDS`,
  `MAX_BUILDER_TOOL_TURNS`, `RUN_BUDGET_SECONDS`.)

- **The default tab is Graph, not Engineer** — `aria-selected` is on
  `button.tab[data-p="graph"]`. Screenshots land on the graph.

- **Screenshot byte size is the fastest liveness check.** The empty console is
  ~70KB; a drawn graph is ~900KB. `driver.py shot` warns below 120KB.

- **The graph is a force layout that needs time to settle.** `driver.py shot`
  passes `--virtual-time-budget=15000`; a smaller budget captures a tangle
  mid-simulation.

- Sending JSON to `/rpc` by hand from bash is easy to mangle — a bad quote gets
  you `{"error": {"message": "bad JSON"}}` from the server, which looks like a
  server problem and is not. Use `driver.py rpc`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'langgraph'` | venv missing or not used. Re-run Build; `driver.py` picks up `.venv` automatically. |
| Graph tab empty, `rag_stats` nodes=0, but search returns hits | Reindexed under a live server. `driver.py restart` (no reindex needed). |
| Shell command exits 144, server still running | `pkill -f serve.py` matched its own caller. Use `driver.py down`. |
| `{"error": {"message": "bad JSON"}}` | Shell quoting mangled the payload, not a server fault. Use `driver.py rpc`. |
| Seats badge `NOT PULLED`, runs fail | Expected on a fresh box. Pull the tag, or set a seat to a provider you have via `rpc set_seat`. |
| `driver.py shot` says "no chromium on PATH" | Install chromium, or screenshot from a browser against http://localhost:8080. |
