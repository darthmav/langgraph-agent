#!/usr/bin/env python3
"""Frontend server for the 4-Agent Console.

Usage:
    python serve.py

Then open: http://localhost:8080

The console talks to a single `POST /rpc` endpoint taking {method, params};
the `/api/*` routes are thin compatibility wrappers over the same dispatch.
"""

import json
import os
import sys
import tempfile
import threading
import time
import uuid
import warnings
from collections.abc import Callable
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Upstream libraries (langsmith, chromadb) emit DeprecationWarnings on Python 3.14+
# about asyncio.iscoroutinefunction. They are harmless and outside our control,
# so suppress them before importing any third-party code.
warnings.filterwarnings(
    "ignore",
    message=".*asyncio\\.iscoroutinefunction.*",
    category=DeprecationWarning,
)

# Ensure src directory is in Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from langgraph.errors import GraphRecursionError  # noqa: E402

from langgraph_agent import AgentState, create_agent_graph  # noqa: E402
from langgraph_agent.config import (  # noqa: E402
    AGENT_LLM_OPTIONS,
    AGENTS,
    get_agent_status,
    list_ollama_models,
    ollama_model_capabilities,
    set_agent_llm,
    set_agent_thinking,
    unload_local_seat_models,
)
from langgraph_agent.control import ACTIVITY, RUN_CONTROL  # noqa: E402
from langgraph_agent.corpus_health import (  # noqa: E402
    corpus_staleness,
    forget_expected_documents,
)
from langgraph_agent.graph import RECURSION_LIMIT  # noqa: E402
from langgraph_agent.graphrag_server import (  # noqa: E402
    EMBEDDING_MODEL_NAME,
    INDEXABLE_SUFFIXES,
    NO_CORPUS_NOTE,
    GraphRAGKnowledgeBase,
    active_embedding_model,
    calibrate_relevance_floor,
    corpus_state,
    embedding_backend,
    embedding_device_status,
    floor_calibration,
    get_knowledge_base,
    index_project_files,
    iter_project_files,
    open_knowledge_base,
    relevance_floor,
    set_embedding_model,
    store_uploaded_document,
)
from langgraph_agent.web_research import research_online  # noqa: E402

# Initialize graph. The knowledge base is deliberately *not* initialized here.
graph = create_agent_graph()

# The corpus, once something has opened one. There is no preload: this used to
# be filled by a background thread started at import, which meant starting the
# server created a store on disk and loaded the embedding model whether or not
# anyone wanted a corpus. A corpus exists because someone indexed, or it does
# not exist.
kb: GraphRAGKnowledgeBase | None = None

# Whether a run may build the corpus it is about to search, on a machine where
# nobody has built one -- see `_index_the_project_before_the_run`. It switches
# off the way `WEB_SEARCH_ENABLED` does, because the one caller that must never
# have it on is the test suite: `tests/conftest.py` turns it off for every
# test, or a suite run in a checkout with no `knowledge/` -- CI's, every time
# -- would index the whole project once per test that starts a run.
INDEX_PROJECT_BEFORE_RUN = os.getenv(
    "INDEX_PROJECT_BEFORE_RUN", "1"
).strip() not in {"0", "false", "no"}


# --------------------------------------------------------------------------
# Request and parameter checking
# --------------------------------------------------------------------------

# The largest request body read. An upload is the largest thing the console
# sends -- MAX_INDEXABLE_BYTES of text, JSON-escaped -- and this sits well
# above that, so it refuses what is not a console request without reading it.
MAX_REQUEST_BYTES = 8 * 1024 * 1024

# Bounds for the numbers a caller may ask for. Each is past anything the
# console's own controls offer, and short of what would make a call expensive.
MAX_TOP_K = 100
MAX_GRAPH_DEPTH = 10
MAX_MIN_DEGREE = 1_000_000
MAX_LIST_LIMIT = 500
MAX_TOPICS = 100

# Every method used to take its parameters through a bare `int()`, `float()`,
# `bool()` or `str()`. `top_k="abc"` came back as Python's own "invalid literal
# for int() with base 10"; a negative `top_k` as an empty result the console
# showed as "no results", as though the query had found nothing; and the string
# "false" as True -- which on `research_web` fetched and permanently embedded
# pages, and on `expect_failures` stopped failing files blocking approval. These
# refuse by name instead, and the methods parse before they touch anything.


def _refusal(name: str, wanted: str, value: Any) -> ValueError:
    return ValueError(f"{name} must be {wanted}; got {value!r}.")


def _int_param(
    params: dict[str, Any], name: str, default: int, *, low: int, high: int
) -> int:
    """A whole number from `low` to `high`, or `default` when it is absent."""
    value = params.get(name)
    if value is None or value == "":
        return default
    wanted = f"a whole number from {low} to {high}"
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _refusal(name, wanted, value)
    try:
        number = float(value)
    except ValueError:
        raise _refusal(name, wanted, value) from None
    if not number.is_integer() or not low <= number <= high:
        raise _refusal(name, wanted, value)
    return int(number)


def _float_param(
    params: dict[str, Any], name: str, default: float | None, *, low: float, high: float
) -> float | None:
    """A number from `low` to `high`, or `default` when it is absent."""
    value = params.get(name)
    if value is None or value == "":
        return default
    wanted = f"a number from {low:g} to {high:g}"
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise _refusal(name, wanted, value)
    try:
        number = float(value)
    except ValueError:
        raise _refusal(name, wanted, value) from None
    if not low <= number <= high:  # NaN fails this too
        raise _refusal(name, wanted, value)
    return number


def _bool_param(params: dict[str, Any], name: str, default: bool = False) -> bool:
    """A JSON boolean, or `default` when it is absent. Nothing else is coerced."""
    value = params.get(name)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise _refusal(name, "true or false", value)
    return value


def _str_param(params: dict[str, Any], name: str, default: str = "") -> str:
    """A string, or `default` when it is absent."""
    value = params.get(name)
    if value is None:
        return default
    if not isinstance(value, str):
        raise _refusal(name, "a string", value)
    return value


def _open_kb() -> GraphRAGKnowledgeBase | None:
    """The corpus if one has been built, `None` if none has. Never builds one.

    Every read goes through here. The console polls `rag_stats` every five
    seconds, so a read that creates is a corpus nobody asked for, arriving
    within seconds of the server starting and reporting itself as a knowledge
    base thereafter.
    """
    global kb
    if kb is None:
        kb = open_knowledge_base()
    return kb


def _kb_for_indexing() -> GraphRAGKnowledgeBase:
    """The corpus, **created if it does not exist**. Two callers, both writers.

    Indexing is the act that is allowed to bring a corpus into being, because
    it is the act that is a request for one -- and so is uploading a document,
    which is a request for a corpus to hold it. Every *read* goes through
    `_open_kb()` instead: a corpus that appeared because something looked at it
    is a corpus nobody asked for.
    """
    global kb
    if kb is None:
        kb = get_knowledge_base()
    return kb


# --------------------------------------------------------------------------
# RPC methods
# --------------------------------------------------------------------------


def rpc_rag_stats(_: dict[str, Any]) -> dict[str, Any]:
    """Counters for the console header.

    `corpus` is what the header actually renders on: zeros alone cannot tell a
    corpus that was indexed and then emptied from one that was never built,
    and only the second means "press Reindex".
    """
    kb_or_none = _open_kb()
    if kb_or_none is None:
        return {
            "corpus": "absent",
            "note": NO_CORPUS_NOTE,
            "total_documents": 0,
            "total_chunks": 0,
            "total_nodes": 0,
            "total_edges": 0,
            # Same keys as the indexed branch, so the console renders one shape
            # rather than branching on which fields happen to exist. `lambda_2`
            # is None, not 0.0: there is no graph to measure, and 0.0 is a real
            # reading that means "totally disconnected".
            "components": 0,
            "largest_component": 0,
            "isolated_nodes": 0,
            "lambda_2": None,
        }

    stats = kb_or_none.stats()
    stats["corpus"] = "indexed" if stats["total_chunks"] else "empty"

    # A corpus rots in silence: it is a function of what is on disk, nothing
    # rebuilds it automatically, and every counter above stays non-zero and
    # internally consistent while it drifts. On 2026-09-09 this project's store
    # held 8 documents against a walk offering 103 -- every project query
    # scored under the relevance floor and the Researcher's seat answered from
    # memory on every run, with nothing anywhere reporting a problem. The
    # comparison is advisory and must never take the header down with it, so a
    # failure to walk leaves the counters alone rather than replacing them.
    try:
        documents = [
            node
            for node, attrs in kb_or_none.graph.nodes(data=True)
            if attrs.get("type") == "document"
        ]
        report = corpus_staleness(documents)
        # A run in flight is the one time the corpus is *supposed* to be moving:
        # the online research phase writes each page to disk and embeds it as a
        # separate step, so between those two there is a file the walk can see
        # and the store cannot. Observed live -- the header read
        # "stale: 1 not indexed" mid-phase and cleared itself moments later.
        # A verdict that flickers is the credibility problem the exact size test
        # was built to avoid, so the *counts* stay (they are the truth about
        # this instant) and only the accusation is withheld.
        if report.get("stale") and _run_progress.get("running"):
            report = {**report, "stale": False, "settling": True}
        stats["staleness"] = report
    except Exception as exc:  # pragma: no cover - a walk that cannot run
        stats["staleness"] = {"stale": False, "unavailable": str(exc)}
    return stats


def rpc_list_documents(_: dict[str, Any]) -> dict[str, Any]:
    """Every document node; the seed set for a graph sweep."""
    kb_or_none = _open_kb()
    if kb_or_none is None:
        return {"documents": [], "corpus": "absent", "note": NO_CORPUS_NOTE}
    return kb_or_none.list_documents()


def rpc_query_graph(params: dict[str, Any]) -> dict[str, Any]:
    """One node's neighbourhood, as drawable nodes and edges.

    `min_degree` defaults to 2 because the common caller is a sweep, where the
    one-document entities bury the structure. A trace passes 1 explicitly.
    """
    node_id = _str_param(params, "node_id")
    max_depth = _int_param(params, "max_depth", 2, low=1, high=MAX_GRAPH_DEPTH)
    min_degree = _int_param(params, "min_degree", 2, low=1, high=MAX_MIN_DEGREE)
    split = _bool_param(params, "split")
    kb_or_none = _open_kb()
    if kb_or_none is None:
        return {
            "error": NO_CORPUS_NOTE,
            "corpus": "absent",
            "center_node": node_id,
            "related_nodes": [],
            "edges": [],
            "total_nodes": 0,
            "total_edges": 0,
        }

    return kb_or_none.neighborhood(
        node_id,
        max_depth=max_depth,
        min_degree=min_degree,
        # Off unless asked for: it triples the cost of a call the console makes
        # on every click, and only a caller drawing the division wants it.
        split=split,
    )


def rpc_graph_overview(params: dict[str, Any]) -> dict[str, Any]:
    """The whole corpus as one drawable graph, for the console's sweep.

    One call where the console used to make one per document; see
    `GraphRAGKnowledgeBase.overview` for why the result is the same.
    """
    min_degree = _int_param(params, "min_degree", 4, low=1, high=MAX_MIN_DEGREE)
    include_isolated = _bool_param(params, "include_isolated")
    kb_or_none = _open_kb()
    if kb_or_none is None:
        return {
            "corpus": "absent",
            "note": NO_CORPUS_NOTE,
            "nodes": [],
            "edges": [],
            "total_nodes": 0,
            "total_edges": 0,
            "unlinked_documents": 0,
        }
    return kb_or_none.overview(
        min_degree=min_degree,
        include_isolated=include_isolated,
    )


def rpc_search_documents(params: dict[str, Any]) -> dict[str, Any]:
    """Semantic search over the corpus.

    With no corpus this returns no results and says so, rather than building
    one to search. Searching is a read like any other.
    """
    query = _str_param(params, "query")
    top_k = _int_param(params, "top_k", 5, low=1, high=MAX_TOP_K)
    kb_or_none = _open_kb()
    if kb_or_none is None:
        return {"results": [], "source": "no_corpus", "note": NO_CORPUS_NOTE}

    results = kb_or_none.search(query, top_k)
    return {"results": results, "source": "local_graphrag"}


def _refuse_while_a_run_is_in_flight(action: str) -> None:
    """Guard the corpus against being changed underneath a run.

    Shared by the two methods that write to it, because the hazard is the same
    for both and it is not the obvious one. Changing the corpus mid-run does
    not break the Researcher's search -- that is the problem. A cleared corpus
    returns no hits, and a corpus in the middle of a rebuild returns whatever
    fraction of itself has been re-added so far; either reads as
    `no_relevant_knowledge` and routes the run as though the knowledge base had
    simply had nothing useful to say. Nothing raises, nothing is logged, and
    the run goes on to plan around an absence that was manufactured out from
    under it. The seat cannot find out, so the operator is told instead.

    Reads the flag under `_run_lock` and names the goal, the way `rpc_shutdown`
    refuses: a refusal that does not say what is running leaves the operator to
    guess whether they still care about it.

    Args:
        action: Past participle of what was refused -- "cleared", "rebuilt".

    Raises:
        ValueError: if a run is in flight. The console's `rpc()` helper turns
            this into a telemetry line and the tab's status text.
    """
    with _run_lock:
        if not _run_progress["running"]:
            return
        goal = str(_run_progress["goal"])

    detail = f" Running: {goal}" if goal else ""
    raise ValueError(
        f"A run is in flight and the Researcher is searching this corpus, so "
        f"it cannot be {action} right now. Stop the run first.{detail}"
    )


# There is no `rpc_reindex`, and that is the design rather than an omission.
# Rebuilding the corpus was a button on the Corpus tab, which made keeping the
# corpus current a thing the operator had to know to do -- and the failure when
# they did not know is silent: the search answers with whatever the store holds,
# the seats plan around it, and every counter in the header stays non-zero and
# self-consistent. `_index_the_project_before_the_run` does it on every run
# instead, at the moment it matters and against the files as they are then.
# Nor is there a script that does it, an install step, or anything else: two
# things index, and both are a request for a corpus rather than housekeeping --
# a run, which searches one, and embedding a document into the corpus from the
# console, which is a request for a corpus to hold it.


def rpc_upload_document(params: dict[str, Any]) -> dict[str, Any]:
    """Embed one document the operator uploaded, from either upload control.

    One method behind both of them -- the Engineer tab's attach button and the
    Corpus tab's -- so the two entry points cannot come to mean different
    things. The file is written under `uploads/` before it is embedded, which
    is what puts it inside the next reindex rather than underneath it; see
    `store_uploaded_document`.

    Refused mid-run, like the other two writers, but for a different reason
    than theirs. An upload only ever adds, so a Researcher would see more
    rather than less -- what it cannot do is add while a search is reading the
    graph, since `add_document` mutates the same networkx object
    `neighborhood` and the lexical index iterate. And a run should be answered
    by the corpus it started against either way.

    One document per call. The console loops for a multi-file drop, so one
    file that the corpus cannot take -- a PDF among the markdown -- fails on
    its own and says so, instead of taking the batch down with it.
    """
    _refuse_while_a_run_is_in_flight("added to")
    name = params.get("name", "")
    content = params.get("content", "")
    if not isinstance(name, str) or not isinstance(content, str):
        raise ValueError("An upload is a filename and its text; both must be strings.")
    report = store_uploaded_document(_kb_for_indexing(), name, content)
    # An upload puts a *file* on disk, so the cached walk is now behind the
    # corpus rather than ahead of it -- and the staleness report would call the
    # freshly embedded document `extra` until the cache expired. This is the
    # one writer that changes what the walk would find.
    forget_expected_documents()
    return report


def rpc_bottleneck(params: dict[str, Any]) -> dict[str, Any]:
    """The narrowest cut in the knowledge graph, and the nodes bridging it.

    Read-only, and deliberately not part of the five-second poll: it is an
    eigenvector plus a sweep over every edge, and it answers a question the
    operator asks rather than one the header needs. Like `export_corpus` it
    takes no run guard -- reading the corpus takes nothing away from the run
    using it.
    """
    limit = _int_param(params, "limit", 12, low=1, high=MAX_LIST_LIMIT)
    kb_or_none = _open_kb()
    if kb_or_none is None:
        raise ValueError(f"There is no corpus to analyse. {NO_CORPUS_NOTE}")
    return kb_or_none.bottleneck(limit=limit)


def rpc_topics(params: dict[str, Any]) -> dict[str, Any]:
    """Topic communities over the corpus: the whole-corpus map.

    Read-only and off the poll, for the same reasons as `bottleneck`. `k` is
    optional; omitted, the eigengap chooses it and the answer may come back
    `no_clear_structure` rather than a map of a corpus that has no topics.
    """
    # Omitted or blank means "choose one", not k = 0.
    k = None if params.get("k") in (None, "") else _int_param(
        params, "k", 0, low=2, high=MAX_TOPICS
    )
    kb_or_none = _open_kb()
    if kb_or_none is None:
        raise ValueError(f"There is no corpus to cluster. {NO_CORPUS_NOTE}")
    return kb_or_none.topics(k=k)


def rpc_duplicate_entities(params: dict[str, Any]) -> dict[str, Any]:
    """Entities playing the same structural role: candidates to merge.

    Read-only and off the poll, like `topics` and `bottleneck`. It proposes; it
    never merges. `add_document` mints an entity per capitalised token, so the
    duplicates are a property of the corpus rather than a fault to be repaired
    behind the operator's back, and merging one is a decision about meaning
    that the graph alone cannot make.

    Candidates are generated from the names and the structure is the evidence,
    not the other way round -- see `duplicate_entities` for the measurement
    that settled which way that runs.
    """
    limit = _int_param(params, "limit", 20, low=1, high=MAX_LIST_LIMIT)
    name_similarity = _float_param(params, "name_similarity", None, low=0.0, high=1.0)
    containment = _float_param(params, "containment", None, low=0.0, high=1.0)
    kb_or_none = _open_kb()
    if kb_or_none is None:
        raise ValueError(f"There is no corpus to scan. {NO_CORPUS_NOTE}")
    return kb_or_none.duplicate_entities(
        limit=limit, name_similarity=name_similarity, containment=containment
    )


def rpc_export_corpus(_: dict[str, Any]) -> dict[str, Any]:
    """The whole corpus as one JSON document, for the console to save.

    Served over `/rpc` like everything else rather than as a file download, so
    a failure lands on the console's telemetry path instead of replacing the
    page with a JSON error. The browser makes the file at the other end.
    """
    kb_or_none = _open_kb()
    if kb_or_none is None:
        raise ValueError(f"There is no corpus to export. {NO_CORPUS_NOTE}")
    return kb_or_none.export_corpus()


def rpc_clear_corpus(_: dict[str, Any]) -> dict[str, Any]:
    """Empty the knowledge base in place.

    Refused mid-run for the reason `_refuse_while_a_run_is_in_flight` sets out.
    Refused with no corpus for a different one: creating a store in order to
    empty it would leave behind exactly the thing the operator was asking to
    be rid of.
    """
    _refuse_while_a_run_is_in_flight("cleared")
    kb_or_none = _open_kb()
    if kb_or_none is None:
        raise ValueError(f"There is no corpus to clear. {NO_CORPUS_NOTE}")
    return kb_or_none.clear()


def rpc_list_seats(_: dict[str, Any]) -> dict[str, Any]:
    """The four seats and whether each can actually run."""
    return {
        "seats": [{"role": agent, **get_agent_status(agent)} for agent in AGENTS]
    }


def rpc_set_seat(params: dict[str, Any]) -> dict[str, Any]:
    """Reassign one seat for the lifetime of this process."""
    agent = _str_param(params, "agent") or _str_param(params, "role")
    provider = _str_param(params, "provider")
    model = _str_param(params, "model")

    if agent not in AGENTS:
        raise ValueError(f"Unknown agent: {agent!r}")
    if not provider or not model:
        raise ValueError("Provider and model are both required")

    set_agent_llm(agent, provider, model)
    return {"ok": True, "role": agent, **get_agent_status(agent)}


def rpc_set_thinking(params: dict[str, Any]) -> dict[str, Any]:
    """Switch one seat's thinking on or off for the lifetime of this process.

    `thinking` must be a JSON boolean. Anything else is refused rather than
    coerced, because the obvious coercion reads the string "false" as on.
    """
    agent = _str_param(params, "agent") or _str_param(params, "role")
    thinking = params.get("thinking")

    if agent not in AGENTS:
        raise ValueError(f"Unknown agent: {agent!r}")
    if not isinstance(thinking, bool):
        raise ValueError("thinking must be true or false")

    set_agent_thinking(agent, thinking)
    return {"ok": True, "role": agent, **get_agent_status(agent)}


def rpc_llm_options(_: dict[str, Any]) -> dict[str, Any]:
    """Model choices for the seat dropdowns.

    `ollama_tags` is what the daemon actually carries, so a seat can be pointed
    at a tag that is pulled but not in the curated list.
    """
    return {"options": AGENT_LLM_OPTIONS, "ollama_tags": list_ollama_models()}


def _embedding_choice(model: str, group: str) -> dict[str, Any]:
    """One entry of the embedding dropdown, with what choosing it would cost."""
    state, _ = corpus_state(model=model)
    floor = relevance_floor(model)
    if model == EMBEDDING_MODEL_NAME:
        floor_source = "measured"
    elif floor_calibration(model) is None:
        floor_source = "not_measured"
    else:
        floor_source = "calibrated" if floor is not None else "no_gap"
    return {
        "model": model,
        "label": model,
        "group": group,
        "backend": embedding_backend(model),
        "corpus": state,
        "floor": floor,
        "floor_source": floor_source,
    }


def rpc_embedding_options(_: dict[str, Any]) -> dict[str, Any]:
    """The embedding models the console can switch between, and the active one.

    MiniLM, which runs in this process, plus every tag the Ollama daemon says
    can embed -- asked of the daemon's capabilities rather than guessed from a
    name, the way the seat cards ask about thinking. Each entry carries the
    state of that model's own corpus and where its relevance floor came from,
    because those two are what switching costs: a model with no corpus is built
    by the next run, and a model with no floor hands every search to the
    Researcher's model.
    """
    active = active_embedding_model()
    choices = [_embedding_choice(EMBEDDING_MODEL_NAME, "In this process")]
    installed = [
        tag for tag in list_ollama_models()
        if "embedding" in (ollama_model_capabilities(tag) or [])
    ]
    choices += [_embedding_choice(tag, "Ollama (installed)") for tag in installed]
    if all(choice["model"] != active for choice in choices):
        choices.insert(0, _embedding_choice(active, "Current"))
    with _run_lock:
        running = bool(_run_progress["running"])
    return {
        "active": active,
        "options": choices,
        "switchable": not running,
        "device": embedding_device_status(),
    }


def rpc_set_embedding_model(params: dict[str, Any]) -> dict[str, Any]:
    """Switch the embedding model for the lifetime of this process.

    Refused mid-run, because switching models is switching corpora and the
    corpus a run searches must not change underneath it. Refused for a model
    the console does not offer, so a typo cannot point the corpus at a tag that
    embeds nothing. Nothing is built here: the next run builds the new model's
    corpus if it has none, the one act allowed to bring a corpus into being.
    """
    global kb
    model = _str_param(params, "model").strip()
    offered = {choice["model"] for choice in rpc_embedding_options({})["options"]}
    if model not in offered:
        raise ValueError(
            f"{model!r} is not an embedding model this machine offers: "
            f"{', '.join(sorted(offered))}"
        )
    _refuse_while_a_run_is_in_flight("switched to another embedding model")
    set_embedding_model(model)
    kb = None
    return rpc_embedding_options({})


def rpc_status(_: dict[str, Any]) -> dict[str, Any]:
    """Everything the console polls for: seats, embedding model, corpus state.

    `corpus` is `absent`, `empty` or `indexed`. The older `graphrag` boolean is
    kept beside it -- `launch_console.sh` polls this route as its readiness
    check -- but it collapses the first two, and they are the pair worth
    telling apart: nothing was ever indexed, versus a corpus that exists and
    holds nothing.
    """
    try:
        state, embedding_model = corpus_state()
    except Exception:
        state, embedding_model = "absent", "unknown"

    seats = {agent: get_agent_status(agent) for agent in AGENTS}
    architect = seats["architect"]

    return {
        # The embedding model is the one thing that would run on this machine,
        # and it is not loaded until something indexes or searches.
        "embedding": embedding_model,
        # Where it runs, which only a loaded model can say: the setting names a
        # device, and a card with no kernels or no room leaves the model on the
        # CPU. Read without loading anything, so `active` is None until
        # something has embedded.
        "embedding_device": embedding_device_status(),
        "corpus": state,
        # Whether a run would build the corpus itself from here. Reported
        # rather than assumed by the console, which otherwise has to promise
        # on the operator's behalf that the next run will index the project --
        # a promise `INDEX_PROJECT_BEFORE_RUN=0` makes false, in the one place
        # someone looks to find out why retrieval is empty.
        "indexes_on_run": INDEX_PROJECT_BEFORE_RUN,
        # What an upload may be, for the console's pickers and tooltips. From
        # the walk's own list, so the page cannot promise a different one.
        "indexable_suffixes": list(INDEXABLE_SUFFIXES),
        "graphrag": state == "indexed",
        "llm": f"{architect['model']} ({architect['provider']})",
        "agents": seats,
        "selected": seats,
        "degraded": [agent for agent, seat in seats.items() if not seat["live"]],
    }


# What the in-flight run has done so far. A run is many cloud calls long, and
# the POST that started it does not return until all of them are done, so
# without this the console can only show elapsed seconds -- a working run and a
# wedged one look exactly alike from the browser.
_run_progress: dict[str, Any] = {
    "running": False,
    "goal": "",
    "messages": [],
    "node": "",
    # Identifies the run a Stop is aimed at. Without it a Stop click held over
    # from a finished run -- a stale tab, a reload -- would kill whatever
    # happened to be running when it finally landed.
    "run_id": "",
    "stopping": False,
}
_run_lock = threading.Lock()

# The last run's final state, kept so a stop is recoverable. Until now the
# final state was returned to the caller and then dropped: a stopped run whose
# partial plan, research and builder report were lost would be no better than
# killing the server, which is the thing the stop exists to replace. Also
# written to disk, so it survives a page reload and a restart.
_last_run_snapshot: dict[str, Any] | None = None
RUNS_DIR = Path(__file__).parent / "runs"
LAST_RUN_PATH = RUNS_DIR / "last_run.json"


def _save_snapshot(snapshot: dict[str, Any]) -> None:
    """Keep the run's outcome in memory and on disk.

    Written to a temp file and renamed, because the console fetches this on
    load: a reader arriving mid-write would otherwise get half a JSON document.
    A disk failure is swallowed -- the in-memory copy is what the current page
    reads, and losing the file is not worth failing a run that already finished.
    """
    global _last_run_snapshot
    _last_run_snapshot = snapshot
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=RUNS_DIR, suffix=".tmp", delete=False, encoding="utf-8"
        ) as handle:
            json.dump(snapshot, handle, default=str, indent=2)
            temp_name = handle.name
        os.replace(temp_name, LAST_RUN_PATH)
    except OSError:
        pass


def _load_snapshot() -> dict[str, Any] | None:
    """The last run's outcome, from memory or from the file a restart left."""
    global _last_run_snapshot
    if _last_run_snapshot is not None:
        return _last_run_snapshot
    try:
        with open(LAST_RUN_PATH, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(loaded, dict):
        _last_run_snapshot = loaded
        return _last_run_snapshot
    return None

# Wall-clock budget for one run. MAX_STEPS bounds how many times the Architect
# may send work back, which is the right logical bound but says nothing about
# how long that takes: giving the Builder a tool loop raised the cost of a
# single step from about four cloud calls to as many as eleven, and a step is
# only as fast as the slowest model in it. A run that never converges was
# reaching tens of minutes with nothing to show. This bounds the symptom
# directly and does not have to be re-guessed when a seat or a model changes.
RUN_BUDGET_SECONDS = float(os.getenv("RUN_BUDGET_SECONDS", "300"))


def rpc_run_progress(_: dict[str, Any]) -> dict[str, Any]:
    """A snapshot of the run currently in flight, for the console to poll."""
    # Read outside `_run_lock`: `ACTIVITY` carries its own, and the node it
    # names is set by the graph rather than by this module.
    active = ACTIVITY.current()
    active_for = ACTIVITY.busy_for()
    with _run_lock:
        return {
            "running": bool(_run_progress["running"]),
            "goal": str(_run_progress["goal"]),
            "node": str(_run_progress["node"]),
            # `node` is the seat that last *finished* -- `graph.stream` yields
            # on completion -- which is what the feed lists. `active` is the
            # seat whose node is on the stack right now, which is what the
            # console's seat lights mean. They are usually different, and
            # during the slowest node of the run they always are.
            "active": active,
            "active_for": round(active_for, 1),
            "messages": list(_run_progress["messages"]),
            "budget": RUN_BUDGET_SECONDS,
            # The console needs both to reattach after a reload: the id to aim
            # a Stop at this run, and `stopping` to keep the button honest
            # about a stop that has been asked for but not yet landed.
            "run_id": str(_run_progress["run_id"]),
            "stopping": bool(_run_progress["stopping"]),
        }


def rpc_stop_run(params: dict[str, Any]) -> dict[str, Any]:
    """Ask the run in flight to stop at its next safe boundary.

    Cooperative, and returns immediately: this only sets a flag. Work already
    started -- a file write, a staged commit, a test run, a model call -- runs
    to completion, because abandoning it would leave a half-written file or a
    worker still writing into the project. So the run ends within one tool call
    or one model call, not instantly, and the caller is told so.

    Served on a different thread from the run it stops: `rpc_run_goal` blocks
    its own request thread for the whole run, which is why the server is a
    ThreadingHTTPServer.
    """
    run_id = _str_param(params, "run_id")
    reason = _str_param(params, "reason")

    if not RUN_CONTROL.stop(run_id, reason):
        armed = RUN_CONTROL.run_id()
        detail = (
            "No run is in flight."
            if not armed
            else "That Stop was for a run that has already ended."
        )
        return {"stopping": False, "run_id": armed, "detail": detail}

    with _run_lock:
        # The stop above is what sends the run to its teardown, and that
        # teardown clears this flag under this same lock. Setting it blind
        # pinned "STOPPING" on a run that had already ended.
        if _run_progress["running"]:
            _run_progress["stopping"] = True

    return {
        "stopping": True,
        "run_id": RUN_CONTROL.run_id(),
        "detail": (
            "Stopping. Nothing further will be started; the call already in "
            "flight finishes first, so this can take up to a minute."
        ),
    }


def rpc_last_run(_: dict[str, Any]) -> dict[str, Any]:
    """The last run's final state, for a console that reloaded and lost it."""
    return {"snapshot": _load_snapshot()}


# Set when the console's exit button asks the process to end; the main thread
# waits on it, exactly as it waits on Ctrl+C.
_shutdown_requested = threading.Event()

# Set when an exit has to wait for the run in flight to hand its state back
# first. The run's own `finally` is what then requests the shutdown, which is
# what guarantees the snapshot is on disk before the process goes.
_exit_after_run = threading.Event()

# A moment's grace between deciding to exit and closing the socket, so the
# reply that asked for the exit reaches the browser rather than dying with the
# connection. Small enough that the console never feels it.
SHUTDOWN_GRACE_SECONDS = float(os.getenv("SHUTDOWN_GRACE_SECONDS", "0.5"))


def _finish_run() -> None:
    """Release the run and let a deferred exit through. The run's `finally`.

    The `_exit_after_run` read happens under `_run_lock`, and `rpc_shutdown`
    decides under that same lock, because the two are one handshake: whoever
    holds the lock either sees a run still in flight and hands it the exit, or
    sees it finished and takes the exit itself. Exactly one of those happens.
    """
    RUN_CONTROL.disarm()
    # Belt and braces behind each node's own `finally`: a run that ended is a
    # run with no seat working, and a light left on would claim otherwise for
    # as long as the console stayed open.
    ACTIVITY.clear()
    with _run_lock:
        _run_progress["running"] = False
        _run_progress["stopping"] = False
        _run_progress["run_id"] = ""
        # An exit that was waiting on this run happens now, after the snapshot
        # the caller has already written -- which is the whole reason it waited.
        if _exit_after_run.is_set():
            _shutdown_requested.set()


def rpc_shutdown(params: dict[str, Any]) -> dict[str, Any]:
    """Exit the server -- what the console's X asks for.

    A run in flight is not killed by surprise. Without `stop_first` this
    refuses and says what is running, so the console can ask; with it, the run
    is stopped through the ordinary emergency stop and the exit is deferred to
    the run's own `finally`. That deferral is the point: it is what puts the
    snapshot on disk before the process ends, so an exit mid-run is as
    recoverable as a stop.
    """
    stop_first = _bool_param(params, "stop_first")

    # Read the run's state and claim the exit in one hold of `_run_lock`.
    # Claiming it afterwards lost the exit outright: `RUN_CONTROL.stop()` below
    # is what makes the run race for its own `finally`, and a run sitting at a
    # superstep boundary gets there in microseconds. It would read
    # `_exit_after_run` unset, decline to request the shutdown -- correctly, on
    # what it could see -- and by the time this function set the flag there was
    # no longer a run left to honour it. Nothing then set `_shutdown_requested`
    # at all: the process stayed up, the console sat on "Closing..." until its
    # own patience ran out, and `stopping` stayed pinned True on a run that had
    # already finished. Set inside the lock and *before* the stop, so the flag
    # is always in place before the run it defers to can look at it.
    with _run_lock:
        running = bool(_run_progress["running"])
        goal = str(_run_progress["goal"])
        if running and stop_first:
            _exit_after_run.set()
            _run_progress["stopping"] = True

    if running and not stop_first:
        return {
            "exiting": False,
            "running": True,
            "goal": goal,
            "detail": (
                "A run is still in flight. Stop it first, or ask again with "
                "stop_first to stop it and exit once its state is saved."
            ),
        }

    if running:
        # The run may already have finished between the lock above and here.
        # That is safe now and needs no branch: `_exit_after_run` was set while
        # the run was still live, so its `finally` saw the flag and requested
        # the shutdown itself. This stop then lands on a disarmed control and
        # is refused, which is exactly right.
        RUN_CONTROL.stop("", "Stopped so the console could exit.")
        return {
            "exiting": True,
            "running": True,
            "detail": (
                "Stopping the run. The server exits once it has handed its "
                "state back and written its snapshot."
            ),
        }

    _shutdown_requested.set()
    return {"exiting": True, "running": False, "detail": "The server is exiting."}


def _claim_the_embedder_before_the_run() -> dict[str, Any]:
    """Put the embedder on its card before any local seat is fitted beside it.

    The order is the whole point, and it cannot be left to whatever embeds
    first. llama.cpp fits a model around what a card already carries and never
    moves it afterwards, so a 3 GB card shared with a local seat takes both in
    exactly one arrangement: the embedder holds its memory, and the seat loads
    into the rest. Measured with the seat's KV cache at q8_0, that left the
    seat 49 of 49 layers on the GPU at unchanged speed; the other way round,
    the embedder found 11 MiB free and failed. A run's first search is inside
    the Planner's node, after the previous run's seat may still be loaded --
    so the placement is made here, before the corpus phase, which then embeds
    on the card too.

    When the card is full because a seated model is still loaded, that model
    is unloaded first (`unload_local_seat_models`) and reloads around the
    embedder on its next call. That happens once per server rather than once
    per run: an embedder already on its card is left there.

    Silent on `cpu`, and opens nothing there: a CPU embedder has no order to
    get right, and the setting is the machine's, which the header reports.
    Never raises; a failure leaves the embedder on the CPU and the run goes on.
    """
    where = embedding_device_status()
    if where["configured"] in ("cpu", "ollama"):
        # Nothing to order: a CPU embedder takes no card, and an Ollama one is
        # placed by the daemon.
        return {"source": where["configured"]}
    if RUN_CONTROL.stopped():
        return {"source": "stopped"}
    kb_or_none = _open_kb()
    if kb_or_none is None:
        # Nothing indexed yet, so nothing searches before the corpus phase
        # builds the store -- and that phase embeds, which loads the model onto
        # the card by itself.
        return {"source": "no_corpus"}
    if where["active"] != where["configured"]:
        # Said before the work, for the reason the corpus phase says its line
        # first: a load plus an unload takes a while, and `running` with no
        # node and no message is what a wedged run looks like.
        with _run_lock:
            _run_progress["messages"] = [
                f"[Embedder] Placing the embedding model on {where['configured']} "
                "before the run starts."
            ]
    try:
        return kb_or_none.claim_embedding_device(make_room=unload_local_seat_models)
    except Exception as exc:
        return {"source": "error", "configured": where["configured"], "note": str(exc)}


def _embedder_feed_line(report: dict[str, Any]) -> str | None:
    """A line only when placing the embedder did something worth knowing.

    Silence is right here most of the time, unlike the corpus line: where the
    embedder runs is on the console header on every poll, and an embedder
    already on its card did nothing this run. What the header cannot say is
    that a seat was unloaded to make room -- which is why that seat's next call
    is slow -- or why a card was refused.
    """
    source = report.get("source")
    if source == "claimed":
        unloaded = report.get("unloaded") or []
        room = ""
        if unloaded:
            one = len(unloaded) == 1
            room = (
                f" {', '.join(unloaded)} {'was' if one else 'were'} unloaded to make "
                f"room, and {'reloads' if one else 'reload'} around it on the next call."
            )
        return f"[Embedder] Loaded onto {report.get('device')} before the seats open.{room}"
    if source == "unavailable":
        return (
            f"[Embedder] {report.get('configured')} could not take the embedding model, "
            f"so it runs on the CPU: {report.get('note')}"
        )
    if source == "error":
        return (
            f"[Embedder] The embedding model could not be placed: {report.get('note')}. "
            "It loads when first needed, on the CPU if the card still refuses."
        )
    return None


def _calibrate_the_floor_before_the_run(corpus_report: dict[str, Any]) -> dict[str, Any]:
    """Take the active model's relevance floor, the first time its corpus is whole.

    MiniLM's floor was measured by hand and is never re-measured here. Any
    other model has none until its corpus exists, and without one retrieval
    cannot tell an answer from noise -- `_gather_research` hands every search
    to the Researcher's model and the Planner gets no project map. So the run
    that finishes a model's corpus measures its floor before any seat
    searches, with the questions `calibrate_relevance_floor` keeps out of the
    corpus, and the measurement is kept beside that corpus from then on.

    Skipped when the corpus phase did not leave a whole corpus: a stopped or
    failed build would calibrate against a fraction of the project.
    """
    model = active_embedding_model()
    if model == EMBEDDING_MODEL_NAME or floor_calibration(model) is not None:
        return {"source": "known", "model": model}
    if RUN_CONTROL.stopped() or corpus_report.get("stopped"):
        return {"source": "stopped", "model": model}
    if corpus_report.get("source") in ("error", "nothing_to_index"):
        return {"source": "no_corpus", "model": model}
    kb_or_none = _open_kb()
    if kb_or_none is None or corpus_state()[0] != "indexed":
        return {"source": "no_corpus", "model": model}
    with _run_lock:
        _run_progress["messages"] = [
            f"[Embedder] Measuring a relevance floor for {model} on its corpus "
            "before the run starts."
        ]
    try:
        record = calibrate_relevance_floor(kb_or_none)
    except Exception as exc:
        return {"source": "error", "model": model, "note": str(exc)}
    return {"source": "calibrated", **record}


def _calibration_feed_line(report: dict[str, Any]) -> str | None:
    """Said when a floor was measured or could not be: the outcomes that change retrieval."""
    source = report.get("source")
    model = report.get("model")
    if source == "calibrated":
        answered = report.get("answered") or [0.0]
        unanswerable = report.get("unanswerable") or [0.0]
        span = (
            f"questions the corpus answers scored from {min(answered):.3f}, questions "
            f"it cannot up to {max(unanswerable):.3f}"
        )
        if report.get("floor") is None:
            return (
                f"[Embedder] {model} left no gap between the two populations ({span}), "
                "so it has no relevance floor: the Researcher's model answers every "
                "search and the Planner gets no project map."
            )
        return (
            f"[Embedder] Measured a relevance floor for {model}: {span}, so retrieval "
            f"over {report['floor']:.3f} counts as an answer."
        )
    if source == "error":
        return (
            f"[Embedder] The relevance floor for {model} could not be measured: "
            f"{report.get('note')}. Until it is, the Researcher's model answers every "
            "search."
        )
    return None


def _index_the_project_before_the_run() -> dict[str, Any]:
    """Make the corpus match the project, before any seat searches it.

    A fresh install has no corpus, and nothing used to bring one into being
    except the operator asking for it by name. Missing that costs nothing
    visible: `search_knowledge_graph` answers `no_corpus`, `_gather_research`
    falls through to the Researcher's own model, and the Builder works from
    whatever that model remembers. Nothing raises, nothing is logged, and the
    run reports itself finished. The same silence covers the other half of the
    problem -- a corpus that was built once and has been drifting from the
    project ever since. Measured here on 2026-09-09: 8 documents in the store
    against a walk offering 103, every project query under
    `RETRIEVAL_RELEVANCE_FLOOR`, and `rag_stats` reporting `indexed` with every
    counter non-zero and consistent.

    Both were left to a button. There is no button now: this runs on **every**
    run, and it is why there is nothing left for anyone to press.

    Five decisions in it are not interchangeable with the obvious
    alternatives.

    *It rebuilds every time rather than only when the corpus is missing.* That
    is affordable because `index_project_files` keeps the vectors of documents
    whose text still hashes to what the store holds -- measured warm on this
    project, 52.0s to re-embed 77 files and 0.09s when nothing changed, with
    the embedding model never loaded in the second case. A rebuild gated on
    `absent`/`empty` would have been cheap in the same way and would have left
    drift exactly where it was: the state that needs fixing most is the one
    where every counter already looks right.

    *It compares content, not the walk.* `corpus_staleness` answers the
    header's question -- which documents are in one and not the other -- and it
    cannot see an edit, because an edited file is in both. The Builder edits
    files, so that is the common case here rather than the exotic one.

    *It runs before the online research phase, not after.* That phase embeds
    the pages it keeps and writes them under `research/web/`, which is inside
    the walk -- so a rebuild afterwards would re-read them from disk, and a
    rebuild before leaves them to be added on top. Going second also meant a
    corpus holding nothing but fetched pages counted above zero. A test pins
    the order.

    *It counts the walk before it opens the door.* `_kb_for_indexing()` creates
    the store, so calling it on a machine with nothing to index leaves an empty
    corpus behind and every later poll reports `empty` where the truth is
    `absent` -- and only one of those two means anything is wrong. That is
    exactly the mistake `research_online` made by resolving its door on the way
    in, one caller along, and it is guarded here the same way: ask what there
    is to index first.

    *And a run already stopped does not start one.* The check is before the
    work rather than inside it, like every stop check guarding something that
    cannot be half-done: `index_project_files` clears the graph up front, so a
    rebuild abandoned midway is the half-finished corpus
    `_refuse_while_a_run_is_in_flight` exists to stop anyone else from
    manufacturing.

    A discussion run still does it, where it never researches online. That
    phase brings in material from outside and makes it a permanent corpus
    member; this one embeds files already on disk into a runtime artifact a
    reindex reproduces exactly, and adds nothing to the project. "Nothing was
    changed" is a promise about the project, and it still holds.

    Never raises. A corpus that could not be built makes for a worse run, not a
    refused one, and the report says which of these happened.
    """
    state, _ = corpus_state()
    if not INDEX_PROJECT_BEFORE_RUN:
        return {"source": "disabled", "corpus": state}
    if RUN_CONTROL.stopped():
        return {"source": "stopped", "corpus": state}

    files = iter_project_files()
    if not files:
        return {"source": "nothing_to_index", "corpus": state, "root": str(Path.cwd())}

    # Said before the work rather than after it. A first index takes tens of
    # seconds, and for all of them `running` is True with no node on the stack
    # and no messages -- which is precisely what a wedged run looks like from
    # the console. This is the same reason `#run-live` names the last stage
    # instead of only counting seconds. A rebuild that changes nothing
    # overwrites this line in under a second and nobody ever reads it.
    model = active_embedding_model()
    with _run_lock:
        _run_progress["messages"] = [
            f"[Corpus] Checking {len(files)} project file(s) against the {model} "
            "corpus before the run starts."
        ]

    def progress(done: int, total: int) -> None:
        # Rewritten in place as the phase goes, so a build that takes hours --
        # any model but MiniLM -- reads as moving rather than wedged.
        with _run_lock:
            _run_progress["messages"] = [
                f"[Corpus] Indexing with {model}: {done} of {total} project file(s) "
                "checked."
            ]

    started = time.monotonic()
    try:
        report = index_project_files(
            _kb_for_indexing(), progress=progress, should_stop=RUN_CONTROL.stopped
        )
    except Exception as exc:
        return {"source": "error", "corpus": state, "note": str(exc)}

    if report.get("stopped"):
        source = "stopped_midway"
    elif state != "indexed":
        source = "built"
    elif report.get("embedded") or report.get("dropped"):
        source = "updated"
    else:
        source = "current"
    report.update(
        source=source,
        corpus=state,
        model=model,
        elapsed_s=round(time.monotonic() - started, 1),
    )
    return report


def _corpus_feed_line(report: dict[str, Any]) -> str | None:
    """One line for the feed on every run that checked the corpus.

    It used to say nothing when the corpus already matched the project, on the
    argument that a line on every run is a line nobody reads by the third one.
    That was wrong in a way worth recording, because the silence is
    indistinguishable from the phase not having run: a rebuild that re-embeds
    nothing takes ~0.1s and loads no model, so an operator watching a second
    run sees the corpus line from the first one and then, for the rest of the
    project's life, nothing -- and reasonably concludes no embedding ever
    happens. That is the same failure `#run-live` fixed by naming the last
    stage instead of only counting seconds: a working run and a run that
    skipped the work must not look identical. The `current` line is therefore
    the *cheapest* one to write and the most often read, and it says what it
    checked rather than merely that it ran.

    `disabled` is the one state that still says nothing, and for a reason that
    does not apply above: it is a machine-level setting the operator chose,
    which the console header already reports on every poll, rather than
    something that happened to this run.
    """
    source = report.get("source", "")
    if source == "disabled":
        return None
    if source == "current":
        return (
            f"[Corpus] The corpus already matched the project, so nothing was "
            f"re-embedded: {report.get('indexed', 0)} document(s), "
            f"{report.get('total_chunks', 0)} passage(s), checked in "
            f"{report.get('elapsed_s', 0)}s. The Researcher searches it as it "
            "stands."
        )

    errors = report.get("errors") or []
    note = f" {len(errors)} file(s) failed to index." if errors else ""

    if source == "built":
        was = (
            "There was no corpus on this machine"
            if report.get("corpus") == "absent"
            else "The corpus on this machine was empty"
        )
        if not report.get("indexed"):
            return (
                f"[Corpus] {was}, and indexing produced nothing: every file the "
                f"walk offered was too large or unreadable.{note} The Researcher "
                "has nothing to retrieve."
            )
        return (
            f"[Corpus] {was}, so the project was indexed before the run: "
            f"{report['indexed']} document(s), {report.get('total_chunks', 0)} "
            f"passage(s) in {report.get('elapsed_s', 0)}s.{note} The Researcher "
            "searches this like any other corpus."
        )
    if source == "updated":
        changed = report.get("embedded", 0)
        dropped = report.get("dropped", 0)
        parts = []
        if changed:
            parts.append(f"{changed} document(s) re-read")
        if dropped:
            parts.append(f"{dropped} no longer in the project dropped")
        return (
            f"[Corpus] The corpus was behind the project, so it was brought up "
            f"to date first: {', '.join(parts)}, {report.get('reused', 0)} "
            f"unchanged, in {report.get('elapsed_s', 0)}s.{note} The Researcher "
            "searches the project as it is now."
        )
    if source == "stopped_midway":
        return (
            f"[Corpus] Stopped while indexing with {report.get('model')}: "
            f"{report.get('indexed', 0)} document(s) checked and "
            f"{report.get('embedded', 0)} embedded, in {report.get('elapsed_s', 0)}s. "
            "The next run carries on from there: what is already embedded keeps its "
            "vectors."
        )
    if source == "nothing_to_index":
        return (
            "[Corpus] There is nothing here to index -- no "
            f"{', '.join(INDEXABLE_SUFFIXES)} files under "
            f"{report.get('root', '.')}. The Researcher will find nothing; start "
            "the console from the project you mean to work on."
        )
    if source == "error":
        return (
            f"[Corpus] The corpus could not be brought up to date: "
            f"{report.get('note')}. The run continues against it as it stands."
        )
    return "[Corpus] Stopped before the corpus was checked."


def _research_online_before_the_run(goal: str, requested: bool) -> dict[str, Any]:
    """Search the web for the goal and embed what earns a place. Never raises.

    Runs only when the caller asked for it (`requested`), and the caller is the
    operator -- never an agent, exactly as `expect_failures` is. That is a
    measured decision, not caution. Three gates were built and graded against
    thirteen hand-labelled pages from two real runs, and all three failed:
    ranking fetched pages against the project's own documents (a keyword-dense
    marketing page outscores every file in the checkout), skipping the phase
    when the corpus already answers the goal (backwards on the measurement --
    0.351 for a goal needing no web at all, 0.405 for one that did), and the
    calibrated `RETRIEVAL_RELEVANCE_FLOOR` itself (3/13: every page cleared it).
    The lambda run shows why no fourth threshold will do better -- the *wrong*
    pages outscore the right ones on both instruments, 0.553-0.631 for AWS
    Lambda deployment guides against 0.459-0.550 for the Dolphin model pages
    the goal was actually about. "LAMBDA" meant a model on this machine and
    the web means AWS; "local project data" is, as a bag of words, generic
    project-documentation advice. The information that settles it is the
    operator's intent, and it is in no comparison of goal text to page text.

    The cost of guessing wrong is not one bad run. A fetched page becomes a
    permanent corpus member indistinguishable from project knowledge: after the
    run of 2026-09-11 the five blogs it kept took every one of the top five
    retrieval slots for that goal, shutting the project's own files out
    entirely, and would have gone on answering any query near "project
    documentation" for as long as they sat there.

    This runs **before** `graph.stream` and never during it, and that ordering
    is the design rather than a convenience. `_refuse_while_a_run_is_in_flight`
    refuses every other corpus write while a run is live because a corpus
    changing underneath the Researcher manufactures an absence no seat can
    detect: a rebuild half-done returns whatever fraction of itself has been
    re-added, which reads as `no_relevant_knowledge` and routes the run around
    a gap created out from under it. Doing the research first is not a way
    around that rule -- it is the only ordering that obeys it. By the time the
    Architect opens, the corpus is whole and stays that way for the rest of the
    run.

    It reaches the Researcher through the **corpus**, not through state. There
    is deliberately no path by which a fetched page skips retrieval: the pages
    are embedded, and the Researcher finds them with the same search, the same
    hybrid re-rank and the same relevance floor it applies to everything else.
    A web page that cannot be retrieved for this goal should not reach the
    Builder just because it was fetched for it.

    The creating door (`_kb_for_indexing`) is correct here, unlike everywhere
    else the console reads: this *is* indexing, and a goal researched against a
    machine with no corpus should leave one behind holding what it found. The
    staleness report will then say, accurately, that the project's own files
    are still missing from it.

    It is handed over **unopened** for that same sentence to hold. Passing
    `_kb_for_indexing()` created the store on the way into a phase that had not
    yet read a page and might never store one -- switched off, or a goal the
    web cannot answer -- so a machine that had never been indexed came out of
    its first run reporting `empty` instead of `absent`. `research_online`
    calls it on the first page that earns a place.

    Every failure is swallowed into the report. A goal the web cannot answer,
    a search engine that is down, a machine with no network -- none of those is
    a reason to refuse to run against the corpus already on disk, and the
    report distinguishes them so the feed can say which happened.
    """
    if not requested:
        # Kept apart from `disabled` for the reason `search_web` keeps its three
        # empty outcomes apart: "this run did not ask" and "this machine has it
        # switched off" call for different things from the operator, and an
        # empty count reads identically in both.
        return {"source": "not_requested", "documents": 0, "considered": 0, "note":
                "Online research was not requested for this run."}
    if RUN_CONTROL.stopped():
        return {"source": "stopped", "documents": 0, "considered": 0, "note":
                "Stopped before the online research phase began."}
    try:
        report = research_online(_kb_for_indexing, goal)
    except Exception as exc:
        return {"source": "error", "documents": 0, "considered": 0,
                "note": f"The online research phase failed: {exc}"}
    # The phase writes files under research/web/, so the cached walk is now
    # behind what is on disk and the header would call the freshly embedded
    # pages `extra` until it expired.
    forget_expected_documents()
    return report


def _research_feed_line(report: dict[str, Any]) -> str:
    """One line for the feed, saying what the phase actually did.

    Worded so the three empty outcomes cannot be mistaken for each other, for
    the reason `search_web` keeps them apart in the first place: "the web had
    nothing on this" and "we never asked" and "we asked and it broke" call for
    completely different things from the operator, and an empty count reads
    identically in all three.
    """
    source = report.get("source", "")
    considered = report.get("considered", 0)
    kept = report.get("documents", 0)

    if source == "not_requested":
        return (
            "[Research] Online research was not requested; running against the "
            "corpus as it stands. Tick 'Research online' to search the web for "
            "this goal."
        )
    if source == "disabled":
        return "[Research] Online research is switched off; running against the corpus as it stands."
    if source == "stopped":
        return "[Research] Stopped before online research began."
    if source == "error" or (not considered and report.get("errors")):
        return f"[Research] Online research found nothing usable: {report.get('note') or 'the search failed'}"
    if not kept:
        return (
            f"[Research] Read {considered} page(s) and kept none -- none of them "
            "scored well enough against this goal to earn a place in the corpus."
        )
    return (
        f"[Research] Read {considered} page(s), embedded {kept} "
        f"({report.get('chunks', 0)} passages) in {report.get('elapsed_s', 0)}s. "
        "The Researcher retrieves these like any other document."
    )


def rpc_run_goal(params: dict[str, Any]) -> dict[str, Any]:
    """Run a goal through the four-agent loop and return the final state."""
    run_id = uuid.uuid4().hex
    goal = _str_param(params, "goal")
    # Refused before the lock is claimed or the corpus touched. A blank goal
    # otherwise indexed the project and put four seats to work on nothing, and
    # only the console's own textarea stood in the way -- `/api/run` and any
    # other caller went straight through.
    if not goal.strip():
        raise ValueError("A run needs a goal, and this one was empty.")
    # Parsed before anything is claimed, so a refused flag never leaves a run
    # armed with nothing behind it.
    discuss_only = _bool_param(params, "discuss_only")
    research_web = _bool_param(params, "research_web")
    expect_failures = _bool_param(params, "expect_failures")

    # One run at a time, said out loud. The server already assumed it --
    # `_run_progress` is a single global -- and the console could break the
    # assumption, because the textarea's Enter handler was not gated behind the
    # disabled Run button. It also makes the stop unambiguous: there is exactly
    # one run for a Stop to mean.
    #
    # Checked and claimed under one lock, arming included. Two acquisitions
    # would leave a window where two POSTs both pass the check, and a stop
    # armed outside it could reach a run that was then refused.
    with _run_lock:
        if _run_progress["running"]:
            raise ValueError(
                "A run is already in flight. Stop it before starting another."
            )
        RUN_CONTROL.arm(run_id)
        _run_progress.update(
            running=True,
            goal=goal,
            messages=[],
            node="",
            run_id=run_id,
            stopping=False,
        )

    # The embedder goes onto its card before anything else: the corpus phase
    # embeds, and a local seat has to be fitted around the embedder rather than
    # the embedder squeezed in beside the seat -- see
    # `_claim_the_embedder_before_the_run`.
    embedder_report = _claim_the_embedder_before_the_run()
    embedder_line = _embedder_feed_line(embedder_report)
    if embedder_line:
        print(f"[run] embedder -> {embedder_line}")

    # The corpus first, then online research, then the Architect opens. Both
    # phases write to the corpus and both run outside `graph.stream`, because
    # `_refuse_while_a_run_is_in_flight` forbids every other writer from
    # touching it once the stream starts -- doing them here is not a way around
    # that rule but the only ordering that obeys it. Which of the two goes
    # first is itself load-bearing: see `_index_the_project_before_the_run`.
    corpus_report = _index_the_project_before_the_run()
    corpus_line = _corpus_feed_line(corpus_report)
    if corpus_line:
        print(f"[run] corpus -> {corpus_line}")

    # A model other than MiniLM has no relevance floor until its corpus is
    # whole, so it is measured here: after the corpus phase, before any seat
    # searches.
    calibration_report = _calibrate_the_floor_before_the_run(corpus_report)
    calibration_line = _calibration_feed_line(calibration_report)
    if calibration_line:
        print(f"[run] calibration -> {calibration_line}")

    # Online research is off unless the caller asks, the same shape as
    # `expect_failures` below and for the same reason: what this turns on
    # cannot be judged from the goal. A discussion run never researches online
    # whatever the box says -- the phase writes pages under research/web/ and
    # embeds them, which is a change to this machine and to every later run's
    # corpus. "No actions" has to mean that too, so the two flags are resolved
    # here rather than left to the operator to keep consistent.
    research_report = _research_online_before_the_run(
        goal, research_web and not discuss_only
    )
    research_line = _research_feed_line(research_report)
    print(f"[run] research -> {research_line}")
    opening = [
        line
        for line in (embedder_line, corpus_line, calibration_line, research_line)
        if line
    ]
    with _run_lock:
        _run_progress["messages"] = list(opening)

    state: AgentState = {
        "goal": goal,
        # Seeded rather than pushed only to `_run_progress`, which every node
        # update overwrites wholesale: these lines have to survive into the
        # final payload and the snapshot, because what the run was told is part
        # of how its result should be read.
        "messages": list(opening),
        "architecture": "",
        "verdict": "",
        "plan": "",
        "research": "",
        "builder_report": "",
        "next_agent": "Researcher",
        "research_status": "",
        "blockers": "",
        "files_changed": [],
        "failed_verification": [],
        "unverified": [],
        "builder_cut_off": "",
        "lint_failed": [],
        # Opt-out for goals whose product is a file that does not run. Off
        # unless the caller asks, so the default stays strict. Untouched by the
        # stop: a file nobody executed is unproven, not expected-to-fail, and
        # goes on blocking approval either way.
        "expect_failures": expect_failures,
        # Reasoning without acting: the Builder is offered no tools at all.
        # Set by the caller, never by an agent.
        "discuss_only": discuss_only,
        "step_count": 0,
    }
    # Streamed rather than invoked so the last state survives the ceiling.
    # `graph.invoke` raises GraphRecursionError with no partial result, so a run
    # that ran for minutes and produced real work reported nothing at all --
    # from the console it was indistinguishable from a request never sent.
    last = state
    started = time.monotonic()
    over_budget = False
    stopped = False
    node_at_stop = ""

    # Everything below is in a try/finally so the bookkeeping runs even when the
    # graph raises. Without it a run that died left `running` True forever and
    # the console polled a run that no longer existed -- and the snapshot, which
    # is the whole recovery story, would only ever be written by runs that did
    # not need recovering.
    try:
        try:
            for event in graph.stream(state, {"recursion_limit": RECURSION_LIMIT}):
                for node, node_state in event.items():
                    if not isinstance(node_state, dict):
                        continue
                    last = node_state  # type: ignore[assignment]
                    messages = list(node_state.get("messages", []))
                    with _run_lock:
                        _run_progress["node"] = node
                        _run_progress["messages"] = messages
                    # Also on the server's own terminal: a run that is working and
                    # a run that is wedged are otherwise indistinguishable there too.
                    print(f"[run] {node} -> {messages[-1] if messages else '...'}")

                # Both checks sit between supersteps, so the run stops at a node
                # boundary with its state intact rather than mid-call. The nodes
                # carry their own stop checks as well; this one is the backstop
                # that guarantees the run ends however far in the seats got.
                if RUN_CONTROL.stopped():
                    stopped = True
                    node_at_stop = str(_run_progress["node"])
                    break
                if time.monotonic() - started > RUN_BUDGET_SECONDS:
                    over_budget = True
                    break
        except GraphRecursionError:
            last["messages"] = [
                *last.get("messages", []),
                "[Graph] Stopped at the recursion ceiling without an approved "
                "verdict. The work below is what the run produced before it "
                "was cut off.",
            ]

        elapsed = int(time.monotonic() - started)

        # "without an approved verdict" is the usual case and was written as if
        # it were the only one. It is not: both exits are checked *between*
        # supersteps, so a stop or a budget that expires while the Architect is
        # ruling lands after that ruling is already in state. Measured on
        # 2026-09-09 -- a stop sent while the gate was working produced
        # "[Architect] Verdict: approved" immediately above "[Graph] Stopped ...
        # without an approved verdict", the run's own record contradicting
        # itself in the one place anyone finds out how it ended. The verdict is
        # read rather than assumed.
        ruled = str(last.get("verdict", "")) == "approved"
        verdict_clause = (
            "The Architect had already ruled approved; the loop was ending anyway"
            if ruled
            else "without an approved verdict"
        )
        if stopped:
            last["messages"] = [
                *last.get("messages", []),
                f"[Graph] Stopped by the emergency stop after {elapsed}s, at the "
                f"{node_at_stop or 'first'} boundary, {verdict_clause}. "
                "Nothing further was started. Anything already written is listed "
                "below, and anything nobody ran is unproven rather than working.",
            ]
        elif over_budget:
            last["messages"] = [
                *last.get("messages", []),
                f"[Graph] Stopped after {elapsed}s, over the {int(RUN_BUDGET_SECONDS)}s "
                f"budget, {verdict_clause}. The work above is what the run "
                "produced. Raise RUN_BUDGET_SECONDS to give it longer.",
            ]

        # Run-level facts, added to the payload rather than to AgentState:
        # AgentState describes what the agents wrote, not how the run ended.
        payload = dict(last)
        payload.update(
            run_id=run_id,
            stopped=stopped,
            stop_reason=RUN_CONTROL.reason() if stopped else "",
            over_budget=over_budget,
            elapsed_s=elapsed,
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            web_research=research_report,
            embedder=embedder_report,
        )
        _save_snapshot(payload)
        return payload
    except Exception as exc:
        # A run that raised still produced whatever it produced, and that is
        # exactly the case where the operator most wants it back.
        payload = dict(last)
        payload.update(
            run_id=run_id,
            stopped=stopped,
            stop_reason=RUN_CONTROL.reason() if stopped else "",
            over_budget=over_budget,
            elapsed_s=int(time.monotonic() - started),
            finished_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            error=str(exc),
            web_research=research_report,
            embedder=embedder_report,
        )
        _save_snapshot(payload)
        raise
    finally:
        _finish_run()


RPC_METHODS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "rag_stats": rpc_rag_stats,
    "bottleneck": rpc_bottleneck,
    "topics": rpc_topics,
    "duplicate_entities": rpc_duplicate_entities,
    "list_documents": rpc_list_documents,
    "query_graph": rpc_query_graph,
    "graph_overview": rpc_graph_overview,
    "search_documents": rpc_search_documents,
    "upload_document": rpc_upload_document,
    "export_corpus": rpc_export_corpus,
    "clear_corpus": rpc_clear_corpus,
    "list_seats": rpc_list_seats,
    "set_seat": rpc_set_seat,
    "set_thinking": rpc_set_thinking,
    "llm_options": rpc_llm_options,
    "embedding_options": rpc_embedding_options,
    "set_embedding_model": rpc_set_embedding_model,
    "status": rpc_status,
    "run_goal": rpc_run_goal,
    "run_progress": rpc_run_progress,
    "stop_run": rpc_stop_run,
    "last_run": rpc_last_run,
    "shutdown": rpc_shutdown,
}

# Methods the console polls on a timer. Logging these buries everything else.
# `llm_options` is not polled on a timer of its own -- it rides along with
# `list_seats` in the console's `loadCrew`, which is how it was missed when the
# rest of this set was written. It is the same 5s cadence either way: it left
# 851 of the 1233 lines in one session's log, so the run that session was
# started to look at could not be found by reading it.
QUIET_METHODS = {
    "status",
    "rag_stats",
    "list_seats",
    "llm_options",
    "embedding_options",
    "run_progress",
}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory="frontend", **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/status":
            # Compatibility route: launch_console.sh polls this as its
            # readiness check, and it is the same payload as rpc "status".
            self.send_json(rpc_status({}))
        elif parsed.path == "/api/llm-options":
            self.send_json(rpc_llm_options({}))
        elif parsed.path in ("/", "/index.html"):
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        # A body whose length is unusable is refused before it is read, and one
        # that is not a JSON object after. `data.get` on a list or a string
        # raised in `handle_rpc` ahead of its error handling, so a body of `[]`
        # came back as a dropped connection and a traceback in the log.
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._refuse("Content-Length is not a number.")
            return
        if not 0 <= length <= MAX_REQUEST_BYTES:
            self._refuse(
                f"A request body is limited to {MAX_REQUEST_BYTES:,} bytes; "
                f"this one declared {length:,}."
            )
            return
        try:
            data = json.loads(self.rfile.read(length)) if length else {}
        except ValueError:  # bad JSON, or bytes that are not UTF-8
            self._refuse("bad JSON")
            return
        if not isinstance(data, dict):
            self._refuse("The request body must be a JSON object.")
            return

        if parsed.path == "/rpc":
            self.handle_rpc(data)
            return

        # Compatibility routes for the previous /api surface.
        aliases: dict[str, tuple[str, dict[str, Any]]] = {
            "/api/run": ("run_goal", data),
            "/api/stop": ("stop_run", data),
            "/api/search": ("search_documents", data),
            "/api/set-llm": ("set_seat", data),
        }
        if parsed.path in aliases:
            method, params = aliases[parsed.path]
            try:
                self.send_json(RPC_METHODS[method](params))
            except Exception as exc:
                self.send_json({"error": str(exc)})
            return

        self.send_error(404)

    def handle_rpc(self, data: dict[str, Any]) -> None:
        """Dispatch one {method, params} call.

        Errors come back as a 200 with an `error` member rather than an HTTP
        status: the console renders them into its telemetry log, and a failed
        method is not a failed request.

        The reply is written apart from the method's outcome, because the caller
        can be gone by then without anything having failed. A console reloaded
        mid-run drops the request that started the run and reattaches through
        `run_progress` -- a supported path; the run still finishes and saves its
        snapshot. With the write inside the same `try`, the closed socket was
        caught as the method failing: the log read `run_goal FAILED ... Broken
        pipe` for a run that had ended cleanly, then carried two tracebacks,
        because the error reply hit the same socket.
        """
        method = str(data.get("method", ""))
        params = data.get("params") or {}
        if not isinstance(params, dict):
            self._refuse("params must be a JSON object.")
            return
        started = time.perf_counter()

        handler = RPC_METHODS.get(method)
        if handler is None:
            self.send_json(
                {"error": {"message": f"Unknown method: {method}"}, "elapsed_ms": 0}
            )
            return

        try:
            result = handler(params)
            elapsed = int((time.perf_counter() - started) * 1000)
            # Encoded inside the `try`, so a result that will not serialise is
            # still reported as this method failing.
            payload = self._encode({"result": result, "elapsed_ms": elapsed})
            if method not in QUIET_METHODS:
                print(f"[RPC] {method} {elapsed}ms")
        except Exception as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            print(f"[RPC] {method} FAILED {elapsed}ms: {exc}")
            payload = self._encode({"error": {"message": str(exc)}, "elapsed_ms": elapsed})

        try:
            self._send_payload(payload)
        except (BrokenPipeError, ConnectionResetError):
            print(f"[RPC] {method}: the caller disconnected before the reply was sent")

    def _refuse(self, message: str) -> None:
        self.send_json({"error": {"message": message}, "elapsed_ms": 0})

    @staticmethod
    def _encode(obj: Any) -> bytes:
        return json.dumps(obj, default=str).encode()

    def send_json(self, obj: Any) -> None:
        self._send_payload(self._encode(obj))

    def _send_payload(self, payload: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        # RPC calls log themselves in handle_rpc, with their method name and
        # timing; the default line would add a second, less useful entry.
        request_line = args[0] if args else ""
        if isinstance(request_line, str) and (
            "POST /rpc" in request_line or "GET /api/status" in request_line
        ):
            return
        print(f"[API] {request_line}")


def main() -> None:
    """Serve until Ctrl+C or the console asks to exit."""
    # launch_console.sh redirects this process to a log file and tails it, and
    # a redirected stdout is block-buffered -- so the progress and timing lines
    # sat in an 8KB buffer instead of appearing as they happened, which is the
    # opposite of what they are for. A TTY would have line-buffered them, which
    # is why this only shows up under the launcher.
    # (typed as TextIO, which does not declare reconfigure; it is a
    # TextIOWrapper at runtime whenever stdout is a real stream.)
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    port = int(os.getenv("PORT", "8080"))
    print(f"Serving at http://localhost:{port}")
    print("Press Ctrl+C to stop, or use the console's exit button")

    # Threaded: a run takes as long as four cloud models take, and a
    # single-threaded server would stall every status poll behind it.
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    # `serve_forever` moves off the main thread so the main thread is free to
    # wait on both ways out: Ctrl+C, and the console asking to exit. Calling
    # `shutdown()` from the request thread that asked for it would deadlock --
    # it blocks until the serve loop stops, and that loop is what has to
    # deliver the reply.
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()

    asked_from_console = False
    try:
        # Waited in slices rather than indefinitely: an untimed wait swallows
        # the KeyboardInterrupt this loop exists to catch.
        while not _shutdown_requested.wait(0.5):
            pass
        asked_from_console = True
    except KeyboardInterrupt:
        pass

    print("\nShutting down (from the console)..." if asked_from_console else "\nShutting down...")
    if asked_from_console:
        # Let the reply that asked for this reach the browser.
        time.sleep(SHUTDOWN_GRACE_SECONDS)
    try:
        server.shutdown()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
