"""The corpus is built when someone asks for one, and not otherwise.

There used to be no distinction between opening a corpus and creating one:
`GraphRAGKnowledgeBase.__init__` makes the store on disk, and every read went
through it. So a corpus appeared as a side effect of asking whether there was
a corpus. `serve` started a preload thread at import, and the console polls
`rag_stats` every five seconds, which together meant that starting the server
was enough to leave a store behind -- one that then reported itself as a
knowledge base to everything that looked, having never been indexed.

What is pinned here is the split that fixes it: `open_knowledge_base` reads,
`get_knowledge_base` builds, and only an explicit index goes through the
second one. The tests avoid the embedding model throughout, which is also the
claim of the last one in the file.
"""

from __future__ import annotations

import sys

import pytest

import serve
from langgraph_agent import graphrag_server
from langgraph_agent.graphrag_server import (
    GraphRAGKnowledgeBase,
    corpus_exists,
    corpus_state,
    open_knowledge_base,
)


@pytest.fixture
def nowhere(tmp_path, monkeypatch):
    """A machine where nobody has ever indexed.

    The singleton is cleared as well as the directory: it is process-global,
    so a corpus another test built would answer for this one and hide exactly
    the failure under test.
    """
    monkeypatch.setattr(graphrag_server, "_kb_instance", None)
    monkeypatch.setattr(serve, "kb", None)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def store(tmp_path, monkeypatch) -> str:
    """A path for a *real* store, absolute and unique to this test.

    Chroma caches its system by the path string it is handed, so two corpora
    in one process both opened as the relative "./knowledge" quietly share the
    first one's store -- and the second test then sees a directory that was
    never created under its own tmp_path.
    """
    monkeypatch.setattr(graphrag_server, "_kb_instance", None)
    return str(tmp_path / "knowledge")


def _touched(path) -> list[str]:
    return sorted(p.name for p in path.iterdir())


# ---------------------------------------------------------------------------
# the two doors
# ---------------------------------------------------------------------------


def test_no_corpus_reads_as_absent_and_leaves_nothing_behind(nowhere):
    assert corpus_exists(str(nowhere)) is False
    assert corpus_state(str(nowhere))[0] == "absent"
    assert open_knowledge_base(str(nowhere)) is None
    assert _touched(nowhere) == []


def test_indexing_is_the_act_that_creates_the_store(store):
    """`get_knowledge_base` is the one door allowed to bring a corpus into being."""
    assert corpus_exists(store) is False

    kb = graphrag_server.get_knowledge_base(store)

    assert corpus_exists(store) is True
    # And once it exists, the reading door finds it.
    assert open_knowledge_base(store) is kb


def test_a_cleared_corpus_is_empty_not_absent(store):
    """Emptying in place leaves a real store, and the two must not be confused.

    Both answer a search with nothing, and only one of them means "press
    Reindex" -- the other means the operator already did and then cleared it.
    """
    kb = graphrag_server.get_knowledge_base(store)
    kb.clear()

    assert corpus_state(store)[0] == "empty"
    assert corpus_exists(store) is True


# ---------------------------------------------------------------------------
# every read the console makes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, params",
    [
        ("status", {}),
        ("rag_stats", {}),
        ("list_documents", {}),
        ("query_graph", {"node_id": "serve.py"}),
        ("search_documents", {"query": "the planner"}),
    ],
)
def test_a_read_never_creates_a_corpus(nowhere, method, params):
    """The console polls these on a timer. A read that creates is a corpus
    nobody asked for, arriving seconds after the server starts."""
    serve.RPC_METHODS[method](params)

    assert _touched(nowhere) == []
    assert serve.kb is None


def test_the_reads_report_the_absence_rather_than_zeros(nowhere):
    """Four zeros read as a knowledge base that happens to be empty."""
    assert serve.rpc_status({})["corpus"] == "absent"
    assert serve.rpc_status({})["graphrag"] is False

    stats = serve.rpc_rag_stats({})
    assert stats["corpus"] == "absent"
    assert stats["total_chunks"] == 0
    assert "reindex" in stats["note"].lower()

    assert serve.rpc_list_documents({})["corpus"] == "absent"
    assert serve.rpc_query_graph({"node_id": "serve.py"})["corpus"] == "absent"


def test_a_search_with_no_corpus_returns_no_hits_and_says_why(nowhere):
    """Not one fabricated row. The Builder reads this field."""
    result = serve.rpc_search_documents({"query": "the planner"})

    assert result["results"] == []
    assert result["source"] == "no_corpus"
    assert result["note"]


# ---------------------------------------------------------------------------
# the writes
# ---------------------------------------------------------------------------


def test_export_and_clear_refuse_rather_than_create_one_to_act_on(nowhere):
    """Creating a store in order to empty it is the opposite of the ask."""
    with pytest.raises(ValueError, match="no corpus to export"):
        serve.rpc_export_corpus({})
    with pytest.raises(ValueError, match="no corpus to clear"):
        serve.rpc_clear_corpus({})

    assert _touched(nowhere) == []


def test_reindex_goes_through_the_creating_door(nowhere, monkeypatch):
    """The console's Reindex button is the operator asking for a corpus.

    It is the one method that may build one, so it is the one method wired to
    `get_knowledge_base`. Had it read like the others it would be handed
    `None` here and index into nothing.
    """
    built = object()  # stands in for a corpus that did not exist a moment ago
    monkeypatch.setattr(serve, "get_knowledge_base", lambda: built)
    monkeypatch.setattr(serve, "index_project_files", lambda kb: {"indexed": 3, "kb": kb})

    result = serve.rpc_reindex({})

    assert result["indexed"] == 3
    assert result["kb"] is built
    assert _touched(nowhere) == []  # the fake built it; nothing else did


# ---------------------------------------------------------------------------
# the Researcher's own door
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_researcher_is_told_there_is_no_corpus(nowhere):
    """It used to be handed `[GraphRAG not indexed]` at score 0.0 -- a made-up
    retrieval hit sitting in the field real ones arrive in."""
    from langgraph_agent.mcp_client import MCPClient

    client = MCPClient()
    await client.connect()
    result = await client.call_tool("search_knowledge_graph", {"query": "the planner"})

    assert result["results"] == []
    assert result["source"] == "no_corpus"
    assert _touched(nowhere) == []


@pytest.mark.asyncio
async def test_a_graph_query_with_no_corpus_is_empty_not_stubbed(nowhere):
    from langgraph_agent.mcp_client import MCPClient

    client = MCPClient()
    await client.connect()
    result = await client.call_tool("query_knowledge_graph", {"entity": "Planner"})

    assert result["neighbors"] == []
    assert result["source"] == "no_corpus"
    assert _touched(nowhere) == []


# ---------------------------------------------------------------------------
# the local model
# ---------------------------------------------------------------------------


def test_opening_a_corpus_does_not_load_the_embedding_model(store):
    """The one thing that runs on this machine loads when something embeds.

    It used to load in `__init__`, so every header poll paid for it and
    importing the module pulled in torch behind it. Counting documents,
    listing them, drawing the graph and exporting all touch neither.
    """
    kb = GraphRAGKnowledgeBase(store)

    assert kb._embedder is None
    kb.stats()
    kb.list_documents()
    kb.export_corpus()
    assert kb._embedder is None


def test_importing_the_server_does_not_import_sentence_transformers():
    """`serve` is imported by this suite already; the assertion is that its
    import did not drag the model in. Skipped if something else has since."""
    if "sentence_transformers" in sys.modules:
        pytest.skip("another test in this session has already loaded the model")

    assert "serve" in sys.modules
    assert "torch" not in sys.modules
