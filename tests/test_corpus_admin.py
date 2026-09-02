"""Tests for clearing and exporting the corpus.

Two layers, both without the embedding model. `GraphRAGKnowledgeBase.__init__`
loads sentence-transformers and opens Chroma, which is the slow part of this
suite and has nothing to do with what is under test here: `clear` and
`export_corpus` touch only `self.graph`, `self.collection` and
`self.persist_dir`. So the knowledge base is built field by field around a fake
collection, and the RPC layer is called directly the way `test_console_stop.py`
calls it.
"""

from __future__ import annotations

import json
from typing import Any

import networkx as nx
import pytest

import serve
from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase


class _FakeCollection:
    """The slice of the Chroma collection API these two methods use.

    `get(include=[])` returning ids is the idiom `index_project_files` already
    relies on, so the fake has to honour it: ids always come back, `documents`
    and `metadatas` only when asked for.
    """

    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, dict[str, Any]]] = {}
        self.fail_on_delete = False

    def add(self, doc_id: str, content: str, metadata: dict[str, Any]) -> None:
        self.rows[doc_id] = (content, metadata)

    def get(self, include: list[str] | None = None) -> dict[str, Any]:
        include = include or []
        ids = list(self.rows)
        out: dict[str, Any] = {"ids": ids}
        if "documents" in include:
            out["documents"] = [self.rows[i][0] for i in ids]
        if "metadatas" in include:
            out["metadatas"] = [self.rows[i][1] for i in ids]
        return out

    def delete(self, ids: list[str]) -> None:
        if self.fail_on_delete:
            raise RuntimeError("chroma is unavailable")
        for doc_id in ids:
            self.rows.pop(doc_id, None)

    def count(self) -> int:
        return len(self.rows)


def _edges(node_link: dict[str, Any]) -> list[dict[str, Any]]:
    """The edge list out of `node_link_data`, whatever this networkx calls it.

    The default key moved from `links` to `edges` in networkx 3.4. Reading both
    keeps these tests honest across the range the project can be installed
    against, rather than passing only on the version that happens to be here.
    """
    return node_link.get("edges", node_link.get("links", []))


def _make_kb(tmp_path) -> GraphRAGKnowledgeBase:
    """A knowledge base with a real graph and a fake vector store."""
    kb = object.__new__(GraphRAGKnowledgeBase)
    kb.persist_dir = tmp_path
    kb.collection = _FakeCollection()
    kb.graph = nx.DiGraph()

    for path, content in (
        ("serve.py", "The Architect rules on the plan."),
        ("README.md", "Ambiguity console."),
    ):
        kb.collection.add(path, content, {"path": path, "type": "python"})
        kb.graph.add_node(path, type="document", content=content[:200], path=path)
        kb.graph.add_node("Architect", type="entity")
        kb.graph.add_edge(path, "Architect", relation="mentions")

    kb._save_graph()
    return kb


@pytest.fixture
def kb(tmp_path):
    return _make_kb(tmp_path)


# ---------------------------------------------------------------------------
# clear
# ---------------------------------------------------------------------------


def test_clear_empties_both_halves_and_reports_what_went(kb):
    before = kb.stats()
    assert before["total_chunks"] and before["total_nodes"]

    result = kb.clear()

    assert result["removed_chunks"] == before["total_chunks"]
    assert result["removed_nodes"] == before["total_nodes"]
    assert result["removed_edges"] == before["total_edges"]
    assert result["total_chunks"] == 0
    assert result["total_nodes"] == 0
    assert result["total_edges"] == 0
    assert result["total_documents"] == 0


def test_clear_survives_a_restart(kb, tmp_path):
    """The clear has to reach disk, not just memory.

    `index_project_files` has this hole: its `graph.clear()` is persisted only
    as a side effect of indexing something afterwards. A clear that indexes
    nothing afterwards would leave the old graph on disk and the corpus would
    come back at the next process start.
    """
    kb.clear()

    reloaded = json.loads((tmp_path / "knowledge_graph.json").read_text())
    assert reloaded["nodes"] == []
    assert _edges(reloaded) == []


def test_a_chroma_failure_leaves_the_graph_alone(kb):
    """Half a wipe is worse than none: the two halves have to agree."""
    kb.collection.fail_on_delete = True
    nodes_before = kb.graph.number_of_nodes()

    with pytest.raises(RuntimeError):
        kb.clear()

    assert kb.graph.number_of_nodes() == nodes_before
    assert kb.collection.count() == 2


def test_clearing_an_empty_corpus_is_not_an_error(tmp_path):
    kb = object.__new__(GraphRAGKnowledgeBase)
    kb.persist_dir = tmp_path
    kb.collection = _FakeCollection()
    kb.graph = nx.DiGraph()

    result = kb.clear()

    assert result["removed_chunks"] == 0
    assert result["total_nodes"] == 0


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


def test_export_carries_the_graph_and_every_chunk(kb):
    export = kb.export_corpus()

    exported_ids = {chunk["id"] for chunk in export["chunks"]}
    assert exported_ids == set(kb.collection.rows)

    graph_nodes = {node["id"] for node in export["graph"]["nodes"]}
    assert graph_nodes == set(kb.graph.nodes())
    assert len(_edges(export["graph"])) == kb.graph.number_of_edges()

    assert export["stats"] == kb.stats()
    assert export["errors"] == []

    chunk = next(c for c in export["chunks"] if c["id"] == "serve.py")
    assert chunk["content"] == "The Architect rules on the plan."
    assert chunk["metadata"]["path"] == "serve.py"


def test_export_omits_embeddings_and_says_so(kb):
    """Asserted structurally, on the keys.

    Grepping the serialised blob for the word looks equivalent and is not: the
    corpus is source files, and `graphrag_server.py` is itself a document in it,
    so the word appears in a chunk's *text* on any real corpus. That check
    passes here only because this fixture's text happens not to say it.
    """
    export = kb.export_corpus()

    assert "embeddings" not in export
    assert all("embeddings" not in chunk for chunk in export["chunks"])
    assert all("embeddings" not in node for node in export["graph"]["nodes"])

    assert "Embeddings are omitted" in export["note"]
    assert export["embedding_model"] == "all-MiniLM-L6-v2"


def test_export_is_json_serialisable(kb):
    """The RPC layer will do exactly this, and a failure there is a 500."""
    payload = json.dumps(kb.export_corpus())
    assert json.loads(payload)["stats"]["total_chunks"] == 2


def test_a_chroma_failure_still_exports_the_graph(kb):
    """The graph lives in memory; losing Chroma must not cost us both."""

    def boom(**_: Any) -> dict[str, Any]:
        raise RuntimeError("chroma is unavailable")

    kb.collection.get = boom  # type: ignore[method-assign]

    export = kb.export_corpus()

    assert export["chunks"] == []
    assert any("chroma is unavailable" in err for err in export["errors"])
    assert export["graph"]["nodes"]


def test_export_of_an_empty_corpus_is_still_a_document(tmp_path):
    kb = object.__new__(GraphRAGKnowledgeBase)
    kb.persist_dir = tmp_path
    kb.collection = _FakeCollection()
    kb.graph = nx.DiGraph()

    export = kb.export_corpus()

    assert export["chunks"] == []
    assert export["graph"]["nodes"] == []
    assert export["exported_at"]


# ---------------------------------------------------------------------------
# the RPC layer
# ---------------------------------------------------------------------------


@pytest.fixture
def console_kb(kb, monkeypatch):
    """Point the console's `_kb()` at the fake, and start with no run armed."""
    monkeypatch.setattr(serve, "kb", kb)
    with serve._run_lock:
        serve._run_progress.update(running=False, goal="", node="", run_id="")
    yield kb
    with serve._run_lock:
        serve._run_progress.update(running=False, run_id="")


def test_rpc_export_returns_a_serialisable_corpus(console_kb):
    result = serve.rpc_export_corpus({})
    assert json.loads(json.dumps(result))["stats"]["total_chunks"] == 2


def test_rpc_clear_empties_the_corpus(console_kb):
    result = serve.rpc_clear_corpus({})

    assert result["removed_chunks"] == 2
    assert console_kb.graph.number_of_nodes() == 0
    assert console_kb.collection.count() == 0


def test_rpc_clear_is_refused_while_a_run_is_in_flight(console_kb):
    """An emptied corpus does not fail the Researcher's search, it just answers
    nothing -- which reads as `no_relevant_knowledge`. So the operator is the
    only one who can be told."""
    with serve._run_lock:
        serve._run_progress.update(running=True, goal="build the thing")

    with pytest.raises(ValueError, match="run is in flight"):
        serve.rpc_clear_corpus({})

    assert console_kb.graph.number_of_nodes() == 3
    assert console_kb.collection.count() == 2


def test_rpc_reindex_is_refused_while_a_run_is_in_flight(console_kb, monkeypatch):
    """Same hazard as the clear, and it does not even need to finish to cause it.

    A rebuild clears the graph up front and re-adds documents one at a time, so
    a search landing partway through is answered from a corpus that is neither
    the old one nor the new one -- and is answered, not failed.
    """
    called = []
    monkeypatch.setattr(serve, "index_project_files", lambda kb: called.append(kb))

    with serve._run_lock:
        serve._run_progress.update(running=True, goal="build the thing")

    with pytest.raises(ValueError, match="run is in flight"):
        serve.rpc_reindex({})

    assert called == []  # refused before it touched the corpus
    assert console_kb.graph.number_of_nodes() == 3
    assert console_kb.collection.count() == 2


def test_rpc_reindex_runs_when_nothing_is_in_flight(console_kb, monkeypatch):
    """The guard must not be a permanent block on the button."""
    monkeypatch.setattr(
        serve, "index_project_files", lambda kb: {"indexed": 7, "errors": []}
    )

    assert serve.rpc_reindex({})["indexed"] == 7


def test_the_refusal_names_the_run_it_is_protecting(console_kb):
    """A refusal that does not say what is running leaves the operator to guess
    whether they still care about it -- the same reason `rpc_shutdown` names it."""
    with serve._run_lock:
        serve._run_progress.update(running=True, goal="port the console")

    with pytest.raises(ValueError, match="port the console"):
        serve.rpc_clear_corpus({})


def test_the_refusal_survives_a_run_with_no_goal(console_kb):
    """`goal` is empty until a run sets it; the guard must not trip on that."""
    with serve._run_lock:
        serve._run_progress.update(running=True, goal="")

    with pytest.raises(ValueError, match="Stop the run first."):
        serve.rpc_clear_corpus({})


def test_export_is_allowed_while_a_run_is_in_flight(console_kb):
    """Reading the corpus takes nothing away from the run using it."""
    with serve._run_lock:
        serve._run_progress.update(running=True, goal="build the thing")

    assert serve.rpc_export_corpus({})["stats"]["total_chunks"] == 2


def test_both_methods_are_registered_and_not_quiet():
    """These are operator actions; the telemetry feed has to show them."""
    assert serve.RPC_METHODS["export_corpus"] is serve.rpc_export_corpus
    assert serve.RPC_METHODS["clear_corpus"] is serve.rpc_clear_corpus
    assert "export_corpus" not in serve.QUIET_METHODS
    assert "clear_corpus" not in serve.QUIET_METHODS
