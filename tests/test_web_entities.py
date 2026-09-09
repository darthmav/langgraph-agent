"""Tests for keeping fetched web pages out of the knowledge graph's entities.

A page the research phase fetched is a *retrieval source*, not knowledge-graph
material, and `add_document`'s capital rule is far worse on web prose than on
this project's own files. `ENTITY_STOPWORDS` is a hand-audited list and it was
audited against source code and numpy docstrings, where the forced capitals are
`Returns`, `Every`, `False`. Web prose opens sentences with a different
vocabulary entirely and none of it is on the list.

Measured on 2026-09-09, after a few runs of the phase: 15 fetched pages had
minted **551 entities no project document mentions — 19% of the whole graph, at
36.7 per page** — `Although`, `Afterward`, `Altogether`, `Again`, `Accessed`.
Skipping them took the graph from 2,830 entities to 2,285 and duplicate
candidates from 205 to 171. Those edges are read as evidence by `neighborhood`,
`topics` and `duplicate_entities`, so the graph degraded as the corpus grew.

The decision is made from the document's **path**, never from the call site,
which is the load-bearing part: a reindex re-reads these files as ordinary
markdown, so a rule applied only where a page is first stored would be silently
reversed by the next rebuild.
"""

from __future__ import annotations

import pytest
from test_chunking import _make_kb

from langgraph_agent.graphrag_server import WEB_RESEARCH_DIR, _is_web_document

PROSE = (
    "Although the parameter saturates, Afterward the ranking changes. "
    "Altogether the Builder reads Documents from Chroma. Again and Again."
)


@pytest.fixture
def kb(tmp_path):
    return _make_kb(tmp_path)


def _entities(kb):
    return {n for n, a in kb.graph.nodes(data=True) if a.get("type") == "entity"}


# ---------------------------------------------------------------------------
# recognising one
# ---------------------------------------------------------------------------


def test_a_web_page_is_recognised_by_its_directory():
    assert _is_web_document(f"{WEB_RESEARCH_DIR}/example-com-guide-a1b2c3d4.md")


def test_an_absolute_root_still_resolves():
    """Every test builds its corpus under an absolute tmp_path."""
    assert _is_web_document(f"/tmp/whatever/{WEB_RESEARCH_DIR}/page-00000000.md")


def test_a_project_file_is_not_a_web_page():
    for path in ("CLAUDE.md", "src/langgraph_agent/nodes.py", "uploads/notes.md",
                 "research/spectral_graph_theory.md", "docs/research/web-notes.md"):
        assert not _is_web_document(path), path


def test_the_directory_itself_is_not_a_document():
    """`research/web` names the folder, not a page inside it."""
    assert not _is_web_document(WEB_RESEARCH_DIR)


# ---------------------------------------------------------------------------
# what that changes
# ---------------------------------------------------------------------------


def test_a_web_page_mints_no_entities(kb):
    kb.add_document(f"{WEB_RESEARCH_DIR}/example-com-guide-a1b2c3d4.md", PROSE,
                    {"path": f"{WEB_RESEARCH_DIR}/example-com-guide-a1b2c3d4.md", "type": "markdown"})

    assert _entities(kb) == set()


def test_the_same_prose_in_a_project_file_still_does(kb):
    """The rule is about where the document came from, not what it says.

    Stated as a pair so the test cannot pass because entity extraction broke.
    """
    kb.add_document("notes.md", PROSE, {"path": "notes.md", "type": "markdown"})

    assert "Although" in _entities(kb)


def test_a_web_page_is_still_a_retrievable_document(kb):
    """It stops voting on entities; it does not stop being in the corpus."""
    doc = f"{WEB_RESEARCH_DIR}/example-com-guide-a1b2c3d4.md"
    chunks = kb.add_document(doc, PROSE, {"path": doc, "type": "markdown"})

    assert chunks >= 1
    assert kb.collection.count() >= 1
    documents = [n for n, a in kb.graph.nodes(data=True) if a.get("type") == "document"]
    assert documents == [doc]


def test_a_reindex_makes_the_same_decision(kb):
    """The trap this is designed against: a rebuild re-reads it as plain markdown.

    `index_project_files` calls `add_document` with nothing but the path and the
    text, so a rule that lived at the storing call site would be undone here —
    silently, in a pass reporting success.
    """
    doc = f"{WEB_RESEARCH_DIR}/example-com-guide-a1b2c3d4.md"
    kb.add_document(doc, PROSE, {"path": doc, "type": "markdown"})
    kb.graph.clear()
    kb.add_document(doc, PROSE, {"path": doc, "type": "markdown"})

    assert _entities(kb) == set()


def test_entity_free_pages_do_not_read_as_a_broken_extractor(kb):
    """The side effect of the rule above, and why `connectivity` excludes them.

    A page with no entities has no edges, so it is an isolated node — which is
    precisely the symptom `connectivity()` exists to detect, and the only place
    an entity-extraction regression is visible at all. Counting deliberately
    entity-free pages there would sit 15 permanent false isolates on top of the
    one reading that matters.
    """
    kb.add_document("notes.md", PROSE, {"path": "notes.md", "type": "markdown"})
    before = kb.connectivity()

    for n in range(3):
        doc = f"{WEB_RESEARCH_DIR}/example-com-{n}-0000000{n}.md"
        kb.add_document(doc, PROSE, {"path": doc, "type": "markdown"})
    after = kb.connectivity()

    assert after["isolated_nodes"] == before["isolated_nodes"] == 0
    assert after["components"] == before["components"]
    # Excluded, but said out loud rather than silently dropped.
    assert after["web_documents"] == 3
