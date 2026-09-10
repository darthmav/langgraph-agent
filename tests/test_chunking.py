"""Tests for splitting a document into passages the embedder can read whole.

Before chunking, `add_document` embedded a file in one `encode()` call while
the model's window is 256 tokens, so everything past roughly the first thousand
characters was silently discarded: measured on this project's own corpus, 73 of
77 documents truncated and 91.5% of the corpus unreachable by search, with the
vector for all 46,094 characters of CLAUDE.md identical to the vector for its
first 1,000. Nothing raised, and the counters read the same either way -- which
is why these are tests and not a note in the README.

No test here loads sentence-transformers. The suite goes out of its way to keep
the model out (`test_stats_does_not_load_the_embedder_for_the_health_check`
asserts as much), so the packing, the trimming and the collapse back onto
documents are all exercised against a stand-in tokenizer that reproduces the
one behaviour that matters: a slice re-tokenizes to a different length than it
had inside its parent.
"""

from __future__ import annotations

import re
from typing import Any

import networkx as nx
import pytest

from langgraph_agent.graphrag_server import (
    CHUNK_ID_SEPARATOR,
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    ENTITY_STOPWORDS,
    GraphRAGKnowledgeBase,
    _chunk_windows,
    _document_id_of,
    index_project_files,
)


class _FakeTokenizer:
    """A word-piece stand-in: runs of non-space text, split every `piece` chars.

    The splitting is what makes it a useful fake rather than a simple one. A
    real word-piece tokenizer decides on context, so a fragment cut mid-word
    encodes to a different number of tokens standalone than it did inside the
    document. Aligning the pieces from the *start of each run* reproduces
    exactly that: slice a run down the middle and the pieces re-align, and the
    count moves.
    """

    def __init__(self, piece: int = 4) -> None:
        self.piece = piece

    def _spans(self, text: str) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        for match in re.finditer(r"\S+", text):
            start, end = match.span()
            for i in range(start, end, self.piece):
                spans.append((i, min(i + self.piece, end)))
        return spans

    def __call__(
        self,
        text: str | list[str],
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
        truncation: bool = False,
        verbose: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(text, list):
            spans = [self._spans(one) for one in text]
            out: dict[str, Any] = {"input_ids": [[0] * len(s) for s in spans]}
            if return_offsets_mapping:
                out["offset_mapping"] = spans
            return out

        one_spans = self._spans(text)
        out = {"input_ids": [0] * len(one_spans)}
        if return_offsets_mapping:
            out["offset_mapping"] = one_spans
        return out


class _FakeEmbedder:
    """Enough of a SentenceTransformer for chunking: a tokenizer and a vector."""

    def __init__(self, piece: int = 4) -> None:
        self.tokenizer = _FakeTokenizer(piece)

    def encode(self, text: str | list[str]) -> Any:
        import numpy as np

        if isinstance(text, list):
            return np.zeros((len(text), 3))
        return np.zeros(3)


class _FakeChunkCollection:
    """The slice of the Chroma API the chunked write and read paths use.

    Supports `delete(where=...)` because that is how a document's previous
    chunks are swept before its new ones land; a collection without it is
    covered separately by `_FakeCollection` in `test_corpus_admin.py`.
    """

    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, dict[str, Any], list[float]]] = {}

    def upsert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for i, chunk_id in enumerate(ids):
            self.rows[chunk_id] = (documents[i], metadatas[i], embeddings[i])

    def get(self, include: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        include = include or []
        ids = list(self.rows)
        out: dict[str, Any] = {"ids": ids}
        if "documents" in include:
            out["documents"] = [self.rows[i][0] for i in ids]
        if "metadatas" in include:
            out["metadatas"] = [self.rows[i][1] for i in ids]
        return out

    def delete(
        self, ids: list[str] | None = None, where: dict[str, Any] | None = None
    ) -> None:
        if where is not None:
            doomed = [
                i for i, (_, meta, _) in self.rows.items()
                if all(meta.get(k) == v for k, v in where.items())
            ]
            for chunk_id in doomed:
                self.rows.pop(chunk_id, None)
        for chunk_id in ids or []:
            self.rows.pop(chunk_id, None)

    def count(self) -> int:
        return len(self.rows)

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Hits in insertion order, with a distance that rises down the list.

        The ranking is not what is under test; the collapse back onto documents
        is, and that only needs hits to arrive best-first the way Chroma
        delivers them.
        """
        ids = list(self.rows)[:n_results]
        return {
            "ids": [ids],
            "documents": [[self.rows[i][0] for i in ids]],
            "metadatas": [[self.rows[i][1] for i in ids]],
            "distances": [[0.1 * rank for rank in range(len(ids))]],
        }


def _make_kb(tmp_path, piece: int = 4) -> GraphRAGKnowledgeBase:
    kb = object.__new__(GraphRAGKnowledgeBase)
    kb.persist_dir = tmp_path
    kb.collection = _FakeChunkCollection()
    kb.graph = nx.DiGraph()
    kb._embedder = _FakeEmbedder(piece)  # type: ignore[assignment]
    return kb


@pytest.fixture
def kb(tmp_path):
    return _make_kb(tmp_path)


# ---------------------------------------------------------------------------
# the packing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_tokens", [1, 50, 253, 254, 255, 500, 1000, 11760])
def test_every_token_lands_in_a_window_and_none_exceeds_the_budget(n_tokens):
    """The two halves of the contract, and the bug is the first one.

    Full coverage is what stops the truncation. The budget is what stops each
    chunk from being truncated in turn, which would reintroduce the same bug
    one level down.
    """
    windows = _chunk_windows(n_tokens, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS)

    covered: set[int] = set()
    for start, end in windows:
        assert start < end, "an empty window carries no tokens"
        assert end - start <= CHUNK_MAX_TOKENS
        covered.update(range(start, end))

    assert covered == set(range(n_tokens))


def test_an_empty_document_produces_no_windows():
    assert _chunk_windows(0, CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS) == []


def test_an_overlap_wider_than_the_window_still_terminates():
    """A stride at or below zero would loop forever rather than fail loudly.

    A misconfigured overlap is a caller's mistake, but hanging the reindex is a
    worse answer to it than degrading to a narrower overlap.
    """
    windows = _chunk_windows(100, 10, 999)

    assert windows, "the document still has to be covered"
    covered: set[int] = set()
    for start, end in windows:
        covered.update(range(start, end))
    assert covered == set(range(100))


# ---------------------------------------------------------------------------
# the slicing
# ---------------------------------------------------------------------------


def test_a_document_inside_the_window_is_left_whole(kb):
    content = "one two three four five"

    assert kb.chunk_text(content) == [content]


def test_an_empty_document_yields_no_chunks(kb):
    assert kb.chunk_text("") == []
    assert kb.chunk_text("   \n  ") == []


def test_the_chunks_are_literal_substrings_that_span_the_document(kb):
    """Coverage, stated the way a reader can check it.

    The first chunk has to start the document and the last has to end it, or
    something was dropped off an end -- which is the original bug, just at a
    smaller scale.
    """
    content = " ".join(f"word{i}" for i in range(2000))

    chunks = kb.chunk_text(content)

    assert len(chunks) > 1, "this document is far past one window"
    for chunk in chunks:
        assert chunk in content
    assert content.startswith(chunks[0])
    assert content.endswith(chunks[-1])


def test_neighbouring_chunks_overlap(kb):
    """The overlap is what keeps a passage cut down the middle findable.

    It is also what makes trimming safe in `_fit_chunks`, so a run of chunks
    that did not actually overlap would break that argument silently.
    """
    content = " ".join(f"word{i}" for i in range(2000))

    chunks = kb.chunk_text(content)

    starts = [content.index(chunk) for chunk in chunks[:3]]
    ends = [start + len(chunk) for start, chunk in zip(starts, chunks[:3], strict=True)]
    assert starts[1] < ends[0], "chunk 2 begins after chunk 1 ended: no overlap"
    assert starts[2] < ends[1]


# ---------------------------------------------------------------------------
# the repair
# ---------------------------------------------------------------------------


def _over_budget_text(extra: int = 5) -> str:
    """A passage that tokenizes past the window under `_FakeTokenizer`."""
    return " ".join(["w"] * (CHUNK_MAX_TOKENS + extra))


def test_a_chunk_that_retokenizes_over_the_window_is_trimmed(kb):
    """Slicing on the parent's token boundaries does not bound the slice.

    Measured against the real tokenizer, 16 of 1,052 full-size chunks came back
    one token longer standalone than they were inside the document, which put
    them at 257 against a 256 window -- handed straight back to the truncation
    this change exists to remove. Padding the constant by the drift observed
    once is guessing; re-encoding and cutting makes the window a fact.
    """
    over = _over_budget_text()

    fitted = kb._fit_chunks([over, "a tail chunk"])

    assert len(kb.embedder.tokenizer(fitted[0])["input_ids"]) == CHUNK_MAX_TOKENS
    assert over.startswith(fitted[0]), "a middle chunk is trimmed at its tail"


def test_the_last_chunk_is_trimmed_at_the_head_so_the_documents_tail_survives(kb):
    """Which end is cut is the whole point.

    A chunk's tail is covered by the next chunk's overlap and its head by the
    previous one's, so trimming into a neighbour loses nothing. The last chunk
    has no next, and trimming *its* tail would drop the final tokens of the
    file without saying so -- the original bug in miniature.
    """
    over = _over_budget_text()

    fitted = kb._fit_chunks(["a head chunk", over])

    assert len(kb.embedder.tokenizer(fitted[-1])["input_ids"]) == CHUNK_MAX_TOKENS
    assert over.endswith(fitted[-1]), "the last chunk must keep the document's end"


def test_a_lone_chunk_is_never_trimmed(kb):
    """It is the whole document, un-sliced, so it cannot have drifted."""
    over = _over_budget_text()

    assert kb._fit_chunks([over]) == [over]


# ---------------------------------------------------------------------------
# chunk ids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chunk_id", "metadata", "expected"),
    [
        ("CLAUDE.md#0007", None, "CLAUDE.md"),
        ("CLAUDE.md#0007", {"doc_id": "CLAUDE.md"}, "CLAUDE.md"),
        # A row written before chunking is keyed by the bare path and carries
        # no doc_id. It has to resolve to itself or a reindex reads it as a
        # stranger and prunes it.
        ("CLAUDE.md", None, "CLAUDE.md"),
        # "#" in a real path is not a chunk number and must be left alone.
        ("weird#name.py", None, "weird#name.py"),
        ("a/b#3.md", None, "a/b#3.md"),
    ],
)
def test_a_stored_row_resolves_to_the_document_it_came_from(
    chunk_id, metadata, expected
):
    assert _document_id_of(chunk_id, metadata) == expected


# ---------------------------------------------------------------------------
# writing and reading
# ---------------------------------------------------------------------------


def test_a_document_is_stored_as_several_rows_but_one_graph_node(kb):
    """Chunking changes what search can find, not what the corpus is shaped like.

    The graph is what every spectral diagnostic runs on, so a document that
    started arriving as forty nodes would quietly change the meaning of every
    one of them.
    """
    content = " ".join(f"word{i}" for i in range(2000))

    kb.add_document("big.md", content, {"path": "big.md", "type": "markdown"})

    assert kb.collection.count() > 1, "a document past the window is several rows"
    documents = [n for n, a in kb.graph.nodes(data=True) if a.get("type") == "document"]
    assert documents == ["big.md"]

    for chunk_id, (_, metadata, _) in kb.collection.rows.items():
        assert chunk_id.startswith(f"big.md{CHUNK_ID_SEPARATOR}")
        assert metadata["doc_id"] == "big.md"
        assert metadata["path"] == "big.md", "the caller's metadata is preserved"


def test_a_shrunken_document_leaves_no_leftover_chunks(kb):
    """`upsert` overwrites 0..2 and leaves 3..9 answering searches.

    Same rule as `index_project_files` follows for whole documents -- a reindex
    rebuilds rather than accumulates -- applied one level down. Without it a
    file that lost a section keeps matching queries with text it no longer
    contains, and nothing raises.
    """
    long_content = " ".join(f"word{i}" for i in range(2000))
    kb.add_document("shrinking.md", long_content)
    assert kb.collection.count() > 3

    kb.add_document("shrinking.md", "now it is short")

    assert kb.collection.count() == 1
    assert list(kb.collection.rows) == [f"shrinking.md{CHUNK_ID_SEPARATOR}0000"]


def test_search_collapses_chunks_onto_their_documents(kb):
    """`top_k` counts documents, as every caller already assumed.

    Without the collapse a query matching six passages of one file returns that
    file six times and crowds out five other sources -- worse than the
    behaviour chunking replaced, rather than better.
    """
    kb.add_document("a.md", " ".join(f"alpha{i}" for i in range(2000)))
    kb.add_document("b.md", " ".join(f"beta{i}" for i in range(2000)))

    results = kb.search("anything", top_k=2)

    assert [r["id"] for r in results] == ["a.md", "b.md"]
    assert all(r["chunks_matched"] > 1 for r in results), (
        "each document matched on several of its chunks"
    )


def test_search_reports_the_document_path_and_the_passage_that_matched(kb):
    """`id` stays the document, `content` becomes the passage.

    Callers index the graph with `id`, print it as a filename and derive an
    entity from it, so it has to keep being a path. `content` is what the
    Researcher forwards 300 characters of to the Builder -- which used to be
    300 characters of whichever file's header scored best.
    """
    content = " ".join(f"word{i}" for i in range(2000))
    kb.add_document("a.md", content)

    result = kb.search("anything", top_k=1)[0]

    assert result["id"] == "a.md"
    assert result["id"] in kb.graph, "the id has to resolve against the graph"
    assert result["chunk_id"].startswith(f"a.md{CHUNK_ID_SEPARATOR}")
    assert result["content"] in content
    assert len(result["content"]) < len(content), "a passage, not the whole file"


def test_search_with_no_hits_returns_nothing(kb):
    assert kb.search("anything", top_k=5) == []


def test_a_reindex_prunes_by_document_not_by_chunk_id(kb, tmp_path, monkeypatch):
    """`CLAUDE.md#0007` is not in `wanted` and never will be.

    Comparing stored ids against the wanted set directly would find every chunk
    stale and delete the whole corpus on each reindex. It happens to end in the
    same place while everything is re-added in the same pass, which is exactly
    why it would sit there unnoticed until something reindexed a subset and
    silently re-embedded the lot.
    """
    (tmp_path / "kept.md").write_text(" ".join(f"word{i}" for i in range(2000)))
    kb.add_document("gone.md", "a document that will not be found on disk")
    kept_before = kb.collection.count()
    assert kept_before == 1

    monkeypatch.chdir(tmp_path)
    kb.persist_dir = tmp_path
    report = index_project_files(kb, str(tmp_path))

    stored = set(kb.collection.rows)
    assert not any(cid.startswith("gone.md") for cid in stored), "stale rows go"
    assert any(cid.startswith(str(tmp_path / "kept.md")) for cid in stored)
    assert report["indexed"] == 1


# ---------------------------------------------------------------------------
# entity extraction
# ---------------------------------------------------------------------------


def test_a_capital_forced_by_position_is_not_an_entity(kb):
    """The capital rule fires on words whose capital is punctuation, not meaning.

    A sentence opener, a numpy docstring heading and a Python literal all
    arrive capitalised for reasons that say nothing about the corpus. Before
    the stopword list `False` was the 4th best-connected node in this
    project's graph and `Returns` the 6th, above `Fiedler` and `Cheeger`: an
    edge through `Returns` claims two documents are related when all it knows
    is that both contain a docstring.
    """
    kb.add_document(
        "doc.py",
        "Returns the Builder. Parameters are ignored. Nothing else. "
        "Every caller checks False against the Laplacian.",
    )

    entities = {n for n, a in kb.graph.nodes(data=True) if a.get("type") == "entity"}

    assert entities == {"Builder", "Laplacian"}


def test_the_stopword_match_is_the_whole_token_not_a_prefix(kb):
    """`Returns` is stopped; `Researcher` merely starts with the same letters."""
    kb.add_document("doc.md", "Returns. Researcher and Verdict and Notes.")

    entities = {n for n, a in kb.graph.nodes(data=True) if a.get("type") == "entity"}

    assert entities == {"Researcher", "Verdict"}


def test_the_stopword_list_holds_no_domain_term(kb):
    """A guard on the list itself, which is hand-maintained and easy to widen.

    Every one of these is a term the corpus is *about*; stopping one would
    delete a real relation from the graph and nothing would raise.

    The last four were nominated by the 2026-09-09 widening and refused. They
    are the ones a future pass is most likely to get wrong, because each looks
    generic and each is load-bearing here: `L_dense` scores zero
    position-free capitals only because an assignment starts its line, and
    `System`, `Search` and `State` are ordinary English words that this corpus
    genuinely uses as terms (`System` carries 22 free capitals on its own).
    """
    for term in (
        "Builder", "Architect", "Researcher", "Planner", "Laplacian", "Fiedler",
        "Cheeger", "GraphRAG", "AgentState", "NetworkX", "Chroma", "Verdict",
        "Spectral", "ValueError", "LangGraph", "Normalized", "Conductance",
        "L_dense", "System", "Search", "State",
    ):
        assert term.lower() not in ENTITY_STOPWORDS, f"{term} is not boilerplate"


def test_the_widened_stopwords_are_actually_stopped(kb):
    """The 2026-09-09 additions, pinned by behaviour rather than by list entry.

    `Measured` is the one that matters: this project's own notes open
    sentences with it constantly, so the document recording a measurement was
    minting an entity for the word. Asserted through `add_document` because a
    membership check would pass on a list the extractor had stopped reading.
    """
    kb.add_document(
        "note.md",
        "Measured on this corpus, the Builder ran clean. Tests pass. "
        "Initialize the Laplacian. Write the report. Degree matters.",
    )

    entities = {n for n, a in kb.graph.nodes(data=True) if a.get("type") == "entity"}

    assert entities == {"Builder", "Laplacian"}


def test_the_stopword_list_is_lowercase_ascii():
    """It is matched with `token.lower()`, so an entry with a capital in it is
    dead weight that reads as coverage."""
    assert all(w == w.lower() and w.isascii() for w in ENTITY_STOPWORDS)
