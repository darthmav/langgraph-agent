"""Tests for the corpus-versus-disk comparison.

This exists because of a failure that had no symptom. The corpus is a function
of what is on disk, nothing rebuilds it automatically, and every counter the
console shows stays non-zero and internally consistent while it drifts. On
2026-09-09 this project's store held 8 documents against a walk offering 103:
every project query scored under `RETRIEVAL_RELEVANCE_FLOOR`, retrieval was
discarded, the Researcher's seat answered from memory on every run — and
`rag_stats` said `indexed`, the full suite passed, and nothing logged anything.

So the assertions here are mostly about *not crying wolf*. A signal the
operator learns to ignore is worse than no signal, and the size limit is where
that would happen: `index_project_files` measures characters while `stat`
counts bytes, and guessing in either direction invents an accusation.
"""

from __future__ import annotations

import pytest

from langgraph_agent.corpus_health import (
    corpus_staleness,
    expected_documents,
    forget_expected_documents,
    oversized_documents,
)
from langgraph_agent.graphrag_server import MAX_INDEXABLE_BYTES


@pytest.fixture(autouse=True)
def _no_cached_walk():
    forget_expected_documents()
    yield
    forget_expected_documents()


def _project(tmp_path, **files: str):
    for name, content in files.items():
        path = tmp_path / name.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return str(tmp_path)


def test_a_corpus_matching_the_walk_is_not_stale(tmp_path):
    root = _project(tmp_path, **{"a.md": "alpha", "src__b.py": "beta"})

    report = corpus_staleness(expected_documents(root, use_cache=False), root, use_cache=False)

    assert not report["stale"]
    assert report["missing_count"] == 0 and report["extra_count"] == 0


def test_a_file_never_indexed_is_reported_missing(tmp_path):
    """The failure that started this: the project grew past its last reindex."""
    root = _project(tmp_path, **{"a.md": "alpha", "b.md": "beta"})
    indexed = [str(tmp_path / "a.md")]

    report = corpus_staleness(indexed, root, use_cache=False)

    assert report["stale"]
    assert report["missing_count"] == 1
    assert report["missing"] == [str(tmp_path / "b.md")]
    assert report["indexed"] == 1 and report["expected"] == 2


def test_a_document_no_longer_on_disk_is_reported_extra(tmp_path):
    """The other direction is staleness too, and still answers searches.

    A deleted or renamed file leaves a document in the store holding text that
    is nowhere in the project, and it goes on matching queries until someone
    rebuilds.
    """
    root = _project(tmp_path, **{"a.md": "alpha"})
    indexed = [str(tmp_path / "a.md"), str(tmp_path / "deleted.md")]

    report = corpus_staleness(indexed, root, use_cache=False)

    assert report["stale"]
    assert report["extra"] == [str(tmp_path / "deleted.md")]


def test_a_file_the_indexer_would_skip_is_never_called_missing(tmp_path):
    """The false accusation this guards against, in the `missing` direction.

    An oversized file is skipped by every reindex, so expecting it in the
    corpus would leave the header permanently stale with nothing to do about
    it — and a permanent warning is one nobody reads.
    """
    root = _project(tmp_path, **{"a.md": "alpha", "huge.md": "x" * (MAX_INDEXABLE_BYTES + 1)})

    report = corpus_staleness([str(tmp_path / "a.md")], root, use_cache=False)

    assert not report["stale"]
    assert str(tmp_path / "huge.md") not in report["missing"]


def test_a_multibyte_file_under_the_limit_is_not_called_extra(tmp_path):
    """The same false accusation pointing the other way.

    `stat` counts bytes and the indexer counts characters, so a file of
    multi-byte characters can exceed the limit in bytes while being perfectly
    indexable. Judging it on bytes alone would call a document that belongs in
    the corpus `extra`, on every poll, forever.
    """
    text = "é" * (MAX_INDEXABLE_BYTES - 10)          # 2 bytes each, 1 char each
    root = _project(tmp_path, **{"wide.md": text})
    assert (tmp_path / "wide.md").stat().st_size > MAX_INDEXABLE_BYTES

    report = corpus_staleness([str(tmp_path / "wide.md")], root, use_cache=False)

    assert not report["stale"], report


def test_the_walk_is_cached_and_the_cache_can_be_dropped(tmp_path):
    """The console polls this every five seconds; the walk is not free."""
    root = _project(tmp_path, **{"a.md": "alpha"})
    first = expected_documents(root)

    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    assert expected_documents(root) == first          # served from the cache

    forget_expected_documents()
    assert len(expected_documents(root)) == 2


def test_an_oversized_file_is_reported_apart_from_staleness(tmp_path):
    """Absent by design, and a rebuild cannot change it — so it is not `stale`.

    This is the gap that opened while writing the check itself: adding it to
    `graphrag_server` pushed that file past the limit, which would have dropped
    the module defining the corpus *out* of the corpus at the next reindex,
    silently, with `missing` and `extra` both empty because an over-limit file
    is excluded from each of them. Counting it separately is what makes a file
    falling off the edge visible at all.
    """
    root = _project(tmp_path, **{"a.md": "alpha", "huge.md": "x" * (MAX_INDEXABLE_BYTES + 1)})

    report = corpus_staleness([str(tmp_path / "a.md")], root, use_cache=False)

    assert not report["stale"]
    assert report["oversized_count"] == 1
    assert report["oversized"] == [str(tmp_path / "huge.md")]
    assert oversized_documents(root, use_cache=False) == (str(tmp_path / "huge.md"),)


# ---------------------------------------------------------------------------
# The one time the corpus is meant to be moving
# ---------------------------------------------------------------------------


def test_the_stale_verdict_is_withheld_while_a_run_is_in_flight(monkeypatch, tmp_path):
    """The research phase writes a page, then embeds it. Between those, it is
    on disk and not in the store — and the header used to accuse it.

    Observed live on 2026-09-09: the chip read "stale: 1 not indexed" mid-phase
    and cleared itself moments later. A verdict that flickers teaches the
    operator to ignore the one that does not, so the counts stay — they are the
    truth about this instant — and only the accusation is withheld.
    """
    import serve

    root = _project(tmp_path, **{"a.md": "alpha", "b.md": "beta"})

    class _KB:
        graph = __import__("networkx").DiGraph()
        def stats(self):
            return {"total_documents": 1, "total_chunks": 1, "total_nodes": 1, "total_edges": 0}
    kb = _KB()
    kb.graph.add_node(str(tmp_path / "a.md"), type="document")

    monkeypatch.setattr(serve, "_open_kb", lambda: kb)
    monkeypatch.setattr(serve, "corpus_staleness", lambda docs: corpus_staleness(docs, root, use_cache=False))

    monkeypatch.setitem(serve._run_progress, "running", False)
    idle = serve.rpc_rag_stats({})["staleness"]
    assert idle["stale"] and idle["missing_count"] == 1

    monkeypatch.setitem(serve._run_progress, "running", True)
    during = serve.rpc_rag_stats({})["staleness"]
    assert not during["stale"]
    assert during["settling"] is True
    # The counts are still the truth about this instant.
    assert during["missing_count"] == 1
