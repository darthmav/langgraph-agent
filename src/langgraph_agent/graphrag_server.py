"""GraphRAG MCP Server.

Provides knowledge base search with entity/relation graph + vector store.

Usage:
    python -m src.langgraph_agent.graphrag_server

Or with stdio transport for MCP:
    mcp dev src/langgraph_agent/graphrag_server.py
"""

import asyncio
import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb
import networkx as nx
from mcp.server import MCPServer

from langgraph_agent.lexical import (
    BM25Index,
    lexical_order,
    reciprocal_rank_fusion,
)

# Force CPU for sentence-transformers (GPU 1060 3GB not compatible). Set at
# import rather than beside the model load below, because it has to be in the
# environment before torch is imported, and that import is now deferred.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

if TYPE_CHECKING:  # pragma: no cover - import cost is the whole point
    from sentence_transformers import SentenceTransformer

# The one model that runs on this machine. Named once because three places
# have to agree on it: the embedder the store is built with, the status check
# that reports it without loading it, and the export that records which model
# produced the corpus it is dumping.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# The score at or below which retrieval is treated as having answered nothing,
# and the run falls through to the Researcher's model. It is a property of
# EMBEDDING_MODEL_NAME and meaningless apart from it -- a cosine similarity has
# no absolute meaning across models -- so it lives here rather than beside the
# comparison in `nodes.py`, where a model swap would leave it behind and
# silently redefine "relevant".
#
# It was 0.3, and 0.3 is inside the off-corpus population rather than below it.
# Measured through `search` on this corpus, twelve questions it answers against
# twelve it cannot:
#
#     on-corpus    min 0.442   median 0.564   max 0.685
#     off-corpus   min 0.144   median 0.208   max 0.306
#
# Those separate with an empty band from 0.306 to 0.442, and 0.3 sat at the top
# of the wrong one. The single question that crossed it is this project's own
# off-corpus probe -- the `offcorpus` exercise in `scripts/diagnose_seats.py`,
# on PostgreSQL vacuum -- which scored 0.306 and was therefore formatted
# straight into the findings as though the corpus had answered it. That
# exercise exists because it is the only team run where the Researcher's model
# is the variable, and the gate was quietly denying it that.
#
# 0.37 is the midpoint of the empty band, picked the way EIGENGAP_DECISIVENESS
# was: a value in open space rather than on an observed boundary. The band is
# narrower than it first measured, and deliberately re-centred since -- the
# hybrid re-rank in `search` promotes the chunk the *lexical* half also likes,
# which is often a better answer carrying a slightly lower cosine, so the
# on-corpus minimum fell from 0.492 to 0.442 as retrieval improved. A floor
# calibrated against dense-only ordering describes code that no longer runs.
RETRIEVAL_RELEVANCE_FLOOR = 0.37

# What every caller says when asked to search a corpus nobody has built. One
# string because three doors report it -- the MCP tools, the Builder's tool
# belt, and the console -- and a corpus that reads as absent in one place and
# as merely empty in another is the confusion this whole path exists to avoid.
NO_CORPUS_NOTE = (
    "No corpus has been indexed. Nothing is retrieved and nothing is loaded "
    "until one is built: run `python scripts/reindex.py`, or press Reindex "
    "project on the console's Corpus tab."
)

# Below this conductance a cut counts as a genuine narrow waist: leaving the
# set, roughly one edge in ten crosses. A convention, not a theorem, and it is
# named here because two things read it -- the verdict below and the console.
# The raw conductance and both Cheeger bounds are always returned alongside it,
# so a caller who wants a different line can draw one.
BOTTLENECK_CONDUCTANCE = 0.1

# How many times larger the winning eigengap must be than the runner-up before
# the number of clusters it implies is worth believing. Not a guess: measured
# in `scripts/spectral_benchmark.py` and again on this graph shape. Across 18
# corpora with a planted topic count the eigengap picked k correctly every
# time, at a decisiveness of 5.1 to 23.4; on graphs with no community structure
# at all -- a grid, a small-world ring, an expander, one dense topic -- it still
# returned some k, at 1.0 to 1.8. Nothing observed lands between 1.8 and 4.5,
# so 3.0 sits in open space rather than on a boundary.
EIGENGAP_DECISIVENESS = 3.0

# The largest k the eigengap is allowed to propose. A whole-corpus map with
# more parts than this is not a map anyone reads, and the heuristic's failures
# in the benchmark were all at the top of its range (k = 10 for a barbell whose
# answer is 2), so the ceiling is also where the bad answers live.
MAX_AUTO_CLUSTERS = 12

# How alike two entity names must read before the pair is worth proposing as a
# merge. Only pairs that are *not* already identical bar their case are held to
# it: `Builder` / `Builders` scores 0.93, `Entity` / `Entities` 0.92, and the
# false positive the old synthetic fixture was built around, `Ent11` / `Ent11x`,
# scores 0.91 -- which is why a lexical pair must clear
# `DUPLICATE_CONTAINMENT` as well, and that is what excludes it.
DUPLICATE_NAME_SIMILARITY = 0.85

# How much of the rarer entity's neighbourhood the commoner one must cover
# before a lexically similar pair is offered. **Containment, not Jaccard**, and
# the difference is the whole reason the old design found nothing: a real
# duplicate is *asymmetric*. `Builder` is mentioned by 29 documents and
# `Builders` by 4, a subset relation that scores containment 1.00 and Jaccard
# 0.14 -- so Jaccard, and the embedding distance built on the same symmetry
# assumption, both rank the true duplicate below thousands of unrelated pairs.
DUPLICATE_CONTAINMENT = 0.5

# Entities are compared only against others sharing this many leading
# characters, case-folded. On this corpus that is 1,528 comparisons instead of
# 948,753 -- 620x fewer -- and it costs nothing this method could otherwise
# find, because a pair that agrees on no prefix cannot clear
# `DUPLICATE_NAME_SIMILARITY` on names of the length the extractor mints.
DUPLICATE_BLOCK_PREFIX = 4

# Capitalised tokens that are not entities. `add_document` mints an entity for
# every capitalised word over four characters, and in a corpus of prose and
# numpy-style docstrings that rule fires constantly on words whose capital is
# an artefact of where they sit rather than of what they mean: a sentence
# opener (`Every`, `Nothing`, `Without`), a docstring section header
# (`Returns`, `Parameters`, `Raises`), a report heading (`Files`, `Status`), or
# a Python literal (`False`, `None`). Measured before this list, `False` was
# the 4th best-connected node in the graph and `Returns` the 6th, above
# `Fiedler`, `Planner` and `Cheeger`: 21 documents share an edge through
# `Returns`, which says only that all 21 contain a docstring.
#
# **This list does not exist to make `topics()` decisive, and it does not.**
# That was the first hypothesis and the measurement refused it: removing every
# one of the 97 entities that bridge this corpus's two topic areas -- the
# theoretical maximum any such filter could achieve -- moves the eigengap
# decisiveness from 1.12x to 1.09x. The corpus's spectrum is a smooth
# continuum because the corpus genuinely has no decisive k, not because
# boilerplate is gluing it together. What the list is for is the graph itself:
# an edge through `Returns` is a false claim that two documents are related,
# and `neighborhood()`, `top_entities` and `duplicate_entities` all read those
# edges as evidence.
#
# Matched case-insensitively against the whole token, never as a prefix. A
# **stopword list, not a heuristic**, on purpose: the obvious alternative is to
# drop a token that only ever appears where a capital is forced (line start,
# after a full stop), and it was built and measured. It removes the same noise
# and severs real edges doing it -- `Planner` 18 documents down to 15,
# `Spectral` 19 to 15, `ValueError` 14 to 11 -- because a term introduced in a
# bulleted list (`- **Planner** -- interprets goals`) never appears anywhere
# else in that document. Silently dropping a true relation to catch a false one
# is the wrong trade here, and a list a reader can audit line by line beats a
# rule whose failures are invisible. Measured on this corpus, this list removes
# 84 entities and 437 edges and costs **no** meaningful term a single edge.
ENTITY_STOPWORDS = frozenset(
    word.lower()
    for word in """
    Returns Return Parameters Parameter Example Examples Provides Raises Notes
    Args Arguments Attributes Yields Warns Usage Summary Description Overview

    Where Which There These Those Their Because Without Within While Since
    Should Would Could Might Cannot Every Nothing Never Always Something
    Anything Everything Instead Otherwise Rather Before After During Between
    Against About Almost Already Still Another Other First Second Third Above
    Below Under Twice Several Given Using Used Uses Value Values Result
    Results Input Inputs Output Outputs Simple Basic Total Default Defaults

    Check Checks Create Creates Compute Computes Number Numbers Optional
    Reading Reads Writes Written Files Status Makes Taken Keeps Change Changed
    Changes Adding Added Running Choose Chooses Verify Verifies Found Finds
    Being Doing Ensure Ensures Consider Include Includes Including Following
    Follows Contains Containing

    False None Import Class Print Assert Except Finally Raise Elif Return
    """.split()
)

# The embedder's context window, minus the two special tokens it adds. This is
# a property of `EMBEDDING_MODEL_NAME` (`max_seq_length` is 256 on
# all-MiniLM-L6-v2), not a tuning knob: a passage longer than this is not
# embedded badly, it is **silently truncated and the tail discarded**.
#
# That is what this constant exists to stop. A document used to be embedded
# whole, in one `encode()` call, with `MAX_INDEXABLE_BYTES` allowing 100 KB --
# so the vector for a 46 KB file was computed from its first ~1,000 characters
# and nothing else. Measured on this project's own corpus before chunking: 73
# of 77 documents over the limit, 224,809 tokens present and 19,147 embedded,
# **91.5% of the corpus unreachable by search**. The embedding of all 46,094
# characters of CLAUDE.md was bit-identical (cosine 1.000000) to the embedding
# of its first 1,000. Two failures came out of that, and neither announces
# itself: retrieval acquired a *length bias*, because a short file is fully
# represented while a long one is represented by its preamble -- so the file
# that actually answers the query loses to a shorter one that merely mentions
# it -- and scores sat low enough that plan-shaped queries fell under the 0.3
# gate in `nodes.py`, discarding retrieval and sending the run to the
# Researcher's model, which is the loop this project already knows is fragile.
CHUNK_MAX_TOKENS = 254

# Tokens carried from the end of one chunk into the start of the next. Chunk
# boundaries are cut on token counts, not on sentences, so a passage can be
# split down the middle; the overlap is what keeps such a passage whole in at
# least one chunk, and it is why boundaries are *not* snapped to line breaks.
# Snapping would have to either shorten a chunk (dropping tokens the model
# could have seen) or lengthen it past the window (truncating again, which is
# the bug), so the cut stays where the arithmetic puts it and the overlap
# absorbs the cosmetic cost.
CHUNK_OVERLAP_TOKENS = 48

# Separates a document id from its chunk number: `CLAUDE.md#0003`. The document
# id is a project-relative path, which cannot contain "#" on any filesystem
# this runs on, so the split back to a document is unambiguous.
CHUNK_ID_SEPARATOR = "#"

# How many chunks to pull per requested result before collapsing them onto
# their documents. A query that matches one document strongly can match several
# of its chunks, and `search` returns documents, so without oversampling a
# top_k of 5 could collapse to 1. Four is enough for the shapes measured here
# without making Chroma do meaningfully more work.
SEARCH_CHUNK_OVERSAMPLE = 4

# The second and last rung of the ladder in `search`, taken only when the first
# window of hits collapsed to fewer documents than the caller asked for. A
# focused query really can match a dozen passages of one large file before it
# matches anything else, and answering a request for five sources with one is
# narrower than the behaviour chunking replaced.
SEARCH_ESCALATION = 8


def _chunk_windows(
    n_tokens: int, max_tokens: int, overlap: int
) -> list[tuple[int, int]]:
    """Token index windows `[start, end)` covering `n_tokens`, with overlap.

    Pure arithmetic, kept out of `chunk_text` so the packing can be tested
    without loading the embedding model -- the thing every other test in this
    project goes out of its way to avoid.

    Every token lands in at least one window and no window is longer than
    `max_tokens`, which together are the whole contract: the first is what
    stops the truncation this exists to fix, and the second is what stops each
    chunk from being truncated in turn.
    """
    if n_tokens <= max_tokens:
        return [(0, n_tokens)] if n_tokens else []

    # A stride at or below zero would never advance and the loop would not
    # terminate. Clamped rather than raised on: an overlap wider than the
    # window is a caller's misconfiguration, and degrading to "no overlap" is
    # better than refusing to index at all.
    stride = max(max_tokens - overlap, 1)

    windows: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + max_tokens, n_tokens)
        windows.append((start, end))
        if end >= n_tokens:
            return windows
        start += stride


def _document_id_of(
    chunk_id: str, metadata: "Mapping[str, Any] | None" = None
) -> str:
    """The document a stored row belongs to.

    Prefers the `doc_id` the chunk carries in its metadata and falls back to
    parsing the id, because both shapes exist in a live store: rows written
    before chunking are keyed by the bare document path and carry no `doc_id`,
    and they must keep resolving to themselves rather than being read as
    strangers and pruned. The suffix is only stripped when it is actually a
    chunk number, so a path that happens to contain "#" is left alone.
    """
    if metadata:
        doc_id = metadata.get("doc_id")
        if isinstance(doc_id, str) and doc_id:
            return doc_id

    head, sep, tail = chunk_id.rpartition(CHUNK_ID_SEPARATOR)
    if sep and head and tail.isdigit():
        return head
    return chunk_id


class GraphRAGKnowledgeBase:
    """Simple GraphRAG: NetworkX graph + Chroma vector store.

    Constructing this **creates the store on disk** -- `mkdir`, plus Chroma's
    own files under `chroma/`. That is why it is not the door most callers go
    through: `open_knowledge_base()` returns the corpus only if one already
    exists, and `get_knowledge_base()` is reserved for the act of building one.
    A corpus that appeared because a status poll happened to run is not a
    corpus anyone asked for.
    """

    # Declared on the class, not assigned in `__init__`, so an instance built
    # field by field around a fake collection -- which is how the corpus tests
    # avoid the model entirely -- still reads as "not loaded yet" rather than
    # raising on the attribute.
    _embedder: "SentenceTransformer | None" = None

    # Same reasoning, and the same construction path: (nodes, edges) -> the
    # connectivity result computed at that shape. A cache must not depend on
    # which door built the object, so it defaults on the class rather than in
    # `__init__`.
    _connectivity_cache: "tuple[tuple[int, int], dict[str, Any]] | None" = None

    # The lexical half of search, built on first use from what the store holds
    # and dropped whenever the store changes. Declared on the class for the
    # same reason as the two above: an instance assembled field by field
    # around a fake collection still reads as "not built yet".
    _lexical_index: "BM25Index | None" = None

    def __init__(self, persist_dir: str = "./knowledge"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Chroma vector store
        self.chroma_client = chromadb.PersistentClient(str(self.persist_dir / "chroma"))
        self.collection = self.chroma_client.get_or_create_collection(
            name="knowledge",
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize knowledge graph
        self.graph = nx.DiGraph()
        self._load_graph()

    @property
    def embedder(self) -> "SentenceTransformer":
        """The local embedding model, loaded the first time something embeds.

        Deferred because loading it is the one heavyweight thing this machine
        does, and it is only needed to add a document or to run a query. It
        used to load in `__init__`, so opening the corpus at all -- a header
        poll, a document list -- paid for it, and importing this module paid
        for pulling in torch behind it.
        """
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return self._embedder

    def _load_graph(self) -> None:
        """Load graph from disk if exists."""
        graph_path = self.persist_dir / "knowledge_graph.json"
        if graph_path.exists():
            self.graph = nx.readwrite.json_graph.node_link_graph(
                json.load(open(graph_path))
            )

    def _save_graph(self) -> None:
        """Save graph to disk."""
        graph_path = self.persist_dir / "knowledge_graph.json"
        node_link_data = nx.readwrite.json_graph.node_link_data(self.graph)
        json.dump(node_link_data, open(graph_path, "w"))

    def chunk_text(self, content: str) -> list[str]:
        """Split a document into passages the embedder can actually read whole.

        One tokenizer pass with an offset mapping, then `_chunk_windows` over
        the token indices; each window's character span runs from the start of
        its first token to the end of its last, so the text between tokens --
        whitespace, indentation, blank lines -- is carried rather than dropped.
        Concatenating the chunks therefore reproduces the document apart from
        the deliberate overlap, and the join is checked in the tests.

        `verbose=False` suppresses the tokenizer's own "sequence longer than
        the maximum" warning. It is silenced only because this function is the
        thing that answers it: the sequence *is* longer than the window, that
        is why it is being cut up, and the warning would otherwise fire once
        per document on every reindex.
        """
        if not content:
            return []

        tokenizer = self.embedder.tokenizer
        encoded = tokenizer(
            content,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )
        offsets = encoded["offset_mapping"]
        if not offsets:
            # Content the tokenizer maps to nothing -- whitespace, or a run of
            # characters with no token of their own. There is no passage to
            # embed, and returning the raw text would put a vector of noise in
            # the store under the document's name.
            return []

        chunks = [
            content[offsets[start][0] : offsets[end - 1][1]]
            for start, end in _chunk_windows(
                len(offsets), CHUNK_MAX_TOKENS, CHUNK_OVERLAP_TOKENS
            )
        ]
        return self._fit_chunks(chunks)

    def _fit_chunks(self, chunks: list[str]) -> list[str]:
        """Trim any chunk that re-tokenizes past the window, and say which end.

        Slicing on the parent document's token boundaries does **not**
        guarantee the slice re-tokenizes to the same length. A word-piece
        tokenizer decides on context, so a fragment cut mid-word encodes
        differently standalone than it did inside the document -- measured on
        this corpus, 16 of 1,052 full-size chunks came back one token longer,
        which put them at 257 against a 256 window and handed them straight
        back to the truncation this whole change exists to remove.

        Measuring the drift and padding the constant would be guessing with an
        extra step: +1 is what this corpus does today, not a bound anybody can
        prove for the next document. So the chunks are re-encoded and the
        overflow is cut, which makes the window a fact rather than an estimate.
        Only the overflowing chunks are touched, and the batched encode of the
        rest is a few tens of milliseconds per reindex.

        **Which end is trimmed is the part that matters.** A chunk's tail is
        covered by the next chunk's overlap and its head by the previous one's,
        so trimming into a neighbour's overlap loses nothing from the corpus --
        except at the two ends of the document, which have no neighbour. The
        last chunk is therefore trimmed at the *head* and every other at the
        tail; trimming the last one's tail would drop the final tokens of the
        file, silently, which is the original bug in miniature.
        """
        if len(chunks) < 2:
            # A lone chunk is the whole document, un-sliced, so it re-tokenizes
            # to exactly what it was measured as and cannot overflow.
            return chunks

        encoded = self.embedder.tokenizer(
            chunks,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            verbose=False,
        )

        fitted: list[str] = []
        for i, (chunk, offsets) in enumerate(zip(chunks, encoded["offset_mapping"], strict=True)):
            if len(offsets) <= CHUNK_MAX_TOKENS:
                fitted.append(chunk)
            elif i == len(chunks) - 1:
                fitted.append(chunk[offsets[-CHUNK_MAX_TOKENS][0] :])
            else:
                fitted.append(chunk[: offsets[CHUNK_MAX_TOKENS - 1][1]])
        return fitted

    def add_document(self, doc_id: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a document to the knowledge base.

        The document is embedded as several chunks and stored as several rows,
        keyed `doc_id#0000`, `doc_id#0001`, ... -- see `CHUNK_MAX_TOKENS` for
        what embedding it as one row cost. The graph is unaffected: it still
        gets exactly one node per document, with entities drawn from the whole
        text, so chunking changes what search can find and not what the corpus
        is shaped like.

        Args:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Optional metadata (path, type, etc.)
        """
        chunks = self.chunk_text(content)

        # A document's previous chunks are deleted before the new ones land,
        # rather than left to be overwritten by `upsert`. A file that shrank
        # between reindexes -- 10 chunks down to 3 -- overwrites 0..2 and
        # leaves 3..9 in the store, still matching queries with text the file
        # no longer contains. That is the same "a reindex rebuilds rather than
        # accumulates" rule `index_project_files` follows for whole documents,
        # applied one level down.
        #
        # Keyed on the `doc_id` metadata, so it also sweeps up the single
        # unsuffixed row a pre-chunking store holds for this document.
        try:
            self.collection.delete(where={"doc_id": doc_id})
        except Exception:
            # A collection that cannot filter by metadata (an older store, or
            # the fakes the corpus tests build) still gets a correct insert
            # below; what is lost is the sweep of rows this call is replacing.
            pass
        self.collection.delete(ids=[doc_id])

        if chunks:
            base = dict(metadata or {})
            # Chunks are embedded in one batched call rather than one per
            # chunk: the model is the expensive thing on this machine and
            # batching is most of what makes a reindex of ~1,100 chunks
            # finish in the time a reindex of 77 documents used to take.
            embeddings = self.embedder.encode(chunks).tolist()
            self.collection.upsert(
                ids=[
                    f"{doc_id}{CHUNK_ID_SEPARATOR}{i:04d}" for i in range(len(chunks))
                ],
                embeddings=embeddings,
                documents=chunks,
                metadatas=[
                    {**base, "doc_id": doc_id, "chunk_index": i, "chunk_count": len(chunks)}
                    for i in range(len(chunks))
                ],
            )

        # Add to graph as a node.
        # `type` on a node is structural -- document vs entity -- and drives how
        # the console draws it. Metadata carries its own `type` (python,
        # markdown), which collides as a duplicate keyword and takes down the
        # whole insert, so it goes in under its own name.
        node_attrs = {k: v for k, v in (metadata or {}).items() if k != "type"}
        if metadata and "type" in metadata:
            node_attrs["doc_type"] = metadata["type"]

        self.graph.add_node(
            doc_id,
            type="document",
            content=content[:200],  # Store snippet
            **node_attrs
        )

        # Extract simple entities (words that look like important terms).
        # The strip set has to cover code punctuation as well as prose: over a
        # corpus that is mostly source files, a prose-only `.,!?;:` leaves
        # entities like `Builder")` standing as graph nodes.
        #
        # `ENTITY_STOPWORDS` is what stops the capital rule firing on words
        # whose capital comes from where they sit rather than what they mean --
        # a sentence opener, a docstring heading, a Python literal. Without it
        # `False` and `Returns` are the 4th and 6th best-connected nodes in
        # this corpus's graph, ahead of `Fiedler` and `Cheeger`, and every
        # document carrying a docstring is joined to every other one through a
        # relation that means nothing.
        entities = []
        for word in content.split():
            token = word.strip("\"'`()[]{}<>.,!?;:*=+-/\\|")
            if (
                len(token) > 4
                and token[0].isupper()
                and token.replace("_", "").isalnum()
                and token.lower() not in ENTITY_STOPWORDS
            ):
                entities.append(token)

        # Add entities and relationships
        for entity in set(entities):
            self.graph.add_node(entity, type="entity")
            self.graph.add_edge(doc_id, entity, relation="mentions")

        # The lexical index describes a corpus that no longer exists. Dropped
        # rather than amended: the next search rebuilds it from the store in
        # milliseconds, and an index maintained in parallel with Chroma is a
        # second account of the same corpus, free to disagree with it.
        self._lexical_index = None

        self._save_graph()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search the knowledge base.

        Matches on chunks and answers in documents. `id` stays the document's
        path -- the thing callers index the graph with, print as a filename and
        derive an entity from -- while `content` becomes the passage that
        actually matched rather than the document's first 200 characters. That
        is the half of chunking the Researcher feels: it forwards
        `content[:300]` to the Builder, which used to be 300 characters of
        whichever file's *header* scored best.

        Chunks are oversampled and then collapsed onto their documents, keeping
        each document's best-scoring chunk. Collapsing is what makes `top_k`
        mean what every caller already assumed it meant: without it a query
        that matches six passages of one file would return that file six times
        and crowd out five other sources, which is worse than the pre-chunking
        behaviour rather than better.

        The retrieval is **hybrid**: the dense window is re-ranked against BM25
        before it is collapsed. Dense similarity is a poor instrument for "this
        passage contains this exact rare identifier", which is what a plan
        naming `BUILDER_DEADLINE_SECONDS` is really asking. Measured through
        this method against the real store, on 541 identifiers each defined in
        exactly one project file, the defining file came first 53.4% of the
        time on dense alone and 65.1% with the re-rank (McNemar p < 0.001, 85
        fixed against 22 broken). The prose case improved too -- 66.0% to 68.2%
        on 400 held-out passages -- so this is not a code-search special case
        bought at the expense of ordinary questions. See `lexical.py` for why
        it re-ranks rather than retrieving in parallel, and why ranks are fused
        rather than scores.

        `score` therefore stays exactly what it was: the dense cosine of the
        chunk that matched. Every candidate comes from the dense window, so
        there is no result whose score had to be invented -- which matters
        because `RETRIEVAL_RELEVANCE_FLOOR` is read off the first one to decide
        whether the corpus answered at all.

        Args:
            query: Search query
            top_k: Number of documents to return

        Returns:
            List of results with content, metadata, and graph context
        """
        if top_k <= 0:
            return []

        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()

        # Widen once if one document monopolised the first window of hits.
        # A focused query genuinely can match a dozen passages of the same
        # large file before it matches anything else, and collapsing those
        # onto one document would answer a request for five sources with one
        # -- narrower than the behaviour chunking replaced. The ladder is two
        # rungs and stops early: there is no point asking a third time, and an
        # unbounded search for breadth would let one query walk the corpus.
        results: list[dict[str, Any]] = []
        wanted = top_k * SEARCH_CHUNK_OVERSAMPLE
        for n_results in (wanted, wanted * SEARCH_ESCALATION):
            raw = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            results = self._collapse_chunk_hits(
                raw, top_k, order=self._rerank_lexically(query, raw)
            )
            hits = len((raw.get("ids") or [[]])[0] or [])
            if len(results) >= top_k or hits < n_results:
                # Either the caller has what it asked for, or the collection
                # returned fewer chunks than requested and has no more to give.
                break

        return results

    @property
    def lexical_index(self) -> "BM25Index | None":
        """The BM25 index over the stored chunks, built on first use.

        Built from the store rather than kept alongside it, so it cannot drift:
        whatever `search` re-ranks is what `search` retrieved. `add_document`
        and `clear` drop it instead of updating it -- rebuilding over this
        corpus's ~1,200 chunks is milliseconds, and an index that edits itself
        in place is a second thing that can disagree with Chroma.

        Returns `None` rather than raising if the store cannot produce its
        documents. A collection too old or too foreign to answer `get` is a
        reason to fall back to dense-only retrieval, not a reason for search to
        stop working -- the lexical half is an improvement to the ranking, and
        the dense half is still a correct answer without it.
        """
        if self._lexical_index is None:
            try:
                stored = self.collection.get(include=["documents"])
                ids = list(stored.get("ids") or [])
                documents = list(stored.get("documents") or [])
            except Exception:
                return None
            if len(ids) != len(documents):
                return None
            self._lexical_index = BM25Index(ids, [d or "" for d in documents])
        return self._lexical_index

    def _rerank_lexically(
        self, query: str, raw: "Mapping[str, Any]"
    ) -> list[str] | None:
        """Fuse the dense window's order with BM25's, or leave it alone.

        Returns `None` -- meaning "use the order Chroma gave" -- when there is
        no index or fewer than two candidates to reorder. A single hit has no
        ranking to improve, and paying for an index build to discover that is
        the sort of cost that lands on the console's five-second poll.
        """
        chunk_ids = [str(c) for c in ((raw.get("ids") or [[]])[0] or [])]
        if len(chunk_ids) < 2:
            return None

        index = self.lexical_index
        if index is None:
            return None

        by_lexical = lexical_order(query, chunk_ids, index)
        if by_lexical == chunk_ids:
            # The lexical half agrees, or had nothing to say. Either way the
            # fusion would return the order we already have.
            return None
        return reciprocal_rank_fusion(chunk_ids, by_lexical)

    def _collapse_chunk_hits(
        self, raw: "Mapping[str, Any]", top_k: int, order: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Fold chunk hits onto the documents they came from, best chunk first.

        Chroma returns hits best-first, so the first chunk seen for a document
        is its best one and every later chunk of it only adds to the count.
        `order`, when given, replaces that ordering with the hybrid one --
        the same hits, re-ranked, so "best chunk first" still holds and every
        result still carries the dense score its chunk was retrieved with.
        """
        ids = raw.get("ids") or [[]]
        documents = raw.get("documents") or [[]]
        metadatas = raw.get("metadatas") or [[]]
        distances = raw.get("distances") or [[]]

        # Guard against empty collections / no hits
        if not ids or not ids[0]:
            return []

        collapsed: list[dict[str, Any]] = []
        by_document: dict[str, dict[str, Any]] = {}

        # Walk the hits in the hybrid order when there is one, and in Chroma's
        # otherwise. A chunk id `order` names that this response does not hold
        # is skipped rather than trusted: the two come from the same query, but
        # the collapse reads `documents`/`distances` positionally and an id
        # without a row here would index the wrong one.
        position = {str(c): i for i, c in enumerate(ids[0])}
        walk = (
            [(position[c], c) for c in order if c in position]
            if order is not None
            else list(enumerate(ids[0]))
        )

        for i, chunk_id in walk:
            metadata = metadatas[0][i] if metadatas[0] else {}
            doc_id = _document_id_of(chunk_id, metadata)

            if doc_id in by_document:
                by_document[doc_id]["chunks_matched"] += 1
                continue

            if len(collapsed) == top_k:
                # Full. Keep draining the hits already paid for so
                # `chunks_matched` counts every match, but admit no new
                # documents.
                continue

            result: dict[str, Any] = {
                "id": doc_id,
                "chunk_id": chunk_id,
                "content": documents[0][i] if documents[0] else "",
                "metadata": metadata,
                "score": 1 - distances[0][i] if distances[0] else 0.0,
                "chunks_matched": 1,
            }

            # Add graph neighbors
            if doc_id in self.graph:
                neighbors = list(self.graph.neighbors(doc_id))[:5]
                result["related_entities"] = neighbors

            by_document[doc_id] = result
            collapsed.append(result)

        return collapsed

    def query_graph(self, entity: str, hops: int = 2) -> dict[str, Any]:
        """Query the knowledge graph for entity relationships.

        Args:
            entity: Entity name to search for
            hops: Number of hops to traverse

        Returns:
            Entity info with relationships
        """
        if entity not in self.graph:
            # Try fuzzy match
            for node in self.graph.nodes():
                if entity.lower() in node.lower():
                    entity = node
                    break

        if entity not in self.graph:
            return {"error": f"Entity '{entity}' not found"}

        # Get neighborhood
        neighbors = nx.single_source_shortest_path_length(
            self.graph, entity, cutoff=hops
        )

        # Build subgraph
        subgraph = self.graph.subgraph(neighbors.keys())

        return {
            "entity": entity,
            "neighbors": list(neighbors.items()),
            "subgraph_nodes": len(subgraph.nodes()),
            "subgraph_edges": len(subgraph.edges()),
        }


    def _resolve_node(self, node_id: str) -> str | None:
        """Resolve a node id, falling back to a substring match.

        Mirrors the fuzzy match in `query_graph` so both entry points accept the
        same loosely-typed ids the console lets a user paste.
        """
        if node_id in self.graph:
            return node_id

        needle = node_id.lower()
        for node in self.graph.nodes():
            if needle in str(node).lower():
                return str(node)
        return None

    def _node_record(self, node_id: str) -> dict[str, Any]:
        """Render one graph node in the shape the console draws."""
        attrs = dict(self.graph.nodes[node_id])
        node_type = attrs.pop("type", "entity")

        # The stored content snippet is for retrieval, not for a properties
        # blob the user hovers over; drop it rather than ship 200 chars per node.
        attrs.pop("content", None)

        # A bare basename collides -- this project has two README.md files, and
        # two nodes labelled the same are unreadable on a graph. Keep the
        # parent directory for anything that is not at the repository root.
        label = str(node_id)
        path = attrs.get("path")
        if node_type == "document" and path:
            as_path = Path(str(path))
            parent = as_path.parent.name
            label = f"{parent}/{as_path.name}" if parent else as_path.name

        return {
            "id": node_id,
            "node_type": node_type,
            "label": label,
            "properties": attrs,
        }

    def list_documents(self) -> dict[str, Any]:
        """List every document node in the knowledge graph.

        Documents are the centres the console sweeps from. A centre never
        appears in its own neighbourhood, so the client seeds its node map from
        this list before it queries anything.
        """
        documents = [
            {"id": node_id, "title": self._node_record(node_id)["label"], "node_type": "document"}
            for node_id, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "document"
        ]
        documents.sort(key=lambda doc: str(doc["title"]))
        return {"documents": documents}

    def _sign_split(self, keep: set[Any], centre: str) -> dict[str, Any]:
        """Fiedler sign bipartition of a swept neighbourhood, oriented on the centre.

        The A4 application from `reports/spectral_applicability.md`, and the
        cheapest technique in it: one eigenvector, 1.7-3.3ms on subgraphs of
        43-291 nodes, against the 1.0ms `neighborhood()` itself costs. That is
        why it is a flag rather than always-on -- it triples the cost of a call
        the console makes on every click, and a caller who only wants the node
        list should not pay it.

        Sides are named `center` and `other` rather than by the sign of the
        eigenvector, whose direction is arbitrary: an eigenvector and its
        negation are equally valid, so a raw sign would swap the two halves
        between runs for no reason. Orienting on the centre also makes the
        answer the one the caller asked for -- "what clusters with the node I
        looked up, and what is peripheral to it".

        `mu_2` is returned because the split is always *available* and only
        sometimes *meaningful*. A sign cut exists for any connected graph; what
        says whether it corresponds to a real division is how small `mu_2` is,
        and on these subgraphs it ranges from 0.134 (a two-hop sweep, barely
        divided) to 0.006 (a five-hop sweep spanning two topic areas). The
        caller is given the number rather than a bare verdict because the
        threshold that matters depends on what the split is being used for --
        here, whether it is worth drawing.

        Runs on the largest component: `min_degree` pruning can disconnect the
        swept subgraph -- measured at depth 3, min_degree 3, which left one
        node isolated -- and on a disconnected graph the Fiedler vector is a
        component indicator, so the "split" would just be that stray node
        against everything else. Nodes outside the largest component are
        reported as `detached` and given no side, which is the truth: they were
        not part of the division.
        """
        undirected = self.graph.to_undirected(as_view=True)
        subgraph = undirected.subgraph(keep)

        if subgraph.number_of_nodes() < 4:
            return {"available": False,
                    "note": "Too few nodes to split.", "sides": {}}

        component = subgraph.subgraph(max(nx.connected_components(subgraph), key=len))
        if component.number_of_nodes() < 4 or centre not in component:
            return {"available": False,
                    "note": "The centre's component is too small to split.", "sides": {}}

        try:
            import numpy as np

            from spectral_graph import compute_spectrum, fiedler_vector

            vector = fiedler_vector(component, normalized=True)
            mu_2 = float(max(compute_spectrum(component, k=2, normalized=True, which="SM")[1], 0.0))
        except ImportError:
            return {"available": False,
                    "note": "spectral_graph is not on sys.path.", "sides": {}}
        except Exception as exc:  # pragma: no cover - solver-dependent
            return {"available": False, "note": f"{type(exc).__name__}: {exc}", "sides": {}}

        nodes = list(component.nodes())
        centre_positive = bool(vector[nodes.index(centre)] >= 0)
        sides = {
            node: ("center" if (bool(value >= 0) == centre_positive) else "other")
            for node, value in zip(nodes, np.asarray(vector), strict=True)
        }

        with_centre = sum(1 for side in sides.values() if side == "center")
        return {
            "available": True,
            "mu_2": mu_2,
            "center_side": with_centre,
            "other_side": len(sides) - with_centre,
            "detached": subgraph.number_of_nodes() - component.number_of_nodes(),
            "sides": sides,
        }

    def neighborhood(
        self, node_id: str, max_depth: int = 2, min_degree: int = 1,
        split: bool = False,
    ) -> dict[str, Any]:
        """Return a node's neighbourhood as drawable nodes and edges.

        `query_graph` answers "how big is this neighbourhood"; this answers
        "what is in it", in the record shape the console renders directly.

        Args:
            node_id: Centre of the traversal; resolved fuzzily.
            max_depth: Hops to traverse out from the centre.
            min_degree: Drop entity nodes with fewer edges than this. A sweep
                passes 2, because `add_document` mints an entity for every
                capitalised word and the one-document ones bury the structure
                worth looking at. A single trace passes 1, so a node the user
                asked for by name never has its neighbours hidden.
            split: Also compute the Fiedler sign bipartition, tagging every
                node `center` or `other`. Off by default: it triples the cost
                of this call, and only a caller that is going to draw the
                division wants it. See `_sign_split`.

        Returns:
            center_node, related_nodes (excluding the centre), edges, and
            totals; plus `split` when asked for.
        """
        centre = self._resolve_node(node_id)
        if centre is None:
            return {
                "error": f"Node '{node_id}' not found",
                "center_node": node_id,
                "related_nodes": [],
                "edges": [],
                "total_nodes": 0,
                "total_edges": 0,
            }

        # Traverse undirected: every edge runs document -> entity, so a directed
        # walk from a document reaches its entities but never the sibling
        # documents that share them, which is the structure worth showing.
        undirected = self.graph.to_undirected(as_view=True)
        reachable = nx.single_source_shortest_path_length(
            undirected, centre, cutoff=max_depth
        )

        keep = {
            found
            for found in reachable
            if found == centre
            or self.graph.nodes[found].get("type") != "entity"
            or undirected.degree(found) >= min_degree
        }

        edges = [
            {
                "id": f"{source}->{target}",
                "source_id": source,
                "target_id": target,
                "relationship": data.get("relation", "related_to"),
                "weight": data.get("weight", 1.0),
            }
            for source, target, data in self.graph.edges(data=True)
            if source in keep and target in keep
        ]

        related = [self._node_record(found) for found in keep if found != centre]

        result = {
            "center_node": centre,
            "related_nodes": related,
            "edges": edges,
            "total_nodes": len(related),
            "total_edges": len(edges),
        }

        if split:
            division = self._sign_split(keep, centre)
            # `sides` is folded onto the node records and dropped from the
            # summary: the console draws nodes, not a lookup table, and
            # shipping both would let the two disagree.
            sides = division.pop("sides", {})
            for record in related:
                record["side"] = sides.get(record["id"])
            result["split"] = division

        return result

    def topics(
        self, k: int | None = None, max_entities: int = 6, max_documents: int = 4
    ) -> dict[str, Any]:
        """Group the corpus into topic communities, or say there are none.

        The A2 application from `reports/spectral_applicability.md`:
        Ng-Jordan-Weiss spectral clustering over the normalized Laplacian,
        which on a bipartite document/entity graph puts documents together with
        the entities that define them -- so each cluster reads as a topic
        rather than as a list of ids. `neighborhood()` shows one node's
        surroundings; this is the whole-corpus map that degree-filtered sweeps
        cannot produce.

        **The number of clusters is where this application was weakest, and it
        is not wired straight to the eigengap.** The report proposes choosing
        `k` from the eigengap; `reports/spectral_architecture_benchmark.md`
        measured that heuristic getting `k` wrong on 3 of 8 architectures,
        including k = 10 for a barbell whose answer is 2. The heuristic always
        returns *some* k, so on a corpus with no topic structure it invents
        one, and clusters presented without that caveat are a fabricated map.

        What makes it usable is that the failures are not merely wrong, they
        are *undecided*: the winning gap barely beats the runner-up. Measured
        across 18 corpora with a planted topic count the eigengap was correct
        every time at a decisiveness of 5.1-23.4, while a grid, a small-world
        ring, an expander and a single dense topic all landed at 1.0-1.8. Below
        `EIGENGAP_DECISIVENESS` the verdict is `no_clear_structure` and no
        clusters are returned, because a map of a corpus that has no topics is
        worse than no map.

        An explicit `k` skips that gate -- a caller asking for six clusters has
        made the decision -- but the decisiveness is still reported, so the
        answer never hides how much the corpus agreed with it.

        Every cluster carries its own conductance, which is the second and
        independent check: a cluster that is genuinely a community has a low
        one, and a `k` that split a real community in half shows up as several
        clusters with high conductance even when the eigengap looked decisive.
        The two signals catch different failures and neither replaces the other.

        Runs on the largest connected component, for the reason
        `connectivity()` and `bottleneck()` do: components are already clusters,
        so on a disconnected graph the eigenvectors would spend themselves
        rediscovering the orphans `connectivity()` already counted.
        """
        undirected = self.graph.to_undirected(as_view=True)
        if undirected.number_of_nodes() == 0:
            return {"verdict": "no_graph", "note": "The graph is empty.", "clusters": []}

        largest = max(nx.connected_components(undirected), key=len)
        if len(largest) < 4:
            return {
                "verdict": "no_graph",
                "note": "The largest component is too small to divide into topics.",
                "clusters": [],
            }
        component = undirected.subgraph(largest)
        n = component.number_of_nodes()

        # A caller's bad k is an error, not a verdict. `unavailable` means the
        # measurement could not be taken; answering a malformed request with it
        # would file the caller's mistake under the solver's failures.
        if k is not None and not 2 <= k <= n:
            raise ValueError(
                f"k must be between 2 and {n} (the largest component), got {k}"
            )

        try:
            # numpy alongside spectral_graph rather than at module scope: it is
            # used only here, and this module is imported by the MCP server and
            # by every test that touches the corpus.
            import numpy as np

            from spectral_graph import compute_spectrum, conductance, spectral_clustering

            # One eigenvalue past the largest k worth proposing, so the gap that
            # would select that k is itself inside the window.
            probe = min(MAX_AUTO_CLUSTERS + 1, n - 1)
            spectrum = np.sort(
                np.maximum(compute_spectrum(component, k=probe, normalized=True, which="SM"), 0.0)
            )
            # Skip the gap out of the trivial eigenvalue: k = 1 is not a finding.
            gaps = np.diff(spectrum)[1:]
            order = np.argsort(gaps)[::-1]
            best = float(gaps[order[0]])
            runner_up = float(gaps[order[1]]) if len(order) > 1 else 0.0
            decisiveness = best / runner_up if runner_up > 1e-12 else float("inf")
            suggested = int(order[0]) + 2

            if k is None:
                if decisiveness < EIGENGAP_DECISIVENESS:
                    return {
                        "verdict": "no_clear_structure",
                        "note": (
                            f"The eigengap suggests {suggested} clusters but only "
                            f"{decisiveness:.1f}x more strongly than the next candidate, "
                            f"under the {EIGENGAP_DECISIVENESS}x this needs to be worth "
                            f"reporting. Corpora with no topic structure still produce a "
                            f"suggestion; this one looks like that. Pass an explicit k to "
                            f"cluster anyway."
                        ),
                        "suggested_k": suggested,
                        "decisiveness": decisiveness,
                        "threshold": EIGENGAP_DECISIVENESS,
                        "clusters": [],
                    }
                k, k_source = suggested, "eigengap"
            else:
                k_source = "requested"

            labels = spectral_clustering(component, k=k, normalized=True)
        except ImportError:
            return {"verdict": "unavailable", "note": "spectral_graph is not on sys.path.",
                    "clusters": []}
        except Exception as exc:  # pragma: no cover - solver-dependent
            return {"verdict": "unavailable", "note": f"{type(exc).__name__}: {exc}",
                    "clusters": []}

        nodes = list(component.nodes())
        # Annotated: the value types are heterogeneous, so without this mypy
        # infers a union from the literal and the sort key below stops typing.
        clusters: list[dict[str, Any]] = []
        for label in range(k):
            members = [nodes[i] for i in range(len(nodes)) if labels[i] == label]
            if not members:
                continue
            member_set = set(members)
            documents = [
                node for node in members
                if self.graph.nodes[node].get("type") == "document"
            ]
            entities = [
                node for node in members
                if self.graph.nodes[node].get("type") == "entity"
            ]
            # The cluster's name, in effect: its best-connected entities are
            # what the documents in it have in common, which is the thing a
            # reader wants and a list of node ids is not.
            top_entities = sorted(
                entities, key=lambda node: (-component.degree(node), str(node))
            )[:max_entities]
            clusters.append(
                {
                    "id": label,
                    "size": len(members),
                    "documents": len(documents),
                    "entities": len(entities),
                    "conductance": (
                        float(conductance(component, member_set))
                        if 0 < len(member_set) < n
                        else None
                    ),
                    "top_entities": top_entities,
                    "sample_documents": sorted(documents, key=str)[:max_documents],
                }
            )

        clusters.sort(key=lambda cluster: -cluster["size"])
        return {
            "verdict": "clustered",
            "k": k,
            "k_source": k_source,
            "suggested_k": suggested,
            "decisiveness": decisiveness,
            "threshold": EIGENGAP_DECISIVENESS,
            "component_size": n,
            "clusters": clusters,
        }

    def duplicate_entities(
        self,
        limit: int = 20,
        name_similarity: float | None = None,
        containment: float | None = None,
    ) -> dict[str, Any]:
        """Entities that are two spellings of one name, as merge candidates.

        `add_document` mints an entity per capitalised token, so the same thing
        arrives under several names: `Builder` and `BUILDER` from a heading,
        `Entity` and `Entities` from a plural, `Builder` and `Builders`. This
        proposes those pairs and reports the structural evidence for each. It
        never merges -- which entities mean the same thing is a decision about
        meaning that the graph cannot make.

        **This used to rank candidates by distance in a spectral embedding, and
        the measurement retired that outright.** On this project's own corpus
        it produced 33,060 candidates of which **67% sat at distance exactly
        0.0000 with a neighbourhood overlap of 1.00** -- the docstring's own
        "strongest merge evidence there is" -- and the top of the list read
        `['LEGAL', 'Virginia']`, `['Canada', 'Professional']`,
        `['Consequences', 'PIPEDA']`. Those are pendant collisions: two
        entities each mentioned by exactly one document, the same one, are
        structurally identical by construction, and 60% of this corpus's
        entities have degree 1. Meanwhile the true duplicates -- `Builder` /
        `Builders`, `Entity` / `Entities`, and all thirty case variants --
        were **not candidates at any rank**. Graded against ground truth the
        spectral ranking scored 0% precision and 0% recall; so did Jaccard, and
        so did containment used as a ranker. Name similarity scored 100%
        precision on its top 20.

        Two things went wrong and only one of them is the pendants.

        *A real duplicate is asymmetric.* `Builder` is mentioned by 29
        documents and `Builders` by 4. That is a subset, not a match, and both
        the embedding distance and the Jaccard overlap are built on symmetry --
        they score the pair 0.14 and rank it below thousands of unrelated ones.
        Containment (`shared / min(degree)`) reads 1.00 on the same pair, and
        is what the evidence here is measured with.

        *And structure cannot generate candidates on a real corpus at all.*
        Tightening it does not help: at Jaccard 1.00 with at least three shared
        documents, the survivors on this corpus are `Oppenheim` / `Schafer`
        (two authors cited in the same three papers), `Nyquist` / `Frequency`,
        and `BUILDER_DEADLINE_SECONDS` / `NODE_DEADLINE_SECONDS`. Every one is
        co-occurrence, not duplication. The synthetic corpus that once
        justified the structural signal assigned entities to documents **at
        random**, which makes an identical neighbourhood astronomically
        improbable and therefore strong evidence. Real corpora are the opposite:
        entities belonging to one topic are mentioned in the same documents --
        that is what a topic *is* -- so identical neighbourhoods are ordinary
        and mean "discussed together". The property the structural test depended
        on is precisely the property a real corpus does not have.

        So names generate the candidates and structure is the evidence, which
        inverts the old docstring's "structure beats names here and names
        actively mislead". That claim was true of the fixture and false of the
        corpus.

        **What this gives up, explicitly: two names for one thing that share no
        characters.** `LanguageModel` for an entity already called something
        else is not found and cannot be, and nothing here should be read as
        looking for it. That case is not merely unimplemented -- it was
        measured, and on real data every method that reaches for it returns
        collocations instead. Anyone reinstating a structural generator should
        re-run that measurement first.

        Pairs come in two kinds, and the difference is how much they need to
        prove. `case` -- the two names are the same token bar capitalisation --
        is certain on the name alone and carries no structural requirement,
        which matters because these are the most asymmetric pairs in the corpus
        (`BUILDER` appears in one document, `Builder` in 29) and any evidence
        floor would drop every one of them. `lexical` is a likeness rather than
        a certainty, so it must also clear `DUPLICATE_CONTAINMENT`; that is
        what separates `Builder` / `Builders` from two merely similar names for
        different things.
        """
        undirected = self.graph.to_undirected(as_view=True)
        if undirected.number_of_nodes() == 0:
            return {"verdict": "no_graph", "note": "The graph is empty.", "pairs": []}

        entities = [
            node for node, attrs in self.graph.nodes(data=True)
            if attrs.get("type") == "entity"
        ]
        if len(entities) < 2:
            return {
                "verdict": "no_graph",
                "note": "Too few entities to compare.",
                "pairs": [],
            }

        min_name = (
            DUPLICATE_NAME_SIMILARITY if name_similarity is None else float(name_similarity)
        )
        min_containment = (
            DUPLICATE_CONTAINMENT if containment is None else float(containment)
        )

        # Blocked by a case-folded prefix so this stays linear in practice. An
        # all-pairs scan is 948,753 comparisons on this corpus against 1,528
        # here, and allocates nothing quadratic as the corpus grows.
        blocks: dict[str, list[str]] = {}
        for entity in entities:
            blocks.setdefault(str(entity).lower()[:DUPLICATE_BLOCK_PREFIX], []).append(
                str(entity)
            )

        neighbours = {
            entity: set(undirected.neighbors(entity)) for entity in entities
        }

        pairs: list[dict[str, Any]] = []
        comparisons = 0
        for block in blocks.values():
            for left, right in combinations(sorted(block), 2):
                comparisons += 1
                same_token = left.lower() == right.lower()
                similarity = (
                    1.0 if same_token
                    else SequenceMatcher(None, left.lower(), right.lower()).ratio()
                )
                if similarity < min_name:
                    continue

                here, there = neighbours[left], neighbours[right]
                if not here or not there:
                    continue
                shared = here & there
                overlap = len(shared) / min(len(here), len(there))

                # A case variant is the same token and needs no corroboration;
                # a mere likeness does. Holding both to the same floor would
                # drop every case variant in this corpus, since the shouted
                # form is typically a single heading in a single document.
                if not same_token and overlap < min_containment:
                    continue

                pairs.append(
                    {
                        "entities": sorted([left, right]),
                        "kind": "case" if same_token else "lexical",
                        "name_similarity": similarity,
                        "shared_documents": len(shared),
                        "containment": overlap,
                        # Jaccard, kept beside containment rather than instead
                        # of it: it is the number that reads low on a real
                        # duplicate, and seeing the two disagree is what shows
                        # the asymmetry rather than hiding it.
                        "neighbourhood_overlap": (
                            len(shared) / len(here | there) if (here | there) else 0.0
                        ),
                        "degrees": [len(here), len(there)],
                    }
                )

        # Certain before likely, then by how alike the names read, then by how
        # much evidence stands behind the pair.
        pairs.sort(
            key=lambda pair: (
                pair["kind"] != "case",
                -pair["name_similarity"],
                -pair["shared_documents"],
                pair["entities"],
            )
        )
        return {
            "verdict": "scanned",
            "pairs": pairs[:limit],
            "total_pairs": len(pairs),
            "entities_compared": len(entities),
            "comparisons": comparisons,
            "name_similarity": min_name,
            "containment": min_containment,
        }

    def bottleneck(self, limit: int = 12) -> dict[str, Any]:
        """The narrowest cut in the corpus, and the nodes that bridge it.

        The A3 application from `reports/spectral_applicability.md`. Sweeps the
        normalized Fiedler vector for the prefix of lowest conductance, then
        names the nodes whose edges actually cross it -- the few entities or
        documents through which two otherwise separate topic areas connect.
        Those are the terms a search should expand on when a query straddles
        both, and the nodes whose removal would fragment the corpus. Degree
        alone does not find them: a bridge entity mentioned by two documents
        has degree 2, which is unremarkable everywhere else in the graph.

        **The verdict has three states, not two, and the middle one is the
        reason this is worth building.** A minimisation always returns
        *something*: ask for the narrowest cut in a perfectly well-knit corpus
        and you get one anyway, and reporting it as a bridge would be a
        fabricated finding of exactly the kind `search` was fixed for. What
        separates them is Cheeger's lower bound, `mu_2 / 2`, which is a proof
        that no cut anywhere in the graph beats it:

        - `certified_none` -- the lower bound is itself above
          `BOTTLENECK_CONDUCTANCE`, so no narrow waist exists *anywhere*. This
          is a theorem about the whole graph, not a statement about the cut
          that was found, and no amount of searching would turn one up.
        - `found` -- the sweep cut came in at or below the line. The bridge
          nodes below are real.
        - `inconclusive` -- the bound permits a bottleneck and the sweep cut did
          not find one. Cheeger brackets the true conductance between
          `mu_2 / 2` and `sqrt(2 * mu_2)`, and that bracket is wide (measured
          from 4x to 546x across graph shapes in
          `reports/spectral_architecture_benchmark.md`), so the sweep cut
          genuinely can miss. Saying so is the honest answer; collapsing it
          into "no bottleneck" would report a gap in the evidence as a finding.

        Runs on the largest connected component, for the same reason
        `connectivity()` measures `lambda_2` there: on a disconnected graph the
        Fiedler vector is a component indicator, so the sweep cut returns one
        component against the rest at conductance 0. That is a true answer to a
        question nobody asked -- "your corpus has an orphan" is what
        `connectivity()` is for, and it would crowd out the real bridge every
        time.
        """
        # Same modelling note as `connectivity()`: every edge runs
        # document -> entity, so reversing one reads "entity is mentioned by
        # document" -- the same relation, not a different claim.
        undirected = self.graph.to_undirected(as_view=True)

        if undirected.number_of_nodes() == 0:
            return {"verdict": "no_graph", "note": "The graph is empty.",
                    "conductance": None, "bridge_nodes": []}

        largest = max(nx.connected_components(undirected), key=len)
        if len(largest) < 2:
            return {
                "verdict": "no_graph",
                "note": "The largest component has a single node; there is nothing to cut.",
                "conductance": None,
                "bridge_nodes": [],
            }

        component = undirected.subgraph(largest)

        try:
            from spectral_graph import cheeger_bounds, compute_spectrum, sweep_cut

            side, phi = sweep_cut(component, normalized=True)
            lower, upper = cheeger_bounds(component)
            # mu_3 as well as mu_2, to detect a tie -- see `tied_cuts` below.
            spectrum = compute_spectrum(component, k=3, normalized=True, which="SM")
        except ImportError:
            return {
                "verdict": "unavailable",
                "note": "spectral_graph is not on sys.path.",
                "conductance": None,
                "bridge_nodes": [],
            }
        except Exception as exc:  # pragma: no cover - solver-dependent
            return {
                "verdict": "unavailable",
                "note": f"{type(exc).__name__}: {exc}",
                "conductance": None,
                "bridge_nodes": [],
            }

        if lower > BOTTLENECK_CONDUCTANCE:
            verdict = "certified_none"
        elif phi <= BOTTLENECK_CONDUCTANCE:
            verdict = "found"
        else:
            verdict = "inconclusive"

        # `mu_2 ~= mu_3` means the graph has more than two topic areas, and the
        # Fiedler vector picks one of several equally-narrow cuts arbitrarily.
        # Worth reporting rather than hiding: running this twice on such a
        # corpus returns different *sides* -- measured 99/198 and 97/200 on
        # alternating runs of the same three-topic graph -- while the
        # conductance (0.008264, all 12 runs) and the bridge entities
        # (BRIDGE0/BRIDGE1, all 12 runs) stay put. An operator who sees the
        # split move and has not been told why will read a working diagnostic
        # as a broken one. It is also a real finding in its own right: a tie
        # says there are three or more areas here, not two.
        mu_2, mu_3 = float(spectrum[1]), float(spectrum[2])
        tied_cuts = bool(mu_3 - mu_2 <= 0.1 * mu_3) if mu_3 > 1e-12 else False

        # The nodes carrying the cut, ranked by how much of it they carry. A
        # node's crossing count is what makes it a bridge; its total degree is
        # reported beside it because the two coming apart is the whole point --
        # a bridge is a node whose few edges happen to be the load-bearing ones.
        crossing: dict[str, int] = {}
        crossing_edges = 0
        for source, target in component.edges():
            if (source in side) != (target in side):
                crossing_edges += 1
                crossing[source] = crossing.get(source, 0) + 1
                crossing[target] = crossing.get(target, 0) + 1

        bridge_nodes = [
            {
                "id": node,
                "type": self.graph.nodes[node].get("type", "unknown"),
                "crossing_edges": count,
                "degree": component.degree(node),
                "side": "a" if node in side else "b",
            }
            for node, count in sorted(
                crossing.items(), key=lambda item: (-item[1], str(item[0]))
            )[:limit]
        ]

        return {
            "verdict": verdict,
            "conductance": float(phi),
            "cheeger_lower": float(lower),
            "cheeger_upper": float(upper),
            "threshold": BOTTLENECK_CONDUCTANCE,
            "component_size": component.number_of_nodes(),
            "side_a": len(side),
            "side_b": component.number_of_nodes() - len(side),
            "crossing_edges": crossing_edges,
            "bridge_nodes": bridge_nodes,
            "total_bridge_nodes": len(crossing),
            "tied_cuts": tied_cuts,
            "mu_2": mu_2,
            "mu_3": mu_3,
        }

    def connectivity(self) -> dict[str, Any]:
        """Structural health of the knowledge graph: components, and lambda_2.

        A reindex that silently drops edges -- an entity-extraction regression
        in `add_document`, say -- does not change the document count and does
        not raise. It shows up here first, as a rising component count or a
        collapsing `lambda_2`, long before it shows up as worse search.

        **Components come from networkx, not from the spectrum.** The textbook
        identity is that the multiplicity of eigenvalue 0 equals the number of
        connected components, and `reports/spectral_applicability.md` proposes
        counting near-zero eigenvalues for exactly that reason. Two measured
        objections, both on this project's own graph shape (920 nodes):

        1. It is 42x the cost of the linear-time answer -- 32ms of
           eigendecomposition against 0.76ms of `number_connected_components`
           -- for a number networkx already computes exactly.
        2. On the *normalized* Laplacian it is simply wrong. The identity holds
           for `L = D - A`; for `I - D^-1/2 A D^-1/2` an isolated node has
           `D^-1/2 = 0`, so the `I` term leaves a bare 1 on its diagonal and it
           contributes eigenvalue **1, not 0**. This graph has 29 isolated
           nodes out of 30 components, so the spectral count returns 1 where
           the truth is 30.

        **lambda_2 is measured on the largest component, and normalized.** Two
        deliberate choices:

        - On the whole graph lambda_2 is identically 0 whenever the corpus is
          disconnected, and it is -- 30 components in the shape measured here.
          A health signal that reads 0.0 every time is not a signal. The
          largest component's lambda_2 is the number that actually moves when
          the body of the corpus knits together or comes apart.
        - Normalized, so it lands in [0, 2] and does not scale with degree.
          The unnormalized lambda_2 grows as documents mention more entities,
          which makes this reindex's value incomparable with last week's --
          and comparing across reindexes is the entire purpose.

        Returns `lambda_2: None` rather than a number when the largest
        component has fewer than two nodes: lambda_2 is undefined there, and 0.0
        would read as "totally disconnected" rather than "nothing to measure".
        """
        # Every edge runs document -> entity, so reversing one reads "entity is
        # mentioned by document" -- the same relation, not a different claim.
        # That is what makes to_undirected() safe to apply on the caller's
        # behalf here, and it is applied explicitly because `spectral_graph`
        # refuses a DiGraph rather than guessing (see `_require_undirected`).
        undirected = self.graph.to_undirected(as_view=True)
        n = undirected.number_of_nodes()

        if n == 0:
            return {"components": 0, "largest_component": 0, "isolated_nodes": 0,
                    "lambda_2": None}

        components = nx.number_connected_components(undirected)
        largest = max(nx.connected_components(undirected), key=len)
        isolated = sum(1 for _, degree in undirected.degree() if degree == 0)

        lambda_2: float | None = None
        unavailable: str | None = None
        if len(largest) < 2:
            unavailable = "largest component has fewer than 2 nodes"
        else:
            try:
                # Imported here, not at module scope. `spectral_graph` lives at
                # the project root and is not part of the installed
                # `langgraph_agent` distribution, so it is importable only when
                # the root is on sys.path -- true for the console and the test
                # suite, false for an MCP server launched from anywhere else. A
                # top-level import would turn a missing diagnostic into a module
                # that will not load at all.
                from spectral_graph import compute_spectrum

                spectrum = compute_spectrum(
                    undirected.subgraph(largest), k=2, normalized=True, which="SM"
                )
                # Clamp solver noise: lambda_1 is 0 and lambda_2 >= 0, so a
                # small negative here is arithmetic, not a finding.
                lambda_2 = max(float(spectrum[1]), 0.0)
            except ImportError:
                unavailable = "spectral_graph is not on sys.path"
            except Exception as exc:  # pragma: no cover - solver-dependent
                # Same posture as the chunk count in `stats()`: this is a
                # diagnostic, and losing it must not cost the console the
                # counters it renders the header from. Named rather than
                # dropped, so "could not measure" never reads as "measured 0".
                unavailable = f"{type(exc).__name__}: {exc}"

        result: dict[str, Any] = {
            "components": components,
            "largest_component": len(largest),
            "isolated_nodes": isolated,
            "lambda_2": lambda_2,
        }
        if unavailable is not None:
            result["lambda_2_unavailable"] = unavailable
        return result

    def stats(self) -> dict[str, Any]:
        """Counters for the console header, plus the connectivity health check."""
        documents = sum(
            1 for _, attrs in self.graph.nodes(data=True) if attrs.get("type") == "document"
        )
        try:
            chunks = self.collection.count()
        except Exception:
            # Chroma is a separate store from the graph; a failure here should
            # not take out the node/edge counts that come from memory.
            chunks = 0

        nodes = self.graph.number_of_nodes()
        edges = self.graph.number_of_edges()

        # The console polls this every five seconds and the eigendecomposition
        # is ~44ms on a 920-node graph, growing with the corpus. It is cached
        # against (nodes, edges) because those are what every mutation path in
        # this class moves: `add_document` only ever adds, and `clear` zeroes
        # both. Re-adding an identical document changes neither count -- and
        # changes no structure either, so the cached answer is still the right
        # one. Nothing here removes an edge without removing a node.
        if self._connectivity_cache is not None and self._connectivity_cache[0] == (nodes, edges):
            connectivity = self._connectivity_cache[1]
        else:
            connectivity = self.connectivity()
            self._connectivity_cache = ((nodes, edges), connectivity)

        return {
            "total_documents": documents,
            "total_chunks": chunks,
            "total_nodes": nodes,
            "total_edges": edges,
            **connectivity,
        }

    def clear(self) -> dict[str, Any]:
        """Empty the knowledge base, keeping the files that hold it.

        The store is emptied in place rather than deleted: Chroma has this
        directory open, and pulling it out from under a live client is a worse
        failure than an empty collection. What is left behind is the same shape
        a reindex leaves -- a real store with nothing in it.

        Chroma goes first, and the graph is only cleared once it has. The two
        halves answer different questions (search, and structure), so a run that
        wiped one and failed on the other would leave the corpus disagreeing
        with itself while reporting success. On failure this raises with both
        intact.

        Returns:
            What was removed, plus the (now zeroed) stats.
        """
        removed_nodes = self.graph.number_of_nodes()
        removed_edges = self.graph.number_of_edges()

        existing = self.collection.get(include=[]).get("ids", [])
        if existing:
            self.collection.delete(ids=existing)

        self.graph.clear()
        self._lexical_index = None
        # Without this the clear lives only in memory: the next process start
        # reloads the old graph off disk and the corpus comes back.
        self._save_graph()

        return {
            "removed_chunks": len(existing),
            "removed_nodes": removed_nodes,
            "removed_edges": removed_edges,
            **self.stats(),
        }

    def export_corpus(self) -> dict[str, Any]:
        """The whole corpus as one JSON-serialisable document.

        Embeddings are left out. They are the bulk of the store by a wide
        margin and the least useful part of a dump: the embedder is local, so
        anything reading this file back can regenerate them, and a reader
        without the same model could not use them anyway. The file says so
        itself rather than leaving the omission to be discovered.

        The graph half is `node_link_data`, which is exactly the on-disk format
        `_save_graph` writes, so it can be compared against
        `knowledge/knowledge_graph.json` directly.
        """
        errors: list[str] = []
        chunks: list[dict[str, Any]] = []
        try:
            stored = self.collection.get(include=["documents", "metadatas"])
            ids = stored.get("ids") or []
            documents = stored.get("documents") or []
            metadatas = stored.get("metadatas") or []
            chunks = [
                {
                    "id": chunk_id,
                    # The document this passage came from, lifted out of the
                    # metadata so the export stands on its own: a row's id is
                    # `path#0007`, and a reader should not have to know how to
                    # parse that to group the file back together.
                    "doc_id": _document_id_of(
                        chunk_id, metadatas[i] if i < len(metadatas) else None
                    ),
                    "content": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
                for i, chunk_id in enumerate(ids)
            ]
        except Exception as exc:
            # Same posture as `stats()`: a Chroma failure must not cost us the
            # graph half of the export as well.
            errors.append(f"reading chunks: {exc}")

        return {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Embeddings are omitted; re-indexing regenerates them locally "
                f"with {EMBEDDING_MODEL_NAME}."
            ),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "stats": self.stats(),
            "graph": nx.readwrite.json_graph.node_link_data(self.graph),
            "chunks": chunks,
            "errors": errors,
        }


# Files worth indexing, and the directories that only add noise. Shared by
# `scripts/reindex.py` and the console's reindex button so the two cannot drift
# into indexing different corpora.
PROJECT_INDEX_PATTERNS = ("**/*.py", "**/*.md", "**/*.txt", "**/*.rst")
# Matched as plain substrings of the path, so no globs: "*.egg-info" never
# matched anything and let build metadata (SOURCES.txt, top_level.txt) into
# the corpus as if it were project knowledge.
PROJECT_INDEX_EXCLUDES = (
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".pytest_cache", ".mypy_cache", "build/", "dist/", ".egg-info",
    "knowledge/", "scripts/", ".qwen/", ".claude/",
)

# Above this size a file is documentation of something else, not a unit of
# knowledge, and it would dominate the embedding budget.
MAX_INDEXABLE_BYTES = 100_000


def iter_project_files(
    root: str = ".", exclude_dirs: tuple[str, ...] | list[str] | None = None
) -> list[Path]:
    """Collect the project files worth indexing."""
    excludes = tuple(exclude_dirs) if exclude_dirs is not None else PROJECT_INDEX_EXCLUDES
    root_path = Path(root)

    files: list[Path] = []
    for pattern in PROJECT_INDEX_PATTERNS:
        for file_path in root_path.glob(pattern):
            if any(excluded in str(file_path) for excluded in excludes):
                continue
            files.append(file_path)
    return sorted(set(files))


def index_project_files(
    kb: "GraphRAGKnowledgeBase", root: str = "."
) -> dict[str, Any]:
    """Index every project file into the knowledge base.

    Returns a report rather than printing one, so both the CLI script and the
    console's reindex button can render it their own way.
    """
    files = iter_project_files(root)
    wanted = {str(path) for path in files}

    # A reindex rebuilds rather than accumulates. A file that no longer
    # qualifies -- renamed, deleted, or newly excluded -- has to leave the
    # corpus, or it keeps answering searches and keeps its graph node long
    # after it stops existing.
    #
    # Stored rows are chunks, so staleness is a question about the *document*
    # a row belongs to, not about the row's own id: `CLAUDE.md#0007` is not in
    # `wanted` and never will be. Comparing ids directly would delete the
    # entire corpus on every reindex and rebuild it from scratch -- which
    # ends in the same place here, but would quietly become a full re-embed of
    # every document the moment anything reindexed a subset.
    try:
        existing = kb.collection.get(include=["metadatas"])
        existing_ids = existing.get("ids") or []
        existing_metadatas = existing.get("metadatas") or []
        stale = [
            chunk_id
            for i, chunk_id in enumerate(existing_ids)
            if _document_id_of(
                chunk_id,
                existing_metadatas[i] if i < len(existing_metadatas) else None,
            )
            not in wanted
        ]
        if stale:
            kb.collection.delete(ids=stale)
    except Exception as exc:  # pragma: no cover - Chroma unavailable
        errors_pre = [f"pruning stale documents: {exc}"]
    else:
        errors_pre = []
    kb.graph.clear()

    indexed, skipped = 0, 0
    errors: list[str] = list(errors_pre)

    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8")
            if len(content) > MAX_INDEXABLE_BYTES:
                skipped += 1
                continue

            kb.add_document(
                str(file_path),
                content,
                {
                    "path": str(file_path),
                    "type": "python" if file_path.suffix == ".py" else "markdown",
                },
            )
            indexed += 1
        except Exception as exc:
            errors.append(f"{file_path}: {exc}")

    report: dict[str, Any] = {"indexed": indexed, "skipped": skipped, "errors": errors}
    report.update(kb.stats())
    return report


# Cached knowledge base instance so repeated MCP calls and test runs do not
# reload the embedding model and Chroma store every time.
_kb_instance: GraphRAGKnowledgeBase | None = None


def get_knowledge_base(persist_dir: str = "./knowledge") -> GraphRAGKnowledgeBase:
    """The singleton knowledge base, **built if it does not exist yet**.

    This is the door for the one act that is allowed to bring a corpus into
    existence: indexing. Everything that only wants to read -- the console's
    header, the document list, the graph sweep, the Researcher's search --
    goes through `open_knowledge_base()` instead, which returns `None` rather
    than manufacturing a store. Reading is not a reason for a corpus to exist.

    `persist_dir` is honoured only on the call that builds the singleton; the
    process holds one corpus, and the argument exists so a caller that opened
    a store somewhere other than the default is not silently handed the
    default one instead.
    """
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = GraphRAGKnowledgeBase(persist_dir)
    return _kb_instance


def corpus_exists(persist_dir: str = "./knowledge") -> bool:
    """Whether a corpus has been built, without building or opening one.

    A store that was emptied by `clear()` still exists -- that is the point of
    emptying it in place -- so this answers "has anyone indexed here", not "is
    there anything in it". `corpus_state()` tells those two apart.
    """
    return (Path(persist_dir) / "chroma").is_dir()


def open_knowledge_base(persist_dir: str = "./knowledge") -> GraphRAGKnowledgeBase | None:
    """The corpus if one has been built, `None` if none has. Never builds one.

    The reason this exists rather than every caller using `get_knowledge_base`:
    constructing a `GraphRAGKnowledgeBase` creates the store on disk. The
    console polls its header every five seconds, so with the creating door
    wired to a read, starting the server was enough to leave a corpus behind --
    an empty one that then reported itself as a knowledge base. A corpus should
    be there because someone indexed, or not be there at all.
    """
    if _kb_instance is not None:
        return _kb_instance
    if not corpus_exists(persist_dir):
        return None
    return get_knowledge_base(persist_dir)


def corpus_state(persist_dir: str = "./knowledge") -> tuple[str, str]:
    """Report the corpus as `absent`, `empty` or `indexed`, plus the model name.

    Deliberately lightweight and deliberately non-creating: it opens Chroma
    read-only and never loads sentence-transformers, so the console can poll it
    on a timer without either blocking on the model or bringing a store into
    being as a side effect of asking about one.

    `absent` and `empty` are kept apart because they call for different things.
    Nothing has ever been indexed here, versus a corpus that exists and was
    emptied -- and a run against either finds nothing, which is exactly why the
    operator has to be told which it was.

    Returns:
        (state, embedding_model_name)
    """
    chroma_dir = Path(persist_dir) / "chroma"
    if not chroma_dir.is_dir():
        return "absent", EMBEDDING_MODEL_NAME

    try:
        client = chromadb.PersistentClient(str(chroma_dir))
        # `get_collection`, not `get_or_create_collection`: asking after the
        # corpus must not create the collection it is asking about.
        collection = client.get_collection(name="knowledge")
        return ("indexed" if collection.count() > 0 else "empty"), EMBEDDING_MODEL_NAME
    except Exception:
        # A store whose collection is missing or unreadable has nothing to
        # answer with, which is what `empty` already means to every caller.
        return "empty", EMBEDDING_MODEL_NAME


def is_knowledge_base_indexed(persist_dir: str = "./knowledge") -> tuple[bool, str]:
    """Whether the knowledge base holds any documents.

    Returns:
        (indexed, embedding_model_name)
    """
    state, model = corpus_state(persist_dir)
    return state == "indexed", model


# Create MCP server
server = MCPServer("graphrag")


# Register tools with the server
@server.tool(name="search_knowledge_graph")
def search_tool(query: str, top_k: int = 5) -> str:
    """Search the knowledge base for relevant documents and passages.

    Searching a corpus nobody built returns no results and says why. It does
    not build one: a retrieval call is not a request for a knowledge base.
    """
    kb = open_knowledge_base()
    if kb is None:
        return json.dumps({"results": [], "source": "no_corpus", "note": NO_CORPUS_NOTE}, indent=2)
    return json.dumps(kb.search(query, top_k), indent=2)


@server.tool(name="query_knowledge_graph")
def query_tool(entity: str, hops: int = 2) -> str:
    """Query the knowledge graph for entity relationships."""
    kb = open_knowledge_base()
    if kb is None:
        return json.dumps(
            {
                "entity": entity,
                "neighbors": [],
                "subgraph_nodes": 0,
                "subgraph_edges": 0,
                "source": "no_corpus",
                "note": NO_CORPUS_NOTE,
            },
            indent=2,
        )
    return json.dumps(kb.query_graph(entity, hops), indent=2)


async def main() -> None:
    """Run the GraphRAG MCP server."""
    server.run(transport="stdio")


if __name__ == "__main__":
    asyncio.run(main())
