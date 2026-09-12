"""GraphRAG MCP Server.

Provides knowledge base search with entity/relation graph + vector store.

Usage:
    python -m src.langgraph_agent.graphrag_server

Or with stdio transport for MCP:
    mcp dev src/langgraph_agent/graphrag_server.py
"""

import asyncio
import hashlib
import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import chromadb
import networkx as nx
from mcp.server import MCPServer

from langgraph_agent.corpus_spectral import (
    BOTTLENECK_CONDUCTANCE,
    DUPLICATE_BLOCK_PREFIX,
    DUPLICATE_CONTAINMENT,
    DUPLICATE_NAME_SIMILARITY,
    EIGENGAP_DECISIVENESS,
    MAX_AUTO_CLUSTERS,
    CorpusSpectralMixin,
)
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

# Where it runs, named rather than left to sentence-transformers, which picks
# `cuda:0` whenever `torch.cuda.is_available()` says True. That answer only
# means a driver answered, not that the installed build carries kernels for
# the card: on 2026-09-10 a Builder swapped the venv to `torch+cu130` on a
# GTX 1060 (compute 6.1, which CUDA 13 dropped), `is_available()` stayed True,
# and every `encode()` raised `no kernel image is available`, taking search
# and indexing down with it. And a 3 GB card is wanted whole for the model
# being offloaded onto it; a 384-dimension MiniLM is not worth any of that.
EMBEDDING_DEVICE = "cpu"

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
    "No corpus has been indexed here, so there is nothing to retrieve. Two "
    "things build one: a run, which indexes the project it was started from "
    "before the Architect opens, and embedding a document into the corpus from "
    "the console. Nothing else does. Reaching this note during a run means the "
    "walk found nothing to index, or INDEX_PROJECT_BEFORE_RUN is off."
)







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
#
# **A hand-audited list drifts, and this one did.** The claim above held when
# it was written and had stopped holding by 2026-09-09, because the corpus it
# was audited against had moved on -- much of it prose written *since*, in this
# file and in CLAUDE.md. `Tests`, `Measured` and `System` were the 6th, 8th and
# 10th best-connected entities in the graph, which is the exact failure the
# list exists to prevent, one vocabulary later. `Measured` is the sharpest:
# CLAUDE.md opens sentences with it ("Measured on this corpus...") twenty-nine
# times, so the prose recording these measurements was minting the entity.
#
# The additions below were chosen by counting, per token, the capitals that
# position does **not** explain -- not at a line start, not after a full stop,
# not the first cell of a table row. A token with zero of those is recording
# where it sits. That is the same heuristic rejected above, used the way it is
# sound: to *nominate* candidates for a human to rule on, never to filter. Two
# nominations were refused on that read and are the reason the pass is a hand
# audit rather than a script. `L_dense` scores zero free capitals because an
# assignment starts its line -- it is a real identifier and stays. `Spectral`
# scores two, both marginal, and stays because it is a term this corpus is
# about; the cost of a wrong removal is a severed true relation, which is
# exactly what the paragraph above refuses. `System`, `Search`, `State` and
# `Verification` were nominated by rank and cleared by the count: `System`
# alone carries 22 free capitals, so the entity is earned.
#
# It moved again on 2026-09-12, which is the point of the guard rather than a
# surprise: `Reported` and `Computed` each crossed the four-document floor at
# zero position-free capitals, as prose was written about what a phase reports
# and about what is computed where. Both are the `Measured` case exactly -- the
# writing-up of a change minting the entity -- and both joined the list by the
# same count. The lesson is not the two words: it is that this list is a claim
# about a vocabulary, and the vocabulary grows every time someone documents
# something.
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

    Measured Initialize Dense Build Maximum Empty Skipping Dictionary Apply
    Refused Whether Split Asserted Degree Shared Based Demonstrates References
    Generate Extract Point Seconds Deliberately Named Built Asking Pinned
    Insert Tests Write Reported Computed Cached Complete Convert Dimension
    Naming
    """.split()
)

# The embedder's context window, minus the two special tokens it adds. This is
# a property of `EMBEDDING_MODEL_NAME` (`max_seq_length` is 256 on
# all-MiniLM-L6-v2), not a tuning knob: a passage longer than this is not
# embedded badly, it is **silently truncated and the tail discarded**.
#
# That is what this constant exists to stop. A document used to be embedded
# whole, in one `encode()` call, with `MAX_INDEXABLE_BYTES` then allowing 100 KB --
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


def _content_sha(content: str) -> str:
    """A fingerprint of a document's text, stored on every chunk it becomes.

    This is what lets a rebuild keep the vectors it already has. Embedding is
    the only expensive part of indexing this project -- measured warm, a full
    rebuild of 77 files and 1,618 chunks takes 52.0s, of which reading every
    file, hashing it, fetching the store's metadata and rebuilding the whole
    entity graph account for 0.1s. So a rebuild that re-embeds only what
    changed costs what the change costs, and one where nothing changed costs
    nothing at all and never loads the model.

    A hash rather than an mtime: a checkout, a `git stash`, a file copied back
    into place all move the timestamp without changing a byte, and re-embedding
    a corpus because someone switched branches is the cost this exists to
    avoid. Truncated to 16 hex characters because it is compared, never
    trusted -- a collision re-uses a stale vector, which the next edit to that
    file corrects, and 64 bits of it is not a risk anyone here will meet.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

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


class GraphRAGKnowledgeBase(CorpusSpectralMixin):
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

            self._embedder = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBEDDING_DEVICE)
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

    def add_document(self, doc_id: str, content: str, metadata: dict[str, Any] | None = None) -> int:
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

        Returns:
            How many chunks this document became. Reported rather than
            discarded because it is the one number that says whether a
            document is *reachable*: chunking is what made a long file
            searchable past its first thousand characters, and a caller adding
            one document at a time -- an upload -- has no other way to see that
            it landed as several passages rather than as a truncated header.
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
            # The fingerprint rides on every chunk, so `index_project_files`
            # can ask "is this document still the one I embedded?" from the
            # metadata it already fetches to prune with. Computed here rather
            # than passed in, for the reason `doc_id` and `chunk_count` are:
            # it describes what the store did with the text, not what the
            # caller knows about the file, and one caller forgetting it would
            # silently cost that document its reuse.
            base["sha"] = _content_sha(content)
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

        self._add_to_graph(doc_id, content, metadata)

        # The lexical index describes a corpus that no longer exists. Dropped
        # rather than amended: the next search rebuilds it from the store in
        # milliseconds, and an index maintained in parallel with Chroma is a
        # second account of the same corpus, free to disagree with it.
        self._lexical_index = None

        self._save_graph()
        return len(chunks)

    def _add_to_graph(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """The half of `add_document` that costs nothing: node and entities.

        Split out because a rebuild that keeps a document's vectors still has
        to put it back in the graph -- `index_project_files` clears the graph
        up front, so a document whose embeddings were reused would otherwise
        vanish from it while staying perfectly searchable. Entity extraction is
        a regex over the text and the whole graph rebuilds in 0.04s, so the
        cheap half is simply always done.
        """
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
        # A fetched web page is a *retrieval source*, not knowledge-graph
        # material, and the capital rule is far worse on it than on this
        # project's own files. `ENTITY_STOPWORDS` is a hand-audited list, and it
        # was audited against source code and numpy docstrings, where the forced
        # capitals are `Returns`, `Every`, `False`. Web prose opens sentences
        # with a different vocabulary entirely, and none of it is on the list.
        # Measured on 2026-09-09: 15 fetched pages minted **551 entities that no
        # project document mentions -- 19% of the whole graph, 36.7 per page** --
        # `Although`, `Afterward`, `Altogether`, `Again`, `Accessed`. Skipping
        # them took the graph from 2,830 entities to 2,285 and duplicate
        # candidates from 205 to 171. Those edges are read as evidence by
        # `neighborhood`, `topics` and `duplicate_entities`, so the graph
        # degrades as the corpus grows.
        # The page is still chunked, embedded and fully retrievable; it simply
        # stops voting on what the entities of this project are.
        entities = []
        for word in ([] if _is_web_document(doc_id) else content.split()):
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
        # An empty query is not a question and must not answer like one. The
        # empty string still embeds, and on this corpus it matched five chunks
        # topping 0.412 -- over `RETRIEVAL_RELEVANCE_FLOOR`, so those would have
        # been formatted as findings and announced as "Research complete". The
        # fabricated retrieval hit, arriving through the query this time.
        if not query or not query.strip():
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
        # One spelling of the fuzzy match, so the console and the Researcher's
        # tool cannot resolve an id differently.
        found = self._resolve_node(entity)
        if found is None:
            return {"error": f"Entity '{entity}' not found"}
        entity = found

        # Undirected, for the reason `neighborhood` is: every edge runs
        # document -> entity, so an entity has in-edges only and a directed walk
        # from one reaches nothing. It returned the entity alone for every
        # entity in the corpus -- `Planner`, 21 edges, reported 0 against a real
        # 2-hop neighbourhood of 614 -- which reads as "this term connects to
        # nothing" rather than as a broken walk. See `test_graph_queries.py`.
        undirected = self.graph.to_undirected(as_view=True)
        neighbors = nx.single_source_shortest_path_length(
            undirected, entity, cutoff=hops
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

        # `"" in anything` is True, so a blank id matched on the first
        # comparison and resolved to whatever the graph enumerated first. The
        # caller asked about nothing and got a real document's neighbourhood.
        if not node_id or not node_id.strip():
            return None

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

        The A4 application of the spectral applicability study, and the
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


# Files worth indexing, and the directories that only add noise. One list, so
# the rebuild a run does and the walk `corpus_health` compares against cannot
# drift into describing different corpora.
PROJECT_INDEX_PATTERNS = ("**/*.py", "**/*.md", "**/*.txt", "**/*.rst")
# Matched as plain substrings of the path, so no globs: "*.egg-info" never
# matched anything and let build metadata (SOURCES.txt, top_level.txt) into
# the corpus as if it were project knowledge.
PROJECT_INDEX_EXCLUDES = (
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    ".pytest_cache", ".mypy_cache", "build/", "dist/", ".egg-info",
    "knowledge/", "scripts/", ".qwen/", ".claude/",
    # Seat-diagnostic sweeps. `.gitignore` already calls these "per-run
    # measurements against non-deterministic models, not history", but the walk
    # is a glob and not git, so being gitignored kept them out of commits and
    # not out of the corpus: five sweeps from one afternoon sat in the store as
    # indexed documents, answering questions with a week-old measurement of a
    # seating nobody runs any more. `reports/*.md` is written deliberately and
    # stays indexed; only the timestamped sweeps under it are excluded.
    "reports/diagnostics/",
)

# Above this size a file is not a document at all -- a data dump, a minified
# bundle, a log -- and reading it into the corpus indexes something nobody
# wrote. It was 100,000, and that number outlived its own justification: it
# said such a file "would dominate the embedding budget", which was true when
# `add_document` embedded a whole file as ONE vector and a long document's
# single embedding competed with everyone else's. Chunking ended that, and
# `search` collapsing chunks back onto documents ended it twice -- a document
# now takes exactly one result slot however many chunks it holds. Measured on
# this corpus: CLAUDE.md carries 7.1% of all 1,687 chunks and returned in
# exactly 1 of 5 slots on every query tried, never more.
#
# What the old number did instead was dictate the shape of the project.
# `corpus_health.py` exists as a separate module because adding the staleness
# check pushed `graphrag_server.py` from 98,920 characters to 104,582 -- the
# module that defines the corpus would have dropped out of it. By 2026-09-11
# `nodes.py` stood at 82% of the limit, `graphrag_server.py` at 75%,
# `test_graph.py` at 73%, and CLAUDE.md had 13 characters left, so the next
# paragraph anyone wrote would have silently cost the project its own
# documentation. A constant that decides how files must be split is not
# measuring anything about knowledge.
#
# The remaining real risk sets the ceiling, and it is why this is 250,000 and
# not unbounded: `search` retrieves a window of chunks *before* collapsing
# them, so a document holding a large enough share of the corpus can fill that
# window with itself and starve every other source -- the case
# `SEARCH_ESCALATION` widens the window for. At 7.1% the largest document here
# is nowhere near it; a file several times this limit would be.
MAX_INDEXABLE_BYTES = 250_000

# Where a document uploaded from the console lands, relative to the project
# root. An upload is written to disk *before* it is embedded, and that ordering
# is the design rather than a convenience: the corpus is a function of what is
# on disk, and `index_project_files` rebuilds it from there -- clearing the
# graph and pruning every stored row whose document is not in the walk. So a
# document embedded straight into the store and nowhere else survives exactly
# until the next reindex, which then deletes it silently, in a pass that
# reports success and a file count that looks right. Writing the file first is
# what makes an upload part of the corpus rather than a guest in it, and it is
# why this directory must stay out of `PROJECT_INDEX_EXCLUDES`.
UPLOADS_DIR = "uploads"

# Where the online research phase writes the pages it fetched, relative to the
# project root. It lives here rather than in `web_research` because
# `add_document` has to recognise one of these on sight -- see
# `_is_web_document` -- and because `web_research` imports this module, so the
# constant could not travel the other way.
WEB_RESEARCH_DIR = "research/web"

# The suffixes an upload may carry, derived from the walk's own patterns rather
# than restated beside them. The two have to agree exactly or an upload is
# accepted, embedded, and then dropped at the next rebuild for a reason nobody
# is told -- the same drift `PROJECT_INDEX_EXCLUDES` had while two scripts kept
# their own copy of it. Compared case-insensitively but *stored* lower-cased,
# because the walk is a glob and a glob is case-sensitive here: `NOTES.MD`
# written as given is a file the reindex cannot see.
INDEXABLE_SUFFIXES = tuple(sorted({Path(pattern).suffix for pattern in PROJECT_INDEX_PATTERNS}))


def _is_web_document(doc_id: str) -> bool:
    """True for a page the online research phase fetched.

    Decided from the **path**, never from the call site, for the same reason
    `_document_metadata` is shared between the upload and the walk: a reindex
    re-reads these files as ordinary markdown, so a decision made only where a
    document is first stored would be silently reversed the next time anyone
    rebuilt. Matching on the parent directories rather than a prefix keeps it
    working when the root is absolute, which it is in every test.
    """
    parts = PurePosixPath(str(doc_id).replace("\\", "/")).parts
    wanted = PurePosixPath(WEB_RESEARCH_DIR).parts
    return len(parts) > len(wanted) and parts[-len(wanted) - 1 : -1] == wanted


def _document_metadata(path: Path) -> dict[str, str]:
    """The metadata a document is stored under. One definition, two callers.

    The reindex and an upload have to agree on this exactly: `path` is what the
    graph node is keyed by and what `search` reports as a filename, and `type`
    is what the console colours a node with. Two spellings of it would file the
    same file twice under two descriptions.
    """
    return {
        "path": str(path),
        "type": "python" if path.suffix == ".py" else "markdown",
    }


def store_uploaded_document(
    kb: "GraphRAGKnowledgeBase",
    name: str,
    content: str,
    root: str = ".",
) -> dict[str, Any]:
    """Write an uploaded document under `uploads/` and embed it, in that order.

    See `UPLOADS_DIR` for why the file comes first. What is left after this
    call is a document indistinguishable from any other in the corpus: the same
    id shape, the same metadata, and a place in the walk, so the next reindex
    re-reads it instead of pruning it.

    Every refusal below is a `ValueError` naming what was wrong, because each
    alternative to refusing is worse than a failed upload and none of them
    announces itself. A file the embedder cannot read becomes a vector of noise
    filed under a real filename. A file over the walk's size limit is embedded
    now and dropped at the next rebuild. A name with a path in it writes
    outside the directory the reindex looks at, so the document is embedded and
    then pruned by the very next pass.

    Args:
        kb: The corpus to add to. This is the *creating* door's knowledge base:
            an upload is a request for a corpus to hold it.
        name: The filename as the browser reported it. Only its last component
            is used.
        content: The document's text, already decoded.
        root: Project root the walk runs from, so the id this stores under is
            the id the walk will produce.

    Returns:
        What was stored -- the path, whether it replaced a document already
        there, how many passages it became -- plus the corpus stats, so the
        console can repaint its header from the same reply.

    Raises:
        ValueError: the name, the type, the size or the content is one the
            corpus cannot take. The message says which, and why.
    """
    # Only the last component, so a name carrying a path cannot place the file
    # outside `uploads/`. `..` survives that (`PurePosixPath("..").name` is
    # `".."`), so it is refused by name rather than left to the suffix check.
    safe = PurePosixPath(name.replace("\\", "/")).name
    if safe in ("", ".", "..") or "\x00" in safe:
        raise ValueError(f"{name!r} is not a filename this can store.")

    suffix = PurePosixPath(safe).suffix.lower()
    if suffix not in INDEXABLE_SUFFIXES:
        accepted = ", ".join(INDEXABLE_SUFFIXES)
        raise ValueError(
            f"{safe!r} is not one of {accepted}. There is no text extractor "
            "here, so a PDF, a .docx or an image would be embedded as whatever "
            "its bytes happen to decode to rather than as what it says -- and "
            "it would look like a document in the corpus afterwards. Convert "
            "it to text first."
        )
    stored_name = PurePosixPath(safe).stem + suffix

    if "\x00" in content:
        raise ValueError(
            f"{stored_name!r} contains null bytes, so it is not text however it "
            "is named. Nothing was stored."
        )
    if not content.strip():
        raise ValueError(f"{stored_name!r} has nothing in it to embed.")
    # The walk's own test, character for character. Accepting a larger file
    # here would embed it now and silently skip it at every reindex after.
    if len(content) > MAX_INDEXABLE_BYTES:
        raise ValueError(
            f"{stored_name!r} is {len(content):,} characters and the limit is "
            f"{MAX_INDEXABLE_BYTES:,} -- the same limit a reindex applies. "
            "Accepting it would embed it now and drop it at the next rebuild. "
            "Split it up."
        )

    directory = Path(root) / UPLOADS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / stored_name
    # Read before the write, so "replaced" is a fact rather than a guess. An
    # upload of a name already there is the ordinary way to correct a document,
    # and `add_document` deletes that document's previous chunks before the new
    # ones land, so nothing of the old version is left behind in the store.
    replaced = path.exists()
    path.write_text(content, encoding="utf-8")

    chunks = kb.add_document(str(path), content, _document_metadata(path))

    report: dict[str, Any] = {
        "path": str(path),
        "name": stored_name,
        "replaced": replaced,
        "chunks": chunks,
        "characters": len(content),
    }
    report.update(kb.stats())
    return report


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
    #
    # The same pass reads each surviving row's fingerprint. A document whose
    # text still hashes to what the store holds keeps the vectors it has: the
    # rebuild is then proportional to what actually changed rather than to how
    # big the corpus is, which is what makes it something a run can do for
    # itself before the Architect opens. Measured warm on this project, 77
    # files and 1,618 chunks: 52.0s re-embedding everything, 0.1s when nothing
    # changed -- and in that case the embedding model is never loaded at all.
    stored_sha: dict[str, str] = {}
    try:
        existing = kb.collection.get(include=["metadatas"])
        existing_ids = existing.get("ids") or []
        existing_metadatas = existing.get("metadatas") or []
        stale: list[str] = []
        stale_documents: set[str] = set()
        for i, chunk_id in enumerate(existing_ids):
            row = existing_metadatas[i] if i < len(existing_metadatas) else None
            document = _document_id_of(chunk_id, row)
            if document not in wanted:
                stale.append(chunk_id)
                stale_documents.add(document)
            elif row and row.get("sha"):
                stored_sha[document] = str(row["sha"])
        # Counted in documents, not rows: "3 chunks pruned" is a fact about
        # the store, and the caller wants to know how many *files* stopped
        # answering searches. It is also the only half of a rebuild's work that
        # `embedded` cannot see -- a pass that deleted a document and re-read
        # nothing did something, and must not report itself as a no-op.
        dropped = len(stale_documents)
        if stale:
            kb.collection.delete(ids=stale)
    except Exception as exc:  # pragma: no cover - Chroma unavailable
        errors_pre = [f"pruning stale documents: {exc}"]
        dropped = 0
    else:
        errors_pre = []
    kb.graph.clear()

    indexed = embedded = reused = skipped = 0
    errors: list[str] = list(errors_pre)

    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8")
            if len(content) > MAX_INDEXABLE_BYTES:
                skipped += 1
                continue

            doc_id = str(file_path)
            metadata = _document_metadata(file_path)
            if stored_sha.get(doc_id) == _content_sha(content):
                # Its chunks and their vectors are still correct and were not
                # pruned above. Only the graph has to come back, because this
                # function cleared it -- and that half is free.
                kb._add_to_graph(doc_id, content, metadata)
                reused += 1
            else:
                kb.add_document(doc_id, content, metadata)
                embedded += 1
            indexed += 1
        except Exception as exc:
            errors.append(f"{file_path}: {exc}")

    # Both halves of the rebuild above are otherwise persisted only as a side
    # effect of `add_document`: the `graph.clear()` reaches disk through its
    # `_save_graph()`, and the pruned rows leave the lexical index through its
    # invalidation. So a reindex that added nothing -- no matching files, or
    # every one of them oversized or unreadable -- cleared the graph in memory
    # and left the old `knowledge_graph.json` for the next process start to
    # reload, while an index built before the prune went on answering with the
    # chunks the prune had just deleted. Unconditional, the way `clear()` ends
    # with the same two lines: the work was done either way, and a reindex is
    # the one operation after which the lexical index must be rebuilt anyway.
    kb._save_graph()
    kb._lexical_index = None

    # `indexed` is how many documents the corpus now holds, which is what every
    # caller has always printed. `embedded` and `reused` split that by cost --
    # the only number that moves when a rebuild is nearly a no-op, and the one
    # thing that says whether a rebuild did any work at all.
    report: dict[str, Any] = {
        "indexed": indexed,
        "embedded": embedded,
        "reused": reused,
        "dropped": dropped,
        "skipped": skipped,
        "errors": errors,
    }
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


# Re-exported from `corpus_spectral`, which is where the four whole-graph
# diagnostics now live. Named here because `serve.py`, the tests and CLAUDE.md
# all reach for them through this module, and moving the code should not have
# moved the vocabulary.
__all__ = [
    "BOTTLENECK_CONDUCTANCE",
    "DUPLICATE_BLOCK_PREFIX",
    "DUPLICATE_CONTAINMENT",
    "DUPLICATE_NAME_SIMILARITY",
    "EIGENGAP_DECISIVENESS",
    "MAX_AUTO_CLUSTERS",
    "CorpusSpectralMixin",
    "GraphRAGKnowledgeBase",
]
