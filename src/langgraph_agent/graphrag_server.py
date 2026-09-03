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
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any

import chromadb
import networkx as nx
from mcp.server import MCPServer

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

# How close two entities' spectral embedding rows must be to be worth offering
# as a merge candidate. Measured on a corpus seeded with known duplicates: an
# entity mentioned by exactly the documents another is mentioned by sits at
# distance 0.0000, one differing by a single document at 0.19, and the nearest
# unrelated pair at 0.59 against a median of 1.47. 0.25 sits in that gap.
DUPLICATE_DISTANCE = 0.25

# Dimensions of the embedding the comparison runs in. Enough to separate
# structural roles, few enough that the rows stay dense and comparable.
DUPLICATE_EMBEDDING_DIM = 10


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

    def add_document(self, doc_id: str, content: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a document to the knowledge base.

        Args:
            doc_id: Unique document identifier
            content: Document text content
            metadata: Optional metadata (path, type, etc.)
        """
        # Generate embedding
        embedding = self.embedder.encode(content).tolist()

        # Add to vector store
        self.collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata or {}]
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
        entities = []
        for word in content.split():
            token = word.strip("\"'`()[]{}<>.,!?;:*=+-/\\|")
            if len(token) > 4 and token[0].isupper() and token.replace("_", "").isalnum():
                entities.append(token)

        # Add entities and relationships
        for entity in set(entities):
            self.graph.add_node(entity, type="entity")
            self.graph.add_edge(doc_id, entity, relation="mentions")

        self._save_graph()

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search the knowledge base.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of results with content, metadata, and graph context
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()

        # Search vector store
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        ids = results.get("ids") or [[]]
        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        # Guard against empty collections / no hits
        if not ids or not ids[0]:
            return []

        # Enrich with graph context
        enriched_results = []
        for i, doc_id in enumerate(ids[0]):
            result = {
                "id": doc_id,
                "content": documents[0][i] if documents[0] else "",
                "metadata": metadatas[0][i] if metadatas[0] else {},
                "score": 1 - distances[0][i] if distances[0] else 0.0,
            }

            # Add graph neighbors
            if doc_id in self.graph:
                neighbors = list(self.graph.neighbors(doc_id))[:5]
                result["related_entities"] = neighbors

            enriched_results.append(result)

        return enriched_results

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
        self, limit: int = 20, distance: float | None = None
    ) -> dict[str, Any]:
        """Entities that play the same structural role, as merge candidates.

        The A5 application from `reports/spectral_applicability.md`.
        `add_document` mints an entity for every capitalised token, so
        "Builder" and "Builders" become two nodes with the same meaning and
        nearly the same neighbours. Two entities close together in the spectral
        embedding are mentioned by nearly the same documents, which is what
        makes them candidates to merge.

        **The report's caveat about cospectral twins points the wrong way, and
        the measurement says so.** It notes that two nodes with identical
        neighbourhoods are indistinguishable to any spectral method -- the
        `C^2` / Weisfeiler-Lehman limit -- and concludes this therefore finds
        near-duplicates but not exact ones. That conflates two different
        questions. You cannot tell an exact twin *apart from* its twin, which
        is true and irrelevant here; what you can do is *find the pair*, and an
        exact twin is the easiest possible case, sitting at distance exactly
        0.0000 and ranking first. Measured: an exact structural twin 0.0000, a
        twin differing by one document 0.19, the nearest unrelated pair 0.59,
        median 1.47. Nothing about the limit obstructs this use.

        **Structure beats names at this, and names actively mislead.** The
        obvious alternative is string similarity on the entity names. On a
        corpus seeded with three known cases it ranked its own false positive
        first -- two similarly-named entities with different neighbourhoods --
        put the same-name duplicate at rank 33, and the differently-named one
        ("LanguageModel" for an entity already called something else) at rank
        503. The spectral ranking put the two real duplicates at ranks 0 and 1.
        `name_similarity` is still reported per pair, because a pair that is
        close *both* ways is nearly certain, but it is reported and never
        filtered on.

        The evidence a human merges on is `shared_documents`, not the distance:
        two entities mentioned by exactly the same six documents is a fact
        anyone can check, while an embedding distance has to be trusted. The
        distance finds the pair; the shared-document count justifies it.

        Pairs come from a KD-tree rather than an all-pairs scan -- 17x faster
        on 1036 entities and, more to the point, allocating nothing quadratic,
        so a corpus that grows does not start building a distance matrix in
        memory.
        """
        undirected = self.graph.to_undirected(as_view=True)
        if undirected.number_of_nodes() == 0:
            return {"verdict": "no_graph", "note": "The graph is empty.", "pairs": []}

        largest = max(nx.connected_components(undirected), key=len)
        component = undirected.subgraph(largest)
        n = component.number_of_nodes()
        radius = DUPLICATE_DISTANCE if distance is None else float(distance)

        entities = [
            node for node in component
            if self.graph.nodes[node].get("type") == "entity"
        ]
        if n < DUPLICATE_EMBEDDING_DIM + 2 or len(entities) < 2:
            return {
                "verdict": "no_graph",
                "note": "Too few connected entities to compare.",
                "pairs": [],
            }

        try:
            import numpy as np
            from scipy.spatial import cKDTree

            from spectral_graph import spectral_embedding

            embedding = spectral_embedding(
                component, dim=DUPLICATE_EMBEDDING_DIM, normalized=True, use_fiedler=True
            )
            # Row-normalized, as in Ng-Jordan-Weiss: what matters is the
            # direction of a node's embedding row, not how far out it sits.
            # Without this, two entities with the same role but different
            # degrees are pushed apart by magnitude alone.
            rows = embedding / np.maximum(
                np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12
            )
        except ImportError:
            return {"verdict": "unavailable",
                    "note": "spectral_graph is not on sys.path.", "pairs": []}
        except Exception as exc:  # pragma: no cover - solver-dependent
            return {"verdict": "unavailable",
                    "note": f"{type(exc).__name__}: {exc}", "pairs": []}

        order = {node: i for i, node in enumerate(component.nodes())}
        points = rows[[order[entity] for entity in entities]]
        close = cKDTree(points).query_pairs(r=radius)

        pairs = []
        for left, right in close:
            a, b = entities[left], entities[right]
            neighbours_a = set(component.neighbors(a))
            neighbours_b = set(component.neighbors(b))
            shared = neighbours_a & neighbours_b
            pairs.append(
                {
                    "entities": sorted([str(a), str(b)]),
                    "distance": float(np.linalg.norm(points[left] - points[right])),
                    "shared_documents": len(shared),
                    "degrees": [component.degree(a), component.degree(b)],
                    # Jaccard on the neighbourhoods: 1.0 means the two are
                    # mentioned by exactly the same documents, which is the
                    # strongest merge evidence there is and is checkable
                    # without trusting the embedding at all.
                    "neighbourhood_overlap": (
                        len(shared) / len(neighbours_a | neighbours_b)
                        if (neighbours_a | neighbours_b) else 0.0
                    ),
                    "name_similarity": SequenceMatcher(None, str(a), str(b)).ratio(),
                }
            )

        pairs.sort(key=lambda pair: (pair["distance"], pair["entities"]))
        return {
            "verdict": "scanned",
            "pairs": pairs[:limit],
            "total_pairs": len(pairs),
            "entities_compared": len(entities),
            "distance": radius,
            "embedding_dim": DUPLICATE_EMBEDDING_DIM,
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
                    "id": doc_id,
                    "content": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
                for i, doc_id in enumerate(ids)
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
    try:
        existing = kb.collection.get(include=[]).get("ids", [])
        stale = [doc_id for doc_id in existing if doc_id not in wanted]
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
