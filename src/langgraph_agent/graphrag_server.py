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
from pathlib import Path
from typing import Any

import chromadb
import networkx as nx
from mcp.server import MCPServer

# Force CPU for sentence-transformers (GPU 1060 3GB not compatible)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from sentence_transformers import SentenceTransformer  # noqa: E402


class GraphRAGKnowledgeBase:
    """Simple GraphRAG: NetworkX graph + Chroma vector store."""

    def __init__(self, persist_dir: str = "./knowledge"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model (local, no API key needed)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        # Initialize Chroma vector store
        self.chroma_client = chromadb.PersistentClient(str(self.persist_dir / "chroma"))
        self.collection = self.chroma_client.get_or_create_collection(
            name="knowledge",
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize knowledge graph
        self.graph = nx.DiGraph()
        self._load_graph()

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

    def neighborhood(
        self, node_id: str, max_depth: int = 2, min_degree: int = 1
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

        Returns:
            center_node, related_nodes (excluding the centre), edges, and totals.
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

        return {
            "center_node": centre,
            "related_nodes": related,
            "edges": edges,
            "total_nodes": len(related),
            "total_edges": len(edges),
        }

    def stats(self) -> dict[str, Any]:
        """Counters for the console header."""
        documents = sum(
            1 for _, attrs in self.graph.nodes(data=True) if attrs.get("type") == "document"
        )
        try:
            chunks = self.collection.count()
        except Exception:
            # Chroma is a separate store from the graph; a failure here should
            # not take out the node/edge counts that come from memory.
            chunks = 0

        return {
            "total_documents": documents,
            "total_chunks": chunks,
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
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


def get_knowledge_base() -> GraphRAGKnowledgeBase:
    """Return the singleton GraphRAG knowledge base instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = GraphRAGKnowledgeBase()
    return _kb_instance


def is_knowledge_base_indexed(persist_dir: str = "./knowledge") -> tuple[bool, str]:
    """Check whether a knowledge base exists and has documents.

    This is intentionally lightweight: it opens the Chroma collection directly
    without loading the sentence-transformers model, so it can be used in
    health/status endpoints without blocking startup.

    Returns:
        (indexed, embedding_model_name)
    """
    persist_path = Path(persist_dir)
    chroma_dir = persist_path / "chroma"
    if not chroma_dir.exists():
        return False, "all-MiniLM-L6-v2"

    try:
        client = chromadb.PersistentClient(str(chroma_dir))
        collection = client.get_or_create_collection(
            name="knowledge",
            metadata={"hnsw:space": "cosine"},
        )
        return collection.count() > 0, "all-MiniLM-L6-v2"
    except Exception:
        return False, "all-MiniLM-L6-v2"


# Create MCP server
server = MCPServer("graphrag")


# Register tools with the server
@server.tool(name="search_knowledge_graph")
def search_tool(query: str, top_k: int = 5) -> str:
    """Search the knowledge base for relevant documents and passages."""
    results = get_knowledge_base().search(query, top_k)
    return json.dumps(results, indent=2)


@server.tool(name="query_knowledge_graph")
def query_tool(entity: str, hops: int = 2) -> str:
    """Query the knowledge graph for entity relationships."""
    result = get_knowledge_base().query_graph(entity, hops)
    return json.dumps(result, indent=2)


async def main() -> None:
    """Run the GraphRAG MCP server."""
    server.run(transport="stdio")


if __name__ == "__main__":
    asyncio.run(main())
