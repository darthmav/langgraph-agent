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

        # Add to graph as a node
        self.graph.add_node(
            doc_id,
            type="document",
            content=content[:200],  # Store snippet
            **(metadata or {})
        )

        # Extract simple entities (words that look like important terms)
        words = content.split()
        entities = [
            w.strip(".,!?;:") for w in words
            if len(w) > 4 and w[0].isupper()
        ]

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


@server.tool(name="add_to_knowledge_base")
def add_tool(doc_id: str, content: str, metadata: dict[str, Any] | None = None) -> str:
    """Add a document to the knowledge base."""
    get_knowledge_base().add_document(doc_id, content, metadata)
    return f"Added document '{doc_id}' to knowledge base"


async def main() -> None:
    """Run the GraphRAG MCP server."""
    server.run(transport="stdio")


if __name__ == "__main__":
    asyncio.run(main())
