"""MCP client integration for GraphRAG and filesystem tools.

This module provides a unified interface to MCP (Model Context Protocol) servers.
For GraphRAG, can use the local knowledge base directly or connect to an MCP server.

Usage:
    from langgraph_agent.mcp_client import MCPClient

    async with MCPClient() as client:
        # List available tools
        tools = await client.list_tools()

        # Call GraphRAG search
        result = await client.call_tool("search_knowledge_base", {"query": "Planner agent"})
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any


class MCPClient:
    """Client for MCP servers (GraphRAG, Filesystem, Git).

    Uses local GraphRAG knowledge base when available, falls back to stubs.
    """

    def __init__(self, server_urls: list[str] | None = None):
        """Initialize MCP client.

        Args:
            server_urls: List of MCP server URLs (default from env vars)
        """
        self.server_urls = server_urls or self._default_servers()
        self._connected = False
        self._tools: dict[str, Any] = {}
        self._kb = None

    def _default_servers(self) -> list[str]:
        """Get default MCP server URLs from environment."""
        servers = []
        if os.getenv("MCP_GRAPHRAG_URL"):
            servers.append(os.getenv("MCP_GRAPHRAG_URL"))
        if os.getenv("MCP_FILESYSTEM_URL"):
            servers.append(os.getenv("MCP_FILESYSTEM_URL"))
        if os.getenv("MCP_GIT_URL"):
            servers.append(os.getenv("MCP_GIT_URL"))
        return servers

    async def connect(self) -> None:
        """Connect to MCP servers and initialize local GraphRAG (lazy)."""
        # Lazy init GraphRAG - only when actually needed for search
        self._kb = None
        self._connected = True
        self._tools = self._discover_tools()

    async def disconnect(self) -> None:
        """Disconnect from MCP servers."""
        self._connected = False
        self._tools = {}
        self._kb = None

    def _discover_tools(self) -> dict[str, Any]:
        """Discover available tools from connected servers.

        Returns:
            Dict mapping tool names to tool callables
        """
        tools = {}

        # Always provide GraphRAG tools (local or stub)
        tools["search_knowledge_base"] = self._graphrag_search
        tools["query_knowledge_graph"] = self._graphrag_query_graph
        tools["add_to_knowledge_base"] = self._graphrag_add

        # Real filesystem and git tools
        tools["filesystem_read"] = self._filesystem_read
        tools["filesystem_write"] = self._filesystem_write
        tools["git_status"] = self._git_status
        tools["git_diff"] = self._git_diff

        return tools

    async def list_tools(self) -> list[str]:
        """List all available tools."""
        if not self._connected:
            await self.connect()
        return list(self._tools.keys())

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool by name.

        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments

        Returns:
            Tool result
        """
        if not self._connected:
            await self.connect()

        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool_fn = self._tools[tool_name]
        return await tool_fn(arguments)

    # GraphRAG implementations (use local knowledge base when available)

    async def _graphrag_search(self, args: dict) -> dict:
        """Search the knowledge base (lazy init)."""
        query = args.get("query", "")
        top_k = args.get("top_k", 5)

        # Lazy init GraphRAG
        if self._kb is None:
            try:
                from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase
                self._kb = GraphRAGKnowledgeBase()
            except Exception:
                self._kb = None

        if self._kb:
            results = self._kb.search(query, top_k)
            return {"results": results, "source": "local_graphrag"}
        else:
            return {
                "results": [{"content": "[GraphRAG not indexed]", "score": 0.0, "id": "no_kb"}],
                "source": "stub",
            }

    async def _graphrag_query_graph(self, args: dict) -> dict:
        """Query the knowledge graph."""
        entity = args.get("entity", "")
        hops = args.get("hops", 2)

        if self._kb:
            result = self._kb.query_graph(entity, hops)
            result["source"] = "local_graphrag"
            return result
        else:
            return {
                "entity": entity,
                "neighbors": [],
                "subgraph_nodes": 0,
                "subgraph_edges": 0,
                "source": "stub",
                "note": "Run 'python scripts/index_knowledge.py' to build the knowledge base",
            }

    async def _graphrag_add(self, args: dict) -> dict:
        """Add a document to the knowledge base."""
        doc_id = args.get("doc_id", "")
        content = args.get("content", "")
        metadata = args.get("metadata", {})

        if self._kb:
            self._kb.add_document(doc_id, content, metadata)
            return {"success": True, "doc_id": doc_id, "source": "local_graphrag"}
        else:
            return {
                "success": False,
                "error": "Knowledge base not initialized",
                "note": "Run 'python scripts/index_knowledge.py' first",
            }

    # Real filesystem implementations

    async def _filesystem_read(self, args: dict) -> dict:
        """Read file contents."""
        from pathlib import Path

        path = args.get("path", "")
        try:
            content = Path(path).read_text(encoding="utf-8")
            return {"success": True, "content": content, "path": path}
        except Exception as e:
            return {"success": False, "error": str(e), "path": path}

    async def _filesystem_write(self, args: dict) -> dict:
        """Write file contents."""
        from pathlib import Path

        path = args.get("path", "")
        content = args.get("content", "")
        try:
            Path(path).write_text(content, encoding="utf-8")
            return {"success": True, "path": path, "bytes_written": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e), "path": path}

    # Real git implementations

    async def _git_status(self, args: dict) -> dict:
        """Git status."""
        import subprocess

        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return {"success": True, "status": result.stdout or "Working tree clean"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _git_diff(self, args: dict) -> dict:
        """Git diff."""
        import subprocess

        try:
            result = subprocess.run(
                ["git", "diff", args.get("path", "")],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return {"success": True, "diff": result.stdout or "No changes"}
        except Exception as e:
            return {"success": False, "error": str(e)}


@asynccontextmanager
async def mcp_client(server_urls: list[str] | None = None):
    """Async context manager for MCP client.

    Usage:
        async with mcp_client() as client:
            tools = await client.list_tools()
            result = await client.call_tool("search_knowledge_base", {"query": "Planner"})
    """
    client = MCPClient(server_urls)
    try:
        await client.connect()
        yield client
    finally:
        await client.disconnect()
