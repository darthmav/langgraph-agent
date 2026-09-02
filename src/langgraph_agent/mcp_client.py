"""MCP client integration for GraphRAG and filesystem tools.

This module provides a unified interface to MCP (Model Context Protocol) servers.
For GraphRAG, can use the local knowledge base directly or connect to an MCP server.

Usage:
    from langgraph_agent.mcp_client import MCPClient

    async with MCPClient() as client:
        # List available tools
        tools = await client.list_tools()

        # Call GraphRAG search
        result = await client.call_tool("search_knowledge_graph", {"query": "Planner agent"})
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase


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
        self._kb: GraphRAGKnowledgeBase | None = None

    def _default_servers(self) -> list[str]:
        """Get default MCP server URLs from environment."""
        servers: list[str] = []
        if url := os.getenv("MCP_GRAPHRAG_URL"):
            servers.append(url)
        if url := os.getenv("MCP_FILESYSTEM_URL"):
            servers.append(url)
        if url := os.getenv("MCP_GIT_URL"):
            servers.append(url)
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

        # Always provide GraphRAG read-only tools (local or stub).
        # GraphRAG is read-only per the 4-Agent System specification; adding
        # documents is done through indexing scripts, not the Researcher tool belt.
        tools["search_knowledge_graph"] = self._graphrag_search
        tools["query_knowledge_graph"] = self._graphrag_query_graph

        # Real filesystem, git, terminal, and test tools
        tools["filesystem_read"] = self._filesystem_read
        tools["filesystem_write"] = self._filesystem_write
        tools["git_status"] = self._git_status
        tools["git_diff"] = self._git_diff
        tools["terminal_execute"] = self._terminal_execute
        tools["run_tests"] = self._run_tests

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

    async def _graphrag_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Search the knowledge base (cached singleton)."""
        query = args.get("query", "")
        top_k = args.get("top_k", 5)

        # Use the cached knowledge base singleton to avoid reloading models.
        if self._kb is None:
            try:
                from langgraph_agent.graphrag_server import get_knowledge_base

                self._kb = get_knowledge_base()
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

    async def _graphrag_query_graph(self, args: dict[str, Any]) -> dict[str, Any]:
        """Query the knowledge graph."""
        entity = args.get("entity", "")
        hops = args.get("hops", 2)

        if self._kb is None:
            try:
                from langgraph_agent.graphrag_server import get_knowledge_base

                self._kb = get_knowledge_base()
            except Exception:
                self._kb = None

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

    # Real filesystem implementations

    async def _filesystem_read(self, args: dict[str, Any]) -> dict[str, Any]:
        """Read file contents."""
        path = args.get("path", "")
        try:
            content = Path(path).read_text(encoding="utf-8")
            return {"success": True, "content": content, "path": path}
        except Exception as e:
            return {"success": False, "error": str(e), "path": path}

    async def _filesystem_write(self, args: dict[str, Any]) -> dict[str, Any]:
        """Write file contents, creating parent directories if needed."""
        path = args.get("path", "")
        content = args.get("content", "")
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return {"success": True, "path": path, "bytes_written": len(content)}
        except Exception as e:
            return {"success": False, "error": str(e), "path": path}

    # Real git implementations

    async def _git_status(self, args: dict[str, Any]) -> dict[str, Any]:
        """Git status."""
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

    async def _git_diff(self, args: dict[str, Any]) -> dict[str, Any]:
        """Git diff."""
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

    # Terminal / test tools

    async def _terminal_execute(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run a shell command in the project workspace.

        Safety: rejects shell metacharacters and only allows simple commands.

        `env` overlays the current environment for this one command; a key
        mapped to None is removed rather than set. It is not offered to the
        Builder in BUILDER_TOOLS -- only callers inside the process set it,
        which today means the verification pass asking for a headless run.
        """
        command = args.get("command", "")
        # Allow only simple commands: alphanumerics, dashes, underscores, dots,
        # slashes, spaces, and a few safe flags/punctuation.
        if not re.fullmatch(r"[A-Za-z0-9_./\s\-:'\"=,]+", command):
            return {
                "success": False,
                "error": "Command contains disallowed shell metacharacters",
                "command": command,
            }

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=args.get("timeout", 30),
                env=_child_env(args.get("env")),
                # No human is at the keyboard behind a Builder tool call, so a
                # command that reads stdin must get EOF and fail, never block
                # until its timeout and report as a hang.
                stdin=subprocess.DEVNULL,
            )
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": command,
            }
        except subprocess.TimeoutExpired as e:
            # Keep what the command managed to print. `str(e)` alone says only
            # that it timed out, and a caller with no output to look at cannot
            # tell a command that hung immediately from one that did all its
            # work and then blocked at the end -- so it guesses, and pays the
            # full timeout again on a retry that was never going to differ.
            return {
                "success": False,
                "error": str(e),
                "timed_out": True,
                "stdout": _as_captured_text(e.stdout),
                "stderr": _as_captured_text(e.stderr),
                "command": command,
            }
        except Exception as e:
            return {"success": False, "error": str(e), "command": command}

    async def _run_tests(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run the pytest test suite."""
        target = args.get("path", "tests/")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", target, "-q"],
                capture_output=True,
                text=True,
                timeout=args.get("timeout", 600),
                # Same reason as _terminal_execute: a suite that stops to ask
                # something would otherwise hang until its timeout.
                stdin=subprocess.DEVNULL,
            )
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except subprocess.TimeoutExpired as e:
            return {
                "success": False,
                "error": str(e),
                "timed_out": True,
                "stdout": _as_captured_text(e.stdout),
                "stderr": _as_captured_text(e.stderr),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


def _as_captured_text(captured: str | bytes | None) -> str:
    """Normalise output hung off a TimeoutExpired to text.

    `capture_output=True` with `text=True` gives str, but the attribute is
    typed to allow bytes and is None when nothing was read before the kill.
    """
    if captured is None:
        return ""
    if isinstance(captured, bytes):
        return captured.decode("utf-8", "replace")
    return captured


def _child_env(overrides: dict[str, str | None] | None) -> dict[str, str] | None:
    """Build a child environment from os.environ plus `overrides`.

    A key mapped to None is removed. Returns None when there is nothing to
    override, so the child simply inherits ours.
    """
    if not overrides:
        return None
    env = dict(os.environ)
    for key, value in overrides.items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


@asynccontextmanager
async def mcp_client(server_urls: list[str] | None = None) -> AsyncGenerator[MCPClient, None]:
    """Async context manager for MCP client.

    Usage:
        async with mcp_client() as client:
            tools = await client.list_tools()
            result = await client.call_tool("search_knowledge_graph", {"query": "Planner"})
    """
    client = MCPClient(server_urls)
    try:
        await client.connect()
        yield client
    finally:
        await client.disconnect()
