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

import math
import os
import shlex
import subprocess
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase



# How long one `terminal_execute` command may run before it is killed. It was
# 30, which is under what this project's own scripts take: `verify_and_test.py`
# finishes clean in ~33s and was reported FAILED for the 3s difference. That is
# the failure `MIN_VERIFY_SLICE_SECONDS` describes, one level up -- a working
# command comes back labelled broken, and the Builder spends its next turns
# repairing code that was never wrong. 60 matches `VERIFY_TIMEOUT_SECONDS`, the
# closest sibling: both bound a single command the Builder is waiting on, and
# the Builder's whole node budget (`BUILDER_DEADLINE_SECONDS`, 240) has to
# cover several of them plus the verification reserve.
TERMINAL_TIMEOUT_SECONDS = float(os.getenv("TERMINAL_TIMEOUT_SECONDS", "60"))

# The ceiling on a *requested* timeout. A tool call is never abandoned -- the
# Builder's deadline may interrupt the model's turn but never a running tool --
# so a seat that asks for 99999 seconds does not overrun a deadline, it hangs
# the pass past every deadline there is. The ceiling is what stops an exposed
# knob from becoming that. 600 matches `_run_tests`, the longest thing the
# tool belt legitimately does.
TERMINAL_TIMEOUT_MAX_SECONDS = float(os.getenv("TERMINAL_TIMEOUT_MAX_SECONDS", "600"))


def _resolve_timeout(requested: Any) -> float:
    """Clamp a requested command timeout to [1, TERMINAL_TIMEOUT_MAX_SECONDS].

    A missing or malformed value falls back to the default instead of raising.
    The number arrives as JSON from a model, and refusing the call to complain
    about it costs a whole tool turn to say what a clamp says for nothing.
    Non-finite is rejected here rather than left to `min`: `min(nan, 600)` is
    `nan`, which reaches `subprocess.run` as no timeout at all.
    """
    if requested is None:
        return TERMINAL_TIMEOUT_SECONDS
    try:
        seconds = float(requested)
    except (TypeError, ValueError):
        return TERMINAL_TIMEOUT_SECONDS
    if not math.isfinite(seconds) or seconds <= 0:
        return TERMINAL_TIMEOUT_SECONDS
    return min(seconds, TERMINAL_TIMEOUT_MAX_SECONDS)


def _resolve_cwd(requested: Any) -> tuple[str | None, str | None]:
    """Resolve a requested working directory to `(cwd, error)`.

    An error is returned *instead of* a directory, never alongside one; both
    are None when nothing was requested and the command inherits ours.

    The check is here rather than left to `subprocess.run` because of what
    that raises: a missing `cwd` comes back as `FileNotFoundError`, which is
    the same exception a missing *program* raises and lands in the handler
    that answers `Command not found: 'python'` -- naming the one thing that
    was fine. A `cwd` that exists but is a file raises `NotADirectoryError`,
    which is not a `FileNotFoundError` at all and falls through to a bare
    errno string. Both are the false accusation this module keeps having to
    design against, so the directory is checked while we still know it is the
    directory being complained about.

    A relative path resolves against the server's working directory, which is
    the project root -- the same base `filesystem_read` and `filesystem_write`
    use, so one relative path means one place across the whole tool belt.
    `~` is not expanded, for the reason no other shell syntax is: there is no
    shell here, and a `cwd` that quietly expanded what an argument on the same
    line would not is a worse surprise than a refusal naming the path.
    """
    if requested is None:
        return None, None
    if not isinstance(requested, str) or not requested.strip():
        return None, f"Invalid cwd: {requested!r}. Pass a directory path."
    path = Path(requested)
    if not path.is_dir():
        detail = "exists but is not a directory" if path.exists() else "does not exist"
        return None, f"Cannot run in {requested!r}: it {detail}."
    return str(path), None


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

    def _open_kb(self) -> GraphRAGKnowledgeBase | None:
        """The corpus if one has been built, `None` otherwise.

        `open_knowledge_base`, never `get_knowledge_base`: the second one
        creates the store, and a Researcher's query is not a request for a
        knowledge base. Cached on the client so repeated calls in one run do
        not reopen Chroma. The cache is only ever filled with a real corpus,
        so a search that happens before the operator indexes does not pin
        `None` for the rest of the process.
        """
        if self._kb is None:
            try:
                from langgraph_agent.graphrag_server import open_knowledge_base

                self._kb = open_knowledge_base()
            except Exception:
                self._kb = None
        return self._kb

    async def _graphrag_search(self, args: dict[str, Any]) -> dict[str, Any]:
        """Search the knowledge base (cached singleton).

        With no corpus this returns *no* results. It used to return one
        fabricated row -- `[GraphRAG not indexed]`, score 0.0 -- which is a
        made-up retrieval hit sitting in the same field real ones arrive in,
        and the Builder reads that field.
        """
        query = args.get("query", "")
        top_k = args.get("top_k", 5)

        kb = self._open_kb()
        if kb is None:
            from langgraph_agent.graphrag_server import NO_CORPUS_NOTE

            return {"results": [], "source": "no_corpus", "note": NO_CORPUS_NOTE}

        return {"results": kb.search(query, top_k), "source": "local_graphrag"}

    async def _graphrag_query_graph(self, args: dict[str, Any]) -> dict[str, Any]:
        """Query the knowledge graph."""
        entity = args.get("entity", "")
        hops = args.get("hops", 2)

        kb = self._open_kb()
        if kb is None:
            from langgraph_agent.graphrag_server import NO_CORPUS_NOTE

            return {
                "entity": entity,
                "neighbors": [],
                "subgraph_nodes": 0,
                "subgraph_edges": 0,
                "source": "no_corpus",
                "note": NO_CORPUS_NOTE,
            }

        result = kb.query_graph(entity, hops)
        result["source"] = "local_graphrag"
        return result

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
        """Run a command in the project workspace.

        There is no shell. The command is split with `shlex` and handed to
        `subprocess` as an argv list, so `;`, `|`, `>`, `&&`, `$(...)` and
        globs are inert *data* rather than syntax -- `echo hi; rm -rf /` runs
        `echo` with four literal arguments and deletes nothing.

        This replaced a whitelist of permitted characters guarding
        `shell=True`. That filter refused the ordinary way to write a
        one-liner (`python -c "import x; print(y)"` trips on `;` `(` `)`) and
        any path containing parentheses, while still admitting a bare
        `rm -rf /` -- it never guarded against a destructive command, only
        against chaining one onto another. Removing the shell removes the
        thing the chaining needed, so the characters no longer have to be
        refused to be harmless.

        The trade is that shell *features* are gone rather than rejected: a
        pipe is now accepted and passed to the program as the literal argument
        `|`. That is stated in the tool description, because a silently
        meaningless pipe is worse than a refused one.

        `cwd` runs the command somewhere other than the project root, and is
        offered to the Builder because without it there is no way to express
        it at all: `cd` is a shell builtin, so `cd somewhere && python x.py`
        does not run in the wrong directory, it fails with
        `Command not found: 'cd'` -- and the Builder, having no other spelling
        to try, spends turns rediscovering absolute paths. It is the same gap
        `timeout` was: a thing the tool can do that the schema never offered.

        `env` overlays the current environment for this one command; a key
        mapped to None is removed rather than set. It is not offered to the
        Builder in BUILDER_TOOLS -- only callers inside the process set it,
        which today means the verification pass asking for a headless run.

        `timeout` is offered to the Builder, and is clamped rather than
        trusted (`_resolve_timeout`). The verification pass always passes one
        computed from its remaining deadline, so it never sees the default.
        """
        command = args.get("command", "")
        try:
            argv = shlex.split(command)
        except ValueError as exc:
            # Unbalanced quotes. Say so plainly: the caller cannot see the
            # parse, and "No closing quotation" alone reads like the program
            # failed rather than like the command was never built.
            return {
                "success": False,
                "error": f"Could not parse command ({exc}). Check the quoting.",
                "command": command,
            }
        if not argv:
            return {
                "success": False,
                "error": "Empty command.",
                "command": command,
            }

        cwd, cwd_error = _resolve_cwd(args.get("cwd"))
        if cwd_error is not None:
            return {"success": False, "error": cwd_error, "command": command}

        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=_resolve_timeout(args.get("timeout")),
                env=_child_env(args.get("env")),
                cwd=cwd,
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
        except FileNotFoundError:
            # With a shell this came back as rc=127 and a message on stderr.
            # Without one it raises, and a bare OSError repr does not say which
            # of the words was the program -- so name it, or the caller reads
            # "not found" as its file argument being missing.
            return {
                "success": False,
                "error": f"Command not found: {argv[0]!r}",
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
