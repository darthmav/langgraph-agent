"""Tests for the MCP tool bindings used by the 4-Agent System.

Verifies that the documented tool belts are exposed and functional:
- Researcher: search_knowledge_graph, query_knowledge_graph
- Builder: filesystem_read, filesystem_write, git_status, git_diff,
  terminal_execute, run_tests
"""

import tempfile
from pathlib import Path

import pytest
import pytest_asyncio

from langgraph_agent.mcp_client import (
    TERMINAL_TIMEOUT_MAX_SECONDS,
    TERMINAL_TIMEOUT_SECONDS,
    MCPClient,
    _resolve_timeout,
    mcp_client,
)


@pytest_asyncio.fixture
async def client():
    """Yield a connected MCP client."""
    async with mcp_client() as c:
        yield c


@pytest.mark.asyncio
async def test_list_tools(client: MCPClient):
    """All documented tools are exposed."""
    tools = await client.list_tools()

    expected = {
        "search_knowledge_graph",
        "query_knowledge_graph",
        "filesystem_read",
        "filesystem_write",
        "git_status",
        "git_diff",
        "terminal_execute",
        "run_tests",
    }
    assert expected.issubset(set(tools))


@pytest.mark.asyncio
async def test_filesystem_write_and_read(client: MCPClient):
    """Builder can write and read files through MCP tools."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.txt"
        content = "Hello from MCP filesystem tool"

        write_result = await client.call_tool(
            "filesystem_write", {"path": str(path), "content": content}
        )
        assert write_result["success"]
        assert path.read_text(encoding="utf-8") == content

        read_result = await client.call_tool("filesystem_read", {"path": str(path)})
        assert read_result["success"]
        assert read_result["content"] == content


@pytest.mark.asyncio
async def test_git_tools(client: MCPClient):
    """Builder can call git status and diff."""
    status = await client.call_tool("git_status", {})
    assert status["success"]
    assert "status" in status

    diff = await client.call_tool("git_diff", {})
    assert diff["success"]
    assert "diff" in diff


@pytest.mark.asyncio
async def test_terminal_execute(client: MCPClient):
    """Builder can run safe shell commands."""
    result = await client.call_tool("terminal_execute", {"command": "echo hello"})
    assert result["success"]
    assert "hello" in result["stdout"]


@pytest.mark.asyncio
async def test_terminal_execute_rejects_unsafe_characters(client: MCPClient):
    """terminal_execute rejects commands with shell metacharacters."""
    result = await client.call_tool(
        "terminal_execute", {"command": "echo hello; rm -rf /"}
    )
    assert not result["success"]
    error = result.get("error", "")
    assert "disallowed" in error.lower() or "shell metacharacter" in error.lower()


@pytest.mark.asyncio
async def test_terminal_execute_honours_a_requested_timeout(client: MCPClient):
    """A caller-supplied timeout bounds the command and reports the kill."""
    result = await client.call_tool(
        "terminal_execute", {"command": "sleep 30", "timeout": 1}
    )
    assert not result["success"]
    assert result["timed_out"] is True


def test_terminal_timeout_clears_this_project_own_scripts():
    """The default must outlast the scripts this repo tells people to run.

    Pinned against the measurement rather than the literal number, the way
    RETRIEVAL_RELEVANCE_FLOOR is: `scripts/verify_and_test.py` completes clean
    in ~33s, and the old default of 30 reported it FAILED for the difference.
    A default at or under that is the false accusation coming back.
    """
    assert TERMINAL_TIMEOUT_SECONDS > 35
    assert TERMINAL_TIMEOUT_MAX_SECONDS >= TERMINAL_TIMEOUT_SECONDS


def test_terminal_timeout_is_clamped_not_trusted():
    """A requested timeout is bounded; a malformed one falls back."""
    # Honoured between the floor and the ceiling.
    assert _resolve_timeout(120) == 120

    # Capped: a tool call is never abandoned, so an unbounded request would
    # hang the pass past every deadline there is.
    assert _resolve_timeout(99999) == TERMINAL_TIMEOUT_MAX_SECONDS

    # Malformed, absent or non-positive falls back rather than raising.
    for bad in (None, "not-a-number", 0, -5, float("nan"), float("inf")):
        assert _resolve_timeout(bad) == TERMINAL_TIMEOUT_SECONDS


def test_builder_can_ask_for_a_longer_timeout():
    """The schema must expose `timeout`, or the default is a hard ceiling.

    This is what actually broke: `_terminal_execute` read
    `args.get("timeout")`, but BUILDER_TOOLS offered only `command`, so the
    Builder could not raise it and had no way to learn the limit existed.
    """
    from langgraph_agent.nodes import BUILDER_TOOLS

    tool = next(
        t for t in BUILDER_TOOLS if t["function"]["name"] == "terminal_execute"
    )
    params = tool["function"]["parameters"]
    assert "timeout" in params["properties"]
    assert "timeout" not in params.get("required", [])
    # The limit is stated where the seat reading the tool can see it.
    assert str(int(TERMINAL_TIMEOUT_SECONDS)) in tool["function"]["description"]


@pytest.mark.asyncio
async def test_run_tests(client: MCPClient):
    """Builder can run the pytest suite via the test tool."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal passing test so the tool has something to execute.
        test_file = Path(tmpdir) / "test_dummy.py"
        test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")

        result = await client.call_tool("run_tests", {"path": str(tmpdir)})
        assert result["success"], result.get("stderr", "")
        assert "passed" in result.get("stdout", "")


def test_sync_tool_call():
    """The sync helper in nodes.py can call MCP tools."""
    from langgraph_agent.nodes import _call_mcp_tool_sync

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sync.txt"
        result = _call_mcp_tool_sync(
            "filesystem_write", {"path": str(path), "content": "sync ok"}
        )
        assert result["success"]
        assert path.read_text(encoding="utf-8") == "sync ok"
