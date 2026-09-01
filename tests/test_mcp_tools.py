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

from langgraph_agent.mcp_client import MCPClient, mcp_client


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
