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
    _resolve_cwd,
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
async def test_terminal_execute_neutralises_injection(client: MCPClient):
    """A chained command is inert data, never a second command.

    There is no shell, so `;` separates nothing. This replaced a character
    filter that refused the input outright -- the canary surviving is a
    stronger guarantee than that refusal was, and unlike the refusal it does
    not also block `python -c "...; ..."`. Note the old filter admitted a bare
    `rm -rf /` quite happily: it never guarded destruction, only chaining.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        canary = Path(tmpdir) / "canary.txt"
        canary.write_text("still here", encoding="utf-8")

        result = await client.call_tool(
            "terminal_execute", {"command": f"echo hello; rm -rf {tmpdir}"}
        )

        # `echo` received the rest as literal arguments and printed them.
        assert result["success"]
        assert "rm -rf" in result["stdout"]

        # The part that matters: nothing was deleted.
        assert canary.exists()
        assert canary.read_text(encoding="utf-8") == "still here"


@pytest.mark.asyncio
async def test_terminal_execute_runs_a_python_one_liner(client: MCPClient):
    """The exact shape the old filter refused: `;` and `()` in a -c argument."""
    result = await client.call_tool(
        "terminal_execute",
        {"command": 'python -c "import sys; print(sys.version_info[0])"'},
    )
    assert result["success"], result.get("error") or result.get("stderr")
    assert result["stdout"].strip() == "3"


@pytest.mark.asyncio
async def test_terminal_execute_passes_shell_syntax_through_literally(
    client: MCPClient,
):
    """A pipe is an argument, not a pipeline -- accepted, and inert.

    Worth pinning because it is the one place removing the shell is a
    downgrade in behaviour rather than an upgrade: the old filter refused a
    pipe, and this accepts one that will not do what it looks like. The tool
    description says so; this proves the description true.
    """
    result = await client.call_tool(
        "terminal_execute", {"command": "echo a | wc -l"}
    )
    assert result["success"]
    assert result["stdout"].strip() == "a | wc -l"


@pytest.mark.asyncio
async def test_terminal_execute_reports_unparseable_and_missing_commands(
    client: MCPClient,
):
    """Two failures the shell used to fold into a return code."""
    unbalanced = await client.call_tool(
        "terminal_execute", {"command": 'python -c "print(1)'}
    )
    assert not unbalanced["success"]
    assert "quoting" in unbalanced["error"].lower()

    empty = await client.call_tool("terminal_execute", {"command": "   "})
    assert not empty["success"]

    missing = await client.call_tool(
        "terminal_execute", {"command": "no-such-program-xyzzy --help"}
    )
    assert not missing["success"]
    # Names the program, not its arguments.
    assert "no-such-program-xyzzy" in missing["error"]


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


@pytest.mark.asyncio
async def test_terminal_execute_runs_in_a_requested_cwd(client: MCPClient):
    """`cwd` is the replacement for a `cd` that cannot exist.

    Removing the shell removed the only spelling the Builder had for "run this
    somewhere else": `cd there && python x.py` reports `Command not found:
    'cd'`, which is true and useless.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "marker.txt").write_text("here", encoding="utf-8")

        result = await client.call_tool(
            "terminal_execute",
            {
                "command": 'python -c "import pathlib; print(pathlib.Path.cwd())"',
                "cwd": tmpdir,
            },
        )
        assert result["success"], result.get("error") or result.get("stderr")
        assert Path(result["stdout"].strip()).samefile(tmpdir)

        # And the command sees that directory's files by relative path.
        read = await client.call_tool(
            "terminal_execute",
            {"command": "cat marker.txt", "cwd": tmpdir},
        )
        assert read["success"], read.get("error") or read.get("stderr")
        assert read["stdout"].strip() == "here"


@pytest.mark.asyncio
async def test_terminal_execute_blames_a_bad_cwd_not_the_program(
    client: MCPClient,
):
    """The directory is named, and the program is not accused of missing.

    An unchecked `cwd` reaches `subprocess.run`, which raises
    `FileNotFoundError` for a missing directory -- indistinguishable, at the
    handler, from a missing program, and answered `Command not found:
    'python'` while python was fine.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        missing = str(Path(tmpdir) / "no-such-dir")
        result = await client.call_tool(
            "terminal_execute", {"command": "python --version", "cwd": missing}
        )
        assert not result["success"]
        assert missing in result["error"]
        assert "python" not in result["error"].lower()

        # A file is not a directory, and raises something else again.
        a_file = Path(tmpdir) / "file.txt"
        a_file.write_text("x", encoding="utf-8")
        on_file = await client.call_tool(
            "terminal_execute", {"command": "python --version", "cwd": str(a_file)}
        )
        assert not on_file["success"]
        assert str(a_file) in on_file["error"]
        assert "not a directory" in on_file["error"].lower()


def test_resolve_cwd_returns_a_directory_or_an_error_never_both():
    """Absent means inherit; anything malformed is refused rather than raised."""
    assert _resolve_cwd(None) == (None, None)

    with tempfile.TemporaryDirectory() as tmpdir:
        resolved, error = _resolve_cwd(tmpdir)
        assert error is None
        assert resolved is not None and Path(resolved).samefile(tmpdir)

    # The value arrives as JSON from a model, so a wrong type is a refusal
    # with a message, not a TypeError out of the tool call.
    for bad in ("", "   ", 5, ["/tmp"], {}):
        resolved, error = _resolve_cwd(bad)
        assert resolved is None
        assert error


def test_builder_can_ask_for_a_working_directory():
    """The schema must expose `cwd`, or the tool can do what the seat cannot ask.

    Exactly the shape the `timeout` gap had: `_terminal_execute` honours the
    argument, and a BUILDER_TOOLS entry omitting it leaves the Builder with no
    way to know it exists -- reaching for `cd` instead, which cannot work.
    """
    from langgraph_agent.nodes import BUILDER_TOOLS

    tool = next(
        t for t in BUILDER_TOOLS if t["function"]["name"] == "terminal_execute"
    )
    params = tool["function"]["parameters"]
    assert "cwd" in params["properties"]
    assert "cwd" not in params.get("required", [])
    # Says why, where the seat reading the tool can see it.
    assert "cd" in tool["function"]["description"]


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
