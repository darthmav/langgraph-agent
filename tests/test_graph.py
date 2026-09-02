"""Tests for the 4-Agent System.

Verifies:
- State schema matches documentation
- Architect opens the run and holds the approval gate
- Planner routes correctly based on task type
- Researcher sets proper status
- Builder reports changes and blockers
- Graph loops on the Architect's verdict and stops at the step limit
"""

from types import SimpleNamespace

import pytest

from langgraph_agent import AgentState, ResearchStatus, Verdict, create_agent_graph
from langgraph_agent.config import StubLLM


@pytest.fixture
def agent_graph(monkeypatch):
    """Create a fresh agent graph for testing using the deterministic StubLLM."""
    # Force every agent node to use the canned StubLLM so tests do not depend
    # on a live cloud LLM endpoint or API keys.
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: StubLLM(),
    )
    return create_agent_graph()


def initial_state(goal: str) -> AgentState:
    """Create initial state per the 4-Agent System specification."""
    return {
        "goal": goal,
        "messages": [],
        "architecture": "",
        "verdict": "",
        "plan": "",
        "research": "",
        "builder_report": "",
        "next_agent": "Researcher",
        "research_status": "",
        "blockers": "",
        "files_changed": [],
        "failed_verification": [],
        "expect_failures": False,
        "step_count": 0,
    }


def test_state_schema_initialization():
    """Test that state schema matches documentation."""
    state = initial_state("Test goal")

    # Required fields per documentation
    assert state["goal"] == "Test goal"
    assert state["messages"] == []
    assert state["architecture"] == ""
    assert state["verdict"] == ""
    assert state["plan"] == ""
    assert state["research"] == ""
    assert state["builder_report"] == ""
    assert state["next_agent"] == "Researcher"
    assert state["research_status"] == ""
    assert state["blockers"] == ""
    assert state["files_changed"] == []
    assert state["step_count"] == 0


def test_planner_routes_to_researcher(agent_graph):
    """Test that Planner routes to Researcher for knowledge-heavy tasks."""
    state = initial_state("Research Python async best practices")

    result = agent_graph.invoke(state)

    # Planner should create a plan
    assert result["plan"] != ""
    # Should route to Researcher for research tasks
    # Note: LLM may choose Builder if task seems straightforward
    assert result["next_agent"] in ["Researcher", "Builder"]
    assert len(result["messages"]) > 0


def test_planner_routes_to_builder(agent_graph):
    """Test that Planner routes to Builder for clear tasks."""
    state = initial_state("Create a file named hello.txt with 'Hello World'")

    result = agent_graph.invoke(state)

    # Planner should create a plan
    assert result["plan"] != ""
    # Should route to Builder (task is clear, no research needed)
    # Note: LLM may still choose Researcher, so we just verify a plan exists
    assert result["plan"] != ""


def test_researcher_sets_status(agent_graph):
    """Test that Researcher sets research_status field."""
    state = initial_state("Research async error handling patterns")

    result = agent_graph.invoke(state)

    # Researcher should set research field
    assert result["research"] != ""
    # Status should be set (even if LLM simulation)
    assert "research_status" in result


def test_builder_reports_changes(agent_graph):
    """Test that Builder reports changes and files."""
    state = initial_state("Create hello.txt with 'Hello World'")

    result = agent_graph.invoke(state)

    # Builder should report something
    assert result["builder_report"] != "" or result["messages"] != []
    # Step count should increment
    assert result["step_count"] > 0


def test_full_graph_execution(agent_graph):
    """Test that the full 4-agent flow executes."""
    state = initial_state("Create a simple Python module with a greet function")

    result = agent_graph.invoke(state)

    # Verify all fields are populated
    assert result["goal"] == "Create a simple Python module with a greet function"
    assert result["plan"] != ""
    assert result["messages"] != []
    # Graph should complete (step_count incremented)
    assert result["step_count"] > 0


def test_max_steps_limit():
    """Test that graph stops at MAX_STEPS (8)."""
    # This test would require mocking to force loops
    # For now, verify the constant exists
    from langgraph_agent.graph import MAX_STEPS

    assert MAX_STEPS == 8


def test_research_status_enum():
    """Test that ResearchStatus enum has correct values."""
    assert ResearchStatus.READY_FOR_BUILDER.value == "ready_for_builder"
    assert ResearchStatus.NEED_REPLAN.value == "need_replan"
    assert ResearchStatus.NO_RELEVANT_KNOWLEDGE.value == "no_relevant_knowledge"


def test_simple_task_skips_research(agent_graph):
    """Test that a straightforward file-creation task skips research.

    Per the documentation happy-path diagram, the Planner should route a
    fully-specified task directly to the Builder.
    """
    state = initial_state("Create a hello.txt file containing 'Hello World'")

    result = agent_graph.invoke(state)

    assert result["plan"] != ""
    assert result["research"] == ""
    assert result["next_agent"] == "Builder"
    assert result["step_count"] > 0


def test_state_injection_shows_empty():
    """Test that empty fields in state injection render as (empty)."""
    from langgraph_agent.nodes import _get_state_injection

    state = initial_state("Test goal")
    injection = _get_state_injection(state)

    assert "Architecture: (empty)" in injection
    assert "Verdict: (empty)" in injection
    assert "Plan: (empty)" in injection
    assert "Research: (empty)" in injection
    assert "Builder report: (empty)" in injection
    assert "Blockers: (empty)" in injection
    assert "Files changed: (empty)" in injection


def test_architect_opens_the_run(agent_graph):
    """The Architect runs before anything is planned and sets direction."""
    result = agent_graph.invoke(initial_state("Create hello.txt with 'Hello World'"))

    assert result["architecture"] != ""
    # The Architect's opening message has to precede the Planner's: it is the
    # authority that sets the constraints the plan is written against.
    roles = [m.split("]")[0].lstrip("[") for m in result["messages"]]
    assert roles[0] == "Architect"
    assert "Planner" in roles


def test_architect_gate_ends_the_run(agent_graph):
    """The run terminates on the Architect's approval, not the Builder's."""
    result = agent_graph.invoke(initial_state("Create hello.txt with 'Hello World'"))

    assert result["verdict"] == Verdict.APPROVED.value
    # The Architect both opens and closes, so it appears twice and last.
    roles = [m.split("]")[0].lstrip("[") for m in result["messages"]]
    assert roles[-1] == "Architect"
    assert roles.count("Architect") == 2


def test_step_count_incremented_by_the_gate(agent_graph):
    """Every completed cycle passes the gate, so every cycle counts a step."""
    result = agent_graph.invoke(initial_state("Create hello.txt with 'Hello World'"))

    assert result["step_count"] == 1


@pytest.mark.parametrize(
    "verdict,expected",
    [
        (Verdict.APPROVED.value, "__end__"),
        (Verdict.REVISE.value, "planner"),
        (Verdict.NEED_RESEARCH.value, "researcher"),
        (Verdict.PLAN.value, "planner"),
    ],
)
def test_architect_routing(verdict, expected):
    """Each verdict sends the loop where the Architect said to send it."""
    from langgraph_agent.graph import MAX_STEPS, _route_from_architect

    state = initial_state("anything")
    state["verdict"] = verdict
    assert _route_from_architect(state) == expected

    # The step ceiling outranks every verdict, including one asking to loop.
    state["step_count"] = MAX_STEPS
    assert _route_from_architect(state) == "__end__"


def test_verdict_enum():
    """Verdict values are the strings the parser and router agree on."""
    assert Verdict.PLAN.value == "plan"
    assert Verdict.APPROVED.value == "approved"
    assert Verdict.REVISE.value == "revise"
    assert Verdict.NEED_RESEARCH.value == "need_research"


def test_architect_parser_defaults_by_pass():
    """An unparseable verdict must not strand the loop.

    Before the Builder reports the only sound fallback is `plan`; after it, the
    sound fallback is `approved`, so a malformed response ends the run instead
    of looping on it until the step ceiling.
    """
    from langgraph_agent.nodes import _parse_architect_output

    assert _parse_architect_output("garbage", reviewing=False)["verdict"] == "plan"
    assert _parse_architect_output("garbage", reviewing=True)["verdict"] == "approved"

    parsed = _parse_architect_output(
        "## Architecture\nKeep it small.\n\n## Constraints\n- No new deps\n\n"
        "## Verdict\nrevise",
        reviewing=True,
    )
    assert parsed["verdict"] == "revise"
    assert "Keep it small." in parsed["architecture"]
    assert "No new deps" in parsed["architecture"]


class _ToolCallingLLM:
    """A Builder seat that calls a tool once, then reports.

    Stands in for a real tool-capable model: the first invoke asks for a
    write, the second closes the report. `bind_tools` returns self so the
    node's bind step behaves like a live seat's.
    """

    def __init__(self, path):
        self._path = path
        self.calls = 0
        self.bound = None

    def bind_tools(self, tools, **kwargs):
        self.bound = tools
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "filesystem_write",
                    "args": {"path": str(self._path), "content": "written by the tool loop\n"},
                    "id": "call_1",
                }],
            )
        return AIMessage(
            content=(
                "## Changes Made\nWrote the file.\n\n"
                f"## Files Modified\n- {self._path}\n\n"
                "## Next Steps / Blockers\nnone\n"
            )
        )


def test_builder_writes_through_a_tool_call(monkeypatch, tmp_path):
    """The Builder's file changes come from real tool calls, not from prose."""
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "written.txt"
    llm = _ToolCallingLLM(target)
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    state = initial_state("Write a file")
    state["plan"] = "1. Write written.txt"
    result = builder_node(state)

    # The tool actually ran: the file is on disk with the tool's content.
    assert target.read_text() == "written by the tool loop\n"
    assert result["files_changed"] == [str(target)]
    assert "filesystem_write" in result["builder_report"]
    assert result["blockers"] == ""


def test_builder_offers_only_its_own_tools(monkeypatch, tmp_path):
    """GraphRAG must never reach the Builder; retrieval is the Researcher's."""
    from langgraph_agent.nodes import builder_node

    llm = _ToolCallingLLM(tmp_path / "x.txt")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )
    builder_node(initial_state("Write a file"))

    offered = {tool["function"]["name"] for tool in llm.bound}
    assert "filesystem_write" in offered
    assert not offered & {"search_knowledge_graph", "query_knowledge_graph"}


def test_builder_does_not_credit_unwritten_files(monkeypatch):
    """A seat that cannot call tools reports, but claims no file changes."""
    from langgraph_agent.config import StubLLM
    from langgraph_agent.nodes import builder_node

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: StubLLM()
    )
    result = builder_node(initial_state("Create app.py"))

    assert result["files_changed"] == []


def test_moving_a_seat_clears_its_recorded_failure():
    """A failure belongs to the seat that produced it, not to the agent."""
    from langgraph_agent import config

    config.set_agent_llm("architect", "anthropic", "claude-opus-5")
    config._seat_failures["architect"] = "Anthropic credit balance too low"

    # Moving the seat to a different provider retires that verdict.
    config.set_agent_llm("architect", "ollama", "kimi-k3:cloud")
    status = config.get_agent_status("architect")

    assert "architect" not in config._seat_failures
    assert status["live"] is True
    assert status["reason"] == ""

    config._agent_llm_overrides.pop("architect", None)
    config._seat_failures.pop("architect", None)


def test_reselecting_the_same_seat_keeps_its_failure():
    """Picking the seat you already have must not launder a real failure."""
    from langgraph_agent import config

    config.set_agent_llm("architect", "ollama", "kimi-k3:cloud")
    config._seat_failures["architect"] = "Provider unreachable"

    config.set_agent_llm("architect", "ollama", "kimi-k3:cloud")
    status = config.get_agent_status("architect")

    assert status["live"] is False
    assert status["reason"] == "Provider unreachable"
    assert status["badge"] == "FAILING"

    config._agent_llm_overrides.pop("architect", None)
    config._seat_failures.pop("architect", None)


def test_default_seats_need_no_api_key():
    """A fresh checkout must run without the user holding a provider key.

    The Architect used to default to Anthropic, which made the entry node --
    and so the whole run -- depend on billable credit nobody had configured.
    """
    from langgraph_agent.config import AGENTS, DEFAULT_SEATS

    assert set(DEFAULT_SEATS) == set(AGENTS)
    assert all(seat["provider"] == "ollama" for seat in DEFAULT_SEATS.values())


class _WritesFileLLM(_ToolCallingLLM):
    """Writes a caller-supplied file body, then reports success."""

    def __init__(self, path, body):
        super().__init__(path)
        self._body = body

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "filesystem_write",
                    "args": {"path": str(self._path), "content": self._body},
                    "id": "call_1",
                }],
            )
        return AIMessage(
            content=(
                "## Changes Made\nWrote it.\n\n"
                f"## Files Modified\n- {self._path}\n\n"
                "## Next Steps / Blockers\nnone\n"
            )
        )


def test_builder_runs_the_python_it_writes(monkeypatch, tmp_path):
    """A written module that raises must come back as a blocker, not success.

    This is the failure that shipped: a file written, reported complete, and
    approved -- which raised an AssertionError the first time it was run.
    """
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "broken.py"
    llm = _WritesFileLLM(target, "assert False, 'boom'\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write a module"))

    # The write happened, and is still reported as a change...
    assert result["files_changed"] == [str(target)]
    # ...but the run failed, so the Builder does not get to claim completion.
    assert "do not run" in result["blockers"]
    assert "boom" in result["blockers"] or "AssertionError" in result["blockers"]
    assert "FAILED" in result["builder_report"]
    assert "do not run" in result["messages"][-1]
    assert "Implementation complete" not in result["messages"][-1]


def test_builder_reports_clean_when_the_file_runs(monkeypatch, tmp_path):
    """A file that executes cleanly verifies, and sets no blocker."""
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "fine.py"
    llm = _WritesFileLLM(target, "print('ok')\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write a module"))

    assert result["blockers"] == ""
    assert "ran clean" in result["builder_report"]
    assert "Implementation complete" in result["messages"][-1]


def test_builder_only_executes_runnable_files(monkeypatch, tmp_path):
    """Markdown has nothing to run; the verification pass must skip it."""
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "notes.md"
    llm = _WritesFileLLM(target, "# just prose\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write notes"))

    assert result["files_changed"] == [str(target)]
    assert result["blockers"] == ""
    assert "Verification" not in result["builder_report"]


class _ApprovingLLM:
    """An Architect seat that always rules approved."""

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        return AIMessage(
            content="## Architecture\nfine\n\n## Verdict\napproved\n"
        )


def test_failed_verification_blocks_approval(monkeypatch):
    """An Architect that approves work which does not run is overruled."""
    from langgraph_agent.nodes import architect_node

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _ApprovingLLM(),
    )

    state = initial_state("Write a module")
    state["plan"] = "1. Write it"
    state["builder_report"] = "wrote it"
    state["failed_verification"] = ["broken.py"]

    result = architect_node(state)

    assert result["verdict"] == Verdict.REVISE.value
    assert "approval blocked" in result["messages"][-1]
    # revise routes back to the Planner rather than ending the run.
    from langgraph_agent.graph import _route_from_architect

    assert _route_from_architect(result) == "planner"


def test_approval_stands_once_verification_passes(monkeypatch):
    """With nothing failing, the Architect's approval is left alone."""
    from langgraph_agent.nodes import architect_node

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _ApprovingLLM(),
    )

    state = initial_state("Write a module")
    state["plan"] = "1. Write it"
    state["builder_report"] = "wrote it"
    state["failed_verification"] = []

    result = architect_node(state)

    assert result["verdict"] == Verdict.APPROVED.value
    assert "blocked" not in result["messages"][-1]


def test_step_ceiling_still_ends_a_permanently_failing_run():
    """The block must not create a run that cannot end."""
    from langgraph_agent.graph import MAX_STEPS, _route_from_architect

    state = initial_state("anything")
    state["verdict"] = Verdict.REVISE.value
    state["failed_verification"] = ["broken.py"]
    state["step_count"] = MAX_STEPS

    assert _route_from_architect(state) == "__end__"


class _WritesNothingLLM:
    """A Builder pass that calls no tools and reports success anyway."""

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        return AIMessage(
            content=(
                "## Changes Made\nNothing to do.\n\n"
                "## Files Modified\n\n\n"
                "## Next Steps / Blockers\nnone\n"
            )
        )


def test_a_pass_that_writes_nothing_cannot_clear_a_failure(monkeypatch, tmp_path):
    """A broken file stays broken until it runs, not until it is ignored.

    The Builder could otherwise clear a failed verification by doing nothing on
    the next pass: the file stayed on disk, the failure list came back empty
    and the Architect approved.
    """
    from langgraph_agent.nodes import builder_node

    broken = tmp_path / "broken.py"
    broken.write_text("assert False, 'still broken'\n")

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _WritesNothingLLM(),
    )

    state = initial_state("Fix it")
    state["plan"] = "1. Fix the module"
    state["failed_verification"] = [str(broken)]

    result = builder_node(state)

    assert result["files_changed"] == []            # nothing written this pass
    assert result["failed_verification"] == [str(broken)]  # still failing
    assert "do not run" in result["blockers"]


def test_a_deleted_file_clears_its_failure(monkeypatch, tmp_path):
    """Deleting the broken file is a fix, not a permanent failure.

    A carried path that no longer exists used to be re-run anyway, and
    `python <missing path>` fails forever: failed_verification never emptied,
    the gate rewrote every `approved` to `revise`, and the run could only end
    at the step ceiling.
    """
    from langgraph_agent.nodes import builder_node

    gone = tmp_path / "_tmp_scratch.py"  # never created

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _WritesNothingLLM(),
    )

    state = initial_state("Drop the scratch script")
    state["plan"] = "1. Remove it"
    state["failed_verification"] = [str(gone)]

    result = builder_node(state)

    assert result["failed_verification"] == []
    assert result["blockers"] == ""
    assert str(gone) not in result["builder_report"]


def test_a_fixed_file_clears_its_failure(monkeypatch, tmp_path):
    """Once the file actually runs, the failure is retired."""
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "broken.py"
    target.write_text("assert False\n")

    llm = _WritesFileLLM(target, "print('fixed')\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    state = initial_state("Fix it")
    state["plan"] = "1. Fix the module"
    state["failed_verification"] = [str(target)]

    result = builder_node(state)

    assert result["failed_verification"] == []
    assert result["blockers"] == ""


def test_expect_failures_lets_the_gate_approve(monkeypatch):
    """With the opt-out set, a failing file no longer overrules the Architect."""
    from langgraph_agent.nodes import architect_node

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _ApprovingLLM(),
    )

    state = initial_state("Write a deliberate fixture")
    state["plan"] = "1. Write it"
    state["builder_report"] = "wrote it"
    state["failed_verification"] = ["fixture.py"]
    state["expect_failures"] = True

    result = architect_node(state)

    assert result["verdict"] == Verdict.APPROVED.value
    assert "blocked" not in result["messages"][-1]
    # The failure is not hidden -- it stays in state for the report.
    assert result["failed_verification"] == ["fixture.py"]


def test_expect_failures_still_runs_and_reports_the_file(monkeypatch, tmp_path):
    """The opt-out suppresses the block, not the check."""
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "fixture.py"
    llm = _WritesFileLLM(target, "assert False, 'by design'\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    state = initial_state("Write a deliberate fixture")
    state["expect_failures"] = True
    result = builder_node(state)

    # Executed and reported...
    assert "FAILED" in result["builder_report"]
    assert result["failed_verification"] == [str(target)]
    assert "expected for this run" in result["messages"][-1]
    # ...but not treated as a blocker.
    assert result["blockers"] == ""


def test_a_package_module_is_skipped_not_failed(monkeypatch, tmp_path):
    """`python pkg/mod.py` cannot import pkg, so running it proves nothing.

    A module that imports its own package absolutely dies with
    ModuleNotFoundError under direct execution however correct it is. Treating
    that as a failure pinned failed_verification open on a working package and
    the gate rewrote every `approved` to `revise` until the step ceiling.
    """
    from langgraph_agent.nodes import builder_node

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    target = pkg / "mod.py"

    llm = _WritesFileLLM(target, "from pkg.missing import nothing\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write a package module"))

    assert result["files_changed"] == [str(target)]
    assert result["failed_verification"] == []
    assert result["blockers"] == ""
    # Skipped loudly: the Architect must see that nothing ran it.
    assert "SKIPPED" in result["builder_report"]
    assert "root-level script" in result["builder_report"]


def test_a_root_level_script_is_still_executed(monkeypatch, tmp_path):
    """The skip is for package modules only; a loose script still has to run."""
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "verify_it.py"  # no __init__.py beside it
    llm = _WritesFileLLM(target, "assert False, 'boom'\n")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write a verification script"))

    assert result["failed_verification"] == [str(target)]
    assert "do not run" in result["blockers"]


def test_package_module_skip_does_not_hide_a_failing_script(monkeypatch, tmp_path):
    """A skipped module beside a failing script still leaves the run blocked."""
    from langgraph_agent.nodes import _verify_written_files

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    module = pkg / "mod.py"
    module.write_text("from pkg.missing import nothing\n")
    script = tmp_path / "verify_it.py"
    script.write_text("assert False\n")

    log: list[str] = []
    results = _verify_written_files([str(module), str(script)], log)
    statuses = {path: status for path, status, _ in results}

    assert statuses[str(module)] == "skipped"
    assert statuses[str(script)] == "failed"
    assert f"verify({module}) -> skipped" in log


def test_expect_failures_defaults_off():
    """The strict behaviour is what you get without asking for otherwise."""
    assert initial_state("anything")["expect_failures"] is False


@pytest.mark.parametrize(
    "text",
    [
        "none",
        "None.",
        "n/a",
        "N/A - all good",
        "no blockers",
        "No blockers.",
        "nothing",
        "none - see the report",
        "none — Note: this file intentionally raises AssertionError by design",
        "",
    ],
)
def test_a_blockers_section_saying_none_is_not_a_blocker(text):
    """"none" followed by commentary still means there are no blockers."""
    from langgraph_agent.nodes import _clean_blockers

    assert _clean_blockers(text) == ""


@pytest.mark.parametrize(
    "text",
    [
        "none of the tests pass",
        "nothing works after the refactor",
        "Missing API key for the provider",
        "no blockers were resolved; the import still fails",
    ],
)
def test_a_real_blocker_starting_with_none_survives(text):
    """The match must not swallow a sentence that only begins with the word.

    Dropping "none of the tests pass" would be exactly the silent success the
    verification pass exists to prevent.
    """
    from langgraph_agent.nodes import _clean_blockers

    assert _clean_blockers(text) == text


# --- Timeouts -------------------------------------------------------------
#
# A stalled seat used to hang a run outright. RUN_BUDGET_SECONDS is checked
# between graph supersteps, so a node that never returns never reaches the
# check, and no provider client carried a request timeout of its own.


def test_a_hung_researcher_gives_up_and_routes_to_builder(monkeypatch):
    """The node returns on its deadline instead of blocking the run forever."""
    import threading

    from langgraph_agent import nodes

    entered = threading.Event()
    release = threading.Event()

    def _hang(state):
        entered.set()
        release.wait(30)  # released in the finally below, never on its own
        return "should never be used", "ready_for_builder"

    monkeypatch.setattr(nodes, "_gather_research", _hang)
    monkeypatch.setattr(nodes, "NODE_DEADLINE_SECONDS", 0.5)

    state = initial_state("Research something the seat will not answer")
    state["plan"] = "1. Look it up"

    try:
        result = nodes.researcher_node(state)
    finally:
        release.set()

    assert entered.is_set()                                   # it really ran
    assert result["research_status"] == "no_relevant_knowledge"
    assert result["next_agent"] == "Builder"                  # run continues
    assert "did not finish" in result["research"]
    assert any("No response within" in m for m in result["messages"])


def test_a_late_researcher_cannot_write_into_state(monkeypatch):
    """The abandoned worker must not land its result after the node returned.

    It cannot be cancelled -- Python cannot interrupt a thread blocked on a
    socket -- so the only protection is that `_gather_research` returns its
    findings rather than writing them, and the timed-out caller drops them.
    """
    import threading

    from langgraph_agent import nodes

    release = threading.Event()
    finished = threading.Event()

    def _slow(state):
        release.wait(30)
        finished.set()
        return "LATE FINDINGS", "ready_for_builder"

    monkeypatch.setattr(nodes, "_gather_research", _slow)
    monkeypatch.setattr(nodes, "NODE_DEADLINE_SECONDS", 0.3)

    state = initial_state("Research something slow")
    result = nodes.researcher_node(state)

    release.set()
    assert finished.wait(10)          # the worker did complete, just too late
    assert "LATE FINDINGS" not in result["research"]
    assert "LATE FINDINGS" not in state["research"]


def test_a_failing_researcher_still_raises(monkeypatch):
    """A seat that errors is not a timeout and must not be swallowed.

    `_SeatLLM` records the reason from the exception; converting it into a
    quiet "no research" would take the failing seat off the console.
    """
    from langgraph_agent import nodes

    def _boom(state):
        raise RuntimeError("credit balance too low")

    monkeypatch.setattr(nodes, "_gather_research", _boom)

    with pytest.raises(RuntimeError, match="credit balance"):
        nodes.researcher_node(initial_state("Research something"))


class _SilentSeat:
    """A seat whose model answers with nothing at all."""

    def __init__(self, content=""):
        self.content = content

    def invoke(self, messages):
        return SimpleNamespace(content=self.content)


@pytest.mark.parametrize(
    "content",
    [
        "",                                  # nothing at all
        "   \n\n  ",                         # whitespace
        "## Key Findings\n\n## Status\nready_for_builder",   # headings, no content
        [],                                  # empty content blocks
    ],
)
def test_a_silent_researcher_is_not_reported_as_research(monkeypatch, content):
    """An empty answer must not route as `ready_for_builder`.

    The parsed status defaults to `ready_for_builder`, so silence was being
    announced as "Research complete" while `research` reached the Builder
    empty. The Builder then reports an empty store, the gate rules
    `need_research`, and the run goes back round to the same silent seat --
    a loop that burns a step per cycle up to the ceiling.
    """
    from langgraph_agent import nodes

    monkeypatch.setattr(nodes, "get_agent_llm", lambda agent: _SilentSeat(content))
    monkeypatch.setattr(
        nodes, "_call_mcp_tool_sync", lambda *a, **k: {"results": []}
    )

    state = initial_state("Research something the seat will not answer")
    state["plan"] = "1. Look it up"

    result = nodes.researcher_node(state)

    assert result["research_status"] == "no_relevant_knowledge"
    assert result["next_agent"] == "Builder"          # not back to the Planner
    assert "returned no findings" in result["research"]
    assert not any("Research complete" in m for m in result["messages"])
    assert any("check the Researcher's model" in m for m in result["messages"])


def test_real_findings_still_route_as_research(monkeypatch):
    """The guard must not swallow a seat that actually answered."""
    from langgraph_agent import nodes

    answer = (
        "## Key Findings\nThe Laplacian is D - A.\n\n"
        "## Relevant Context\nSpectral graph theory.\n\n"
        "## Recommendations for Builder\nStart from the Laplacian.\n\n"
        "## Status\nready_for_builder"
    )
    monkeypatch.setattr(nodes, "get_agent_llm", lambda agent: _SilentSeat(answer))
    monkeypatch.setattr(
        nodes, "_call_mcp_tool_sync", lambda *a, **k: {"results": []}
    )

    state = initial_state("Research spectral graph theory")
    state["plan"] = "1. Look it up"

    result = nodes.researcher_node(state)

    assert result["research_status"] == "ready_for_builder"
    assert "Laplacian" in result["research"]
    assert any("Research complete" in m for m in result["messages"])


def test_every_provider_client_carries_a_request_timeout(monkeypatch):
    """No seat may be built without a deadline, whatever the provider.

    Each spells it differently, so a single missed keyword silently restores
    the unbounded wait for that one provider only.
    """
    from langgraph_agent import config

    monkeypatch.setattr(config, "LLM_TIMEOUT_SECONDS", 42.0)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    ollama = config.get_llm(provider="ollama", model="kimi-k3:cloud")
    assert ollama._client._client.timeout.read == 42.0

    anthropic = config.get_llm(provider="anthropic", model="claude-sonnet-5")
    assert anthropic.default_request_timeout == 42.0

    openai = config.get_llm(provider="openai", model="gpt-4o-mini")
    assert openai.request_timeout == 42.0


def test_a_timed_out_seat_reads_as_a_failure_not_a_blank(monkeypatch):
    """httpx raises ReadTimeout with an empty message.

    Left alone that produced an empty chip, which reads as a healthy seat.
    """
    import httpx

    from langgraph_agent.config import _failure_reason

    assert "No response within" in _failure_reason(httpx.ReadTimeout(""))


# --- The Builder's deadline -----------------------------------------------
#
# Unlike the Researcher, this node has side effects: abandoning a worker
# mid-write would keep writing into the project after the node returned. So the
# model's call is bounded, the tool calls underneath it always finish, and the
# budget is re-checked between turns.


class _SlowToolCallingLLM(_ToolCallingLLM):
    """A Builder seat whose *model* call hangs on the turn after its write."""

    def __init__(self, path, release):
        super().__init__(path)
        self._release = release

    def invoke(self, messages):
        if self.calls >= 1:
            self._release.wait(30)
        return super().invoke(messages)


def test_a_hung_builder_keeps_the_files_it_already_wrote(monkeypatch, tmp_path):
    """The deadline ends the turn; it does not discard completed work."""
    import threading

    from langgraph_agent import nodes

    target = tmp_path / "written.py"
    release = threading.Event()
    llm = _SlowToolCallingLLM(target, release)
    monkeypatch.setattr(nodes, "get_agent_llm", lambda agent, temperature=0.1: llm)
    monkeypatch.setattr(nodes, "BUILDER_DEADLINE_SECONDS", 1.0)
    monkeypatch.setattr(nodes, "VERIFY_RESERVE_SECONDS", 0.5)

    state = initial_state("Write a file")
    state["plan"] = "1. Write it"

    try:
        result = nodes.builder_node(state)
    finally:
        release.set()

    # The write completed before the hang, so it counts.
    assert str(target) in result["files_changed"]
    assert target.exists()
    assert "deadline" in result["blockers"]
    assert not any("Implementation complete" in m for m in result["messages"])


def test_a_tool_call_is_never_abandoned_midway(monkeypatch, tmp_path):
    """A slow tool runs to completion even past the deadline.

    Only the model's own call may be abandoned. A worker cut off inside
    `filesystem_write` would go on writing into the project after the node had
    returned, which is worse than the hang the deadline exists to stop.
    """
    import time

    from langgraph_agent import nodes

    target = tmp_path / "slow.py"
    started = []
    finished = []

    real_call = nodes._call_mcp_tool_sync

    def _slow_call(name, args):
        if name == "filesystem_write":
            started.append(name)
            time.sleep(1.2)          # outlives the deadline set below
            result = real_call(name, args)
            finished.append(name)
            return result
        return real_call(name, args)

    monkeypatch.setattr(nodes, "_call_mcp_tool_sync", _slow_call)
    monkeypatch.setattr(
        nodes, "get_agent_llm", lambda agent, temperature=0.1: _ToolCallingLLM(target)
    )
    monkeypatch.setattr(nodes, "BUILDER_DEADLINE_SECONDS", 1.0)
    monkeypatch.setattr(nodes, "VERIFY_RESERVE_SECONDS", 0.5)

    state = initial_state("Write a file slowly")
    result = nodes.builder_node(state)

    assert started and finished          # it ran all the way through
    assert str(target) in result["files_changed"]


def test_an_unverified_file_does_not_count_as_passing(tmp_path):
    """A file the deadline stopped us running is unproven, not working."""
    from langgraph_agent import nodes

    good = tmp_path / "fine.py"
    good.write_text("print('fine')\n")

    log = []
    spent = nodes._Deadline(0.0)         # already expired
    results = nodes._verify_written_files([str(good)], log, spent)

    assert results == [
        (str(good), "unverified", nodes.VERIFY_DEADLINE_SKIP_REASON)
    ]
    assert "not run (deadline)" in log[0]


def test_a_sliver_of_time_left_does_not_fail_a_working_file(tmp_path):
    """A slice too small to run in must read as unrun, not as broken.

    `min(60, remaining)` rounded down to a zero-second timeout, so a perfectly
    good file came back FAILED with "timed out after 0 seconds" and set the
    blocker that says the file does not run.
    """
    from langgraph_agent import nodes

    good = tmp_path / "fine.py"
    good.write_text("print('fine')\n")

    log = []
    sliver = nodes._Deadline(nodes.MIN_VERIFY_SLICE_SECONDS / 2)
    statuses = [status for _, status, _ in nodes._verify_written_files(
        [str(good)], log, sliver
    )]

    assert statuses == ["unverified"]


def _builder_with_unverified_file(monkeypatch, tmp_path, **state_overrides):
    """Run builder_node with the verification pass reporting one unrun file.

    The result is forced rather than produced by racing the real clock: making
    the tool loop finish while leaving verification under its floor takes
    timings this suite cannot hold steady. What is under test here is the
    wiring around that result, not the clock, which
    `test_an_unverified_file_does_not_count_as_passing` covers directly.
    """
    from langgraph_agent import nodes

    target = tmp_path / "written.py"
    monkeypatch.setattr(
        nodes, "get_agent_llm", lambda agent, temperature=0.1: _ToolCallingLLM(target)
    )
    monkeypatch.setattr(
        nodes,
        "_verify_written_files",
        lambda paths, log, deadline=None: [
            (str(target), "unverified", nodes.VERIFY_DEADLINE_SKIP_REASON)
        ],
    )

    state = initial_state("Write a file")
    state.update(state_overrides)
    return str(target), nodes.builder_node(state)


def test_an_unverified_file_blocks_and_is_carried(monkeypatch, tmp_path):
    """It reaches failed_verification so the next cycle re-runs it.

    Clearing by omission is the failure the verification pass exists to
    prevent, and a file nobody executed is the purest case of it.
    """
    target, result = _builder_with_unverified_file(monkeypatch, tmp_path)

    assert result["failed_verification"] == [target]
    assert "Not executed before the deadline" in result["blockers"]
    assert "NOT RUN" in result["builder_report"]
    assert not any("Implementation complete" in m for m in result["messages"])


def test_expect_failures_does_not_excuse_an_unverified_file(monkeypatch, tmp_path):
    """The opt-out covers a file meant to fail, not one never executed."""
    target, result = _builder_with_unverified_file(
        monkeypatch, tmp_path, expect_failures=True
    )

    assert result["failed_verification"] == [target]
    assert "Not executed before the deadline" in result["blockers"]


# --- The Architect's and Planner's deadlines -------------------------------
#
# Both are single-call, tool-free nodes, so they can be wrapped whole the way
# the Researcher is. What needed thought was not the mechanism but the fallback
# each one leaves behind.


class _HangingLLM:
    """A seat that never answers in time, until the test releases it.

    It returns rather than raising once released: an exception would travel
    back out of `_with_deadline` on purpose, which is a different behaviour
    from a hang and would mask what these tests are checking.
    """

    def __init__(self, release):
        self._release = release
        self.calls = 0

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls += 1
        self._release.wait(30)
        return AIMessage(content="")   # abandoned by then; never read


def _hang(monkeypatch, seconds=0.4):
    """Point every seat at a hanging model and shorten every node deadline.

    The Builder keeps its own, larger budget, so patching only
    NODE_DEADLINE_SECONDS would leave that node waiting out the real one.
    """
    import threading

    from langgraph_agent import nodes

    release = threading.Event()
    monkeypatch.setattr(
        nodes, "get_agent_llm", lambda agent, temperature=0.1: _HangingLLM(release)
    )
    monkeypatch.setattr(nodes, "NODE_DEADLINE_SECONDS", seconds)
    monkeypatch.setattr(nodes, "BUILDER_DEADLINE_SECONDS", seconds * 2)
    monkeypatch.setattr(nodes, "VERIFY_RESERVE_SECONDS", seconds)
    return release


def test_a_hung_architect_never_approves(monkeypatch):
    """The gate ends the run, so a stalled seat must not be able to end one.

    This is the one fallback in the system that could turn a hang into a
    false success, which is why it is asserted on both passes rather than
    just the one that happens to be reachable first.
    """
    from langgraph_agent import nodes

    release = _hang(monkeypatch)
    try:
        opening = nodes.architect_node(initial_state("Do the thing"))

        reviewing = initial_state("Do the thing")
        reviewing["plan"] = "1. Do it"
        reviewing["builder_report"] = "Implementation complete"
        gate = nodes.architect_node(reviewing)
    finally:
        release.set()

    assert opening["verdict"] == Verdict.PLAN.value
    assert gate["verdict"] == Verdict.REVISE.value
    for result in (opening, gate):
        assert result["verdict"] != Verdict.APPROVED.value
        assert any("never an approval" in m for m in result["messages"])


def test_a_hung_architect_keeps_the_architecture_it_had(monkeypatch):
    """Losing it mid-run would strip the constraints out of every later prompt."""
    from langgraph_agent import nodes

    release = _hang(monkeypatch)
    state = initial_state("Do the thing")
    state["plan"] = "1. Do it"
    state["architecture"] = "Layered, with the parser kept separate."
    try:
        result = nodes.architect_node(state)
    finally:
        release.set()

    assert result["architecture"] == "Layered, with the parser kept separate."


def test_a_hung_architect_still_counts_its_step(monkeypatch):
    """Otherwise a stalling gate loops forever instead of reaching MAX_STEPS."""
    from langgraph_agent import nodes

    release = _hang(monkeypatch)
    state = initial_state("Do the thing")
    state["plan"] = "1. Do it"
    try:
        result = nodes.architect_node(state)
    finally:
        release.set()

    assert result["step_count"] == 1


def test_a_hung_planner_leaves_a_plan_behind(monkeypatch):
    """An empty plan would loop uncounted to the recursion limit.

    step_count is incremented only while a plan exists, so a blank fallback
    would send Planner -> Builder -> Architect round without ever counting,
    until LangGraph killed the run by exception and discarded its messages.
    """
    from langgraph_agent import nodes

    release = _hang(monkeypatch)
    try:
        result = nodes.planner_node(initial_state("Do the thing"))
    finally:
        release.set()

    assert result["plan"]                      # the load-bearing part
    assert "did not respond" in result["plan"]
    assert result["next_agent"] == "Builder"
    assert any("No response within" in m for m in result["messages"])


def test_a_hung_planner_keeps_a_real_plan_over_the_placeholder(monkeypatch):
    """On a revise cycle the existing plan is better information."""
    from langgraph_agent import nodes

    release = _hang(monkeypatch)
    state = initial_state("Do the thing")
    state["plan"] = "1. The plan from the previous cycle"
    try:
        result = nodes.planner_node(state)
    finally:
        release.set()

    assert result["plan"] == "1. The plan from the previous cycle"


def test_a_stalling_architect_run_still_terminates(monkeypatch):
    """End to end: every seat hangs, and the run ends at the step ceiling.

    The point of every fallback above is that the graph keeps moving and the
    counter keeps counting; this asserts the property they exist to give.
    """
    from langgraph_agent import graph as graph_module
    from langgraph_agent import nodes

    release = _hang(monkeypatch, seconds=0.05)
    try:
        result = create_agent_graph().invoke(
            initial_state("Do the thing"),
            {"recursion_limit": graph_module.RECURSION_LIMIT},
        )
    finally:
        release.set()

    assert result["step_count"] >= graph_module.MAX_STEPS
    assert result["verdict"] != Verdict.APPROVED.value
    assert nodes.NODE_DEADLINE_SECONDS == 0.05   # the fixture really applied


# ---------------------------------------------------------------------------
# The emergency stop
#
# Cooperative by construction: nothing here interrupts work in flight. The
# checks decline to start the *next* piece of work, which is why a stopped run
# never leaves a half-written file behind -- and why what it did write is still
# on disk and still reported.
# ---------------------------------------------------------------------------


class _StopsAfterWritingLLM(_ToolCallingLLM):
    """Writes a file, then trips the emergency stop before the next turn."""

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        from langgraph_agent.control import RUN_CONTROL

        self.calls += 1
        if self.calls == 1:
            RUN_CONTROL.arm("run-under-test")
            RUN_CONTROL.stop("run-under-test", "Stopped from the console.")
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "filesystem_write",
                    "args": {"path": str(self._path), "content": "half a job\n"},
                    "id": "call_1",
                }],
            )
        raise AssertionError("the stop should have ended the turn loop")


def test_a_stopped_builder_keeps_the_files_it_already_wrote(monkeypatch, tmp_path):
    """The write in flight completes; the next turn never starts."""
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "half.txt"
    llm = _StopsAfterWritingLLM(target)
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write a file"))

    # The tool call that was already issued ran to completion -- abandoning it
    # is what would leave a half-written file in the project.
    assert target.read_text() == "half a job\n"
    assert result["files_changed"] == [str(target)]
    # ...and the loop stopped rather than taking another turn.
    assert llm.calls == 1
    assert "emergency stop" in result["blockers"]
    assert "Implementation complete" not in result["messages"][-1]
    assert "Stopped by the operator" in result["messages"][-1]


def test_a_stopped_builder_is_not_described_as_out_of_time(monkeypatch, tmp_path):
    """A run the operator stopped must not be reported as one that overran.

    The two are separate flags for exactly this reason: the report is the only
    place anyone finds out which of them happened.
    """
    from langgraph_agent.nodes import BUILDER_DEADLINE_SECONDS, builder_node

    llm = _StopsAfterWritingLLM(tmp_path / "half.txt")
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm", lambda agent, temperature=0.1: llm
    )

    result = builder_node(initial_state("Write a file"))

    assert f"{int(BUILDER_DEADLINE_SECONDS)}s" not in result["blockers"]
    assert "deadline" not in result["messages"][-1]


def test_a_stopped_run_leaves_written_files_unproven(monkeypatch, tmp_path):
    """Files nobody executed come back unverified, not clear."""
    from langgraph_agent.control import RUN_CONTROL
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "works.py"
    target.write_text("print('fine')\n")

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: StubLLM(),
    )
    state = initial_state("Fix the module")
    # Carried from an earlier pass: the file exists and would pass if run.
    state["failed_verification"] = [str(target)]

    RUN_CONTROL.arm("run-under-test")
    RUN_CONTROL.stop("run-under-test")
    result = builder_node(state)

    assert result["failed_verification"] == [str(target)]
    assert "NOT RUN" in result["builder_report"]
    assert "the run was stopped" in result["builder_report"]


def test_expect_failures_does_not_clear_a_file_the_stop_left_unrun(
    monkeypatch, tmp_path
):
    """The safeguard checkbox cannot make a stopped run look finished.

    `expect_failures` is for a file the run *meant* to fail -- still executed,
    still reported. A file the stop prevented anyone from executing is a gap in
    the evidence, not an expected result, so it goes on blocking either way.
    """
    from langgraph_agent.control import RUN_CONTROL
    from langgraph_agent.nodes import builder_node

    target = tmp_path / "works.py"
    target.write_text("print('fine')\n")

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: StubLLM(),
    )
    state = initial_state("Fix the module")
    state["failed_verification"] = [str(target)]
    state["expect_failures"] = True

    RUN_CONTROL.arm("run-under-test")
    RUN_CONTROL.stop("run-under-test")
    result = builder_node(state)

    assert result["failed_verification"] == [str(target)]
    assert "Not executed before the stop" in result["blockers"]


def test_a_stopped_seat_never_calls_its_model(monkeypatch):
    """The tool-free nodes bail before spending a cloud call."""
    from langgraph_agent.control import RUN_CONTROL
    from langgraph_agent.nodes import architect_node, planner_node, researcher_node

    class _Counting(StubLLM):
        calls = 0

        def invoke(self, messages):
            type(self).calls += 1
            return super().invoke(messages)

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: _Counting(),
    )

    RUN_CONTROL.arm("run-under-test")
    RUN_CONTROL.stop("run-under-test")

    for node, role in (
        (architect_node, "Architect"),
        (planner_node, "Planner"),
        (researcher_node, "Researcher"),
    ):
        result = node(initial_state("anything"))
        assert f"[{role}] Stopped by the emergency stop" in result["messages"][-1]

    assert _Counting.calls == 0


def test_a_stopped_architect_never_approves(monkeypatch):
    """The gate is what ends the run, so a stop must not end one successfully."""
    from langgraph_agent.control import RUN_CONTROL
    from langgraph_agent.nodes import architect_node

    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: StubLLM(),
    )

    state = initial_state("Ship it")
    state["plan"] = "1. Do it"
    state["builder_report"] = "Implementation complete."  # the gate pass
    before = state["step_count"]

    RUN_CONTROL.arm("run-under-test")
    RUN_CONTROL.stop("run-under-test")
    result = architect_node(state)

    assert result["verdict"] != Verdict.APPROVED.value
    # A pass that did no work does not count as a step.
    assert result["step_count"] == before


def test_a_stale_stop_cannot_kill_the_run_that_replaced_it():
    """A Stop held over from a finished run must not hit the current one."""
    from langgraph_agent.control import RUN_CONTROL

    RUN_CONTROL.arm("run-two")
    assert RUN_CONTROL.stop("run-one") is False
    assert RUN_CONTROL.stopped() is False

    assert RUN_CONTROL.stop("run-two") is True
    assert RUN_CONTROL.stopped() is True


def test_stopping_nothing_is_refused_not_faked():
    """With no run armed there is nothing to stop, and saying so is the answer."""
    from langgraph_agent.control import RUN_CONTROL

    assert RUN_CONTROL.stop() is False
    assert RUN_CONTROL.stopped() is False
