"""Tests for the 4-Agent System.

Verifies:
- State schema matches documentation
- Architect opens the run and holds the approval gate
- Planner routes correctly based on task type
- Researcher sets proper status
- Builder reports changes and blockers
- Graph loops on the Architect's verdict and stops at the step limit
"""

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
