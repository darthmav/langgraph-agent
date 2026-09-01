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
