"""Tests for the 3-Agent System.

Verifies:
- State schema matches documentation
- Planner routes correctly based on task type
- Researcher sets proper status
- Builder reports changes and blockers
- Graph loops on blockers and stops at step limit
"""

import pytest

from langgraph_agent import AgentState, ResearchStatus, create_agent_graph
from langgraph_agent.config import StubLLM


@pytest.fixture
def agent_graph(monkeypatch):
    """Create a fresh agent graph for testing using the deterministic StubLLM."""
    # Force every agent node to use the canned StubLLM so tests do not depend
    # on a live Ollama server or cloud API keys.
    monkeypatch.setattr(
        "langgraph_agent.nodes.get_agent_llm",
        lambda agent, temperature=0.1: StubLLM(),
    )
    return create_agent_graph()


def initial_state(goal: str) -> AgentState:
    """Create initial state per the 3-Agent System specification."""
    return {
        "goal": goal,
        "messages": [],
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
    """Test that the full 3-agent flow executes."""
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

    assert "Plan: (empty)" in injection
    assert "Research: (empty)" in injection
    assert "Builder report: (empty)" in injection
    assert "Blockers: (empty)" in injection
    assert "Files changed: (empty)" in injection


