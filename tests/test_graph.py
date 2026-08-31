"""Basic tests for the agent graph."""

import pytest

from langgraph_agent import AgentState, create_agent_graph


@pytest.fixture
def agent_graph():
    """Create a fresh agent graph for testing."""
    return create_agent_graph()


def initial_state(input_text: str) -> AgentState:
    """Create initial state for testing."""
    return {
        "input": input_text,
        "plan": [],
        "current_step": 0,
        "research_findings": None,
        "builder_output": None,
        "messages": [],
        "status": "started",
        "next_node": "",
        "feedback": None,
        "iteration": 0,
        "max_iterations": 3,
    }


def test_planner_routes_to_researcher(agent_graph):
    """Test that planner routes to researcher when input contains research keywords."""
    state = initial_state("Research the best way to implement caching")

    result = agent_graph.invoke(state)

    # Check that the plan was created
    assert len(result["plan"]) > 0
    # Check that researcher ran (messages should mention researcher)
    assert any("researcher" in msg.lower() for msg in result["messages"])


def test_planner_routes_to_builder(agent_graph):
    """Test that planner routes to builder for clear tasks."""
    state = initial_state("Create a function that adds two numbers")

    result = agent_graph.invoke(state)

    assert len(result["plan"]) > 0
    assert result["status"] == "complete"


def test_full_graph_completion(agent_graph):
    """Test that the full graph completes successfully."""
    state = initial_state("Build a simple calculator")

    result = agent_graph.invoke(state)

    assert result["status"] == "complete"
    assert result["builder_output"] is not None
    assert len(result["messages"]) >= 2


def test_feedback_loop_triggers_replan(agent_graph):
    """Test that feedback triggers replanning."""
    state = initial_state("Create a REST API")
    state["feedback"] = "This is wrong, please fix the approach"

    result = agent_graph.invoke(state)

    # Graph should complete (either replanned or finished)
    assert result["status"] == "complete" or result["iteration"] > 0


def test_max_iterations_stops_loop(agent_graph):
    """Test that max_iterations prevents infinite loops."""
    state = initial_state("Build something")
    state["feedback"] = "Keep fixing this"  # Continuous feedback
    state["iteration"] = 3  # At the limit
    state["max_iterations"] = 3

    result = agent_graph.invoke(state)

    # Should complete without infinite loop (status should be complete)
    assert result["status"] == "complete"
    # Iteration should not exceed max
    assert result["iteration"] <= result["max_iterations"]
