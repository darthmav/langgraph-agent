"""LangGraph graph definition."""

from typing import Literal

from langgraph.graph import END, StateGraph

from langgraph_agent.nodes import builder_node, planner_node, researcher_node
from langgraph_agent.state import AgentState


def create_agent_graph(max_iterations: int = 3) -> StateGraph:
    """Create and compile the agent graph.

    Graph structure:
        START -> Planner -> conditional -> Researcher -> Builder -> END
                                      ^                    |
                                      +------ loop --------+

        Builder can loop back to Planner if feedback indicates replanning needed.

    Args:
        max_iterations: Maximum times through the loop (prevents infinite loops)
    """
    # Initialize the graph with our state schema
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("planner", planner_node)
    graph_builder.add_node("researcher", researcher_node)
    graph_builder.add_node("builder", builder_node)

    # Set entry point
    graph_builder.set_entry_point("planner")

    # Add edges with conditional routing from planner
    def route_from_planner(state: AgentState) -> Literal["researcher", "builder"]:
        return state.get("next_node", "builder")

    graph_builder.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "researcher": "researcher",
            "builder": "builder",
        },
    )

    # Researcher always goes to builder after research
    graph_builder.add_edge("researcher", "builder")

    # Builder can loop back to planner if feedback indicates issues, or end
    def route_from_builder(state: AgentState) -> Literal["planner", END]:
        # Check if we've hit max iterations
        if state.get("iteration", 0) >= max_iterations:
            state["messages"].append(
                f"[Graph] Max iterations ({max_iterations}) reached. Stopping."
            )
            state["status"] = "complete"
            return END

        # Check if feedback indicates need for replanning
        feedback = state.get("feedback", "")
        if feedback and any(
            kw in feedback.lower()
            for kw in ["redo", "fix", "change", "revise", "replan", "wrong", "error"]
        ):
            state["iteration"] = state.get("iteration", 0) + 1
            state["messages"].append(
                f"[Graph] Feedback detected. Replanning (iteration {state['iteration']}/{max_iterations})"
            )
            return "planner"

        state["status"] = "complete"
        return END

    graph_builder.add_conditional_edges(
        "builder",
        route_from_builder,
        {
            "planner": "planner",
            END: END,
        },
    )

    # Compile
    return graph_builder.compile()
