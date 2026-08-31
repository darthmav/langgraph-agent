"""LangGraph graph definition.

Implements the 3-Agent System architecture:
- Planner → Researcher → Builder flow
- Feedback loops from Builder back to Planner/Researcher
- Step count limit to prevent infinite loops
"""

from typing import Literal

from langgraph.graph import END, StateGraph

from langgraph_agent.nodes import builder_node, planner_node, researcher_node
from langgraph_agent.state import AgentState, ResearchStatus

# Maximum steps before escalation
MAX_STEPS = 8


def create_agent_graph() -> StateGraph:
    """Create and compile the 3-agent system graph.

    Graph structure:
        START → Planner → conditional → Researcher → Builder → END
                                    ^             |
                                    +---- loop ---+

    Routing logic:
    - Planner chooses Researcher (needs knowledge) or Builder (task is clear)
    - Researcher sets status: ready_for_builder | need_replan | no_relevant_knowledge
    - Builder sets blockers or completes
    - Graph loops on blockers or need_replan status
    - Stops at MAX_STEPS or when complete
    """
    # Initialize the graph with our state schema
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("planner", planner_node)
    graph_builder.add_node("researcher", researcher_node)
    graph_builder.add_node("builder", builder_node)

    # Set entry point
    graph_builder.set_entry_point("planner")

    # Route from Planner based on next_agent field
    # First turn: always go to Researcher to ensure knowledge gathering
    # Subsequent turns: respect the Planner's routing decision
    def route_from_planner(state: AgentState) -> Literal["researcher", "builder"]:
        # Check if this is the first turn (research not yet done)
        if not state.get("research"):
            return "researcher"  # Always research first
        
        # On subsequent turns, respect the Planner's decision
        next_agent = state.get("next_agent", "Researcher")
        return next_agent.lower()

    graph_builder.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "researcher": "researcher",
            "builder": "builder",
        },
    )

    # Route from Researcher based on research_status
    def route_from_researcher(state: AgentState) -> Literal["planner", "builder"]:
        status = state.get("research_status", ResearchStatus.READY_FOR_BUILDER.value)

        if status == ResearchStatus.NEED_REPLAN.value:
            return "planner"
        return "builder"

    graph_builder.add_conditional_edges(
        "researcher",
        route_from_researcher,
        {
            "planner": "planner",
            "builder": "builder",
        },
    )

    # Route from Builder based on blockers and step count
    def route_from_builder(state: AgentState) -> Literal["planner", "researcher", END]:
        # Check step limit
        if state.get("step_count", 0) >= MAX_STEPS:
            state["messages"].append(f"[Graph] Max steps ({MAX_STEPS}) reached. Stopping.")
            return END

        # Check for blockers
        blockers = state.get("blockers", "")
        if blockers:
            # Determine if we need research or replanning
            if "need" in blockers.lower() and (
                "info" in blockers.lower() or "research" in blockers.lower()
            ):
                state["messages"].append("[Graph] Blockers require research")
                return "researcher"
            else:
                state["messages"].append("[Graph] Blockers require replanning")
                return "planner"

        # No blockers, task complete
        return END

    graph_builder.add_conditional_edges(
        "builder",
        route_from_builder,
        {
            "planner": "planner",
            "researcher": "researcher",
            END: END,
        },
    )

    # Compile
    return graph_builder.compile()
