"""LangGraph graph definition.

Implements the 4-Agent System architecture:
- Architect → Planner → Researcher → Builder → Architect flow
- The Architect is the leading authority: it opens the run and holds the gate
- Step count limit to prevent infinite loops
"""

from typing import Any, Literal, Protocol

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from langgraph_agent.control import ACTIVITY
from langgraph_agent.nodes import (
    architect_node,
    builder_node,
    planner_node,
    researcher_node,
)
from langgraph_agent.state import AgentState, ResearchStatus, Verdict

# Maximum cycles through the Architect gate before escalation
MAX_STEPS = 8

# Supersteps LangGraph will run before it gives up. A counted step costs at
# most four supersteps (architect, planner, researcher, builder), but a
# Researcher that asks for a replan bounces back to the Planner without passing
# the gate, so the true cost per step is higher than four. At exactly 4 *
# MAX_STEPS the recursion limit fired before the gate ever reached MAX_STEPS,
# which meant the run always died by exception instead of ending on its own
# terms. The gate is the intended stop; this is only the backstop behind it.
RECURSION_LIMIT = 6 * MAX_STEPS + 4


# The routing functions live at module scope rather than inside the factory:
# they are pure functions of state, and as closures they could only be
# exercised by running the whole graph.
def _route_from_architect(state: AgentState) -> Literal["planner", "researcher", "__end__"]:
    """Route on the Architect's verdict. Opening move and terminating gate."""
    # The step ceiling is checked first so a stuck loop cannot outvote it.
    if state.get("step_count", 0) >= MAX_STEPS:
        return "__end__"

    verdict = state.get("verdict", Verdict.PLAN.value)

    if verdict == Verdict.APPROVED.value:
        return "__end__"
    if verdict == Verdict.NEED_RESEARCH.value:
        return "researcher"
    # plan and revise both go to the Planner; the difference is that on a
    # revise there is already a plan, research and a report in state for it
    # to work from.
    return "planner"


def _route_from_planner(state: AgentState) -> Literal["researcher", "builder"]:
    """Respect the Planner's explicit routing decision on every turn.

    The Planner chooses Researcher when knowledge is needed and Builder when
    the task is already fully specified.
    """
    next_agent = state.get("next_agent", "Researcher")
    if next_agent.lower() == "builder":
        return "builder"
    return "researcher"


def _route_from_researcher(state: AgentState) -> Literal["planner", "builder"]:
    """Route on research_status; only a need_replan goes back to the Planner."""
    status = state.get("research_status", ResearchStatus.READY_FOR_BUILDER.value)

    if status == ResearchStatus.NEED_REPLAN.value:
        return "planner"
    return "builder"


class _NodeFn(Protocol):
    """A graph node: `(state) -> state`, with the parameter named `state`.

    Spelled out rather than written as `Callable[[AgentState], AgentState]`
    because LangGraph's own node protocol names its parameter, and a bare
    `Callable` -- whose parameter is positional and nameless -- does not
    satisfy it. `add_node` then rejects a perfectly good wrapper.
    """

    def __call__(self, state: AgentState) -> AgentState: ...


def _tracked(name: str, node: _NodeFn) -> _NodeFn:
    """Mark a seat as working for exactly as long as its node is on the stack.

    This is the only honest source for "is this seat doing work": the graph's
    stream reports a node when it *ends*, so a light driven from there is always
    one seat behind. The `finally` is what makes it safe -- a node that raises,
    times out, or returns early on the emergency stop still puts its light out.
    """

    def run(state: AgentState) -> AgentState:
        ACTIVITY.enter(name)
        try:
            return node(state)
        finally:
            ACTIVITY.leave(name)

    return run


def create_agent_graph() -> CompiledStateGraph[AgentState, Any, AgentState, AgentState]:
    """Create and compile the 4-agent system graph.

    Graph structure:
        START → Architect → Planner → (Researcher | Builder) → Architect → END
                     ^                                              |
                     +--------- revise / need_research -------------+

    Routing logic:
    - Architect opens with `plan`, then rules on the Builder's report
    - Planner chooses Researcher (needs knowledge) or Builder (task is clear)
    - Researcher sets status: ready_for_builder | need_replan | no_relevant_knowledge
    - Builder always reports back to the Architect; it does not decide it is done
    - Stops on an `approved` verdict, or at MAX_STEPS
    """
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("architect", _tracked("architect", architect_node))
    graph_builder.add_node("planner", _tracked("planner", planner_node))
    graph_builder.add_node("researcher", _tracked("researcher", researcher_node))
    graph_builder.add_node("builder", _tracked("builder", builder_node))

    # The Architect is the entry point: nothing is planned before the
    # architectural direction and its constraints exist.
    graph_builder.set_entry_point("architect")

    graph_builder.add_conditional_edges(
        "architect",
        _route_from_architect,
        {
            "planner": "planner",
            "researcher": "researcher",
            END: END,
        },
    )

    graph_builder.add_conditional_edges(
        "planner",
        _route_from_planner,
        {
            "researcher": "researcher",
            "builder": "builder",
        },
    )

    graph_builder.add_conditional_edges(
        "researcher",
        _route_from_researcher,
        {
            "planner": "planner",
            "builder": "builder",
        },
    )

    # The Builder always reports to the Architect. It no longer decides that the
    # run is over -- that ruling belongs to the authority that set the
    # constraints, and routing back through the gate is also what counts a step.
    graph_builder.add_edge("builder", "architect")

    return graph_builder.compile()
