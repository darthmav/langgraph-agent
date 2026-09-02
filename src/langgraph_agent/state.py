"""State schema for the agent graph.

The 4-Agent System:
- Architect writes: architecture, verdict
- Planner writes: plan, next_agent
- Researcher writes: research, research_status
- Builder writes: builder_report, files_changed, blockers, failed_verification
"""

from enum import Enum
from typing import TypedDict


class ResearchStatus(str, Enum):
    """Status from Researcher to guide next steps."""

    READY_FOR_BUILDER = "ready_for_builder"
    NEED_REPLAN = "need_replan"
    NO_RELEVANT_KNOWLEDGE = "no_relevant_knowledge"


class Verdict(str, Enum):
    """The Architect's ruling, which decides where the loop goes next.

    The Architect runs twice per cycle -- once to set direction, once as the
    approval gate -- so PLAN is the opening ruling and the other three are
    what it can say about work the Builder has already reported.
    """

    PLAN = "plan"
    APPROVED = "approved"
    REVISE = "revise"
    NEED_RESEARCH = "need_research"


class AgentState(TypedDict):
    """Shared state passed between all nodes.

    Attributes:
        goal: The user's original goal/objective
        messages: Conversation history / log
        architecture: Architectural direction and constraints from the Architect
        verdict: The Architect's ruling (Verdict enum value)
        plan: Structured plan from Planner
        research: Findings from Researcher
        builder_report: Implementation report from Builder
        next_agent: Which agent runs next ("Researcher" | "Builder" | "END")
        research_status: Status from Researcher (ready_for_builder | need_replan | no_relevant_knowledge)
        blockers: What's blocking progress (set by Builder when stuck)
        files_changed: List of file paths modified by Builder
        failed_verification: Files the Builder wrote that did not run. Set by
            the Builder every pass, so it clears once a file is fixed. A
            non-empty list blocks the Architect from approving.
        step_count: Number of steps taken (for loop limit)
    """

    goal: str
    messages: list[str]
    architecture: str
    verdict: str  # Verdict enum value
    plan: str
    research: str
    builder_report: str
    next_agent: str  # "Researcher" | "Builder" | "END"
    research_status: str  # ResearchStatus enum value
    blockers: str
    files_changed: list[str]
    failed_verification: list[str]
    step_count: int
