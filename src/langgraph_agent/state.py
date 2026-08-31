"""State schema for the agent graph.

Matches the 3-Agent System specification:
- Planner writes: plan, next_agent
- Researcher writes: research, research_status
- Builder writes: builder_report, files_changed, blockers
"""

from enum import Enum
from typing import TypedDict


class ResearchStatus(str, Enum):
    """Status from Researcher to guide next steps."""

    READY_FOR_BUILDER = "ready_for_builder"
    NEED_REPLAN = "need_replan"
    NO_RELEVANT_KNOWLEDGE = "no_relevant_knowledge"


class AgentState(TypedDict):
    """Shared state passed between all nodes.

    As specified in the 3-Agent System documentation:

    Attributes:
        goal: The user's original goal/objective
        messages: Conversation history / log
        plan: Structured plan from Planner
        research: Findings from Researcher
        builder_report: Implementation report from Builder
        next_agent: Which agent runs next ("Researcher" | "Builder" | "END")
        research_status: Status from Researcher (ready_for_builder | need_replan | no_relevant_knowledge)
        blockers: What's blocking progress (set by Builder when stuck)
        files_changed: List of file paths modified by Builder
        step_count: Number of steps taken (for loop limit)
    """

    goal: str
    messages: list[str]
    plan: str
    research: str
    builder_report: str
    next_agent: str  # "Researcher" | "Builder" | "END"
    research_status: str  # ResearchStatus enum value
    blockers: str
    files_changed: list[str]
    step_count: int
