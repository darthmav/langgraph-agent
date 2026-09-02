"""Internal state management module.

This module contains the internal implementation of agent state classes
and related types. These are exposed publicly via the parent package.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Verdict(Enum):
    """Verdict types for agent decisions.

    Attributes:
        APPROVE: The agent approves the current state or action.
        REJECT: The agent rejects the current state or action.
        DEFER: The agent defers the decision to another agent or later.
    """

    APPROVE = "approve"
    REJECT = "reject"
    DEFER = "defer"


class ResearchStatus(Enum):
    """Status values for research operations.

    Attributes:
        PENDING: Research has not started yet.
        IN_PROGRESS: Research is currently being conducted.
        COMPLETE: Research has finished successfully.
        FAILED: Research encountered an error and failed.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class AgentState:
    """Represents the current state of an agent in the system.

    This class holds all the state information needed for an agent
    to perform its duties, including conversation history, research
    results, and decision tracking.

    Attributes:
        agent_name: The name of the agent (Architect, Planner, Researcher, Builder).
        messages: List of message dictionaries in the conversation.
        research_status: Current status of any ongoing research.
        verdict: The agent's verdict on the current operation.
        context: Additional context data for the agent.
        errors: List of error messages encountered during execution.

    Example:
        >>> state = AgentState(agent_name="Builder")
        >>> state.research_status = ResearchStatus.IN_PROGRESS
    """

    agent_name: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    research_status: ResearchStatus = ResearchStatus.PENDING
    verdict: Optional[Verdict] = None
    context: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            role: The role of the message sender (e.g., 'user', 'assistant').
            content: The message content.
        """
        self.messages.append({"role": role, "content": content})

    def add_error(self, error_message: str) -> None:
        """Record an error that occurred during execution.

        Args:
            error_message: The error message to record.
        """
        self.errors.append(error_message)

    def has_errors(self) -> bool:
        """Check if any errors have been recorded.

        Returns:
            True if there are recorded errors, False otherwise.
        """
        return len(self.errors) > 0
