#!/usr/bin/env python3
"""Example usage of the 4-Agent System.

Demonstrates:
- Architect → Planner → Researcher → Builder → Architect flow
- State injection on every turn
- Strict output format parsing
- Cloud LLM support (Anthropic by default, OpenAI optional)
"""

import os

from langgraph_agent import AgentState, create_agent_graph


def run_example(goal: str, max_steps: int = 8):
    """Run the 4-agent system with a goal.

    Args:
        goal: The goal to achieve
        max_steps: Maximum steps before stopping (default 8)
    """
    # Cloud-only default: Anthropic. Set ANTHROPIC_API_KEY in your environment.
    # For optional OpenAI execution, set OPENAI_API_KEY and OPENAI_MODEL.
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("Note: ANTHROPIC_API_KEY and OPENAI_API_KEY are not set. The LLM will fall back to the StubLLM.")
        print("Set ANTHROPIC_API_KEY to use the default cloud provider, or OPENAI_API_KEY for the optional provider.")

    graph = create_agent_graph()

    # Initialize state per the 4-Agent System specification
    initial_state: AgentState = {
        "goal": goal,
        "messages": [],
        "architecture": "",
        "verdict": "",
        "plan": "",
        "research": "",
        "builder_report": "",
        "next_agent": "Researcher",  # Default, Planner will set
        "research_status": "",
        "blockers": "",
        "files_changed": [],
        "failed_verification": [],
        "expect_failures": False,
        "step_count": 0,
    }

    print(f"\n{'=' * 70}")
    print(f"4-Agent System - Goal: {goal}")
    print(f"{'=' * 70}\n")

    # Run the graph
    result = graph.invoke(initial_state)

    # Display results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n## Plan")
    print(result.get("plan", "(empty)") or "(empty)")

    print("\n## Research Findings")
    print(result.get("research", "(empty)") or "(empty)")

    print("\n## Builder Report")
    print(result.get("builder_report", "(empty)") or "(empty)")

    print("\n## Files Changed")
    files = result.get("files_changed", [])
    if files:
        for f in files:
            print(f"  - {f}")
    else:
        print("  (none)")

    print("\n## Blockers")
    blockers = result.get("blockers", "")
    print(blockers if blockers else "  (none)")

    print("\n## Execution Log")
    for msg in result.get("messages", []):
        print(f"  - {msg}")

    print(f"\n## Steps Taken: {result.get('step_count', 0)}")

    return result


if __name__ == "__main__":
    # Example 1: Simple build task (should skip research)
    run_example("Create a hello.txt file containing 'Hello World'")

    # Example 2: Research-heavy task
    print("\n\n")
    run_example("Research the best practices for Python async error handling")
