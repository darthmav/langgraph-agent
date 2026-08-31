#!/usr/bin/env python3
"""Example usage of the 3-Agent System.

Demonstrates:
- Planner → Researcher → Builder flow
- State injection on every turn
- Strict output format parsing
- Ollama (local) or OpenAI (cloud) LLM support
"""

import os

from langgraph_agent import AgentState, create_agent_graph


def run_example(goal: str, max_steps: int = 8):
    """Run the 3-agent system with a goal.

    Args:
        goal: The goal to achieve
        max_steps: Maximum steps before stopping (default 8)
    """
    # Set Ollama as default for local execution
    if not os.getenv("OPENAI_API_KEY"):
        print("No OPENAI_API_KEY found, using Ollama (ensure 'ollama pull qwen3:8b' first)")
        os.environ["OLLAMA_BASE_URL"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    graph = create_agent_graph()

    # Initialize state per the 3-Agent System specification
    initial_state: AgentState = {
        "goal": goal,
        "messages": [],
        "plan": "",
        "research": "",
        "builder_report": "",
        "next_agent": "Researcher",  # Default, Planner will set
        "research_status": "",
        "blockers": "",
        "files_changed": [],
        "step_count": 0,
    }

    print(f"\n{'=' * 70}")
    print(f"3-Agent System - Goal: {goal}")
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
