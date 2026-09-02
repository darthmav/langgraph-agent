#!/usr/bin/env python3
"""Cloud LLM end-to-end test.

This script verifies that the 4-Agent system can run using a cloud LLM.
Set ANTHROPIC_API_KEY (default) or OPENAI_API_KEY in your environment.
"""

import os
from pathlib import Path

from langgraph_agent import AgentState, create_agent_graph
from langgraph_agent.graph import RECURSION_LIMIT

# Default to Anthropic if no cloud key is configured.
if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    print("ANTHROPIC_API_KEY and OPENAI_API_KEY are not set. Skipping cloud LLM test.")
    print("Set one of them to run this test.")
    raise SystemExit(0)

graph = create_agent_graph()

# Simple file creation task
state: AgentState = {
    "goal": "Create cloud_test.txt with content 'Cloud LLM works!'",
    "messages": [],
    "architecture": "",
    "verdict": "",
    "plan": "",
    "research": "",
    "builder_report": "",
    "next_agent": "Researcher",
    "research_status": "",
    "blockers": "",
    "files_changed": [],
    "failed_verification": [],
    "expect_failures": False,
    "step_count": 0,
}

print("Running 4-Agent with cloud LLM...")
result = graph.invoke(state, {"recursion_limit": RECURSION_LIMIT})

print("\n=== RESULTS ===")
print(f"Full plan: {result.get('plan', 'N/A')!r}")
print(f"Files changed: {result.get('files_changed', [])}")
print(f"Builder report: {result.get('builder_report', 'N/A')!r}")
print(f"Messages: {result.get('messages', [])}")

# Verify file was created
if Path("cloud_test.txt").exists():
    print(f"\n✓ File created! Content: {Path('cloud_test.txt').read_text()!r}")
else:
    print("\n✗ File not created")
