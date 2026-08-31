#!/usr/bin/env python3
"""Minimal cloud LLM test - uses cloud credits efficiently."""

import os
from langgraph_agent import create_agent_graph, AgentState

# Use cloud LLM
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["OLLAMA_MODEL"] = "qwen3.5:397b-cloud"

graph = create_agent_graph()

# Simple file creation task
state: AgentState = {
    "goal": "Create cloud_test.txt with content 'Cloud LLM works!'",
    "messages": [],
    "plan": "",
    "research": "",
    "builder_report": "",
    "next_agent": "Researcher",
    "research_status": "",
    "blockers": "",
    "files_changed": [],
    "step_count": 0,
}

print("Running 3-Agent with cloud LLM (qwen3.5:397b-cloud)...")
result = graph.invoke(state, {"recursion_limit": 5})

print("\n=== RESULTS ===")
print(f"Full plan: {result.get('plan', 'N/A')!r}")
print(f"Files changed: {result.get('files_changed', [])}")
print(f"Builder report: {result.get('builder_report', 'N/A')!r}")
print(f"Messages: {result.get('messages', [])}")

# Verify file was created
from pathlib import Path
if Path("cloud_test.txt").exists():
    print(f"\n✓ File created! Content: {Path('cloud_test.txt').read_text()!r}")
else:
    print("\n✗ File not created")
