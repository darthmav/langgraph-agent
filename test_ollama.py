#!/usr/bin/env python3
"""Minimal local Ollama LLM test using the documented default model."""

import os
from langgraph_agent import create_agent_graph, AgentState

# Use the local model recommended by the 3-Agent System guide.
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
os.environ["OLLAMA_MODEL"] = "qwen3:8b"

graph = create_agent_graph()

# Simple file creation task
state: AgentState = {
    "goal": "Create ollama_test.txt with content 'Local Ollama LLM works!'",
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

print("Running 3-Agent with local Ollama LLM (qwen3:8b)...")
result = graph.invoke(state, {"recursion_limit": 5})

print("\n=== RESULTS ===")
print(f"Full plan: {result.get('plan', 'N/A')!r}")
print(f"Files changed: {result.get('files_changed', [])}")
print(f"Builder report: {result.get('builder_report', 'N/A')!r}")
print(f"Messages: {result.get('messages', [])}")

# Verify file was created
from pathlib import Path
if Path("ollama_test.txt").exists():
    print(f"\n✓ File created! Content: {Path('ollama_test.txt').read_text()!r}")
else:
    print("\n✗ File not created")
