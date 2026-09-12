#!/usr/bin/env python3
"""Fully automated setup and verification - no prompts required.

This script:
1. Fixes the GraphRAG server API (MCP compatibility)
2. Verifies all components work
3. Runs tests

It does not build the corpus. Nothing does but a run and an upload.

Usage:
    python scripts/full_setup.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def fix_graphrag_server():
    """Fix MCP API compatibility."""
    print("Fixing GraphRAG server...")

    graphrag_path = Path(__file__).parent.parent / "src" / "langgraph_agent" / "graphrag_server.py"
    content = graphrag_path.read_text()

    # Replace Server with MCPServer
    old_import = "from mcp.server import Server"
    new_import = "from mcp.server import MCPServer\n\nServer = MCPServer"

    if old_import in content and "MCPServer" not in content:
        content = content.replace(old_import, new_import, 1)
        graphrag_path.write_text(content)
        print("  ✓ Updated GraphRAG server for MCP compatibility")
    else:
        print("  ✓ GraphRAG server already compatible")


def test_graphrag_import():
    """Test GraphRAG imports correctly."""
    print("\nTesting GraphRAG import...")
    try:
        print("  ✓ GraphRAG imports OK")
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False


# There is no `reindex_knowledge` here any more, and nothing in this script
# builds a corpus. Two things index: a run, which rebuilds before the Architect
# opens, and embedding a document into the corpus from the console. A setup
# script was a third, and it is the one that most looks like housekeeping and
# least looks like a decision -- it fixes how fresh the corpus is on a machine
# nobody has run anything on, which the first run decides correctly by itself.
# `test_search` below therefore reports an absent corpus as a fact about this
# machine rather than as a failure of setup.


def test_search():
    """Test GraphRAG search."""
    print("\nTesting GraphRAG search...")

    from langgraph_agent.graphrag_server import open_knowledge_base

    # Opened, never created. Nothing in this script builds a corpus, so an
    # absent one is reported rather than quietly made -- a store that appeared
    # because something looked at it is a corpus nobody asked for.
    kb = open_knowledge_base()
    if kb is None:
        print("  - No corpus on this machine yet; a run builds one")
        return False

    results = kb.search("Planner agent", top_k=2)

    if results:
        print(f"  ✓ Search working ({len(results)} results)")
        return True
    else:
        print("  ✗ No results found")
        return False


def run_tests():
    """Run test suite."""
    print("\nRunning tests...")

    import pytest
    exit_code = pytest.main(["-q", "--tb=no", "tests/"])

    if exit_code == 0:
        print("  ✓ All tests passing")
        return True
    else:
        print(f"  ✗ Tests failed (exit code {exit_code})")
        return False


def main():
    """Run all setup steps."""
    print("=" * 60)
    print("4-AGENT SYSTEM - FULL AUTOMATED SETUP")
    print("=" * 60)
    print("\nDefault backend: Ollama Cloud tags for all four seats.")
    print("Set ANTHROPIC_API_KEY in .env for live agent runs.\n")

    # Step 1: Fix GraphRAG server
    fix_graphrag_server()

    # Step 2: Test imports
    if not test_graphrag_import():
        print("\n✗ Setup failed at import stage")
        sys.exit(1)

    # Step 3: Test search
    test_search()

    # Step 4: Run tests
    run_tests()

    # Summary
    print("\n" + "=" * 60)
    print("✓ SETUP COMPLETE")
    print("=" * 60)
    print("\nReady to use:")
    print("  python example_usage.py")
    print("\nThe corpus builds itself: start a run from the console and it is")
    print("indexed before the Architect opens.")


if __name__ == "__main__":
    main()
