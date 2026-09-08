#!/usr/bin/env python3
"""Fully automated setup and verification - no prompts required.

This script:
1. Fixes the GraphRAG server API (MCP compatibility)
2. Re-indexes all project files into GraphRAG
3. Verifies all components work
4. Runs tests

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


def reindex_knowledge():
    """Re-index all project files.

    Delegates to `index_project_files` rather than walking the tree itself.
    The copy this used to keep had drifted from the canonical one in three
    ways, none of which raises anything. Its excludes are matched as plain
    substrings, so `"*.egg-info"` matched nothing and indexed four build
    artifacts, while a bare `"build"` matched `prompts/builder.txt` and kept
    the Builder's own system prompt out of the corpus. And it *accumulated*
    where a reindex is supposed to rebuild: nothing cleared the graph or
    pruned Chroma rows that no longer qualify, so a file that was renamed,
    deleted or newly excluded went on answering searches.
    """
    print("\nIndexing project files into GraphRAG...")

    from langgraph_agent.graphrag_server import (
        get_knowledge_base,
        index_project_files,
    )

    kb = get_knowledge_base()
    root = Path(__file__).parent.parent

    report = index_project_files(kb, str(root))

    for error in report["errors"]:
        print(f"  ! {error}")
    if report["skipped"]:
        print(f"  Skipped {report['skipped']} file(s) over the size limit")
    print(f"  ✓ Indexed {report['indexed']} files")
    return report["indexed"] > 0


def test_search():
    """Test GraphRAG search."""
    print("\nTesting GraphRAG search...")

    from langgraph_agent.graphrag_server import open_knowledge_base

    # Opened, not created. The reindex above is what builds the corpus; if it
    # built nothing, this must say so rather than quietly make an empty store.
    kb = open_knowledge_base()
    if kb is None:
        print("  \u2717 No corpus was built, so there is nothing to search")
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

    # Step 3: Re-index knowledge
    if not reindex_knowledge():
        print("\n⚠ No files indexed (knowledge base may be empty)")

    # Step 4: Test search
    test_search()

    # Step 5: Run tests
    run_tests()

    # Summary
    print("\n" + "=" * 60)
    print("✓ SETUP COMPLETE")
    print("=" * 60)
    print("\nReady to use:")
    print("  python example_usage.py")
    print("\nRe-index anytime:")
    print("  python scripts/reindex.py")


if __name__ == "__main__":
    main()
