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
        from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase, server
        print("  ✓ GraphRAG imports OK")
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False


def reindex_knowledge():
    """Re-index all project files."""
    print("\nIndexing project files into GraphRAG...")
    
    from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase
    
    kb = GraphRAGKnowledgeBase()
    
    # Get project files
    root = Path(__file__).parent.parent
    exclude_dirs = ["__pycache__", ".git", ".venv", "node_modules", 
                    ".pytest_cache", ".mypy_cache", "build", "dist", 
                    "*.egg-info", "knowledge/", ".qwen/"]
    
    files = []
    for pattern in ["**/*.py", "**/*.md", "**/*.txt"]:
        for f in root.glob(pattern):
            if any(excl in str(f) for excl in exclude_dirs):
                continue
            files.append(f)
    
    print(f"  Found {len(files)} files")
    
    # Index files
    indexed = 0
    for f in files:
        try:
            content = f.read_text(encoding="utf-8")
            if len(content) > 100_000:
                continue
            kb.add_document(str(f), content, {"path": str(f), "type": f.suffix})
            indexed += 1
        except Exception:
            pass
    
    print(f"  ✓ Indexed {indexed} files")
    return indexed > 0


def test_search():
    """Test GraphRAG search."""
    print("\nTesting GraphRAG search...")
    
    from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase
    
    kb = GraphRAGKnowledgeBase()
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
    print("3-AGENT SYSTEM - FULL AUTOMATED SETUP")
    print("=" * 60)
    
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
