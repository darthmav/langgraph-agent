#!/usr/bin/env python3
"""Fully automated verification - no prompts, just run and report.

Usage:
    python scripts/auto_verify.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Run all verification steps silently."""
    results = []

    # 1. Dependencies
    try:
        import langgraph, langchain_core, chromadb, sentence_transformers, networkx, faiss
        results.append("✓ Dependencies OK")
    except ImportError as e:
        results.append(f"✗ Missing dependency: {e}")
        print("\n".join(results))
        sys.exit(1)

    # 2. Import agent modules
    try:
        from langgraph_agent import state, nodes, graph, config, mcp_client
        results.append("✓ Agent modules import OK")
    except Exception as e:
        results.append(f"✗ Import error: {e}")
        print("\n".join(results))
        sys.exit(1)

    # 3. Test GraphRAG
    try:
        from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase
        kb = GraphRAGKnowledgeBase()
        kb.search("test", top_k=1)
        results.append("✓ GraphRAG working")
    except Exception as e:
        results.append(f"✗ GraphRAG error: {e}")

    # 4. Run tests
    try:
        import pytest
        exit_code = pytest.main(["-q", "--tb=no", "tests/"])
        if exit_code == 0:
            results.append("✓ All tests passing")
        else:
            results.append(f"✗ Some tests failed")
    except Exception as e:
        results.append(f"✗ Test error: {e}")

    # Summary
    print("\n" + "=" * 50)
    print("3-AGENT SYSTEM STATUS")
    print("=" * 50)
    for r in results:
        print(r)
    print("=" * 50)

    # Quick usage example
    print("\nQuick test:")
    print("  python example_usage.py")
    print("\nRe-index knowledge:")
    print("  python scripts/reindex.py")


if __name__ == "__main__":
    main()
