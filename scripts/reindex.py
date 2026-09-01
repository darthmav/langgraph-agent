#!/usr/bin/env python3
"""Re-index knowledge base and verify it works.

Usage:
    python scripts/reindex.py

The embedding model runs on-device for GraphRAG and does not require an API key.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph_agent.graphrag_server import (
    GraphRAGKnowledgeBase,
    index_project_files,
    iter_project_files,
)


def main():
    """Re-index all project files."""
    print("Initializing GraphRAG Knowledge Base...")
    kb = GraphRAGKnowledgeBase()

    print("\nScanning project files...")
    files = iter_project_files()
    print(f"Found {len(files)} files to index\n")

    print("Indexing files...")
    report = index_project_files(kb)

    for failure in report["errors"]:
        print(f"  \u2717 {failure}")

    print(f"\n\u2705 Indexed {report['indexed']} files ({report['skipped']} skipped)")
    if report["errors"]:
        print(f"\u26a0\ufe0f  {len(report['errors'])} file(s) failed to index")
    print(
        f"Graph: {report['total_nodes']} nodes, {report['total_edges']} edges, "
        f"{report['total_documents']} documents"
    )
    print(f"Knowledge base stored in: {kb.persist_dir}")

    # Test queries
    print("\n" + "=" * 60)
    print("TESTING SEARCH")
    print("=" * 60)

    test_queries = [
        "What is the Planner agent?",
        "How does GraphRAG integration work?",
        "What are the node types?",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = kb.search(query, top_k=2)

        if results:
            for i, result in enumerate(results[:1], 1):
                print(f"  {i}. Score: {result['score']:.3f} | {Path(result['id']).name}")
        else:
            print("  No results found")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
