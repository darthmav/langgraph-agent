#!/usr/bin/env python3
"""Re-index knowledge base and verify it works.

Usage:
    python scripts/reindex.py
"""

from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase


def get_project_files(root: str = ".", exclude_dirs: list[str] | None = None) -> list[Path]:
    """Get all relevant project files."""
    if exclude_dirs is None:
        exclude_dirs = [
            "__pycache__", ".git", ".venv", "venv", "node_modules",
            ".pytest_cache", ".mypy_cache", "build", "dist", "*.egg-info",
            "knowledge/", "scripts/", ".qwen/"
        ]

    files = []
    root_path = Path(root)

    for pattern in ["**/*.py", "**/*.md", "**/*.txt", "**/*.rst"]:
        for file_path in root_path.glob(pattern):
            if any(excl in str(file_path) for excl in exclude_dirs):
                continue
            files.append(file_path)

    return files


def main():
    """Re-index all project files."""
    print("Initializing GraphRAG Knowledge Base...")
    kb = GraphRAGKnowledgeBase()

    print("\nScanning project files...")
    files = get_project_files()
    print(f"Found {len(files)} files to index\n")

    print("Indexing files...")
    for file_path in files:
        try:
            content = file_path.read_text(encoding="utf-8")

            if len(content) > 100_000:
                print(f"  Skipping large file: {file_path}")
                continue

            metadata = {
                "path": str(file_path),
                "type": "python" if file_path.suffix == ".py" else "markdown",
            }

            kb.add_document(str(file_path), content, metadata)
            print(f"  ✓ {file_path}")

        except Exception as e:
            print(f"  ✗ Error {file_path}: {e}")

    print(f"\n✅ Indexed {len(files)} files")
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
