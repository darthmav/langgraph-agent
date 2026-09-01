#!/usr/bin/env python3
"""Index project files into the GraphRAG knowledge base.

Usage:
    python scripts/index_knowledge.py

This will:
1. Scan the project for .py, .md, .txt files
2. Extract content and metadata
3. Add to Chroma vector store + NetworkX graph

The embedding model runs on-device for GraphRAG and does not require an API key.
"""

from pathlib import Path

from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase


def get_project_files(root: str = ".", exclude_dirs: list[str] | None = None) -> list[Path]:
    """Get all relevant project files.

    Args:
        root: Project root directory
        exclude_dirs: Directories to exclude

    Returns:
        List of file paths
    """
    if exclude_dirs is None:
        exclude_dirs = [
            "__pycache__", ".git", ".venv", "venv", "node_modules",
            ".pytest_cache", ".mypy_cache", "build", "dist", "*.egg-info"
        ]

    files = []
    root_path = Path(root)

    for pattern in ["**/*.py", "**/*.md", "**/*.txt", "**/*.rst"]:
        for file_path in root_path.glob(pattern):
            # Check if any exclude dir is in path
            if any(excl in str(file_path) for excl in exclude_dirs):
                continue
            files.append(file_path)

    return files


def index_file(kb: GraphRAGKnowledgeBase, file_path: Path) -> None:
    """Index a single file into the knowledge base.

    Args:
        kb: Knowledge base instance
        file_path: Path to file
    """
    try:
        content = file_path.read_text(encoding="utf-8")

        # Skip very large files (>100KB)
        if len(content) > 100_000:
            print(f"  Skipping large file: {file_path} ({len(content)} bytes)")
            return

        # Create metadata
        metadata = {
            "path": str(file_path),
            "type": "python" if file_path.suffix == ".py" else "markdown" if file_path.suffix == ".md" else "text",
            "size": len(content),
        }

        # Add to knowledge base
        doc_id = str(file_path)
        kb.add_document(doc_id, content, metadata)

        print(f"  Indexed: {file_path} ({len(content)} bytes)")

    except Exception as e:
        print(f"  Error indexing {file_path}: {e}")


def main():
    """Index all project files."""
    print("Initializing GraphRAG Knowledge Base...")
    kb = GraphRAGKnowledgeBase()

    print("\nScanning project files...")
    files = get_project_files()
    print(f"Found {len(files)} files to index\n")

    print("Indexing files...")
    for file_path in files:
        index_file(kb, file_path)

    print(f"\n✅ Indexed {len(files)} files")
    print(f"Knowledge base stored in: {kb.persist_dir}")

    # Test a query
    print("\nTesting search...")
    test_query = "What is the Planner agent?"
    results = kb.search(test_query, top_k=3)

    print(f"\nQuery: '{test_query}'")
    print(f"Found {len(results)} results\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['id']}")
        print(f"   Score: {result['score']:.3f}")
        print(f"   Content: {result['content'][:100]}...")
        if result.get("related_entities"):
            print(f"   Related: {result['related_entities'][:3]}")
        print()


if __name__ == "__main__":
    main()
