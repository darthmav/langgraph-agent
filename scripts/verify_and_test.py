#!/usr/bin/env python3
"""Automated verification and testing for the 4-Agent System.

Usage:
    python scripts/verify_and_test.py [--index] [--test-graphrag] [--run-example]

This script:
1. Verifies all dependencies are installed
2. Optionally indexes knowledge base
3. Optionally tests GraphRAG search
4. Optionally runs the example usage
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_step(text: str) -> None:
    """Print a formatted step."""
    print(f"\n► {text}")


def check_dependencies() -> bool:
    """Verify all required dependencies are installed."""
    print_header("STEP 1: Checking Dependencies")

    required = {
        "langgraph": "LangGraph",
        "langchain_core": "LangChain Core",
        "mcp": "MCP",
        "chromadb": "ChromaDB",
        "sentence_transformers": "Sentence Transformers",
        "networkx": "NetworkX",
        "dotenv": "python-dotenv",
    }

    optional = {
        "langchain_openai": "LangChain OpenAI",
        "langchain_anthropic": "LangChain Anthropic",
    }

    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} (module: {module})")
            missing.append(name)

    for module, name in optional.items():
        try:
            __import__(module)
            print(f"  ✓ {name} (optional)")
        except ImportError:
            print(f"  ○ {name} (optional, not installed)")

    if missing:
        print(f"\n✗ Missing dependencies: {', '.join(missing)}")
        print("\nInstall with: pip install -e '.[dev]'")
        return False

    print("\n✓ All dependencies installed")
    return True


def check_cloud_llm() -> bool:
    """Check whether a cloud LLM API key is configured."""
    print_step("Checking cloud LLM configuration")

    if os.getenv("ANTHROPIC_API_KEY"):
        print("  ✓ Anthropic API key configured")
        return True
    if os.getenv("OPENAI_API_KEY"):
        print("  ✓ OpenAI API key configured")
        return True

    print("  ⚠ No cloud API key configured. Live agent runs require ANTHROPIC_API_KEY or OPENAI_API_KEY.")
    return False


def index_knowledge_base() -> bool:
    """Index project files into GraphRAG."""
    print_header("STEP 2: Indexing Knowledge Base")

    script_path = Path(__file__).parent / "index_knowledge.py"
    if not script_path.exists():
        print(f"  ✗ Index script not found: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=120,
        )

        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  ✗ Indexing timed out (>2 minutes)")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_graphrag_search() -> bool:
    """Test GraphRAG search functionality."""
    print_header("STEP 3: Testing GraphRAG Search")

    try:
        from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase

        print_step("Loading knowledge base")
        kb = GraphRAGKnowledgeBase()

        print_step("Testing search queries")

        queries = [
            "What is the Planner agent?",
            "How does GraphRAG work?",
            "What are the system requirements?",
        ]

        for query in queries:
            print(f"\n  Query: '{query}'")
            results = kb.search(query, top_k=2)

            if results:
                print(f"    Found {len(results)} result(s)")
                for result in results[:1]:
                    print(f"    - Score: {result['score']:.3f}")
                    print(f"      Source: {result['id']}")
            else:
                print("    No results found")

        print("\n✓ GraphRAG search working")
        return True

    except Exception as e:
        print(f"\n✗ GraphRAG test failed: {e}")
        return False


def run_example_usage() -> bool:
    """Run the example usage script."""
    print_header("STEP 4: Running Example Usage")

    example_path = Path(__file__).parent.parent / "example_usage.py"
    if not example_path.exists():
        print(f"  ✗ Example script not found: {example_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(example_path)],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        print(result.stdout)
        if result.stderr:
            print("Output:", result.stderr)

        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  ✗ Example timed out (>5 minutes)")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def run_tests() -> bool:
    """Run the test suite."""
    print_header("STEP 5: Running Tests")

    try:
        import pytest

        print_step("Running pytest")
        exit_code = pytest.main(
            [
                "-v",
                "--tb=short",
                "tests/",
            ]
        )

        return exit_code == 0
    except Exception as e:
        print(f"  ✗ Error running tests: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify and test the 4-Agent System"
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Index knowledge base",
    )
    parser.add_argument(
        "--test-graphrag",
        action="store_true",
        help="Test GraphRAG search",
    )
    parser.add_argument(
        "--run-example",
        action="store_true",
        help="Run example usage",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run test suite",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all steps",
    )

    args = parser.parse_args()

    # Default to --all if no specific flags
    run_all = args.all or not any(
        [args.index, args.test_graphrag, args.run_example, args.run_tests]
    )

    print_header("4-AGENT SYSTEM VERIFICATION")

    # Step 1: Always check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Check cloud LLM (non-blocking)
    check_cloud_llm()

    # Step 2: Index knowledge base
    if args.index or run_all:
        if not index_knowledge_base():
            print("\n⚠ Indexing failed, continuing anyway...")

    # Step 3: Test GraphRAG
    if args.test_graphrag or run_all:
        if not test_graphrag_search():
            print("\n⚠ GraphRAG test failed, continuing anyway...")

    # Step 4: Run example
    if args.run_example or run_all:
        if not run_example_usage():
            print("\n⚠ Example failed, continuing anyway...")

    # Step 5: Run tests
    if args.run_tests or run_all:
        if not run_tests():
            print("\n⚠ Some tests failed")

    print_header("VERIFICATION COMPLETE")
    print("\nNext steps:")
    print("  - Review output above for any issues")
    print("  - Set ANTHROPIC_API_KEY in .env for live agent runs")
    print("  - Run: python example_usage.py")


if __name__ == "__main__":
    main()
