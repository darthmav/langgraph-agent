#!/usr/bin/env python3
"""Automated verification and testing for the 4-Agent System.

Usage:
    python scripts/verify_and_test.py                # read-only checks
    python scripts/verify_and_test.py --all          # including the ones that write

With no flags this runs only the steps that leave the project alone: the
dependency and seat checks, a GraphRAG search, and the test suite.

Two steps are held back from that default because they are not read-only.
`--index` rebuilds the corpus, and `--run-example` is a live agent run that
writes files into the repository -- so plain `verify_and_test.py` used to
reindex the knowledge base and leave several new files behind, which is not
what "verify" reads as. Ask for them by name, or with `--all`.
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


def check_seats() -> bool:
    """Report what each seat will actually run.

    This used to look for ANTHROPIC_API_KEY / OPENAI_API_KEY and warn when it
    found neither -- a question nobody here asked. No seat uses either provider
    by default, so a correctly configured machine was told live agent runs were
    impossible while all four seats sat live on Ollama cloud.

    `get_agent_status` is what it reads instead, because key presence is not
    liveness: a key can authenticate and the seat still be unusable, and a seat
    with no key at all silently becomes StubLLM while every static check keeps
    reporting the configured model. It is also the only place `stubbed` and
    `live` are kept apart, and they are different failures -- a stubbed seat
    completes the run with canned text, a failing seat kills it.
    """
    print_step("Checking seats")

    # Imported here, not at module scope: step 1 is what reports a missing
    # dependency in readable form, and a top-level import would crash ahead of
    # it with a traceback instead.
    try:
        from langgraph_agent.config import AGENTS, get_agent_status
    except Exception as e:  # pragma: no cover - step 1 already reports this
        print(f"  ✗ Could not read the seat configuration: {e}")
        return False

    all_live = True
    for agent in AGENTS:
        seat = get_agent_status(agent)
        if seat["live"]:
            mark, note = "✓", ""
        else:
            all_live = False
            mark = "○" if seat["stubbed"] else "✗"
            note = f"   !! {seat['badge']}: {seat['reason']}"
        print(
            f"  {mark} {agent:<11}{seat['model']:<24}"
            f"{seat['provider']:<11}{seat['placement']}{note}"
        )

    if all_live:
        print("\n✓ All seats live")
    else:
        print(
            "\n  ⚠ Not every seat can run. A stubbed seat (○) completes the run "
            "with canned text; a failing seat (✗) kills it."
        )
    return all_live


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
        from langgraph_agent.graphrag_server import open_knowledge_base

        print_step("Loading knowledge base")
        # Opened, not created. Checking that search works must not leave a
        # corpus behind on a machine that had none -- an empty store then
        # reports itself as a knowledge base to everything that looks.
        kb = open_knowledge_base()
        if kb is None:
            print("  \u2717 No corpus has been indexed; nothing to search.")
            print("    Run: python scripts/index_knowledge.py")
            return False

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

        # `serve` and `spectral_graph` sit at the project root and are not part
        # of the installed distribution, so three test modules import them by
        # name. `python -m pytest` works because it puts the working directory
        # on sys.path first; pytest.main() from here does not -- sys.path[0] is
        # scripts/ -- so those three failed to collect and this step reported
        # "Some tests failed" against a tree where all 394 pass. A false failure
        # from the verification runner is worse than none: it is the reading
        # someone acts on.
        root = str(Path(__file__).parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)

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
        help="Rebuild the knowledge base (writes to knowledge/)",
    )
    parser.add_argument(
        "--test-graphrag",
        action="store_true",
        help="Test GraphRAG search",
    )
    parser.add_argument(
        "--run-example",
        action="store_true",
        help="Run a live agent run (writes files into the repository)",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run test suite",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every step, including the two that write",
    )

    args = parser.parse_args()

    # No flags runs the read-only steps. The two that write are reached only by
    # naming them or by --all: a verification script that reindexes the corpus
    # and commits a live agent run to the working tree on a bare invocation is
    # a trap, and the caller has no way to find out before it happens.
    read_only = args.all or not any(
        [args.index, args.test_graphrag, args.run_example, args.run_tests]
    )

    # Every step runs even after one fails -- one broken step should not hide
    # the state of the rest -- but the exit code has to carry the result. It
    # was 0 unconditionally, so a caller reading only the status got "verified"
    # from a run that had just printed four warnings.
    failed: list[str] = []

    print_header("4-AGENT SYSTEM VERIFICATION")

    # Step 1: Always check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Non-blocking, as before: a stubbed seat still completes a run, and the
    # steps below are worth reading either way. It reports; it does not judge.
    check_seats()

    # Step 2: Index knowledge base
    if args.index or args.all:
        if not index_knowledge_base():
            print("\n⚠ Indexing failed, continuing anyway...")
            failed.append("indexing")

    # Step 3: Test GraphRAG
    if args.test_graphrag or read_only:
        if not test_graphrag_search():
            print("\n⚠ GraphRAG test failed, continuing anyway...")
            failed.append("GraphRAG search")

    # Step 4: Run example
    if args.run_example or args.all:
        if not run_example_usage():
            print("\n⚠ Example failed, continuing anyway...")
            failed.append("example run")

    # Step 5: Run tests
    if args.run_tests or read_only:
        if not run_tests():
            print("\n⚠ Some tests failed")
            failed.append("tests")

    print_header("VERIFICATION COMPLETE")

    if failed:
        print("\n\u2717 Failed: " + ", ".join(failed))
        sys.exit(1)

    print("\nNext steps:")
    print("  - Review output above for any issues")
    print("  - Any seat not live above: start Ollama and `ollama signin` for")
    print("    a :cloud tag, or set that provider's key for an Anthropic/OpenAI seat")
    print("  - Run: python example_usage.py")


if __name__ == "__main__":
    main()
