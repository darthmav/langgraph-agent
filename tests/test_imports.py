"""Verification script for langgraph_agent package imports.

This script validates that the package structure is correct and all
public interfaces can be imported properly. It ensures:

1. The main package imports work correctly
2. Only public interfaces are exported (no _internal modules)
3. Custom exceptions inherit from the base exception class
4. Type hints and docstrings are present on public API

Run this script with: python -m pytest tests/test_imports.py -v
Or directly: python tests/test_imports.py
"""

import sys


def test_package_imports():
    """Test that the main package can be imported."""
    import langgraph_agent

    assert hasattr(langgraph_agent, "__version__")
    assert hasattr(langgraph_agent, "__all__")
    print(f"✓ Package version: {langgraph_agent.__version__}")
    print(f"✓ Exported symbols: {langgraph_agent.__all__}")


def test_public_api_exports():
    """Test that only public interfaces are exported."""
    import langgraph_agent

    # The exported surface is pinned, not sampled: a symbol added to or dropped
    # from __all__ is a change to the package's public API and has to be made
    # here as well.
    expected_public = {
        "create_agent_graph",
        "AgentState",
        "ResearchStatus",
        "Verdict",
        "CompiledStateGraph",
        "get_agent_graph",
    }

    all_symbols = set(langgraph_agent.__all__)
    assert all_symbols == expected_public, (
        f"__all__ drifted: added {sorted(all_symbols - expected_public)}, "
        f"removed {sorted(expected_public - all_symbols)}"
    )

    # Verify no internal modules are exported
    for symbol in all_symbols:
        assert not symbol.startswith("_"), f"Internal symbol '{symbol}' should not be exported"

    print(f"✓ All exported symbols are public: {all_symbols}")


def test_create_agent_graph_import():
    """Test that create_agent_graph can be imported and is callable."""
    from langgraph_agent import create_agent_graph

    assert callable(create_agent_graph)
    print("✓ create_agent_graph is callable")


def test_state_types_import():
    """Test that state types can be imported."""
    from langgraph_agent import AgentState, ResearchStatus, Verdict

    # Verify AgentState is a type
    assert AgentState is not None

    # Verify enums have expected values
    assert hasattr(ResearchStatus, "READY_FOR_BUILDER")
    assert hasattr(Verdict, "PLAN")
    assert hasattr(Verdict, "APPROVED")

    print("✓ AgentState, ResearchStatus, Verdict imported successfully")


def test_exceptions_base_class():
    """Test that all custom exceptions inherit from LangGraphAgentError."""
    from langgraph_agent.exceptions import (
        ConfigurationError,
        GraphError,
        InferenceError,
        LangGraphAgentError,
        StateError,
        ToolError,
    )

    # Verify base class exists
    assert LangGraphAgentError is not None
    assert issubclass(LangGraphAgentError, Exception)

    # Verify all custom exceptions inherit from base
    assert issubclass(ConfigurationError, LangGraphAgentError)
    assert issubclass(StateError, LangGraphAgentError)
    assert issubclass(ToolError, LangGraphAgentError)
    assert issubclass(GraphError, LangGraphAgentError)
    assert issubclass(InferenceError, LangGraphAgentError)

    print("✓ All exceptions inherit from LangGraphAgentError")


def test_exception_instantiation():
    """Test that exceptions can be instantiated with proper arguments."""
    from langgraph_agent.exceptions import (
        ConfigurationError,
        LangGraphAgentError,
    )

    # Test base exception
    base_error = LangGraphAgentError("Test message")
    assert "Test message" in str(base_error)

    base_error_with_details = LangGraphAgentError("Test message", "Additional details")
    assert "Test message" in str(base_error_with_details)
    assert "Additional details" in str(base_error_with_details)

    # Test derived exception
    config_error = ConfigurationError("Missing API key", "Check .env file")
    assert "Missing API key" in str(config_error)

    print("✓ Exceptions can be instantiated correctly")


def test_get_agent_graph():
    """Test that get_agent_graph helper function exists."""
    from langgraph_agent import get_agent_graph

    assert callable(get_agent_graph)
    print("✓ get_agent_graph helper function exists")


def test_no_internal_imports_in_public_api():
    """Verify that _internal modules are not directly accessible from public API."""
    import langgraph_agent

    # hasattr is not the test: importing langgraph_agent.exceptions binds
    # _internal as an attribute of the parent package, so whether it is there
    # depends on what ran first. What __all__ says does not.
    assert "_internal" not in langgraph_agent.__all__
    for symbol in langgraph_agent.__all__:
        obj = getattr(langgraph_agent, symbol)
        module = getattr(obj, "__module__", "")
        assert "_internal" not in module, f"'{symbol}' is exported straight out of {module}"

    print("✓ _internal module is not exposed in public API")


def test_docstrings_present():
    """Test that public API has docstrings."""
    import langgraph_agent

    # Check module docstring
    assert langgraph_agent.__doc__ is not None
    assert len(langgraph_agent.__doc__) > 50

    # Check function docstrings
    from langgraph_agent import create_agent_graph, get_agent_graph

    assert create_agent_graph.__doc__ is not None
    assert get_agent_graph.__doc__ is not None

    print("✓ Docstrings are present on public API")


if __name__ == "__main__":
    # Run all tests when executed directly
    print("Running langgraph_agent package import verification...\n")

    tests = [
        test_package_imports,
        test_public_api_exports,
        test_create_agent_graph_import,
        test_state_types_import,
        test_exceptions_base_class,
        test_exception_instantiation,
        test_get_agent_graph,
        test_no_internal_imports_in_public_api,
        test_docstrings_present,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)
    else:
        print("All verification tests passed!")
        sys.exit(0)
