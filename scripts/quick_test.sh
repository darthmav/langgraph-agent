#!/bin/bash
# Quick verification and testing for the 4-Agent System
# Usage: ./scripts/quick_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "========================================"
echo "  4-AGENT SYSTEM - QUICK TEST"
echo "========================================"

echo -e "\n► Checking dependencies..."
python -c "import langgraph, langchain_core, chromadb, sentence_transformers, networkx; print('  ✓ Core deps OK')"

echo -e "\n► Checking LLM configuration..."
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "  ℹ Using Anthropic cloud backend"
elif [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "  ℹ Using OpenAI cloud backend"
else
    echo "  ⚠ No cloud API key detected. Tests will use the StubLLM."
fi

echo -e "\n► Testing GraphRAG..."
python -c "
from langgraph_agent.graphrag_server import GraphRAGKnowledgeBase
kb = GraphRAGKnowledgeBase()
results = kb.search('Planner agent', top_k=2)
print(f'  ✓ Found {len(results)} result(s)')
"

echo -e "\n► Running tests..."
python -m pytest tests/ -v --tb=short -q

echo -e "\n========================================"
echo "  ✓ VERIFICATION COMPLETE"
echo "========================================"
echo -e "\nRun full example: python example_usage.py"
