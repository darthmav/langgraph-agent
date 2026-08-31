#!/bin/bash
# Quick verification and testing for the 3-Agent System
# Usage: ./scripts/quick_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "========================================"
echo "  3-AGENT SYSTEM - QUICK TEST"
echo "========================================"

echo -e "\n► Checking dependencies..."
python -c "import langgraph, langchain_core, chromadb, sentence_transformers, networkx, faiss; print('  ✓ All deps OK')"

echo -e "\n► Checking Ollama..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ✓ Ollama running"
else
    echo "  ⚠ Ollama not running (skip local LLM tests)"
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
