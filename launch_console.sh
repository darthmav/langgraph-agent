#!/bin/bash
# Launch the 3-Agent Console with frontend

set -e

cd "$(dirname "$0")"

echo "============================================"
echo "  Ambiguity 3-Agent Console"
echo "============================================"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env 2>/dev/null || true
fi

# Load environment
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
    echo "✓ Loaded .env"
    echo "  LLM: ${OLLAMA_MODEL:-${OPENAI_MODEL:-unknown}}"
fi

# Check Ollama
if command -v ollama &> /dev/null; then
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama running"
    else
        echo "⚠️  Ollama not running. Start with: ollama serve"
    fi
fi

echo ""
echo "Starting frontend server..."
echo "Open: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop"
echo "============================================"
echo ""

python serve.py
