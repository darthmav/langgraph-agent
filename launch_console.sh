#!/bin/bash
# Launch the 3-Agent Console with frontend and auto-open browser

set -e

cd "$(dirname "$0")"

PORT=8080
URL="http://localhost:${PORT}"

echo "============================================"
echo "  Ambiguity 3-Agent Console"
echo "============================================"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env 2>/dev/null || true
fi

# Load environment safely
if [ -f ".env" ]; then
    set -a
    # shellcheck source=/dev/null
    . ./.env
    set +a
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

is_server_ready() {
    if command -v curl &> /dev/null; then
        curl -s "${URL}/api/status" > /dev/null 2>&1
    else
        python3 -c "import urllib.request; urllib.request.urlopen('${URL}/api/status', timeout=1)" > /dev/null 2>&1
    fi
}

open_browser() {
    local url="$1"
    if command -v xdg-open &> /dev/null; then
        xdg-open "$url" &
    elif command -v open &> /dev/null; then
        open "$url" &
    elif command -v python3 &> /dev/null; then
        python3 -c "import webbrowser; webbrowser.open('$url')" &
    elif command -v python &> /dev/null; then
        python -c "import webbrowser; webbrowser.open('$url')" &
    else
        echo "Please open your browser manually: $url"
        return 1
    fi
}

echo ""
echo "Starting frontend server on ${URL}..."

echo "  (Knowledge base preloads in the background; first run may take a moment)"

python serve.py > /tmp/ambiguity-console.log 2>&1 &
SERVER_PID=$!

# Wait for server to be ready (generous timeout for slower CPUs)
echo -n "  Waiting for server"
for _ in $(seq 1 60); do
    if is_server_ready; then
        echo ""
        echo "✓ Server ready"
        break
    fi
    echo -n "."
    sleep 0.5
done

if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "✗ Server failed to start. Log:"
    cat /tmp/ambiguity-console.log
    exit 1
fi

echo ""
echo "Opening browser..."
open_browser "${URL}" || true

echo ""
echo "Press Ctrl+C to stop"
echo "============================================"
echo ""

# Tail server log so output is visible, then stop with the server
trap 'kill "$SERVER_PID" 2>/dev/null || true; exit 0' INT TERM
wait "$SERVER_PID"
