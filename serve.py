#!/usr/bin/env python3
"""Simple frontend server for 3-Agent Console.

Usage:
    python serve.py

Then open: http://localhost:8080
"""

import json
import os
import sys
import threading
import warnings
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Upstream libraries (langsmith, chromadb) emit DeprecationWarnings on Python 3.14+
# about asyncio.iscoroutinefunction. They are harmless and outside our control,
# so suppress them before importing any third-party code.
warnings.filterwarnings(
    "ignore",
    message=".*asyncio\\.iscoroutinefunction.*",
    category=DeprecationWarning,
)

# Ensure src directory is in Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from langgraph_agent import AgentState, create_agent_graph  # noqa: E402
from langgraph_agent.config import (  # noqa: E402
    AGENT_LLM_OPTIONS,
    get_agent_model_info,
    set_agent_llm,
)
from langgraph_agent.graphrag_server import (  # noqa: E402
    get_knowledge_base,
    is_knowledge_base_indexed,
)

# Initialize graph and knowledge base
graph = create_agent_graph()
kb = None
_kb_loaded = threading.Event()


def _preload_kb() -> None:
    """Load the knowledge base in the background so the first run is fast."""
    global kb
    try:
        kb = get_knowledge_base()
    except Exception:
        # If the knowledge base cannot be loaded, the first run will attempt
        # again and fall back to LLM-only research. The server stays usable.
        pass
    finally:
        _kb_loaded.set()


# Start background preload as soon as the server module loads.
threading.Thread(target=_preload_kb, daemon=True).start()


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory="frontend", **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == '/api/status':
            # Use the lightweight index check so this endpoint never blocks on
            # loading the embedding model.
            try:
                kb_indexed, embedding_model = is_knowledge_base_indexed()
            except Exception:
                kb_indexed = False
                embedding_model = "unknown"

            # Determine a sensible display string for the active LLM.
            planner_info = get_agent_model_info("planner")
            provider = planner_info.get("provider", "anthropic")
            model = planner_info.get("model", "claude-3-5-sonnet-20241022")
            active_llm = f"{model} ({provider})"

            self.send_json({
                "llm": active_llm,
                "embedding": embedding_model,
                "graphrag": kb_indexed,
                "agents": {
                    "planner": planner_info,
                    "researcher": get_agent_model_info("researcher"),
                    "builder": get_agent_model_info("builder"),
                },
                "selected": {
                    "planner": planner_info,
                    "researcher": get_agent_model_info("researcher"),
                    "builder": get_agent_model_info("builder"),
                },
            })
        elif parsed.path == '/api/llm-options':
            self.send_json({"options": AGENT_LLM_OPTIONS})
        elif parsed.path == '/' or parsed.path == '/index.html':
            self.path = '/index.html'
            return super().do_GET()
        else:
            super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get('Content-Length', 0))
        data: dict[str, Any] = json.loads(self.rfile.read(length)) if length else {}

        if parsed.path == '/api/run':
            goal = data.get('goal', '')
            state: AgentState = {
                "goal": goal, "messages": [], "plan": "", "research": "",
                "builder_report": "", "next_agent": "Researcher",
                "research_status": "", "blockers": "", "files_changed": [],
                "step_count": 0,
            }
            result = graph.invoke(state, {"recursion_limit": 10})
            self.send_json(result)

        elif parsed.path == '/api/search':
            # Use local GraphRAG directly instead of MCP client
            global kb
            if kb is None:
                kb = get_knowledge_base()

            query = data.get("query", "")
            top_k = data.get("top_k", 5)

            try:
                results = kb.search(query, top_k)
                self.send_json({"results": results, "source": "local_graphrag"})
            except Exception as e:
                self.send_json({"error": str(e), "results": []})

        elif parsed.path == '/api/set-llm':
            agent = data.get("agent", "")
            provider = data.get("provider", "")
            model = data.get("model", "")
            if agent not in {"planner", "researcher", "builder"}:
                self.send_error(400, explain="Invalid agent")
                return
            if not provider or not model:
                self.send_error(400, explain="Provider and model required")
                return
            set_agent_llm(str(agent), str(provider), str(model))
            self.send_json({
                "ok": True,
                "agent": agent,
                "provider": provider,
                "model": model,
            })

        else:
            self.send_error(404)

    def send_json(self, obj: Any) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(obj, default=str).encode())

    def log_message(self, format: str, *args: Any) -> None:
        request_line = args[0] if args else ""
        # Suppress noisy routine status-poll logs; everything else is still logged.
        if isinstance(request_line, str) and "GET /api/status" in request_line:
            return
        print(f"[API] {request_line}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"Serving at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    server = None
    try:
        server = HTTPServer(("0.0.0.0", port), Handler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if server:
            server.shutdown()
