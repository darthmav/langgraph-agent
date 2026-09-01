#!/usr/bin/env python3
"""Frontend server for the 4-Agent Console.

Usage:
    python serve.py

Then open: http://localhost:8080

The console talks to a single `POST /rpc` endpoint taking {method, params};
the `/api/*` routes are thin compatibility wrappers over the same dispatch.
"""

import json
import os
import sys
import threading
import time
import warnings
from collections.abc import Callable
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
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
    AGENTS,
    get_agent_status,
    list_ollama_models,
    set_agent_llm,
)
from langgraph_agent.graph import RECURSION_LIMIT  # noqa: E402
from langgraph_agent.graphrag_server import (  # noqa: E402
    GraphRAGKnowledgeBase,
    get_knowledge_base,
    index_project_files,
    is_knowledge_base_indexed,
)

# Initialize graph and knowledge base
graph = create_agent_graph()
kb: GraphRAGKnowledgeBase | None = None
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


def _kb() -> GraphRAGKnowledgeBase:
    """The knowledge base, loading it on demand if the preload has not landed."""
    global kb
    if kb is None:
        kb = get_knowledge_base()
    return kb


# --------------------------------------------------------------------------
# RPC methods
# --------------------------------------------------------------------------


def rpc_rag_stats(_: dict[str, Any]) -> dict[str, Any]:
    """Counters for the console header."""
    return _kb().stats()


def rpc_list_documents(_: dict[str, Any]) -> dict[str, Any]:
    """Every document node; the seed set for a graph sweep."""
    return _kb().list_documents()


def rpc_query_graph(params: dict[str, Any]) -> dict[str, Any]:
    """One node's neighbourhood, as drawable nodes and edges.

    `min_degree` defaults to 2 because the common caller is a sweep, where the
    one-document entities bury the structure. A trace passes 1 explicitly.
    """
    return _kb().neighborhood(
        str(params.get("node_id", "")),
        max_depth=int(params.get("max_depth", 2)),
        min_degree=int(params.get("min_degree", 2)),
    )


def rpc_search_documents(params: dict[str, Any]) -> dict[str, Any]:
    """Semantic search over the corpus."""
    results = _kb().search(str(params.get("query", "")), int(params.get("top_k", 5)))
    return {"results": results, "source": "local_graphrag"}


def rpc_reindex(_: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the knowledge base from the project files."""
    return index_project_files(_kb())


def rpc_list_seats(_: dict[str, Any]) -> dict[str, Any]:
    """The four seats and whether each can actually run."""
    return {
        "seats": [{"role": agent, **get_agent_status(agent)} for agent in AGENTS]
    }


def rpc_set_seat(params: dict[str, Any]) -> dict[str, Any]:
    """Reassign one seat for the lifetime of this process."""
    agent = str(params.get("agent") or params.get("role", ""))
    provider = str(params.get("provider", ""))
    model = str(params.get("model", ""))

    if agent not in AGENTS:
        raise ValueError(f"Unknown agent: {agent!r}")
    if not provider or not model:
        raise ValueError("Provider and model are both required")

    set_agent_llm(agent, provider, model)
    return {"ok": True, "role": agent, **get_agent_status(agent)}


def rpc_llm_options(_: dict[str, Any]) -> dict[str, Any]:
    """Model choices for the seat dropdowns.

    `ollama_tags` is what the daemon actually carries, so a seat can be pointed
    at a tag that is pulled but not in the curated list.
    """
    return {"options": AGENT_LLM_OPTIONS, "ollama_tags": list_ollama_models()}


def rpc_status(_: dict[str, Any]) -> dict[str, Any]:
    """Everything the console polls for: seats, embedding model, corpus state."""
    try:
        kb_indexed, embedding_model = is_knowledge_base_indexed()
    except Exception:
        kb_indexed, embedding_model = False, "unknown"

    seats = {agent: get_agent_status(agent) for agent in AGENTS}
    architect = seats["architect"]

    return {
        # The embedding model is the one thing that runs on this machine.
        "embedding": embedding_model,
        "graphrag": kb_indexed,
        "llm": f"{architect['model']} ({architect['provider']})",
        "agents": seats,
        "selected": seats,
        "degraded": [agent for agent, seat in seats.items() if not seat["live"]],
    }


def rpc_run_goal(params: dict[str, Any]) -> dict[str, Any]:
    """Run a goal through the four-agent loop and return the final state."""
    state: AgentState = {
        "goal": str(params.get("goal", "")),
        "messages": [],
        "architecture": "",
        "verdict": "",
        "plan": "",
        "research": "",
        "builder_report": "",
        "next_agent": "Researcher",
        "research_status": "",
        "blockers": "",
        "files_changed": [],
        "step_count": 0,
    }
    result = graph.invoke(state, {"recursion_limit": RECURSION_LIMIT})
    return dict(result)


RPC_METHODS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "rag_stats": rpc_rag_stats,
    "list_documents": rpc_list_documents,
    "query_graph": rpc_query_graph,
    "search_documents": rpc_search_documents,
    "reindex": rpc_reindex,
    "list_seats": rpc_list_seats,
    "set_seat": rpc_set_seat,
    "llm_options": rpc_llm_options,
    "status": rpc_status,
    "run_goal": rpc_run_goal,
}

# Methods the console polls on a timer. Logging these buries everything else.
QUIET_METHODS = {"status", "rag_stats", "list_seats"}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory="frontend", **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/api/status":
            # Compatibility route: launch_console.sh polls this as its
            # readiness check, and it is the same payload as rpc "status".
            self.send_json(rpc_status({}))
        elif parsed.path == "/api/llm-options":
            self.send_json(rpc_llm_options({}))
        elif parsed.path in ("/", "/index.html"):
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0))
        try:
            data: dict[str, Any] = json.loads(self.rfile.read(length)) if length else {}
        except json.JSONDecodeError:
            self.send_json({"error": {"message": "bad JSON"}, "elapsed_ms": 0})
            return

        if parsed.path == "/rpc":
            self.handle_rpc(data)
            return

        # Compatibility routes for the previous /api surface.
        aliases: dict[str, tuple[str, dict[str, Any]]] = {
            "/api/run": ("run_goal", data),
            "/api/search": ("search_documents", data),
            "/api/set-llm": ("set_seat", data),
        }
        if parsed.path in aliases:
            method, params = aliases[parsed.path]
            try:
                self.send_json(RPC_METHODS[method](params))
            except Exception as exc:
                self.send_json({"error": str(exc)})
            return

        self.send_error(404)

    def handle_rpc(self, data: dict[str, Any]) -> None:
        """Dispatch one {method, params} call.

        Errors come back as a 200 with an `error` member rather than an HTTP
        status: the console renders them into its telemetry log, and a failed
        method is not a failed request.
        """
        method = str(data.get("method", ""))
        params = data.get("params") or {}
        started = time.perf_counter()

        handler = RPC_METHODS.get(method)
        if handler is None:
            self.send_json(
                {"error": {"message": f"Unknown method: {method}"}, "elapsed_ms": 0}
            )
            return

        try:
            result = handler(params)
            elapsed = int((time.perf_counter() - started) * 1000)
            if method not in QUIET_METHODS:
                print(f"[RPC] {method} {elapsed}ms")
            self.send_json({"result": result, "elapsed_ms": elapsed})
        except Exception as exc:
            elapsed = int((time.perf_counter() - started) * 1000)
            print(f"[RPC] {method} FAILED {elapsed}ms: {exc}")
            self.send_json({"error": {"message": str(exc)}, "elapsed_ms": elapsed})

    def send_json(self, obj: Any) -> None:
        payload = json.dumps(obj, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        # RPC calls log themselves in handle_rpc, with their method name and
        # timing; the default line would add a second, less useful entry.
        request_line = args[0] if args else ""
        if isinstance(request_line, str) and (
            "POST /rpc" in request_line or "GET /api/status" in request_line
        ):
            return
        print(f"[API] {request_line}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"Serving at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    server = None
    try:
        # Threaded: a run takes as long as four cloud models take, and a
        # single-threaded server would stall every status poll behind it.
        server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if server:
            server.shutdown()
