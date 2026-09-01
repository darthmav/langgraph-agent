#!/usr/bin/env python3
"""Simple frontend server for 3-Agent Console.

Usage:
    python serve.py

Then open: http://localhost:8080
"""

import json
import os
import sys
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

# Ensure src directory is in Python path
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from langgraph_agent import create_agent_graph, AgentState
from langgraph_agent.config import get_agent_model_info
from langgraph_agent.graphrag_server import get_knowledge_base

# Initialize graph and knowledge base
graph = create_agent_graph()
kb = None

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="frontend", **kwargs)
    
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/status':
            # Check if knowledge base is indexed
            global kb
            kb_indexed = False
            embedding_model = "unknown"
            
            try:
                if kb is None:
                    kb = get_knowledge_base()
                kb_indexed = kb.collection.count() > 0 if kb.collection else False
                embedding_model = getattr(kb, 'embedder_model_name', 'all-MiniLM-L6-v2')
            except Exception:
                kb_indexed = False
            
            self.send_json({
                "llm": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
                "embedding": embedding_model,
                "graphrag": kb_indexed,
                "agents": {
                    "planner": get_agent_model_info("planner"),
                    "researcher": get_agent_model_info("researcher"),
                    "builder": get_agent_model_info("builder"),
                },
            })
        elif parsed.path == '/' or parsed.path == '/index.html':
            self.path = '/index.html'
            return super().do_GET()
        else:
            super().do_GET()
    
    def do_POST(self):
        parsed = urlparse(self.path)
        length = int(self.headers.get('Content-Length', 0))
        data = json.loads(self.rfile.read(length)) if length else {}

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

        else:
            self.send_error(404)
    
    def send_json(self, obj):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(obj, default=str).encode())
    
    def log_message(self, format, *args):
        request_line = args[0] if args else ""
        # Suppress noisy routine status-poll logs; everything else is still logged.
        if "GET /api/status" in request_line:
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
