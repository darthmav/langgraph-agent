#!/usr/bin/env python3
"""Simple frontend server for 3-Agent Console.

Usage:
    python -m http.server 8080 --directory frontend
    
Then in another terminal, test API:
    curl -X POST http://localhost:8080/api/run -H "Content-Type: application/json" -d '{"goal": "test"}'
"""

import json
import os
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse

from langgraph_agent import create_agent_graph, AgentState
from langgraph_agent.mcp_client import mcp_client

graph = create_agent_graph()

class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/status':
            self.send_json({
                "llm": os.getenv("OLLAMA_MODEL", "qwen3.5:397b-cloud"),
                "graphrag": True,
            })
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
            import asyncio
            async def search():
                async with mcp_client() as c:
                    return await c.call_tool("search_knowledge_base", {
                        "query": data.get("query", ""), "top_k": data.get("top_k", 5)
                    })
            self.send_json(asyncio.run(search()))
        
        else:
            self.send_error(404)
    
    def send_json(self, obj):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(obj, default=str).encode())
    
    def log_message(self, format, *args):
        print(f"[API] {args[0]}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"Serving at http://localhost:{port}")
    print("Press Ctrl+C to stop")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
