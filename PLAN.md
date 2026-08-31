# Plan: Align Project with 3-Agent System Full Consolidated Guide

## Goal
Bring `langgraph-agent` into full compliance with `/home/darthmaverus/Downloads/3-Agent-System-Full-Consolidated-Guide.md` and ensure the web console displays accurate system information.

## Current Status
- The three agents, shared `AgentState`, and strict system prompts are implemented.
- All 8 tests pass (with stub LLM).
- However, several documented behaviors are missing or incorrect (see gaps below).

## Identified Gaps vs. Documentation

1. **Planner routing is overridden by the graph**
   - *Doc*: Planner chooses next agent; simple tasks go `Planner → Builder`.
   - *Code*: `route_from_planner` always sends the first turn to Researcher if no research exists, ignoring the Planner's `next_agent` decision.
   - *Fix*: Respect `state["next_agent"]` on every turn; remove the forced-first-research branch.

2. **Builder never sets `blockers`, so feedback loops never trigger**
   - *Doc*: Builder reports blockers so the graph can loop to Planner/Researcher.
   - *Code*: `builder_node` leaves `state["blockers"]` empty. The LLM fallback parses `changes_made` and `files_modified` but not blockers.
   - *Fix*: Parse `## Next Steps / Blockers` and copy any non-"none" text into `state["blockers"]`; also set a blocker when direct file execution fails.

3. **Researcher bypasses MCP and talks directly to GraphRAG**
   - *Doc*: Researcher calls the GraphRAG MCP server; MCP is the only I/O path.
   - *Code*: `researcher_node` directly instantiates `GraphRAGKnowledgeBase`.
   - *Fix*: Route Researcher calls through `MCPClient` (or the exposed `search_knowledge_graph` / `query_knowledge_graph` tools) so the tool boundary matches the docs.

4. **Builder bypasses MCP for file writes**
   - *Doc*: Builder uses filesystem / git / test tools via MCP; `write_file` is the only tool the Builder may use to write files.
   - *Code*: Builder uses raw `Path.write_text()`.
   - *Fix*: Use `MCPClient.call_tool("filesystem_write", ...)` for file creation; expand to `git_status`/`git_diff` and test-tool stubs if needed.

5. **Frontend shows an impossible / non-local LLM**
   - *Doc*: Recommended local models are `qwen3:8b` or `qwen2.5:7b`; the system is "fully local · zero service fees".
   - *Code*: `frontend/index.html` and `serve.py` hardcode `qwen3.5:397b-cloud`, which is neither local nor runnable on 32 GB RAM / 3 GB GPU.
   - *Fix*: Replace all references with the documented default (`qwen3:8b`) and surface the actual configured `OLLAMA_MODEL` from the backend.

6. **State injection does not show `(empty)` for empty strings**
   - *Doc*: "Empty fields should explicitly say `(empty)`."
   - *Code*: `_get_state_injection` uses `state.get(key, "(empty)")`, which keeps `""` instead of replacing it with `(empty)`.
   - *Fix*: Normalize empty strings to `(empty)` in the injection block.

7. **Suggested project structure is partially missing**
   - *Doc*: Suggests `agents/`, `prompts/`, `mcp_servers/` directories.
   - *Code*: Prompts are inline in `nodes.py`; no `prompts/` or `mcp_servers/` directories.
   - *Fix*: (Optional but recommended) Extract prompts to `prompts/planner.txt`, `prompts/researcher.txt`, `prompts/builder.txt` and load them at import time. Add a brief `mcp_servers/README.md` explaining how to launch the GraphRAG MCP server.

8. **Default LLM in `config.py` does not match docs**
   - *Doc*: Default recommendation is `qwen3:8b`.
   - *Code*: Fallback in `config.py` is `llama3.1` when `OLLAMA_MODEL` is unset.
   - *Fix*: Change default Ollama model to `qwen3:8b`.

9. **README/frontend understate MCP gaps**
   - README calls MCP "scaffolding". The user wants a system that follows the docs, so we should make the MCP tool path real (or clearly documented if still stubbed).
   - *Fix*: Update README after fixes above to remove "scaffolding" language and describe actual tool binding.

## Proposed Implementation Order

1. Fix state injection empty-string handling (`nodes.py`).
2. Fix `route_from_planner` to respect Planner's decision (`graph.py`).
3. Fix `builder_node` to set `blockers` from parsed output or execution failures.
4. Refactor `researcher_node` to use `MCPClient` GraphRAG tools.
5. Refactor `builder_node` to use `filesystem_write` (and git tools) through `MCPClient`.
6. Update default Ollama model in `config.py` to `qwen3:8b`.
7. Update frontend and `serve.py` to report the real configured LLM and show documented local defaults.
8. (Optional) Extract prompts to `prompts/*.txt` and add `mcp_servers/README.md`.
9. Update tests if behavior changes (e.g., simple task now goes Planner → Builder; blockers trigger loops).
10. Run full test suite and lint/type checks.

## Notes
- The current tests pass because they assert weak conditions. After fixing routing and blockers, tests may need to be strengthened to verify the documented behavior.
- The sentence-transformers model (`all-MiniLM-L6-v2`) is fine and matches the local embedding recommendation.
- Keep changes minimal and focused on doc compliance; avoid adding new optional features (human-in-the-loop, persistence) unless explicitly requested.
