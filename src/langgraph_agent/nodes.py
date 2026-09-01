"""Agent nodes: Planner, Researcher, Builder.

Implements the 3-Agent System specification with strict prompts and tool binding:
- Planner: reasoning only, no tools
- Researcher: GraphRAG MCP tools only
- Builder: filesystem, git, terminal tools only
"""

import asyncio
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_agent.config import get_agent_llm
from langgraph_agent.mcp_client import mcp_client
from langgraph_agent.state import AgentState, ResearchStatus


def _call_mcp_tool_sync(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Call an MCP tool from a synchronous LangGraph node.

    The MCP client is async, so this helper bridges sync and async contexts.
    """

    async def _call() -> Any:
        async with mcp_client() as client:
            return await client.call_tool(tool_name, arguments)

    try:
        return asyncio.run(_call())
    except RuntimeError:
        # Already running inside an event loop (e.g., some test runners).
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_call())


def _load_prompt(name: str, fallback: str) -> str:
    """Load a system prompt from the prompts/ directory.

    Falls back to the inline string so the module works even if the prompt
    files are not present (e.g., during distribution).
    """
    from pathlib import Path

    try:
        prompt_path = Path(__file__).parent.parent.parent / "prompts" / f"{name}.txt"
        return prompt_path.read_text(encoding="utf-8")
    except Exception:
        return fallback


# System prompts from the 3-Agent System documentation.
# Loaded from prompts/ when available, with inline fallbacks.
_PLANNER_PROMPT_INLINE = """You are the Planner agent in a three-agent software system.
The other agents are Researcher and Builder. You are not them.

Your only job:
1. Understand the user's goal.
2. Break it into clear, ordered steps.
3. Choose the next agent: Researcher or Builder.
4. Write the plan in the exact format below.

You must:
- Be concise and specific.
- Name concrete artifacts (files, endpoints, behaviors) when the user named them.
- Choose Researcher when the work depends on existing code, docs, architecture, or relationships across files.
- Choose Builder when the task is fully specified, research is already in state, or retrieval would not help.
- If this is a loop (you can see prior research or a builder report), update the plan; do not ignore what already happened.
- If blockers are present, address them in the new plan.

You must not:
- Write or propose code.
- Call tools.
- Retrieve knowledge yourself.
- Invent files or APIs that were not in the user goal or in prior state.
- Talk to the user in a conversational way. Output only the plan format.

Output exactly this format and nothing else:

## Goal
[one sentence]

## Steps
1. ...
2. ...
3. ...

## Next Agent
Researcher
OR
Builder

## Notes
[assumptions, risks, why this next agent, what to skip]
"""

_RESEARCHER_PROMPT_INLINE = """You are the Researcher agent in a three-agent software system.
The other agents are Planner and Builder. You are not them.

Your only job:
Gather high-quality, relevant knowledge so the Builder can implement the plan.
Use tools. Prefer the GraphRAG tool for anything involving relationships, architecture, or multiple files.

You must:
- Call GraphRAG (or the knowledge-search tool) before answering, unless the plan is purely about creating a brand-new isolated file with no existing context.
- Search for entities, relationships, and source passages that match the plan's steps.
- Summarize what you found. Quote or cite paths and entity names.
- Give the Builder concrete recommendations: what to change, where, what to reuse, what not to break.
- If the graph has little or nothing relevant, say so explicitly. Do not invent a codebase.
- If you need a different question or a replan, say so in Recommendations.

You must not:
- Write or modify code.
- Call filesystem write tools or git write tools.
- Make the final implementation decision as if you were the Builder.
- Pretend you read files you did not retrieve.

Output exactly this format and nothing else:

## Key Findings
- ...

## Relevant Context
[summary of retrieved entities, relationships, and passages]

## Recommendations for Builder
[what to implement, which files/entities, constraints]

## Status
ready_for_builder
OR
need_replan
OR
no_relevant_knowledge
"""

_BUILDER_PROMPT_INLINE = """You are the Builder agent in a three-agent software system.
The other agents are Planner and Researcher. You are not them.

Your only job:
Implement the plan using the research provided. Use tools to make real changes.

You must:
- Follow the plan and the research. Do not freelance a new design if research already specified one.
- Use filesystem / git / test tools to actually change files. Do not only describe code.
- Write clean, working code consistent with the existing project.
- After changes, report what you did.
- If you cannot finish, set blockers clearly so the graph can loop to Researcher or Planner.
- Prefer the smallest change that satisfies the plan.

You must not:
- Call GraphRAG or knowledge-search tools.
- Ignore the plan or the research.
- Claim you changed a file if you did not call a write tool.
- Expand scope beyond the plan.

Output exactly this format after you finish using tools:

## Changes Made
- ...

## Files Modified
- path/to/file
- ...

## Next Steps / Blockers
[none | what is blocked and what information is needed]
"""

PLANNER_PROMPT = _load_prompt("planner", _PLANNER_PROMPT_INLINE)
RESEARCHER_PROMPT = _load_prompt("researcher", _RESEARCHER_PROMPT_INLINE)
BUILDER_PROMPT = _load_prompt("builder", _BUILDER_PROMPT_INLINE)


def _get_state_injection(state: AgentState) -> str:
    """Create the state injection block sent every turn.

    As specified in the documentation:
    Empty fields should explicitly say `(empty)`.
    """

    def _fmt(key: str, default: str = "(empty)") -> str:
        value = state.get(key)
        if value is None or value == "" or value == []:
            return default
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)
        return str(value)

    return f"""## Current state
Goal: {_fmt("goal")}
Plan: {_fmt("plan")}
Research: {_fmt("research")}
Builder report: {_fmt("builder_report")}
Blockers: {_fmt("blockers")}
Files changed: {_fmt("files_changed")}
Step: {state.get("step_count", 0)}
"""


def _parse_planner_output(content: str) -> dict[str, Any]:
    """Parse Planner output into structured format.

    Expected format:
    ## Goal
    ...

    ## Steps
    1. ...

    ## Next Agent
    Researcher | Builder

    ## Notes
    ...
    """
    result = {"plan": "", "next_agent": "Builder", "notes": ""}

    # Extract goal
    goal_match = re.search(r"## Goal\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE)
    if goal_match:
        result["goal"] = goal_match.group(1).strip()

    # Extract steps
    steps_match = re.search(r"## Steps\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE)
    if steps_match:
        result["plan"] = steps_match.group(1).strip()

    # Extract next agent
    agent_match = re.search(
        r"## Next Agent\s*\n(Researcher|Builder)(?:\s|$)", content, re.IGNORECASE
    )
    if agent_match:
        result["next_agent"] = agent_match.group(1).strip()

    # Extract notes
    notes_match = re.search(r"## Notes\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE)
    if notes_match:
        result["notes"] = notes_match.group(1).strip()

    return result


def _parse_researcher_output(content: str) -> dict[str, Any]:
    """Parse Researcher output into structured format.

    Expected format:
    ## Key Findings
    ...

    ## Relevant Context
    ...

    ## Recommendations for Builder
    ...

    ## Status
    ready_for_builder | need_replan | no_relevant_knowledge
    """
    result = {
        "key_findings": "",
        "relevant_context": "",
        "recommendations": "",
        "status": ResearchStatus.READY_FOR_BUILDER.value,
    }

    # Extract sections
    findings_match = re.search(
        r"## Key Findings\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if findings_match:
        result["key_findings"] = findings_match.group(1).strip()

    context_match = re.search(
        r"## Relevant Context\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if context_match:
        result["relevant_context"] = context_match.group(1).strip()

    recs_match = re.search(
        r"## Recommendations for Builder\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if recs_match:
        result["recommendations"] = recs_match.group(1).strip()

    status_match = re.search(
        r"## Status\s*\n(ready_for_builder|need_replan|no_relevant_knowledge)",
        content,
        re.IGNORECASE,
    )
    if status_match:
        result["status"] = status_match.group(1).lower()

    return result


def _parse_builder_output(content: str) -> dict[str, Any]:
    """Parse Builder output into structured format.

    Expected format:
    ## Changes Made
    ...

    ## Files Modified
    - path/to/file
    ...

    ## Next Steps / Blockers
    ...
    """
    result = {"changes_made": "", "files_modified": [], "next_steps_blockers": ""}

    changes_match = re.search(
        r"## Changes Made\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if changes_match:
        result["changes_made"] = changes_match.group(1).strip()

    files_match = re.search(
        r"## Files Modified\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if files_match:
        # Extract file paths from bullet list
        lines = files_match.group(1).strip().split("\n")
        result["files_modified"] = [line.lstrip("- ").strip() for line in lines if line.strip()]

    blockers_match = re.search(
        r"## Next Steps / Blockers\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if blockers_match:
        result["next_steps_blockers"] = blockers_match.group(1).strip()

    return result


def planner_node(state: AgentState) -> AgentState:
    """Planner: Interpret goal, create structured plan, choose next agent.

    As specified:
    - No tools
    - Output strict format
    - Routes to Researcher when knowledge needed, Builder when task is clear
    """
    # Build messages with state injection
    state_injection = _get_state_injection(state)
    goal = state.get("goal", "")

    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"{state_injection}\n\nUser goal: {goal}"),
    ]

    llm = get_agent_llm("planner")
    response = llm.invoke(messages)

    # Parse the structured output
    parsed = _parse_planner_output(response.content)

    # Update state
    state["plan"] = parsed.get("plan", "")
    state["next_agent"] = parsed.get("next_agent", "Builder")
    state["messages"].append(f"[Planner] Plan created. Next agent: {state['next_agent']}")

    return state


def researcher_node(state: AgentState) -> AgentState:
    """Researcher: Query GraphRAG, summarize findings, recommend approach.

    As specified:
    - GraphRAG tools only
    - Output strict format with status
    - Status guides next steps (ready_for_builder | need_replan | no_relevant_knowledge)

    Calls the GraphRAG MCP tool (`search_knowledge_graph`) rather than importing
    the knowledge base directly, preserving the documented tool boundary.
    Falls back to the LLM if the knowledge base is empty or the MCP tool fails.
    """
    # Build messages with state injection
    state_injection = _get_state_injection(state)
    plan = state.get("plan", "")

    # Call GraphRAG through the MCP tool boundary
    graphrag_results: dict[str, Any] | None = None

    try:
        search_response: dict[str, Any] = _call_mcp_tool_sync(
            "search_knowledge_graph", {"query": plan, "top_k": 5}
        )
        results = search_response.get("results", [])

        # Check if we got real results
        if results and len(results) > 0 and results[0].get("score", 0) > 0.3:
            graphrag_results = {"results": results, "source": "local_graphrag"}

            # Try to get graph info too via the MCP query tool
            try:
                first_doc = results[0].get("id", "")
                if first_doc:
                    entity = (
                        first_doc.split("/")[-1].split(".")[0]
                        if "/" in first_doc
                        else plan.split()[0] if plan else "code"
                    )
                    graph_response = _call_mcp_tool_sync(
                        "query_knowledge_graph", {"entity": entity, "hops": 2}
                    )
                    graphrag_results["graph"] = graph_response
            except Exception:
                pass  # Graph query is optional
    except Exception as e:
        # GraphRAG MCP tool unavailable or failed; will use LLM fallback
        graphrag_results = {"error": str(e)}

    # Check if we got real results from GraphRAG
    has_real_results = (
        graphrag_results
        and graphrag_results.get("source") == "local_graphrag"
        and graphrag_results.get("results")
        and len(graphrag_results["results"]) > 0
    )

    if has_real_results:
        assert graphrag_results is not None
        # Format GraphRAG results into Researcher output format
        results = graphrag_results.get("results", [])
        research_findings = "## Key Findings\n"

        for i, result in enumerate(results[:3], 1):
            content = result.get('content', '')[:300]
            score = result.get('score', 0)
            research_findings += f"\n{i}. {content}"
            if result.get("related_entities"):
                research_findings += f"\n   Related: {result['related_entities'][:3]}"
            research_findings += f"\n   Score: {score:.2f}\n"

        research_findings += "\n## Relevant Context\n"
        research_findings += f"Found {len(results)} relevant documents in knowledge base.\n"

        graph_response = graphrag_results.get("graph", {})
        if graph_response and graph_response.get("subgraph_nodes", 0) > 0:
            research_findings += (
                f"\nKnowledge graph has {graph_response['subgraph_nodes']} nodes "
                f"and {graph_response['subgraph_edges']} relationships.\n"
            )

        research_findings += "\n## Recommendations for Builder\n"
        research_findings += "Use the retrieved documentation as reference for implementation.\n"
        research_findings += "\n## Status\nready_for_builder"

        research_status = "ready_for_builder"

    else:
        # Fallback to LLM when GraphRAG has no real data
        messages = [
            SystemMessage(content=RESEARCHER_PROMPT),
            HumanMessage(content=f"{state_injection}\n\nPlan to research:\n{plan}"),
        ]

        llm = get_agent_llm("researcher")
        response = llm.invoke(messages)
        research_findings = response.content

        # Parse status from LLM response
        parsed = _parse_researcher_output(response.content)
        research_status = parsed.get("status", "ready_for_builder")

    # Update state
    state["research"] = research_findings
    state["research_status"] = research_status

    # Route based on status
    if research_status == "need_replan":
        state["next_agent"] = "Planner"
        state["messages"].append("[Researcher] Needs replan")
    elif research_status == "no_relevant_knowledge":
        state["next_agent"] = "Builder"
        state["messages"].append("[Researcher] No relevant knowledge, proceeding")
    else:
        state["next_agent"] = "Builder"
        state["messages"].append("[Researcher] Research complete, routing to Builder")

    return state


def builder_node(state: AgentState) -> AgentState:
    """Builder: Implement plan using tools, report changes.

    As specified:
    - Filesystem, git, test tools only (no GraphRAG)
    - Actually make changes via tools
    - Report in strict format
    - Set blockers if stuck

    Uses direct file operations. Falls back to LLM if tools fail.
    """
    import re

    # Build messages with state injection
    state_injection = _get_state_injection(state)
    plan = state.get('plan', '')
    research = state.get('research', '')
    goal = state.get('goal', '')

    # Try to execute the plan directly
    files_changed = []
    builder_report_parts = []
    execution_error = None
    blockers = ""

    try:
        # Pattern: "Create <filename>" or "Builder creates <filename>" etc.
        # Supports backticks, quotes, or bare filenames
        create_matches = re.findall(
            r'[Cc]reate(?:s)?\s+(?:the\s+)?(?:a\s+)?(?:new\s+)?(?:file\s+)?(?:named|called)?\s*[`"\']?([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+)[`"\']?',
            plan
        )

        # Pattern: content in quotes - multiple patterns to try
        content_matches = []

        # Pattern 1: "containing 'content'" or "with 'content'" - most specific
        content_matches += re.findall(r"(?:containing|with)\s*['\"`]([^'\"`]+)['\"`]", plan, re.IGNORECASE)

        # Pattern 2: "write 'content' to it" or "writes 'content'"
        content_matches += re.findall(r"(?:write|writes?)\s+(?:to\s+it\s+)?['\"`]([^'\"`]+)['\"`]", plan, re.IGNORECASE)

        # Pattern 3: Backtick-quoted text that's NOT a filename (LLM often uses backticks)
        all_backtick = re.findall(r'`([^`]+)`', plan)
        # Filter out filenames (things that look like file paths)
        for match in all_backtick:
            if not re.match(r'^[a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+$', match):
                content_matches.append(match)

        # Pattern 4: Double-quoted text (not filenames)
        all_double = re.findall(r'"([^"]+)"', plan)
        for match in all_double:
            if not re.match(r'^[a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+$', match):
                content_matches.append(match)

        # Pattern 5: Single-quoted text (not filenames)
        all_single = re.findall(r"'([^']+)'", plan)
        for match in all_single:
            if not re.match(r'^[a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+$', match):
                content_matches.append(match)

        # Pattern 6: Just standalone quoted strings in the goal (fallback)
        if not content_matches and goal:
            goal_content = re.findall(r"['\"`]([^'\"`]+)['\"`]", goal)
            # Filter out filenames
            for match in goal_content:
                if not re.match(r'^[a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+$', match):
                    content_matches.append(match)

        if create_matches:
            for filename in create_matches:
                # Clean up filename
                filename = filename.strip().strip('`"\'')

                # Get content (use first match or default)
                content = content_matches[0] if content_matches else f"Content for {filename}"

                # Write the file through the MCP filesystem tool, preserving
                # the documented tool boundary (Builder only uses filesystem/git
                # tools, never direct file operations or GraphRAG).
                try:
                    result = _call_mcp_tool_sync(
                        "filesystem_write", {"path": filename, "content": content}
                    )
                    if result.get("success"):
                        files_changed.append(filename)
                        builder_report_parts.append(
                            f"Created {filename} with content: {content[:50]}..."
                        )
                    else:
                        error = result.get("error", "Unknown filesystem_write error")
                        builder_report_parts.append(f"Failed to create {filename}: {error}")
                        execution_error = error
                except Exception as e:
                    builder_report_parts.append(f"Failed to create {filename}: {e}")
                    execution_error = str(e)

        builder_report = "\n".join(builder_report_parts) if builder_report_parts else "No file operations identified in plan"

    except Exception as e:
        # Fallback to LLM-only mode
        execution_error = str(e)

    # If execution failed or no file ops identified, use LLM
    if not files_changed or execution_error:
        messages = [
            SystemMessage(content=BUILDER_PROMPT),
            HumanMessage(
                content=f"{state_injection}\n\nPlan to implement:\n{plan}\n\n"
                f"Research findings:\n{research}"
            ),
        ]

        llm = get_agent_llm("builder")
        response = llm.invoke(messages)
        parsed = _parse_builder_output(response.content)

        if not files_changed:
            builder_report = parsed.get("changes_made", response.content)
            files_changed = parsed.get("files_modified", [])

        # Capture any blockers reported by the LLM, but treat "none" as no
        # blockers so the graph can finish cleanly.
        parsed_blockers = parsed.get("next_steps_blockers", "")
        if parsed_blockers and parsed_blockers.strip().lower() not in {"none", "n/a", ""}:
            blockers = parsed_blockers

    # If direct execution failed and no actionable blocker was parsed, set one
    # so the graph can loop back to Planner or Researcher per the specification.
    if execution_error and not blockers:
        blockers = f"Execution failed: {execution_error}. Need replan or additional information."

    # Update state
    state["builder_report"] = builder_report
    state["files_changed"] = files_changed
    state["blockers"] = blockers
    state["messages"].append(f"[Builder] Implementation complete. Files: {len(files_changed)}")
    state["step_count"] = state.get("step_count", 0) + 1

    return state
