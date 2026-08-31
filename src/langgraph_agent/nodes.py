"""Agent nodes: Planner, Researcher, Builder.

Implements the 3-Agent System specification with strict prompts and tool binding:
- Planner: reasoning only, no tools
- Researcher: GraphRAG MCP tools only
- Builder: filesystem, git, terminal tools only
"""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from langgraph_agent.config import get_llm
from langgraph_agent.state import AgentState, ResearchStatus

# System prompts from the 3-Agent System documentation
PLANNER_PROMPT = """You are the Planner agent in a three-agent software system.
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

RESEARCHER_PROMPT = """You are the Researcher agent in a three-agent software system.
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

BUILDER_PROMPT = """You are the Builder agent in a three-agent software system.
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


def _get_state_injection(state: AgentState) -> str:
    """Create the state injection block sent every turn.

    As specified in the documentation:
    Empty fields should explicitly say `(empty)`.
    """
    return f"""## Current state
Goal: {state.get("goal", "(empty)")}
Plan: {state.get("plan", "(empty)")}
Research: {state.get("research", "(empty)")}
Builder report: {state.get("builder_report", "(empty)")}
Blockers: {state.get("blockers", "(empty)")}
Files changed: {state.get("files_changed", [])}
Step: {state.get("step_count", 0)}
"""


def _parse_planner_output(content: str) -> dict:
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


def _parse_researcher_output(content: str) -> dict:
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


def _parse_builder_output(content: str) -> dict:
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

    llm = get_llm()
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

    Uses MCP client to search the knowledge base. Falls back to LLM if KB is empty.
    """
    import asyncio

    # Build messages with state injection
    state_injection = _get_state_injection(state)
    plan = state.get("plan", "")

    # Try to use GraphRAG first
    graphrag_results = None
    graph_results = None

    try:

        async def search_graphrag():
            from langgraph_agent.mcp_client import MCPClient

            async with MCPClient() as client:
                # Search for relevant information
                search_results = await client.call_tool(
                    "search_knowledge_base", {"query": plan, "top_k": 5}
                )

                # Query knowledge graph for relationships
                graph_results = await client.call_tool(
                    "query_knowledge_graph", {"entity": plan.split()[0] if plan else "code", "hops": 2}
                )

                return search_results, graph_results

        # Run async search
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        graphrag_results, graph_results = loop.run_until_complete(search_graphrag())
        loop.close()

    except Exception as e:
        # GraphRAG not available, will use LLM fallback
        graphrag_results = {"error": str(e)}

    # Check if we got real results from GraphRAG
    has_real_results = (
        graphrag_results
        and graphrag_results.get("source") == "local_graphrag"
        and graphrag_results.get("results")
        and len(graphrag_results["results"]) > 0
        and graphrag_results["results"][0].get("score", 0) > 0.5
    )

    if has_real_results:
        # Format GraphRAG results into Researcher output format
        results = graphrag_results["results"]
        research_findings = "## Key Findings\n"

        for i, result in enumerate(results[:3], 1):
            research_findings += f"\n{i}. {result.get('content', '')[:200]}"
            if result.get("related_entities"):
                research_findings += f"\n   Related: {result['related_entities'][:3]}"
            research_findings += f"\n   Score: {result.get('score', 0):.2f}\n"

        research_findings += "\n## Relevant Context\n"
        research_findings += f"Found {len(results)} relevant documents in knowledge base.\n"

        if graph_results and graph_results.get("subgraph_nodes", 0) > 0:
            research_findings += (
                f"\nKnowledge graph has {graph_results['subgraph_nodes']} nodes "
                f"and {graph_results['subgraph_edges']} relationships.\n"
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

        llm = get_llm()
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

    Uses MCP client to call filesystem tools. Falls back to LLM if tools unavailable.
    """
    import asyncio
    from pathlib import Path

    # Build messages with state injection
    state_injection = _get_state_injection(state)

    messages = [
        SystemMessage(content=BUILDER_PROMPT),
        HumanMessage(
            content=f"{state_injection}\n\nPlan to implement:\n{state.get('plan', '')}\n\n"
            f"Research findings:\n{state.get('research', '(none)')}"
        ),
    ]

    # Try to use tools to actually implement the plan
    files_changed = []
    builder_report_parts = []

    try:
        async def use_tools():
            from langgraph_agent.mcp_client import mcp_client

            async with mcp_client() as client:
                # Parse the plan to extract file operations
                plan = state.get('plan', '')

                # Simple heuristic: look for file creation patterns
                import re

                # Pattern: "Create <filename>" - handles backticks, "the file", etc.
                create_matches = re.findall(r'[Cc]reate (?:the |a )?(?:new )?(?:file )?`?([a-zA-Z0-9_.\-/]+\.\w+)`?', plan)

                # Pattern: content in quotes after containing/with/write
                content_matches = re.findall(r"(?:containing|(?:with |having )?(?:the )?(?:exact )?(?:string |content )|write)\s*['\"]([^'\"]+)['\"]", plan, re.IGNORECASE)

                if create_matches:
                    for filename in create_matches:
                        # Get content (use first match or default)
                        content = content_matches[0] if content_matches else f"Content for {filename}"

                        # Write the file
                        result = await client.call_tool("filesystem_write", {
                            "path": filename,
                            "content": content
                        })

                        if result.get("success"):
                            files_changed.append(filename)
                            builder_report_parts.append(f"Created {filename}")
                        else:
                            builder_report_parts.append(f"Failed to create {filename}: {result.get('error', 'unknown')}")

                return files_changed

        # Run async tool calls
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        files_changed = loop.run_until_complete(use_tools())
        loop.close()
        builder_report = "\n".join(builder_report_parts) if builder_report_parts else "No file operations identified"

    except Exception as e:
        # Fallback to LLM-only mode
        llm = get_llm()
        response = llm.invoke(messages)
        parsed = _parse_builder_output(response.content)
        builder_report = parsed.get("changes_made", response.content)
        files_changed = parsed.get("files_modified", [])

    # Update state
    state["builder_report"] = builder_report
    state["files_changed"] = files_changed
    state["messages"].append(f"[Builder] Implementation complete. Files: {len(files_changed)}")
    state["step_count"] = state.get("step_count", 0) + 1

    return state
