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
    """
    # Build messages with state injection
    state_injection = _get_state_injection(state)

    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=f"{state_injection}\n\nPlan to research:\n{state.get('plan', '')}"),
    ]

    # TODO: Call GraphRAG MCP tool here
    # For now, use LLM to simulate (will be replaced with actual GraphRAG calls)
    llm = get_llm()
    response = llm.invoke(messages)

    # Parse the structured output
    parsed = _parse_researcher_output(response.content)

    # Update state
    state["research"] = (
        f"## Key Findings\n{parsed['key_findings']}\n\n"
        f"## Relevant Context\n{parsed['relevant_context']}\n\n"
        f"## Recommendations\n{parsed['recommendations']}"
    )
    state["research_status"] = parsed.get("status", ResearchStatus.READY_FOR_BUILDER.value)

    # Route based on status
    if parsed["status"] == ResearchStatus.NEED_REPLAN.value:
        state["next_agent"] = "Researcher"  # Will trigger replan
        state["messages"].append("[Researcher] Needs replan")
    elif parsed["status"] == ResearchStatus.NO_RELEVANT_KNOWLEDGE.value:
        state["next_agent"] = "Builder"  # Proceed without research
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
    """
    # Build messages with state injection
    state_injection = _get_state_injection(state)

    messages = [
        SystemMessage(content=BUILDER_PROMPT),
        HumanMessage(
            content=f"{state_injection}\n\nPlan to implement:\n{state.get('plan', '')}\n\n"
            f"Research findings:\n{state.get('research', '(none)')}"
        ),
    ]

    # TODO: Call actual MCP tools (filesystem, git, etc.)
    # For now, use LLM to generate implementation plan
    llm = get_llm()
    response = llm.invoke(messages)

    # Parse the structured output
    parsed = _parse_builder_output(response.content)

    # Update state
    state["builder_report"] = parsed.get("changes_made", response.content)
    state["files_changed"] = parsed.get("files_modified", [])

    # Check for blockers
    blockers = parsed.get("next_steps_blockers", "")
    if blockers and blockers.lower() != "none":
        state["blockers"] = blockers
        state["messages"].append("[Builder] Blockers detected")
    else:
        state["blockers"] = ""
        state["messages"].append("[Builder] Implementation complete")

    state["step_count"] = state.get("step_count", 0) + 1

    return state
