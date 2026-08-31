"""Agent nodes: Planner, Researcher, Builder."""

import json
from typing import Literal
from langgraph_agent.state import AgentState, NodeStatus
from langgraph_agent.config import get_llm
from langchain_core.messages import HumanMessage, SystemMessage


# System prompts for each node
PLANNER_PROMPT = """You are a planning agent. Your job is to:
1. Understand the user's request
2. Break it down into clear, actionable steps
3. Decide if research is needed (route to researcher) or if it's a clear build task (route to builder)

Research is needed when the request involves:
- Investigating unknown information
- Exploring options or comparing approaches
- Analyzing requirements or constraints
- Finding best practices

Output your response as JSON:
{
    "plan": ["step 1", "step 2", ...],
    "next_node": "researcher" or "builder"
}"""

RESEARCHER_PROMPT = """You are a research agent. Your job is to:
1. Gather relevant information for the current step
2. Summarize findings clearly
3. Recommend approaches based on findings

Output a concise summary of research findings."""

BUILDER_PROMPT = """You are a builder agent. Your job is to:
1. Implement the solution based on the plan and research
2. Write clean, working code
3. Report what was created/modified

Output a summary of what was built."""


def planner_node(state: AgentState) -> AgentState:
    """Planner: Interpret input, create a plan, route to next node.
    
    Uses LLM to parse intent and generate structured plan.
    Clears feedback on replanning to prevent infinite loops.
    """
    user_input = state["input"]
    
    # Clear feedback when replanning (prevents infinite loops)
    if state.get("feedback"):
        state["messages"].append(f"[Planner] Replanning with feedback: {state['feedback'][:80]}...")
        state["feedback"] = None
    
    llm = get_llm()
    
    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=f"Plan this request: {user_input}"),
    ]
    
    response = llm.invoke(messages)
    
    # Parse JSON response
    try:
        parsed = json.loads(response.content)
        plan = parsed.get("plan", [f"Understand: {user_input}", "Implement solution"])
        next_node = parsed.get("next_node", "builder")
    except (json.JSONDecodeError, AttributeError):
        # Fallback to simple heuristic if LLM fails
        research_keywords = ["research", "investigate", "find", "explore", "analyze"]
        needs_research = any(kw in user_input.lower() for kw in research_keywords)
        plan = [f"Understand: {user_input}", "Gather context if needed", "Implement the solution"]
        next_node = "researcher" if needs_research else "builder"
    
    state["plan"] = plan
    state["current_step"] = 0
    state["next_node"] = next_node
    state["messages"].append(f"[Planner] Created plan with {len(plan)} steps. Routing to: {next_node}")
    
    return state


def researcher_node(state: AgentState) -> AgentState:
    """Researcher: Query knowledge base, summarize, recommend approaches.
    
    Uses LLM for research simulation. MCP GraphRAG integration is scaffolding in mcp_client.py.
    """
    current_step = state["plan"][state["current_step"]]
    llm = get_llm()
    
    # Use LLM to simulate research (MCP integration is scaffolding - see mcp_client.py)
    messages = [
        SystemMessage(content=RESEARCHER_PROMPT),
        HumanMessage(content=f"Research for this step: {current_step}\n\nContext: {state['input']}"),
    ]
    
    response = llm.invoke(messages)
    state["research_findings"] = response.content
    state["messages"].append(f"[Researcher] Completed research for step {state['current_step'] + 1}")
    
    # Move to next step or mark done
    state["current_step"] += 1
    
    if state["current_step"] >= len(state["plan"]):
        state["next_node"] = "builder"
    else:
        # Check if next step needs research
        next_step = state["plan"][state["current_step"]]
        research_keywords = ["research", "investigate", "find", "explore", "analyze"]
        state["next_node"] = "researcher" if any(kw in next_step.lower() for kw in research_keywords) else "builder"
    
    return state


def builder_node(state: AgentState) -> AgentState:
    """Builder: Write, edit, execute code.
    
    Uses LLM to generate code. MCP filesystem/git integration is scaffolding in mcp_client.py.
    """
    llm = get_llm()
    
    # Build context from plan and research
    context = f"Input: {state['input']}\n"
    context += f"Plan: {' | '.join(state['plan'])}\n"
    if state["research_findings"]:
        context += f"Research: {state['research_findings']}\n"
    
    messages = [
        SystemMessage(content=BUILDER_PROMPT),
        HumanMessage(content=f"Build based on:\n{context}"),
    ]
    
    response = llm.invoke(messages)
    state["builder_output"] = response.content
    state["status"] = "complete"
    state["messages"].append("[Builder] Build completed")
    
    return state
