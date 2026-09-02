"""Agent nodes: Architect, Planner, Researcher, Builder.

Implements the 4-Agent System with strict prompts and tool binding:
- Architect: reasoning only, no tools; sets direction and holds the approval gate
- Planner: reasoning only, no tools
- Researcher: GraphRAG MCP tools only
- Builder: filesystem, git, terminal tools only
"""

import asyncio
import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langgraph_agent.config import get_agent_llm
from langgraph_agent.mcp_client import mcp_client
from langgraph_agent.state import AgentState, ResearchStatus, Verdict


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


# System prompts from the 4-Agent System documentation.
# Loaded from prompts/ when available, with inline fallbacks.
_ARCHITECT_PROMPT_INLINE = """You are the Architect. You are the leading authority
on this project: you set the architectural direction before any work starts, and
you decide when work is finished.

You run twice per cycle. Before the Planner sees the goal, you decide how the
work should be shaped and rule `plan`. After the Builder reports, you judge the
result against the goal and your own constraints and rule `approved` (the goal
is met -- this ends the run), `revise` (the approach needs replanning), or
`need_research` (blocked on knowledge nobody has gathered).

You must not write code, call tools, or gather knowledge yourself.

Output exactly this format and nothing else:

## Architecture
<how this work should be shaped>

## Constraints
- <a constraint the Planner and Builder must respect>

## Verdict
plan | approved | revise | need_research"""


_PLANNER_PROMPT_INLINE = """You are the Planner agent in a four-agent software system.
The other agents are Architect, Researcher and Builder. You are not them.
The Architect is the authority on architecture and rules on when the
work is done; treat its constraints as binding.

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

_RESEARCHER_PROMPT_INLINE = """You are the Researcher agent in a four-agent software system.
The other agents are Architect, Planner and Builder. You are not them.
The Architect is the authority on architecture and rules on when the
work is done; treat its constraints as binding.

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

_BUILDER_PROMPT_INLINE = """You are the Builder agent in a four-agent software system.
The other agents are Architect, Planner and Researcher. You are not them.
The Architect is the authority on architecture and rules on when the
work is done; treat its constraints as binding.

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

ARCHITECT_PROMPT = _load_prompt("architect", _ARCHITECT_PROMPT_INLINE)
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
Architecture: {_fmt("architecture")}
Verdict: {_fmt("verdict")}
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


def _parse_architect_output(content: str, reviewing: bool = False) -> dict[str, Any]:
    """Parse Architect output into architecture text and a verdict.

    Args:
        content: Raw model output.
        reviewing: Whether this was the gate pass. It picks the fallback when
            no verdict parses: before the Builder has reported the only sound
            ruling is `plan`, and after it the sound ruling is `approved`, so a
            malformed response ends the run instead of looping on it forever.
    """
    result: dict[str, Any] = {
        "architecture": "",
        "verdict": Verdict.APPROVED.value if reviewing else Verdict.PLAN.value,
    }

    architecture = ""
    arch_match = re.search(
        r"## Architecture\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if arch_match:
        architecture = arch_match.group(1).strip()

    # Constraints ride along in the same field: they are the half the Planner
    # and Builder actually have to obey, and state injection renders one value.
    constraints_match = re.search(
        r"## Constraints\s*\n(.*?)(?=##|$)", content, re.DOTALL | re.IGNORECASE
    )
    if constraints_match and constraints_match.group(1).strip():
        constraints = constraints_match.group(1).strip()
        architecture = f"{architecture}\n\nConstraints:\n{constraints}".strip()

    result["architecture"] = architecture

    verdicts = "|".join(verdict.value for verdict in Verdict)
    verdict_match = re.search(
        rf"## Verdict\s*\n({verdicts})\b", content, re.IGNORECASE
    )
    if verdict_match:
        result["verdict"] = verdict_match.group(1).strip().lower()

    return result


def architect_node(state: AgentState) -> AgentState:
    """Architect: set direction, then rule on whether the work is done.

    No tools. Runs twice per cycle -- as the entry authority before the Planner,
    and as the approval gate after the Builder reports. A populated builder
    report is what tells the two passes apart.
    """
    reviewing = bool(state.get("builder_report"))

    state_injection = _get_state_injection(state)
    goal = state.get("goal", "")
    task = (
        "The Builder has reported. Rule on the work."
        if reviewing
        else "Set the architectural direction for this goal."
    )

    messages = [
        SystemMessage(content=ARCHITECT_PROMPT),
        HumanMessage(content=f"{state_injection}\n\nUser goal: {goal}\n\n{task}"),
    ]

    llm = get_agent_llm("architect")
    response = llm.invoke(messages)
    parsed = _parse_architect_output(response.content, reviewing=reviewing)

    # Keep the opening architecture if the gate pass did not restate it --
    # losing it mid-run would strip the constraints out of every later prompt.
    if parsed["architecture"]:
        state["architecture"] = parsed["architecture"]
    state["verdict"] = parsed["verdict"]

    # A file that does not run cannot be approved work, whatever the Architect
    # concluded. This is the one place the gate's ruling is overridden, and it
    # is deliberate: the Architect reads the failure in `blockers` and had
    # approved past it. The verdict is rewritten rather than the routing
    # patched, so the state says what actually happened.
    #
    # Note the cost: a goal that legitimately calls for a failing file can no
    # longer be approved, and will run to MAX_STEPS before the ceiling ends it.
    blocked = list(state.get("failed_verification") or [])
    if state.get("expect_failures"):
        # The caller asked for a failing file, so a failure is the product.
        # The list stays in state and the report still shows it; it just does
        # not overrule the gate.
        blocked = []
    overridden = bool(blocked) and state["verdict"] == Verdict.APPROVED.value
    if overridden:
        state["verdict"] = Verdict.REVISE.value

    # The gate is the one point every cycle passes through, so it is where the
    # loop counter belongs. The Builder used to own it, which let a
    # Planner/Researcher loop run without ever counting a step.
    # Every pass but the opening one closes a cycle, so that is what counts a
    # step. Keying this off `reviewing` instead undercounted: a Builder that
    # reports nothing leaves `reviewing` False, the gate reads the cycle as a
    # fresh opening pass and sends the work round again, and that shape repeats
    # uncounted until LangGraph hits its recursion limit and kills the run --
    # discarding every message the run had produced. A missing plan is what
    # actually marks the entry pass; a missing report does not.
    if state.get("plan"):
        state["step_count"] = state.get("step_count", 0) + 1

    if overridden:
        note = f" (approval blocked: {len(blocked)} file(s) do not run)"
    elif blocked:
        note = f" ({len(blocked)} file(s) do not run)"
    else:
        note = ""
    state["messages"].append(f"[Architect] Verdict: {state['verdict']}{note}")

    return state


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


# The tools the Builder is allowed to call. GraphRAG is deliberately absent:
# retrieval belongs to the Researcher, and a Builder that can search the corpus
# stops working from the plan it was handed.
BUILDER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "filesystem_read",
            "description": (
                "Read a UTF-8 text file and return its contents. Read a file "
                "before rewriting it, so the rewrite keeps what is already there."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path, relative to the project root.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filesystem_write",
            "description": (
                "Write the COMPLETE new contents of a file, creating parent "
                "directories as needed. This replaces the entire file, so to "
                "change an existing file call filesystem_read first and send "
                "the whole modified text back -- never send only the new lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write."},
                    "content": {
                        "type": "string",
                        "description": "The entire contents the file should have afterwards.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_status",
            "description": "Show the working tree status as porcelain output.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git_diff",
            "description": "Show the unstaged diff, optionally limited to one path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Optional path to diff."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal_execute",
            "description": (
                "Run a simple shell command in the project. Shell "
                "metacharacters are rejected, so pipes and redirection do not work."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "The command to run."}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the pytest suite, optionally limited to one path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Optional test path."}},
            },
        },
    },
]

BUILDER_TOOL_NAMES = {tool["function"]["name"] for tool in BUILDER_TOOLS}

# How many times the Builder may think-and-call before the node gives up. Each
# turn is a cloud round trip, and the Architect gate gets another cycle anyway.
MAX_BUILDER_TOOL_TURNS = 8

# A tool result this long is summarised rather than pasted whole. Large reads
# are the reason: a whole file in the transcript crowds out the plan.
MAX_TOOL_RESULT_CHARS = 20000


def _run_builder_tools(
    llm: Any,
    messages: list[Any],
    files_changed: list[str],
    tool_log: list[str],
) -> tuple[str, bool]:
    """Let the Builder call tools until it stops asking for them.

    Returns the Builder's closing message and whether it ran out of turns.
    `files_changed` is appended to only when a write tool reports success, so
    the list stays a record of what happened rather than what was claimed.
    """
    for _ in range(MAX_BUILDER_TOOL_TURNS):
        response = llm.invoke(messages)
        calls = list(getattr(response, "tool_calls", None) or [])

        if not calls:
            return str(response.content), False

        messages.append(response)

        for call in calls:
            name = str(call.get("name", ""))
            args = dict(call.get("args") or {})

            if name not in BUILDER_TOOL_NAMES:
                # Refused rather than run: the tool split is the whole point,
                # and a Researcher tool reaching the Builder is a real bug
                # worth surfacing in the report instead of silently serving.
                result: Any = {
                    "success": False,
                    "error": f"{name} is not a Builder tool",
                }
            else:
                try:
                    result = _call_mcp_tool_sync(name, args)
                except Exception as exc:
                    result = {"success": False, "error": str(exc)}

            ok = bool(result.get("success")) if isinstance(result, dict) else False

            if name == "filesystem_write" and ok:
                path = str(args.get("path", ""))
                if path and path not in files_changed:
                    files_changed.append(path)

            target = args.get("path") or args.get("command") or ""
            tool_log.append(f"{name}({target}) -> {'ok' if ok else 'failed'}")

            payload = json.dumps(result, default=str)
            if len(payload) > MAX_TOOL_RESULT_CHARS:
                # Flagged, not silently clipped: a truncated read that the model
                # then writes back would delete the tail of the file.
                payload = (
                    payload[:MAX_TOOL_RESULT_CHARS]
                    + '..."TRUNCATED": "Result cut short. Do not write this '
                    'content back to a file -- it is incomplete."'
                )

            messages.append(
                ToolMessage(content=payload, tool_call_id=str(call.get("id", "")))
            )

    return "", True


# Files the Builder writes that can be executed as a script. Anything else it
# produces -- markdown, config, data -- has nothing to run.
RUNNABLE_SUFFIXES = (".py",)

# Per-file ceiling for the verification pass. A written script that hangs is a
# failed verification, not a reason to stall the whole run.
VERIFY_TIMEOUT_SECONDS = 60

# Verification output kept in the report. Enough for the next cycle to see the
# traceback that matters, not so much that it buries the plan.
MAX_VERIFY_DETAIL_CHARS = 800


# A Builder often answers the Blockers section with "none" and then keeps
# writing -- "none - Note: this file raises by design". Only the leading token
# is the answer; the rest is commentary, and keeping the whole string made
# state claim something was blocked while literally saying "none".
#
# Matched only where the token stands as a complete clause: followed by the end
# of the string or a separator. "none of the tests pass" continues into a real
# sentence and stays a blocker -- swallowing that would be the silent success
# this module spends its time preventing. The match is deliberately narrow, so
# an unrecognised phrasing ("none needed") is kept as a blocker rather than
# dropped: a spurious blocker costs a cycle, a dropped one costs the guarantee.
_NO_BLOCKER = re.compile(
    r"^\s*(?:none|n/?a|nothing|no\s+blockers?)\s*(?:$|[-\u2014\u2013:;.,])",
    re.IGNORECASE,
)


def _clean_blockers(text: str) -> str:
    """The Blockers section as an actual blocker, or empty when it says none."""
    if not text or _NO_BLOCKER.match(text):
        return ""
    return text.strip()


def _verify_written_files(
    files_changed: list[str], tool_log: list[str]
) -> list[tuple[str, bool, str]]:
    """Execute the runnable files the Builder wrote and report what happened.

    Writing a file is not evidence that it works. The Builder previously
    reported "Implementation complete" for a module it had never executed, and
    the Architect approved it -- the file raised an AssertionError the first
    time anyone ran it. Running it here means a broken file comes back as a
    blocker the loop can act on, rather than as a success nobody checked.

    Returns one (path, ok, detail) per runnable file.
    """
    results: list[tuple[str, bool, str]] = []

    for path in files_changed:
        if not path.endswith(RUNNABLE_SUFFIXES):
            continue

        try:
            result = _call_mcp_tool_sync(
                "terminal_execute",
                {"command": f"python {path}", "timeout": VERIFY_TIMEOUT_SECONDS},
            )
        except Exception as exc:
            results.append((path, False, str(exc)))
            tool_log.append(f"verify({path}) -> failed")
            continue

        ok = bool(result.get("success")) if isinstance(result, dict) else False
        detail = ""
        if isinstance(result, dict) and not ok:
            # stderr first: a traceback is what the next cycle needs to see.
            detail = str(
                result.get("stderr") or result.get("error") or result.get("stdout") or ""
            ).strip()[:MAX_VERIFY_DETAIL_CHARS]

        results.append((path, ok, detail))
        tool_log.append(f"verify({path}) -> {'ok' if ok else 'failed'}")

    return results


def builder_node(state: AgentState) -> AgentState:
    """Builder: implement the plan by actually calling tools.

    As specified:
    - Filesystem, git, terminal and test tools only (no GraphRAG)
    - Actually make changes via tools
    - Report in strict format
    - Set blockers if stuck

    The model drives the tools. An earlier version regex-scraped the plan for
    `create <file>` plus a quoted string and wrote that, which meant it could
    only ever create whole new files from a plan phrased just so -- every other
    goal, including editing an existing file, reported "Implementation
    complete" having changed nothing.
    """
    state_injection = _get_state_injection(state)
    plan = state.get("plan", "")
    research = state.get("research", "")

    files_changed: list[str] = []
    tool_log: list[str] = []
    exhausted = False

    messages: list[Any] = [
        SystemMessage(content=BUILDER_PROMPT),
        HumanMessage(
            content=f"{state_injection}\n\nPlan to implement:\n{plan}\n\n"
            f"Research findings:\n{research}"
        ),
    ]

    llm = get_agent_llm("builder")
    try:
        tool_llm = llm.bind_tools(BUILDER_TOOLS)
    except AttributeError:
        # A seat whose model cannot call tools at all -- StubLLM, or a tag
        # without tool support. It still reports; it just cannot change a file.
        tool_llm = None

    if tool_llm is None:
        content = str(llm.invoke(messages).content)
    else:
        content, exhausted = _run_builder_tools(
            tool_llm, messages, files_changed, tool_log
        )

    # Every runnable file the Builder wrote is executed before it gets to claim
    # the work is done. This is in code rather than left to the prompt for the
    # same reason files_changed is: the Builder's own account of its work is
    # not evidence.
    #
    # Files that failed on an earlier pass are re-checked even when this pass
    # did not touch them. Verifying only what was just written let the Builder
    # clear a failure by doing nothing: the broken file stayed on disk, the
    # next pass wrote nothing, the failure list came back empty and the gate
    # approved. A file clears only by running clean.
    carried = [
        path
        for path in (state.get("failed_verification") or [])
        if path not in files_changed
    ]
    verification = _verify_written_files(files_changed + carried, tool_log)
    failed = [(path, detail) for path, ok, detail in verification if not ok]

    parsed = _parse_builder_output(content)
    builder_report = parsed.get("changes_made") or content or "No report produced."

    if verification:
        builder_report += "\n\nVerification (each runnable file was executed):\n"
        # fall through to the per-file lines below
        builder_report += "\n".join(
            f"- {path}: {'ran clean' if ok else 'FAILED'}"
            + (f"\n{detail}" if detail else "")
            for path, ok, detail in verification
        )

    if tool_log:
        builder_report += "\n\nTool calls:\n" + "\n".join(f"- {c}" for c in tool_log)

    # files_changed is deliberately NOT taken from the model's prose. A file
    # counts as changed only when a write tool reported success for it; a model
    # that describes writing a module it never wrote would otherwise have the
    # console report "changed this machine" for work that never touched disk.
    claimed = [
        path for path in parsed.get("files_modified", []) if path not in files_changed
    ]
    if claimed:
        builder_report += (
            "\n\nDescribed but not written (no successful write call): "
            + ", ".join(claimed)
        )

    blockers = _clean_blockers(parsed.get("next_steps_blockers", ""))

    # A failed verification outranks whatever the model concluded: it wrote a
    # file that does not run, and the Architect must see that as unfinished --
    # unless the run was asked for one, in which case it is the product, not a
    # defect. It is still executed and still reported either way.
    expected = bool(state.get("expect_failures"))
    if failed and not expected:
        blockers = "Files that do not run: " + "; ".join(
            f"{path} ({detail.splitlines()[-1] if detail else 'no output'})"
            for path, detail in failed
        )

    if exhausted and not blockers:
        blockers = (
            f"Builder stopped after {MAX_BUILDER_TOOL_TURNS} tool turns without "
            "finishing. Narrow the plan or split it into smaller steps."
        )

    state["builder_report"] = builder_report
    state["files_changed"] = files_changed
    # Written every pass, including empty, so a file that gets fixed on a later
    # cycle stops blocking the gate.
    state["failed_verification"] = [path for path, _ in failed]
    state["blockers"] = blockers
    # The feed line has to carry the verification result too. "Implementation
    # complete" beside a file that does not run is the same false claim this
    # pass exists to catch, and it is what the Architect reads in state.
    if failed:
        suffix = " (expected for this run)" if expected else ""
        summary = (
            f"Wrote {len(files_changed)} file(s); {len(failed)} do not run{suffix}"
        )
    else:
        summary = f"Implementation complete. Files: {len(files_changed)}"
    state["messages"].append(f"[Builder] {summary}")

    # step_count is incremented by the Architect gate, not here: every cycle
    # passes through the gate, but a Planner/Researcher loop never reaches the
    # Builder, and counting here let those loops run uncounted.

    return state
