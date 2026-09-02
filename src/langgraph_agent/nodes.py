"""Agent nodes: Architect, Planner, Researcher, Builder.

Implements the 4-Agent System with strict prompts and tool binding:
- Architect: reasoning only, no tools; sets direction and holds the approval gate
- Planner: reasoning only, no tools
- Researcher: GraphRAG MCP tools only
- Builder: filesystem, git, terminal tools only
"""

import asyncio
import json
import os
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar, cast

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langgraph_agent.config import get_agent_llm
from langgraph_agent.control import RUN_CONTROL
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


_T = TypeVar("_T")

# How long one node may spend gathering its answer. This is the second half of
# the timeout story and is not made redundant by `LLM_TIMEOUT_SECONDS`: that one
# bounds the socket, so it catches a connection that goes quiet but not a model
# that streams tokens slowly and indefinitely, and not a node that makes several
# calls each of which finishes just inside its own limit.
NODE_DEADLINE_SECONDS = float(os.getenv("NODE_DEADLINE_SECONDS", "150"))


class _Deadline:
    """A monotonic countdown shared across the several calls one node makes.

    The Researcher runs a single retrieval and can be bounded by wrapping it.
    A node that makes many calls in sequence -- the Builder's turn loop, then
    its verification pass -- needs the budget to travel with it, or each call
    gets the full allowance and the node as a whole is bounded by nothing.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._end = time.monotonic() + seconds

    def remaining(self) -> float:
        return max(0.0, self._end - time.monotonic())

    def expired(self) -> bool:
        return self.remaining() <= 0.0


def _with_deadline(work: Callable[[], _T], seconds: float, fallback: _T) -> _T:
    """Run `work`, giving up on it after `seconds` and returning `fallback`.

    The abandoned call cannot actually be cancelled -- Python cannot interrupt a
    thread blocked on a socket -- so the worker is left to unwind on its own
    when the client timeout fires. Two consequences shape this code.

    First, `work` must not write to state: a late finisher would otherwise land
    its result in a state the graph had already moved past. Callers pass a
    function that only reads and apply what it returns themselves.

    Second, the worker is a bare daemon thread rather than a
    `ThreadPoolExecutor`. Pool threads are non-daemon and the module's atexit
    hook joins them, so one abandoned worker would hold up interpreter shutdown
    for as long as it stayed blocked -- turning a bounded node into an
    unkillable server.
    """
    box: list[Any] = []
    error: list[BaseException] = []

    def _run() -> None:
        try:
            box.append(work())
        except BaseException as exc:  # re-raised on the caller's thread below
            error.append(exc)

    thread = threading.Thread(target=_run, daemon=True, name="node-deadline")
    thread.start()
    thread.join(timeout=seconds)

    if thread.is_alive():
        return fallback
    if error:
        # A seat that failed outright is not a timeout; let it raise so
        # `_SeatLLM` records the reason and the console can show it.
        raise error[0]
    return cast("_T", box[0])


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


# A `## Files Modified` line is prose, and the paths in `files_changed` are raw
# `filesystem_write` arguments. The "Described but not written" check compares
# the two, so every way the model can spell a path it really did write turns
# into a false accusation -- and it is the worst one the report makes: the
# mirror of a Builder claiming a file it never wrote, pointing the other way,
# read by an Architect that rules on the report. The model decorates the line
# in all the ordinary ways -- `- \`test_spectral_graph.py\``, `- **file.py**`,
# `- ./file.py`, an absolute path, `* file.py`, `1. file.py` -- and above all
# it annotates: `- test_spectral_graph.py (new file)`, which put a written,
# executed, passing file under "Described but not written".
_LIST_MARKER = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_MARKDOWN_WRAP = re.compile(r"^(?:\*\*|__|`|\*|_)+|(?:\*\*|__|`|\*|_)+$")
# A whitelist rather than "strip any trailing parenthetical": this project has
# real filenames that carry parentheses (examples/filter_band_pass_(40_60_hz).png),
# and mangling one of those would reintroduce the same false accusation.
_PATH_ANNOTATION = re.compile(
    r"\s+\((?:new|newly|created|create|added|modified|updated|edited|changed|"
    r"rewritten|rewrote|overwritten|existing|unchanged|deleted|removed)\b[^()]*\)$",
    re.IGNORECASE,
)
# Lines that are an answer of "nothing", not a path. Without these a Builder
# that honestly reported writing no files was accused of not writing "None".
_NOT_A_PATH = {"", ".", "-", "none", "n/a", "na", "(none)", "nothing", "no files"}


def _report_path_key(text: str) -> str:
    """Normalize one `## Files Modified` entry for comparison against tool records.

    Returns "" for a line that is not naming a file at all. Normalization is
    deliberately one-directional in its risk: over-stripping could only ever
    hide a real accusation, while under-stripping invents one, and an invented
    one is what reaches the Architect as evidence.
    """
    path = _LIST_MARKER.sub("", text).strip()
    path = _MARKDOWN_WRAP.sub("", path).strip()
    path = _PATH_ANNOTATION.sub("", path).strip()
    path = _MARKDOWN_WRAP.sub("", path).strip().rstrip(":,;").strip()

    if not path or path.lower() in _NOT_A_PATH:
        return ""
    # Prose, not a path: a sentence in place of a bullet list. Five words is
    # well clear of any real filename, and missing one costs a warning we did
    # not print rather than one we made up.
    if len(path.split()) >= 5:
        return ""

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(Path.cwd())
        except ValueError:
            # Outside the project; keep it absolute and let it compare as-is.
            pass
    return os.path.normpath(str(candidate))


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
        # Extract file paths from bullet list, normalized: the raw line carries
        # list markers, markdown and "(new file)"-style annotations that no
        # recorded tool path will ever match.
        seen: set[str] = set()
        paths: list[str] = []
        for line in files_match.group(1).strip().split("\n"):
            key = _report_path_key(line)
            if key and key not in seen:
                seen.add(key)
                paths.append(key)
        result["files_modified"] = paths

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


def _rule_on_state(state: AgentState, reviewing: bool) -> dict[str, str]:
    """Ask the Architect for direction or a ruling; return the parsed output.

    Reads state; never writes it, so it is safe to run under `_with_deadline`.
    """
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
    return _parse_architect_output(response.content, reviewing=reviewing)


def architect_node(state: AgentState) -> AgentState:
    """Architect: set direction, then rule on whether the work is done.

    No tools. Runs twice per cycle -- as the entry authority before the Planner,
    and as the approval gate after the Builder reports. A populated builder
    report is what tells the two passes apart.

    Bounded by `NODE_DEADLINE_SECONDS`. The fallback verdict is never
    `approved`, and that is the whole point of choosing one here: this gate is
    what ends the run, so a seat that stalled must not be able to end one
    successfully. `revise` on the gate pass and `plan` on the opening pass both
    route back to the Planner, and the step counter below carries a repeatedly
    stalling Architect to MAX_STEPS instead of letting it spin.
    """
    reviewing = bool(state.get("builder_report"))

    if RUN_CONTROL.stopped():
        # A stopped gate can no more end the run successfully than a timed-out
        # one can, and for the same reason: this node is what ends it. Nothing
        # was ruled here, so nothing is approved. `step_count` is deliberately
        # not advanced -- this pass did no work -- and the architecture already
        # in state is kept, since the recovered run is read with it.
        if reviewing:
            state["verdict"] = Verdict.REVISE.value
        state["messages"].append(
            "[Architect] Stopped by the emergency stop before this seat ran; "
            "no verdict was reached."
        )
        return state

    parsed = _with_deadline(
        lambda: _rule_on_state(state, reviewing), NODE_DEADLINE_SECONDS, None
    )
    timed_out = parsed is None
    if parsed is None:
        parsed = {
            "architecture": "",
            "verdict": (
                Verdict.REVISE.value if state.get("plan") else Verdict.PLAN.value
            ),
        }

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

    if timed_out:
        # Said plainly, because the run reads this line to know what happened:
        # a verdict nobody actually reached is not the same as a ruling.
        note = (
            f" (no response within {int(NODE_DEADLINE_SECONDS)}s -- "
            "not a ruling, and never an approval)"
        )
    elif overridden:
        note = f" (approval blocked: {len(blocked)} file(s) do not run)"
    elif blocked:
        note = f" ({len(blocked)} file(s) do not run)"
    else:
        note = ""
    state["messages"].append(f"[Architect] Verdict: {state['verdict']}{note}")

    return state


def _make_plan(state: AgentState) -> dict[str, Any]:
    """Ask the Planner for a plan; return the parsed output.

    Reads state; never writes it, so it is safe to run under `_with_deadline`.
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
    return _parse_planner_output(response.content)


# What a timed-out Planner leaves in `plan`. It must not be empty, and that is
# load-bearing rather than cosmetic: the Architect increments `step_count` only
# while a plan exists, so an empty one would send the run round the
# Planner/Builder loop uncounted until LangGraph's recursion limit killed it by
# exception -- discarding every message the run had produced. That is the exact
# failure the counter was moved to the gate to prevent, and an empty fallback
# here would reintroduce it through the back door.
_PLANNER_TIMED_OUT = (
    "1. The Planner did not respond within {seconds}s, so this goal was never "
    "broken into steps.\n"
    "2. Do not guess at the plan. Report the goal as unplanned, and set that "
    "as a blocker so the Architect sees why nothing was implemented.\n"
)


def planner_node(state: AgentState) -> AgentState:
    """Planner: Interpret goal, create structured plan, choose next agent.

    As specified:
    - No tools
    - Output strict format
    - Routes to Researcher when knowledge needed, Builder when task is clear

    Bounded by `NODE_DEADLINE_SECONDS`; see `_PLANNER_TIMED_OUT` for why the
    fallback plan is a real string rather than an empty one.
    """
    if RUN_CONTROL.stopped():
        # Any plan already in state is kept -- see `_PLANNER_TIMED_OUT` for why
        # an empty one here is load-bearing rather than cosmetic. `next_agent`
        # is left alone so the recovered state still records the last real
        # routing decision rather than one nobody made.
        state["messages"].append(
            "[Planner] Stopped by the emergency stop before this seat ran."
        )
        return state

    parsed = _with_deadline(
        lambda: _make_plan(state), NODE_DEADLINE_SECONDS, None
    )

    if parsed is None:
        # An existing plan beats the placeholder: on a revise cycle state
        # already holds a real one, and re-running it is better information
        # than a note saying the seat stalled.
        state["plan"] = state.get("plan") or _PLANNER_TIMED_OUT.format(
            seconds=int(NODE_DEADLINE_SECONDS)
        )
        # Builder rather than Researcher: it is the shorter path back to the
        # Architect, which is the only node that can end the run, and a second
        # slow seat in between is the last thing a stalling run needs.
        state["next_agent"] = "Builder"
        state["messages"].append(
            f"[Planner] No response within {int(NODE_DEADLINE_SECONDS)}s; "
            "routing to Builder"
        )
        return state

    # Update state
    state["plan"] = parsed.get("plan", "")
    state["next_agent"] = parsed.get("next_agent", "Builder")
    state["messages"].append(f"[Planner] Plan created. Next agent: {state['next_agent']}")

    return state


def _as_text(content: Any) -> str:
    """Flatten a message's content to text.

    Providers differ: some return a plain string, some a list of content
    blocks. `len()` and the section regexes both read a list as truthy
    non-empty, so an answer that carried no text at all still looked like
    findings.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        ]
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def _said_nothing(content: str, parsed: dict[str, Any]) -> bool:
    """True when the Researcher's answer carries no findings.

    Two shapes count as nothing: an empty (or whitespace) response, and one
    that filled in the headings and the status line but left every section
    blank. Both reach the Builder as an empty `research`.
    """
    if not content.strip():
        return True
    return not any(
        parsed.get(section, "").strip()
        for section in ("key_findings", "relevant_context", "recommendations")
    )


# Internal marker, never a `research_status` in state: `researcher_node` maps
# it to `no_relevant_knowledge` and emits its own message. It exists for the
# same reason the deadline's does -- a seat that answered with nothing and a
# corpus with nothing to say both reach the Builder empty-handed, and the feed
# is where an operator finds out which one happened.
_SEAT_EMPTY = "seat_empty"


# What the Researcher hands the Builder when its seat answered with nothing.
# Worded apart from the timeout and from a genuinely empty corpus: all three
# arrive with no findings, and only this one means the seat is not working.
# Saying so in the text is what stops the Builder's report from implying the
# corpus was searched and found wanting.
_RESEARCH_EMPTY = (
    "## Key Findings\nNone -- the Researcher's seat returned no findings.\n\n"
    "## Relevant Context\nThe model answered with nothing usable, so no "
    "retrieval was summarized. This says nothing about the knowledge base: "
    "the corpus may hold relevant material that was never reported. Check "
    "the seat's model on the console before reading this as an empty "
    "corpus.\n\n"
    "## Recommendations for Builder\nWork from the plan alone, and say in the "
    "report that it was built without research because the Researcher seat "
    "returned nothing.\n\n"
    "## Status\nno_relevant_knowledge"
)


def _gather_research(state: AgentState) -> tuple[str, str]:
    """Retrieve for the Researcher and return `(findings, status)`.

    Reads state; never writes it. Split out of `researcher_node` so it can run
    under `_with_deadline`, which may abandon it still running -- see that
    function for why a worker that writes state is a bug.
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
        research_findings = _as_text(response.content)

        # Parse status from LLM response
        parsed = _parse_researcher_output(research_findings)
        research_status = parsed.get("status", "ready_for_builder")

        # A seat that answered with nothing has not done research, whatever the
        # parsed status says -- and the status defaults to `ready_for_builder`,
        # so silence was being announced as success. That is the same rule the
        # Builder is held to: the seat's account of its own work is not
        # evidence. The cost of missing it is not one bad cycle but a loop --
        # empty `research` reaches the Builder, whose report says the store is
        # empty, so the gate rules `need_research` and sends it back to the
        # same silent seat, burning a step at the ceiling every time.
        if _said_nothing(research_findings, parsed):
            research_findings = _RESEARCH_EMPTY
            research_status = _SEAT_EMPTY

    return research_findings, research_status


# What the Researcher hands the Builder when it runs out of time. The status is
# `no_relevant_knowledge` rather than `need_replan` because a deadline says
# nothing about the plan -- looping back to the Planner would re-run the same
# slow retrieval and burn the run's budget on it. The Builder is told plainly
# that it has no research, so its report cannot silently imply otherwise.
_RESEARCH_TIMED_OUT = (
    "## Key Findings\nNone -- retrieval did not finish.\n\n"
    "## Relevant Context\nThe Researcher was stopped at its "
    "{seconds}s deadline before it produced findings. Treat this as no "
    "research rather than as an empty corpus: the knowledge base may well "
    "hold relevant material that was not retrieved in time.\n\n"
    "## Recommendations for Builder\nWork from the plan alone, and say in "
    "the report that it was built without research.\n\n"
    "## Status\nno_relevant_knowledge"
)


def researcher_node(state: AgentState) -> AgentState:
    """Researcher: Query GraphRAG, summarize findings, recommend approach.

    As specified:
    - GraphRAG tools only
    - Output strict format with status
    - Status guides next steps (ready_for_builder | need_replan | no_relevant_knowledge)

    Calls the GraphRAG MCP tool (`search_knowledge_graph`) rather than importing
    the knowledge base directly, preserving the documented tool boundary.
    Falls back to the LLM if the knowledge base is empty or the MCP tool fails.

    Bounded by `NODE_DEADLINE_SECONDS`. Without it a stalled seat hung the whole
    run here: `RUN_BUDGET_SECONDS` is checked between graph supersteps, and a
    node that never returns never reaches one, so the run sat inside this
    function indefinitely while the console still showed the Planner as current.
    """
    if RUN_CONTROL.stopped():
        # Distinct from an empty corpus: nothing was retrieved because nothing
        # was attempted. Leaving `research` untouched keeps whatever an earlier
        # cycle found instead of overwriting it with a note.
        state["messages"].append(
            "[Researcher] Stopped by the emergency stop before this seat ran."
        )
        return state

    timed_out = (_RESEARCH_TIMED_OUT.format(seconds=int(NODE_DEADLINE_SECONDS)), "timed_out")
    research_findings, research_status = _with_deadline(
        lambda: _gather_research(state), NODE_DEADLINE_SECONDS, timed_out
    )

    deadline_hit = research_status == "timed_out"
    seat_empty = research_status == _SEAT_EMPTY
    if deadline_hit or seat_empty:
        research_status = ResearchStatus.NO_RELEVANT_KNOWLEDGE.value

    # Update state
    state["research"] = research_findings
    state["research_status"] = research_status

    # Route based on status
    if deadline_hit:
        # Worded apart from the ordinary `no_relevant_knowledge` message: a
        # corpus with nothing to say and a seat that stopped answering both
        # reach the Builder empty-handed, and only one of them is a problem.
        state["next_agent"] = "Builder"
        state["messages"].append(
            f"[Researcher] No response within {int(NODE_DEADLINE_SECONDS)}s; "
            "routing to Builder without research"
        )
    elif seat_empty:
        # Routed to the Builder, not back to the Planner: the plan is not what
        # failed, and re-planning would send the run at the same silent seat
        # again. Named in the feed so the operator can change the seat's model,
        # which is the only thing that actually fixes it.
        state["next_agent"] = "Builder"
        state["messages"].append(
            "[Researcher] Seat returned no findings; routing to Builder "
            "without research (check the Researcher's model)"
        )
    elif research_status == "need_replan":
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

# Wall-clock ceiling for the whole Builder turn, larger than the Researcher's
# because this node legitimately makes many calls: up to MAX_BUILDER_TOOL_TURNS
# round trips, each of which may run tests. The turn cap bounds how many calls
# it makes and says nothing about how long they take -- the same gap that let a
# stalled Researcher hang a run.
BUILDER_DEADLINE_SECONDS = float(os.getenv("BUILDER_DEADLINE_SECONDS", "240"))

# Held back from the tool loop so the verification pass always gets to run.
# Verification is what stops the Builder claiming work it never proved, so
# letting the loop spend the entire budget would trade the guarantee for one
# more tool call. Whatever the loop leaves unused is added to this.
VERIFY_RESERVE_SECONDS = float(os.getenv("VERIFY_RESERVE_SECONDS", "60"))

# A tool result this long is summarised rather than pasted whole. Large reads
# are the reason: a whole file in the transcript crowds out the plan.
MAX_TOOL_RESULT_CHARS = 20000


def _run_builder_tools(
    llm: Any,
    messages: list[Any],
    files_changed: list[str],
    tool_log: list[str],
    deadline: _Deadline,
) -> tuple[str, bool, bool, bool]:
    """Let the Builder call tools until it stops asking for them.

    Returns the Builder's closing message, whether it ran out of turns, whether
    it ran out of time, and whether the operator stopped it. `files_changed` is
    appended to only when a write tool reports success, so the list stays a
    record of what happened rather than what was claimed.

    The deadline is enforced in two places, and only one of them may abandon
    work. The model's own call is wrapped, because discarding a half-received
    response costs nothing but the turn. The tool calls underneath it are not:
    they write files, stage commits and run commands, and a worker abandoned
    mid-`filesystem_write` would keep writing into the project after this node
    returned. So each turn's tools always run to completion, and the budget is
    re-checked at the top of the next turn instead.

    The emergency stop obeys the same rule, and is checked in the same place --
    never inside the batch below. Skipping the remaining calls of a batch would
    leave `ToolMessage` replies missing for `tool_call_id`s the model has
    already been told about, which corrupts the message list rather than ending
    cleanly. It is kept apart from the deadline so the report can say which one
    happened: a run the operator stopped must not be described as one that
    exceeded its own budget.
    """
    for _ in range(MAX_BUILDER_TOOL_TURNS):
        if RUN_CONTROL.stopped():
            return "", False, False, True
        if deadline.expired():
            return "", False, True, False

        # None is the sentinel for "gave up"; a real response is never None.
        response = _with_deadline(
            lambda: llm.invoke(messages), deadline.remaining(), None
        )
        if response is None:
            return "", False, True, False

        calls = list(getattr(response, "tool_calls", None) or [])

        if not calls:
            return str(response.content), False, False, False

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

    return "", True, False, False


# Files the Builder writes that can be executed as a script. Anything else it
# produces -- markdown, config, data -- has nothing to run.
RUNNABLE_SUFFIXES = (".py",)

# Verification runs a file to prove it does not raise, with nobody watching.
# Anything that opens a window waits for a human to close it, so a correct
# script ending in `plt.show()` -- the ordinary way to write a plotting
# example -- burned a full VERIFY_TIMEOUT_SECONDS and came back FAILED. The
# display variables are removed as well as MPLBACKEND being set, because a
# library that checks for a display itself never consults MPLBACKEND.
HEADLESS_VERIFY_ENV: dict[str, str | None] = {
    "MPLBACKEND": "Agg",
    "DISPLAY": None,
    "WAYLAND_DISPLAY": None,
}

# Per-file ceiling for the verification pass. A written script that hangs is a
# failed verification, not a reason to stall the whole run.
VERIFY_TIMEOUT_SECONDS = 60

# Introduces what a hung file printed before it was killed.
_TIMEOUT_OUTPUT_HEADING = "\nOutput before it was killed (tail):\n"

# Verification output kept in the report. Enough for the next cycle to see the
# traceback that matters, not so much that it buries the plan.
MAX_VERIFY_DETAIL_CHARS = 800

# Why a package module is not executed, said in the report so the Architect
# reads a reason rather than a silence.
PACKAGE_MODULE_SKIP_REASON = (
    "not executed: module inside a package, which `python <path>` cannot import. "
    "Cover it with a root-level script that imports the package."
)

# The least time worth starting a file in. Below this the remaining slice
# rounds down to a timeout nothing can finish inside, and the file comes back
# FAILED -- which is not merely useless but wrong: it accuses a working file of
# not running, and sets the blocker that says so. Better to admit it was never
# executed.
MIN_VERIFY_SLICE_SECONDS = 1.0

# Why a file was left unrun when the operator stopped the run. Worded apart
# from the deadline reason because the two are not the same event, and the
# report is the only place anyone finds out which one happened.
VERIFY_STOPPED_REASON = (
    "not executed: the run was stopped before this file was reached. It is "
    "unproven, not passing, and is re-checked on the next cycle."
)

# Why a file was left unrun when the Builder's budget ran out.
VERIFY_DEADLINE_SKIP_REASON = (
    "not executed: the Builder's deadline passed before this file was reached. "
    "It is unproven, not passing, and is re-checked on the next cycle."
)

# How each verification status reads in the report. "SKIPPED" and "NOT RUN" are
# shouted like "FAILED" on purpose: an unexecuted file is not a passing one.
_VERIFY_LABELS = {
    "ok": "ran clean",
    "failed": "FAILED",
    "skipped": "SKIPPED",
    "unverified": "NOT RUN",
}


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


def _is_package_module(path: str) -> bool:
    """True for a .py file that lives inside a Python package.

    `python pkg/mod.py` puts *pkg* on sys.path rather than the project root, so
    a module that imports its own package absolutely -- `from pkg.other import
    x`, the normal way to write one -- dies with ModuleNotFoundError no matter
    how correct it is. Executing it proves nothing about the code and produces
    a failure that cannot be fixed inside the file.

    The test is the one Python itself uses to decide what a package is: the
    directory holding the file has an `__init__.py`.
    """
    parent = Path(path).parent
    return (parent / "__init__.py").exists()


def _timeout_detail(result: dict[str, Any]) -> str:
    """Describe a timed-out verification with the output it produced.

    The timeout message alone says a file hung but not where, which is the
    difference between a script that blocked on its first line and one that
    did all its work and then waited at `plt.show()`. Without it the Builder
    guesses -- it read a bare timeout as a missing dependency once, installed
    a package that was already there, and spent a second full timeout on an
    identical retry. The tail is kept rather than the head: what a hung
    process printed last is how far it got.
    """
    message = str(result.get("error") or "timed out").strip()
    printed = str(result.get("stdout") or "").strip() or str(result.get("stderr") or "").strip()
    if not printed:
        return message[:MAX_VERIFY_DETAIL_CHARS]

    room = MAX_VERIFY_DETAIL_CHARS - len(message) - len(_TIMEOUT_OUTPUT_HEADING)
    if room <= 0:
        return message[:MAX_VERIFY_DETAIL_CHARS]

    tail = printed[-room:]
    if len(tail) < len(printed):
        tail = "..." + tail[3:]
    return f"{message}{_TIMEOUT_OUTPUT_HEADING}{tail}"


def _verify_written_files(
    files_changed: list[str],
    tool_log: list[str],
    deadline: _Deadline | None = None,
) -> list[tuple[str, str, str]]:
    """Execute the runnable files the Builder wrote and report what happened.

    Writing a file is not evidence that it works. The Builder previously
    reported "Implementation complete" for a module it had never executed, and
    the Architect approved it -- the file raised an AssertionError the first
    time anyone ran it. Running it here means a broken file comes back as a
    blocker the loop can act on, rather than as a success nobody checked.

    A file inside a package is skipped instead: see `_is_package_module`. The
    skip is reported, never silent -- a module nobody ran is exactly what this
    pass exists to surface, and the way to cover one is a root-level script
    that imports it, which this pass does execute.

    Each file is bounded by VERIFY_TIMEOUT_SECONDS, but the number of files is
    not, so `deadline` bounds the pass as a whole. Files past it come back
    "unverified" rather than "ok": treating an unrun file as passing is the
    exact false clearance this pass exists to prevent, and the caller keeps
    them in `failed_verification` so the next cycle re-runs them. The emergency
    stop lands in the same place and with the same status, for the same reason.

    Returns one (path, status, detail) per runnable file, where status is
    "ok", "failed", "skipped" or "unverified".
    """
    results: list[tuple[str, str, str]] = []

    for path in files_changed:
        if not path.endswith(RUNNABLE_SUFFIXES):
            continue

        if _is_package_module(path):
            results.append((path, "skipped", PACKAGE_MODULE_SKIP_REASON))
            tool_log.append(f"verify({path}) -> skipped")
            continue

        if RUN_CONTROL.stopped():
            # `unverified`, not `skipped`: nobody ran this file, which is
            # exactly the gap this pass exists to surface. It keeps blocking
            # approval even under `expect_failures`, because that opt-out is
            # for a file the run meant to fail -- still executed, still
            # reported -- not for one that was never executed at all.
            results.append((path, "unverified", VERIFY_STOPPED_REASON))
            tool_log.append(f"verify({path}) -> not run (stopped)")
            continue

        if deadline is not None and deadline.remaining() < MIN_VERIFY_SLICE_SECONDS:
            results.append((path, "unverified", VERIFY_DEADLINE_SKIP_REASON))
            tool_log.append(f"verify({path}) -> not run (deadline)")
            continue

        try:
            result = _call_mcp_tool_sync(
                "terminal_execute",
                {
                    "command": f"python {path}",
                    # Never let one file overrun what is left for the rest, and
                    # never hand it a slice too small to run in -- see
                    # MIN_VERIFY_SLICE_SECONDS.
                    "timeout": (
                        VERIFY_TIMEOUT_SECONDS
                        if deadline is None
                        else max(1, int(min(VERIFY_TIMEOUT_SECONDS, deadline.remaining())))
                    ),
                    "env": HEADLESS_VERIFY_ENV,
                },
            )
        except Exception as exc:
            results.append((path, "failed", str(exc)))
            tool_log.append(f"verify({path}) -> failed")
            continue

        ok = bool(result.get("success")) if isinstance(result, dict) else False
        detail = ""
        if isinstance(result, dict) and not ok:
            # stderr first: a traceback is what the next cycle needs to see.
            detail = str(
                result.get("stderr") or result.get("error") or result.get("stdout") or ""
            ).strip()[:MAX_VERIFY_DETAIL_CHARS]
            if result.get("timed_out"):
                detail = _timeout_detail(result)

        results.append((path, "ok" if ok else "failed", detail))
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
    out_of_time = False
    stopped = False

    # The loop is held to the budget minus the verification reserve; whatever it
    # does not spend is handed on below, so a quick build still gets a long
    # verification pass and a slow one cannot starve it entirely.
    loop_deadline = _Deadline(
        max(0.0, BUILDER_DEADLINE_SECONDS - VERIFY_RESERVE_SECONDS)
    )

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

    if RUN_CONTROL.stopped():
        # Nothing new is started once the stop is in -- no model call, no
        # tools. The verification pass below still runs, and its own stop check
        # brings back every carried file as unproven rather than clear, which
        # is what keeps a stopped run from looking finished.
        content, stopped = "", True
    elif tool_llm is None:
        reply = _with_deadline(
            lambda: str(llm.invoke(messages).content), loop_deadline.remaining(), None
        )
        out_of_time = reply is None
        content = reply or ""
    else:
        content, exhausted, out_of_time, stopped = _run_builder_tools(
            tool_llm, messages, files_changed, tool_log, loop_deadline
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
    # A carried file that is gone from disk drops out instead of failing.
    # Deleting it is a real fix -- a file that does not exist cannot raise --
    # but `python <missing path>` exits non-zero forever, so re-running it
    # pinned failed_verification open and the gate rewrote every `approved` to
    # `revise` until the step ceiling ended the run. Only carried paths get
    # this; a path in files_changed was just written by a tool that reported
    # success, and its absence would be the write lying.
    carried = [
        path
        for path in (state.get("failed_verification") or [])
        if path not in files_changed and Path(path).exists()
    ]
    # The reserve plus whatever the tool loop left unspent.
    verify_deadline = _Deadline(VERIFY_RESERVE_SECONDS + loop_deadline.remaining())
    verification = _verify_written_files(
        files_changed + carried, tool_log, verify_deadline
    )
    failed = [
        (path, detail) for path, status, detail in verification if status == "failed"
    ]
    skipped = [path for path, status, _ in verification if status == "skipped"]
    unverified = [path for path, status, _ in verification if status == "unverified"]

    parsed = _parse_builder_output(content)
    builder_report = parsed.get("changes_made") or content or "No report produced."

    if verification:
        builder_report += (
            "\n\nVerification (each runnable file was executed, except as noted):\n"
        )
        # fall through to the per-file lines below
        builder_report += "\n".join(
            f"- {path}: {_VERIFY_LABELS[status]}" + (f"\n{detail}" if detail else "")
            for path, status, detail in verification
        )
    if skipped:
        builder_report += (
            f"\n\n{len(skipped)} package module(s) were not executed. A package "
            "module only proves itself through a root-level script that imports "
            "it; write one if none of the scripts above cover it."
        )
    if unverified:
        why = (
            "the run was stopped"
            if stopped
            else f"the Builder ran out of its {int(BUILDER_DEADLINE_SECONDS)}s budget"
        )
        builder_report += (
            f"\n\n{len(unverified)} file(s) were not executed: {why}. They are "
            "unproven rather than working, and are re-checked next cycle."
        )

    if tool_log:
        builder_report += "\n\nTool calls:\n" + "\n".join(f"- {c}" for c in tool_log)

    # `files_changed` above is what *this* pass wrote, and the verification
    # logic needs it to stay that way: `carried` leans on it to tell a path
    # this pass rewrote from one only an earlier pass touched. The state field
    # answers a different question -- what the whole run produced -- so it
    # accumulates. Overwriting it per pass meant a run that wrote a file on one
    # cycle and nothing on the next ended reporting it had changed nothing
    # while the file sat on disk, and the Architect ruled on that empty record:
    # a build with a file to its name was approved as having produced none.
    # That is the same false account as a Builder claiming a file it never
    # wrote, pointing the other way.
    previously_changed = list(state.get("files_changed") or [])
    all_files_changed = previously_changed + [
        path for path in files_changed if path not in previously_changed
    ]

    # files_changed is deliberately NOT taken from the model's prose. A file
    # counts as changed only when a write tool reported success for it; a model
    # that describes writing a module it never wrote would otherwise have the
    # console report "changed this machine" for work that never touched disk.
    # Checked against the run's whole record rather than this pass: a file an
    # earlier pass wrote did come from a successful write call, so naming it
    # again is not a claim about work that never happened.
    # Both sides go through the same normalizer before they are compared. The
    # tool side needs it as much as the prose side: `filesystem_write` records
    # whatever argument the model passed, so a pass that wrote `./foo.py` and a
    # report naming `foo.py` are the same file and must not read as a lie.
    written = {_report_path_key(path) for path in all_files_changed}
    claimed = [path for path in parsed.get("files_modified", []) if path not in written]
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

    # An unproven file blocks whatever `expect_failures` says. That opt-out is
    # for a file the run meant to fail -- it is still executed and still
    # reported -- not for one that was never executed at all, which is a gap in
    # the evidence rather than an expected result.
    if unverified:
        lead = "Not executed before the stop" if stopped else "Not executed before the deadline"
        note = f"{lead}: " + ", ".join(unverified)
        blockers = f"{blockers}. {note}" if blockers else note

    if stopped and not blockers:
        blockers = (
            "Stopped by the emergency stop before the Builder finished. "
            "Anything it had already written is kept; anything it did not get "
            "to run is listed above as unproven."
        )

    if out_of_time and not blockers:
        blockers = (
            f"Builder stopped at its {int(BUILDER_DEADLINE_SECONDS)}s deadline "
            "without finishing. Anything it had already written is kept and "
            "verified; narrow the plan or raise BUILDER_DEADLINE_SECONDS."
        )

    if exhausted and not blockers:
        blockers = (
            f"Builder stopped after {MAX_BUILDER_TOOL_TURNS} tool turns without "
            "finishing. Narrow the plan or split it into smaller steps."
        )

    state["builder_report"] = builder_report
    state["files_changed"] = all_files_changed
    # Written every pass, including empty, so a file that gets fixed on a later
    # cycle stops blocking the gate. Unverified paths ride along so the next
    # cycle re-runs them: a file nobody executed must not clear by omission,
    # which is the same rule that keeps a failed file on the list.
    state["failed_verification"] = [path for path, _ in failed] + unverified
    state["blockers"] = blockers
    # The feed line has to carry the verification result too. "Implementation
    # complete" beside a file that does not run is the same false claim this
    # pass exists to catch, and it is what the Architect reads in state.
    if failed:
        suffix = " (expected for this run)" if expected else ""
        summary = (
            f"Wrote {len(files_changed)} file(s); {len(failed)} do not run{suffix}"
        )
    elif unverified:
        why = "stopped" if stopped else "deadline"
        summary = (
            f"Wrote {len(files_changed)} file(s); {len(unverified)} not run "
            f"({why})"
        )
    elif stopped:
        # Never "Implementation complete", for the same reason a deadline-cut
        # Builder never says it: the node was cut off, and the feed line is
        # what the Architect and the operator read.
        summary = f"Stopped by the operator. Files: {len(files_changed)}"
    elif out_of_time:
        # Never "Implementation complete": the node was cut off, and the feed
        # saying otherwise is the same false claim the verification pass exists
        # to catch.
        summary = (
            f"Stopped at the {int(BUILDER_DEADLINE_SECONDS)}s deadline. "
            f"Files: {len(files_changed)}"
        )
    else:
        summary = f"Implementation complete. Files: {len(files_changed)}"
    # Every count above is this pass, which is what just happened and so what
    # the feed should say. But a pass that wrote nothing, on a run that has
    # already produced files, would read as though the run had lost them -- so
    # the run total is named whenever it differs from the pass.
    if len(all_files_changed) > len(files_changed):
        summary += f" ({len(all_files_changed)} changed so far this run)"
    state["messages"].append(f"[Builder] {summary}")

    # step_count is incremented by the Architect gate, not here: every cycle
    # passes through the gate, but a Planner/Researcher loop never reaches the
    # Builder, and counting here let those loops run uncounted.

    return state
