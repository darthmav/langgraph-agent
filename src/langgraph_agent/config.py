"""Configuration and LLM setup.

Supports per-agent LLM selection so Architect, Planner, Researcher, and Builder
can each use a different model/provider. Inference is cloud-only: every seat
defaults to an Ollama Cloud tag, which the local daemon proxies to ollama.com
using credentials it holds itself. Anthropic and OpenAI remain available per
seat, but neither is needed to run the crew.

The only thing that runs on this machine is the embedding model, which belongs
to GraphRAG rather than to any agent seat.
"""

import json
import os
import re
import time
import urllib.request
from typing import Any, Literal, cast

from dotenv import load_dotenv
from pydantic import SecretStr

# Load environment variables from .env file
load_dotenv()


# The four seats, in the order they hold the loop. Anything that iterates
# agents reads this rather than repeating the list.
AgentName = Literal["architect", "planner", "researcher", "builder"]
Provider = Literal["openai", "anthropic", "ollama"]

AGENTS: tuple[AgentName, ...] = ("architect", "planner", "researcher", "builder")


# Default model per (provider, agent) when a per-agent provider is configured
# but no model is supplied. Cloud-first defaults.
_DEFAULT_AGENT_MODELS: dict[tuple[str, str], str] = {
    ("anthropic", "architect"): "claude-opus-5",
    ("anthropic", "planner"): "claude-opus-5",
    ("anthropic", "researcher"): "claude-sonnet-5",
    ("anthropic", "builder"): "claude-sonnet-5",
    ("ollama", "architect"): "kimi-k3:cloud",
    ("ollama", "planner"): "qwen3.5:397b-cloud",
    ("ollama", "researcher"): "nemotron-3-ultra:cloud",
    ("ollama", "builder"): "kimi-k3:cloud",
    ("openai", "architect"): "gpt-4o",
    ("openai", "planner"): "gpt-4o",
    ("openai", "researcher"): "gpt-4o-mini",
    ("openai", "builder"): "gpt-4o-mini",
}


# The seat each agent takes when nothing overrides it. All four are Ollama
# Cloud tags so a fresh checkout runs with no API key of its own -- the daemon
# already holds the ollama.com credentials. Putting the Architect on Anthropic
# meant the entry node, and so the whole run, died without billable credit.
# Point a seat at Anthropic or OpenAI with {ROLE}_PROVIDER / {ROLE}_MODEL, or
# from the console dropdown.
DEFAULT_SEATS: dict[str, dict[str, str]] = {
    "architect": {"provider": "ollama", "model": "kimi-k3:cloud"},
    "planner": {"provider": "ollama", "model": "qwen3.5:397b-cloud"},
    "researcher": {"provider": "ollama", "model": "nemotron-3-ultra:cloud"},
    "builder": {"provider": "ollama", "model": "kimi-k3:cloud"},
}


# Cloud LLM options exposed in the console. `group` drives the <optgroup>
# headings in the seat dropdowns.
AGENT_LLM_OPTIONS: list[dict[str, str]] = [
    {"label": "Claude Opus 5", "provider": "anthropic", "model": "claude-opus-5",
     "group": "Anthropic"},
    {"label": "Claude Sonnet 5", "provider": "anthropic", "model": "claude-sonnet-5",
     "group": "Anthropic"},
    {"label": "Claude Haiku 4.5", "provider": "anthropic", "model": "claude-haiku-4-5",
     "group": "Anthropic"},
    {"label": "Kimi K3", "provider": "ollama", "model": "kimi-k3:cloud",
     "group": "Ollama Cloud"},
    {"label": "Qwen3.5 397B", "provider": "ollama", "model": "qwen3.5:397b-cloud",
     "group": "Ollama Cloud"},
    {"label": "Nemotron 3 Ultra", "provider": "ollama", "model": "nemotron-3-ultra:cloud",
     "group": "Ollama Cloud"},
    {"label": "Kimi K2.7 Code", "provider": "ollama", "model": "kimi-k2.7-code:cloud",
     "group": "Ollama Cloud"},
    {"label": "Gemma 4", "provider": "ollama", "model": "gemma4:cloud",
     "group": "Ollama Cloud"},
    # Optional cloud provider
    {"label": "OpenAI GPT-4o", "provider": "openai", "model": "gpt-4o",
     "group": "OpenAI"},
    {"label": "OpenAI GPT-4o mini", "provider": "openai", "model": "gpt-4o-mini",
     "group": "OpenAI"},
]


# How long one call to a seat may take before the client gives up. Every
# provider client defaults to no deadline at all, so a stalled cloud call
# blocked `llm.invoke` forever -- and RUN_BUDGET_SECONDS could not end it,
# because that is checked between graph supersteps and a node in flight never
# reaches a superstep boundary. A run could therefore hang indefinitely inside
# a single node with the console still naming the previous one.
#
# This bounds the socket, not the call: for Ollama it becomes an httpx timeout
# on a streamed response, so it fires on a connection that goes quiet, not on a
# model that trickles tokens forever. The node-level deadline in nodes.py
# covers that second case; the two are not redundant.
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "120"))


def _ollama_base_url() -> str:
    """Where the local Ollama daemon listens.

    The daemon is a proxy here, not a runtime: every seat that uses it runs a
    `:cloud` tag, which it forwards to ollama.com.
    """
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


_ollama_tags_cache: tuple[float, list[str]] = (0.0, [])


def list_ollama_models() -> list[str]:
    """Tags the local Ollama daemon reports, cached for 30s.

    Feeds both the seat dropdowns and the liveness check, either of which can
    be hit on every status poll -- hence the cache and the short timeout. An
    unreachable daemon is an empty list, never an exception.
    """
    global _ollama_tags_cache

    now = time.monotonic()
    cached_at, cached = _ollama_tags_cache
    if cached and now - cached_at < 30.0:
        return cached

    try:
        with urllib.request.urlopen(
            f"{_ollama_base_url()}/api/tags", timeout=2.0
        ) as response:
            payload = json.loads(response.read())
        tags = sorted(str(entry["name"]) for entry in payload.get("models", []))
    except Exception:
        tags = []

    _ollama_tags_cache = (now, tags)
    return tags


# Why the last call to a seat failed, if it did. A key can be present and the
# seat still unusable -- out of credits, expired, revoked, wrong workspace --
# and only a real call finds that out. Recording the outcome here is what lets
# the console stop claiming a seat is live without spending a probe request on
# every five-second status poll.
_seat_failures: dict[str, str] = {}


def _failure_reason(exc: Exception) -> str:
    """Turn a provider exception into something a seat card can show."""
    text = str(exc)

    # Providers wrap their real message in a dict repr; pull it back out.
    match = re.search(r"'message': '([^']+)'", text)
    message = match.group(1) if match else text

    lowered = message.lower()
    if "credit balance" in lowered:
        return "Anthropic credit balance too low"
    if "authentication" in lowered or "invalid x-api-key" in lowered:
        return "API key rejected"
    if "rate limit" in lowered:
        return "Rate limited"
    # httpx raises ReadTimeout with an empty message, so the class name is the
    # only thing that identifies it -- without this a timed-out seat showed a
    # blank chip, which reads as "fine" rather than "gave up after 120s".
    if "timeout" in lowered or "timed out" in lowered or "Timeout" in type(exc).__name__:
        return f"No response within {int(LLM_TIMEOUT_SECONDS)}s"
    if "not found" in lowered and "model" in lowered:
        return "Model not available on this account"
    if "connect" in lowered or "connection" in lowered:
        return "Provider unreachable"

    return message[:90].strip()


class _SeatLLM:
    """Wraps a seat's chat model so its failures are visible in the console.

    Transparent apart from `invoke`: every other attribute passes through to
    the wrapped model.
    """

    def __init__(self, agent: str, inner: Any) -> None:
        self._agent = agent
        self._inner = inner

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        try:
            result = self._inner.invoke(*args, **kwargs)
        except Exception as exc:
            _seat_failures[self._agent] = _failure_reason(exc)
            raise
        # A call that works clears an older failure, so a seat recovers on its
        # own once credits are topped up or the daemon comes back.
        _seat_failures.pop(self._agent, None)
        return result

    def bind_tools(self, tools: Any, **kwargs: Any) -> "_SeatLLM":
        """Bind tools, keeping the wrapper so the bound model still reports.

        Without this the bound runnable would come back unwrapped through
        `__getattr__`, and every failure the Builder hit while calling tools
        would be invisible to `get_agent_status`. Seats whose model cannot call
        tools at all (`StubLLM`) raise AttributeError here on purpose, so a
        caller wanting the no-tools path catches AttributeError around the
        call itself -- `hasattr` is always True once this method exists.
        """
        inner_bind = getattr(self._inner, "bind_tools", None)
        if inner_bind is None:
            raise AttributeError("bind_tools")
        return _SeatLLM(self._agent, inner_bind(tools, **kwargs))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _accepts_temperature(provider: str, model: str) -> bool:
    """Whether this model still accepts a sampling temperature.

    Anthropic removed temperature/top_p/top_k on the 4.6-and-later families
    (Opus 5, Sonnet 5, Opus 4.6+) -- sending one is rejected with a 400, which
    reads like a credentials problem and is not one. Legacy `claude-3-*` and
    the 4.5 models still take it, as do Ollama and OpenAI.
    """
    if provider != "anthropic":
        return True
    return model.startswith("claude-3") or "-4-5" in model


# Runtime per-agent LLM selections set from the console. These override the
# environment-variable defaults for the lifetime of the process.
_agent_llm_overrides: dict[str, dict[str, str]] = {}


def set_agent_llm(agent: str, provider: str, model: str) -> None:
    """Set the LLM for an agent at runtime.

    The selection is stored in memory only; it does not modify environment
    variables or persist across server restarts.

    Moving a seat clears any failure recorded against it. A recorded failure
    describes the seat that produced it, so leaving it in place made the new
    seat inherit the old one's verdict -- an Architect moved off Anthropic
    still read "Anthropic credit balance too low", which is exactly the
    reading that sends someone to buy credits they do not need. Re-selecting
    the seat it already has is not a move and keeps the failure, so a dead
    seat cannot be made to look live by picking it again.
    """
    current = get_agent_model_info(cast("AgentName", agent))
    if (current["provider"], current["model"]) != (provider, model):
        _seat_failures.pop(agent, None)

    _agent_llm_overrides[agent] = {"provider": provider, "model": model}


def get_llm(
    provider: Provider | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> Any:
    """Get an LLM instance.

    Args:
        provider: LLM provider ("anthropic", "ollama" or "openai").
                  Auto-detected from model name/env if not specified.
        model: Model name (default from provider-specific env var).
        temperature: Sampling temperature, where the model accepts one.
        base_url: Optional API base URL override.
        api_key: Optional API key override.
        timeout: Seconds one call may take; `LLM_TIMEOUT_SECONDS` if omitted.
                 Each provider spells this differently, hence the three
                 separate keyword names below.

    Returns:
        Chat model instance, or `StubLLM` when the provider needs a key and
        none is configured. Callers that need to know which of the two they
        got should ask `get_agent_status()` rather than inspect the result.

    Environment variables:
        ANTHROPIC_API_KEY, ANTHROPIC_MODEL
        OLLAMA_BASE_URL, OLLAMA_MODEL
        OPENAI_API_KEY, OPENAI_MODEL  (optional)
    """
    provider = provider or _detect_provider(model)
    timeout = LLM_TIMEOUT_SECONDS if timeout is None else timeout

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # No key: the daemon holds the ollama.com credentials for `:cloud`
        # tags, so there is nothing for this process to authenticate with.
        # `client_kwargs` reaches the httpx client the ollama SDK builds; there
        # is no `timeout` field on ChatOllama itself.
        return ChatOllama(
            model=str(model or os.getenv("OLLAMA_MODEL", "qwen3.5:397b-cloud")),
            temperature=temperature,
            base_url=base_url or _ollama_base_url(),
            client_kwargs={"timeout": timeout},
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model_name = str(model or os.getenv("ANTHROPIC_MODEL", "claude-opus-5"))
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return StubLLM()
        kwargs: dict[str, Any] = {
            "model": model_name,
            "api_key": SecretStr(key),
            "default_request_timeout": timeout,
        }
        if _accepts_temperature("anthropic", model_name):
            kwargs["temperature"] = temperature
        if base_url:
            kwargs["base_url"] = base_url
        return ChatAnthropic(**kwargs)

    # openai (optional cloud provider)
    from langchain_openai import ChatOpenAI

    model_name = str(model or os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return StubLLM()
    kwargs = {
        "model": str(model_name),
        "temperature": temperature,
        "api_key": SecretStr(key),
        "request_timeout": timeout,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _detect_provider(model: str | None) -> Provider:
    """Detect provider from model name or environment.

    Cloud-first priority: Anthropic, then Ollama, then OpenAI.
    """
    # A tag carrying a colon is an Ollama tag; nothing else names models
    # that way, so it settles the provider before any env var is consulted.
    if model and ":" in model:
        return "ollama"

    # Anthropic is the primary cloud default.
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_MODEL"):
        return "anthropic"

    # OpenAI remains available as an optional cloud provider.
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_MODEL"):
        return "openai"

    # Fallback to model name detection
    if not model:
        return "anthropic"
    if "claude" in model.lower():
        return "anthropic"
    return "openai"


def _resolve_seat(agent: str) -> dict[str, str | None]:
    """Resolve one agent's provider, model, base URL and key.

    Precedence: a console override, then per-agent environment variables, then
    the agent's default seat. `get_agent_llm`, `get_agent_model_info` and
    `get_agent_status` all route through here -- if they resolved seats
    independently the console could name one model while the run used another,
    which is the failure this function exists to prevent.

    Per-agent environment variables, for each of ARCHITECT, PLANNER,
    RESEARCHER and BUILDER:
        {ROLE}_PROVIDER, {ROLE}_MODEL, {ROLE}_BASE_URL, {ROLE}_API_KEY
    """
    override = _agent_llm_overrides.get(agent)
    if override:
        chosen = override["provider"]
        return {
            "provider": chosen,
            "model": override["model"],
            # A console selection carries no key of its own, so the provider's
            # own credentials apply -- the same ones the default seat uses.
            "base_url": os.getenv(f"{chosen.upper()}_BASE_URL"),
            "api_key": None,
        }

    prefix = agent.upper()
    provider: str | None = os.getenv(f"{prefix}_PROVIDER")
    model: str | None = os.getenv(f"{prefix}_MODEL")

    if not provider and not model:
        # Deliberately not consulting a provider-wide {PROVIDER}_MODEL here:
        # the four seats run four different models on purpose, and a single
        # OLLAMA_MODEL would silently collapse three of them onto one. Retune
        # a seat with {ROLE}_MODEL or the console dropdown instead.
        seat = DEFAULT_SEATS.get(agent, DEFAULT_SEATS["builder"])
        provider, model = seat["provider"], seat["model"]
    elif provider and not model:
        model = _DEFAULT_AGENT_MODELS.get((provider, agent))
    elif model and not provider:
        provider = _detect_provider(model)

    return {
        "provider": provider,
        "model": model,
        "base_url": os.getenv(f"{prefix}_BASE_URL"),
        "api_key": os.getenv(f"{prefix}_API_KEY"),
    }


def get_agent_llm(agent: AgentName, temperature: float = 0.1) -> Any:
    """Get the LLM configured for a specific agent role.

    Default seats (cloud only -- see DEFAULT_SEATS):
        Architect  -> Ollama    kimi-k3:cloud      (leading authority)
        Planner    -> Ollama    qwen3.5:397b-cloud
        Researcher -> Ollama    nemotron-3-ultra:cloud
        Builder    -> Ollama    kimi-k3:cloud
    """
    seat = _resolve_seat(agent)
    return _SeatLLM(
        agent,
        get_llm(
            provider=seat["provider"],  # type: ignore[arg-type]
            model=seat["model"],
            temperature=temperature,
            base_url=seat["base_url"],
            api_key=seat["api_key"],
        ),
    )


def get_agent_model_info(agent: AgentName) -> dict[str, str]:
    """Resolve an agent's provider/model without instantiating an LLM."""
    seat = _resolve_seat(agent)
    return {
        "provider": seat["provider"] or "anthropic",
        "model": seat["model"] or "claude-opus-5",
    }


def get_agent_status(agent: AgentName) -> dict[str, Any]:
    """Resolve an agent's seat and say whether it can actually run.

    `get_llm` falls back to `StubLLM` when a key is missing, while
    `get_agent_model_info` keeps reporting the configured model either way --
    so without this the console shows a model name while canned text comes out
    of the run. `live` is the field that tells the truth about that, and it is
    what drives the NO KEY chip and the offline banner.
    """
    info = get_agent_model_info(agent)
    provider, model = info["provider"], info["model"]

    # `stubbed` and `live` are different failures and must not be conflated.
    # A seat with no key at all silently becomes StubLLM and the run completes
    # with canned text; a seat with a key that does not work fails the run
    # outright. The console words those two cases differently.
    live, reason, badge, stubbed = True, "", "", False

    # A real failure from the last call outranks every static check: the key
    # can be present and correct and the seat still unable to run.
    failure = _seat_failures.get(agent)
    if failure:
        live, reason, badge = False, failure, "FAILING"
    elif provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        live, reason, badge, stubbed = False, "ANTHROPIC_API_KEY not set", "NO KEY", True
    elif provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        live, reason, badge, stubbed = False, "OPENAI_API_KEY not set", "NO KEY", True
    elif provider == "ollama":
        tags = list_ollama_models()
        if not tags:
            live, reason, badge = False, "Ollama daemon unreachable", "OFFLINE"
        elif model not in tags:
            live, reason, badge = False, f"{model} not pulled", "NOT PULLED"

    # Where the prompt goes, read off the seat rather than guessed: an Ollama
    # tag ending `:cloud` is proxied to ollama.com by the local daemon, so the
    # transport is local but the prompt still leaves the machine.
    remote = provider in ("anthropic", "openai") or model.endswith((":cloud", "-cloud"))

    return {
        "provider": provider,
        "model": model,
        "live": live,
        "reason": reason,
        "badge": badge,
        "stubbed": stubbed,
        "placement": "REMOTE" if remote else "LOCAL",
    }


class StubLLM:
    """Stub LLM for testing without API key.

    Returns responses in the 4-Agent System format.
    """

    def invoke(self, messages: list[Any]) -> Any:
        """Return canned responses for testing."""
        from langchain_core.messages import AIMessage

        # Role comes from the system message; everything else is data.
        # The Researcher injects retrieved documents into the Builder's prompt,
        # and this repository indexes its own prompt files -- so a corpus hit
        # can put "You are the Architect" inside a Builder call. Reading the
        # role off user content makes the stub answer as the wrong agent.
        system_content = ""
        all_content = ""
        last_content = ""
        for msg in messages:
            content = str(msg.content) if hasattr(msg, "content") else str(msg)
            all_content += content + " "
            last_content = content
            if getattr(msg, "type", "") == "system":
                system_content += content + " "

        all_lower = all_content.lower()
        last_lower = last_content.lower()
        # Fall back to the whole prompt only when no system message was given.
        role_source = system_content.lower() or all_lower

        # Detect which agent is being called. Every prompt names the other
        # roles in order to route between them, so a bare role keyword matches
        # all four -- the Builder's prompt says "Planner" twice. Identify on
        # the self-identifying opening instead, and only fall back to the
        # looser phrases when a caller supplied its own prompt.
        roles = ("architect", "planner", "researcher", "builder")
        identified = next(
            (role for role in roles if f"you are the {role}" in role_source), None
        )

        is_architect = identified == "architect"
        is_planner = identified == "planner"
        is_researcher = identified == "researcher"
        is_builder = identified == "builder"

        if identified is None:
            is_architect = "## verdict" in role_source
            is_planner = "understand the user's goal" in role_source
            is_researcher = "gather high-quality" in role_source
            is_builder = "implement the plan" in role_source

        # For the Planner, decide whether the *user goal* asks for research.
        # Ignore the state-injection block, which contains a "Research:" label.
        user_goal_match = re.search(
            r"User goal:\s*(.+)", last_content, re.IGNORECASE | re.DOTALL
        )
        user_goal = user_goal_match.group(1).lower() if user_goal_match else last_lower
        needs_research = "research" in user_goal

        if is_architect:
            # The Architect runs twice per cycle: once to set direction, and
            # again as the approval gate. A populated builder report is what
            # separates the two.
            reviewing = (
                "builder report:" in all_lower
                and "builder report: (empty)" not in all_lower
            )
            if reviewing:
                response = """## Architecture
Change stays within the existing module boundaries.

## Constraints
- Preserve the existing state schema
- No new external services

## Verdict
approved"""
            else:
                response = """## Architecture
Single-module change against the current structure.

## Constraints
- Preserve the existing state schema
- No new external services

## Verdict
plan"""

        elif is_planner:
            # Planner response format
            if needs_research:
                response = """## Goal
Research Python best practices

## Steps
1. Search for existing documentation
2. Identify key patterns
3. Summarize findings

## Next Agent
Researcher

## Notes
Research needed for knowledge gathering"""
            else:
                response = """## Goal
Create a file with content

## Steps
1. Create the file
2. Write the content
3. Verify the file

## Next Agent
Builder

## Notes
Task is straightforward, no research needed"""

        elif is_researcher:
            # Researcher response format
            response = """## Key Findings
- Found relevant patterns in documentation
- Identified best practices

## Relevant Context
Existing code follows similar patterns

## Recommendations for Builder
Implement using the identified patterns

## Status
ready_for_builder"""

        elif is_builder:
            # Builder response format
            response = """## Changes Made
- Created file with specified content
- Verified file exists

## Files Modified
- hello.txt

## Next Steps / Blockers
none"""

        else:
            # Default fallback
            response = """## Goal
Complete the task

## Steps
1. Understand requirements
2. Implement solution

## Next Agent
Builder

## Notes
Default response"""

        return AIMessage(content=response)
