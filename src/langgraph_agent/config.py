"""Configuration and LLM setup.

Supports per-agent LLM selection so Planner, Researcher, and Builder can each
use a different model/provider. Defaults are cloud-only: Anthropic is the primary
provider. OpenAI remains available as an optional cloud provider.
"""

import os
import re
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import SecretStr

# Load environment variables from .env file
load_dotenv()


# Default model per (provider, agent) when a per-agent provider is configured
# but no model is supplied. Cloud-first defaults.
_DEFAULT_AGENT_MODELS: dict[tuple[str, str], str] = {
    ("anthropic", "planner"): "claude-3-5-sonnet-20241022",
    ("anthropic", "researcher"): "claude-3-5-haiku-20241022",
    ("anthropic", "builder"): "claude-3-5-haiku-20241022",
    ("openai", "planner"): "gpt-4o",
    ("openai", "researcher"): "gpt-4o-mini",
    ("openai", "builder"): "gpt-4o-mini",
}


# Cloud LLM options exposed in the console.
AGENT_LLM_OPTIONS: list[dict[str, str]] = [
    {
        "label": "Anthropic Claude 3.5 Sonnet",
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
    },
    {
        "label": "Anthropic Claude 3.5 Haiku",
        "provider": "anthropic",
        "model": "claude-3-5-haiku-20241022",
    },
    # Optional cloud provider
    {"label": "OpenAI GPT-4o", "provider": "openai", "model": "gpt-4o"},
    {"label": "OpenAI GPT-4o mini", "provider": "openai", "model": "gpt-4o-mini"},
]


# Runtime per-agent LLM selections set from the console. These override the
# environment-variable defaults for the lifetime of the process.
_agent_llm_overrides: dict[str, dict[str, str]] = {}


def set_agent_llm(agent: str, provider: str, model: str) -> None:
    """Set the LLM for an agent at runtime.

    The selection is stored in memory only; it does not modify environment
    variables or persist across server restarts.
    """
    _agent_llm_overrides[agent] = {"provider": provider, "model": model}


def get_llm(
    provider: Literal["openai", "anthropic"] | None = None,
    model: str | None = None,
    temperature: float = 0.1,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Any:
    """Get an LLM instance.

    Args:
        provider: LLM provider ("openai" or "anthropic").
                  Auto-detected from model name/env if not specified.
        model: Model name (default from provider-specific env var).
        temperature: Sampling temperature.
        base_url: Optional API base URL override.
        api_key: Optional API key override.

    Returns:
        Chat model instance.

    Environment variables:
        ANTHROPIC_API_KEY, ANTHROPIC_MODEL
        OPENAI_API_KEY, OPENAI_MODEL  (optional)
    """
    provider = provider or _detect_provider(model)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model_name = model or os.getenv(
            "ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"
        )
        key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not key:
            return StubLLM()
        kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "api_key": SecretStr(key),
        }
        if base_url:
            kwargs["base_url"] = base_url
        return ChatAnthropic(**kwargs)

    # openai (optional cloud provider)
    from langchain_openai import ChatOpenAI

    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        return StubLLM()
    kwargs = {
        "model": str(model_name),
        "temperature": temperature,
        "api_key": SecretStr(key),
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _detect_provider(model: str | None) -> Literal["openai", "anthropic"]:
    """Detect provider from model name or environment.

    Cloud-first priority: Anthropic, then OpenAI.
    """
    # Anthropic is the primary cloud default.
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_MODEL"):
        return "anthropic"

    # OpenAI remains available as an optional cloud provider.
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_MODEL"):
        return "openai"

    # Fallback to model name detection
    if not model:
        return "anthropic"
    model_lower = model.lower()
    if "claude" in model_lower:
        return "anthropic"
    return "openai"


def get_agent_llm(
    agent: Literal["planner", "researcher", "builder"],
    temperature: float = 0.1,
) -> Any:
    """Get the LLM configured for a specific agent role.

    Reads per-agent environment variables:
        PLANNER_PROVIDER, PLANNER_MODEL, PLANNER_BASE_URL, PLANNER_API_KEY
        RESEARCHER_PROVIDER, RESEARCHER_MODEL, RESEARCHER_BASE_URL, RESEARCHER_API_KEY
        BUILDER_PROVIDER, BUILDER_MODEL, BUILDER_BASE_URL, BUILDER_API_KEY

    Falls back to the legacy single-model configuration when per-agent variables
    are not set, preserving backward compatibility.

    Recommended 3-model setup (cloud):
        Planner    -> Anthropic Claude 3.5 Sonnet  (main architect / structural planning)
        Researcher -> Anthropic Claude 3.5 Haiku   (knowledge gathering)
        Builder    -> Anthropic Claude 3.5 Haiku   (code implementation)
    """
    provider: str | None
    model: str | None
    base_url: str | None
    api_key: str | None

    # Runtime console override takes precedence over env vars.
    override = _agent_llm_overrides.get(agent)
    if override:
        provider = override["provider"]
        model = override["model"]
        base_url = os.getenv(f"{provider.upper()}_BASE_URL") if provider else None
        api_key = None
    else:
        prefix = agent.upper()
        provider = os.getenv(f"{prefix}_PROVIDER")
        model = os.getenv(f"{prefix}_MODEL")
        base_url = os.getenv(f"{prefix}_BASE_URL")
        api_key = os.getenv(f"{prefix}_API_KEY")

        per_agent_configured = bool(provider or model)

        if not per_agent_configured:
            # Legacy single-model fallback: every agent uses the same provider/model.
            provider = _detect_provider(None)
            model = None
        elif not model and provider:
            # Per-agent provider was explicitly chosen; apply a sensible default
            # model for that provider + agent combination.
            model = _DEFAULT_AGENT_MODELS.get((provider, agent))

    return get_llm(
        provider=provider,  # type: ignore[arg-type]
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )


def get_agent_model_info(
    agent: Literal["planner", "researcher", "builder"],
) -> dict[str, str]:
    """Resolve the configured provider/model name for an agent without instantiating an LLM.

    Mirrors the fallback logic of `get_agent_llm()` so the UI can display the
    model each agent will use.
    """
    # Runtime console override takes precedence over env vars.
    override = _agent_llm_overrides.get(agent)
    if override:
        return {"provider": override["provider"], "model": override["model"]}

    prefix = agent.upper()
    provider = os.getenv(f"{prefix}_PROVIDER")
    model = os.getenv(f"{prefix}_MODEL")

    per_agent_configured = bool(provider or model)

    if not per_agent_configured:
        provider = _detect_provider(None)
        model = None
    elif not model and provider:
        model = _DEFAULT_AGENT_MODELS.get((provider, agent))

    # Fallback to provider-specific global defaults if no per-agent default applies.
    if not model:
        if provider == "anthropic":
            model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        else:
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    return {"provider": provider or "anthropic", "model": model}


class StubLLM:
    """Stub LLM for testing without API key.

    Returns responses in the 3-Agent System format.
    """

    def invoke(self, messages: list[Any]) -> Any:
        """Return canned responses for testing."""
        from langchain_core.messages import AIMessage

        # Check all messages for agent detection
        all_content = ""
        last_content = ""
        for msg in messages:
            content = str(msg.content) if hasattr(msg, "content") else str(msg)
            all_content += content + " "
            last_content = content

        all_lower = all_content.lower()
        last_lower = last_content.lower()

        # Detect which agent is being called based on prompt content
        is_planner = "planner" in all_lower or "understand the user's goal" in all_lower
        is_researcher = "researcher" in all_lower or "gather high-quality" in all_lower
        is_builder = "builder" in all_lower or "implement the plan" in all_lower

        # For the Planner, decide whether the *user goal* asks for research.
        # Ignore the state-injection block, which contains a "Research:" label.
        user_goal_match = re.search(
            r"User goal:\s*(.+)", last_content, re.IGNORECASE | re.DOTALL
        )
        user_goal = user_goal_match.group(1).lower() if user_goal_match else last_lower
        needs_research = "research" in user_goal

        if is_planner:
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
