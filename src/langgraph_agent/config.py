"""Configuration and LLM setup."""

import os
from typing import Literal

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_llm(
    provider: Literal["openai", "anthropic", "ollama"] | None = None,
    model: str | None = None,
    temperature: float = 0.1,
):
    """Get LLM instance.

    Args:
        provider: LLM provider ("openai", "anthropic", "ollama").
                  Auto-detected from model name if not specified.
        model: Model name (default from env: OPENAI_MODEL, ANTHROPIC_MODEL, etc.)
        temperature: Sampling temperature

    Returns:
        Chat model instance

    Environment variables:
        OPENAI_API_KEY, OPENAI_MODEL
        ANTHROPIC_API_KEY, ANTHROPIC_MODEL
        OLLAMA_BASE_URL, OLLAMA_MODEL
    """
    provider = provider or _detect_provider(model)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model_name = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return StubLLM()
        return ChatAnthropic(model=model_name, temperature=temperature, api_key=api_key)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama

        model_name = model or os.getenv("OLLAMA_MODEL", "llama3.1")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(model=model_name, temperature=temperature, base_url=base_url)

    else:  # openai (default)
        from langchain_openai import ChatOpenAI

        model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return StubLLM()
        return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)


def _detect_provider(model: str | None) -> Literal["openai", "anthropic", "ollama"]:
    """Detect provider from model name or environment."""
    # Check environment variables first
    if os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_MODEL"):
        return "ollama"
    if os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_MODEL"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_MODEL"):
        return "openai"
    
    # Fallback to model name detection
    if not model:
        return "ollama"  # Default to local Ollama
    model_lower = model.lower()
    if "claude" in model_lower:
        return "anthropic"
    if "llama" in model_lower or "mistral" in model_lower or "qwen" in model_lower:
        return "ollama"
    return "openai"


class StubLLM:
    """Stub LLM for testing without API key.

    Returns responses in the 3-Agent System format.
    """

    def invoke(self, messages):
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
        needs_research = "research" in last_lower

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
