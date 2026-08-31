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
    """Detect provider from model name."""
    if not model:
        return "openai"
    model_lower = model.lower()
    if "claude" in model_lower:
        return "anthropic"
    if "llama" in model_lower or "mistral" in model_lower:
        return "ollama"
    return "openai"


class StubLLM:
    """Stub LLM for testing without API key."""

    def invoke(self, messages):
        """Return canned responses for testing."""
        from langchain_core.messages import AIMessage

        content = str(messages) if isinstance(messages, str) else str(messages[-1].content)

        if "research" in content.lower() or "investigate" in content.lower():
            response = """{
                "plan": [
                    "Research existing solutions and best practices",
                    "Identify key requirements and constraints",
                    "Design the architecture",
                    "Implement the solution"
                ],
                "next_node": "researcher"
            }"""
        else:
            response = """{
                "plan": [
                    "Understand the requirements",
                    "Design the solution",
                    "Implement the code"
                ],
                "next_node": "builder"
            }"""

        return AIMessage(content=response)
