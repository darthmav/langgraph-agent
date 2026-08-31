"""Configuration and LLM setup."""

import os
from typing import Optional
from langchain_openai import ChatOpenAI


def get_llm(model: Optional[str] = None, temperature: float = 0.1):
    """Get LLM instance.
    
    Uses OPENAI_API_KEY from environment. Falls back to stub for testing.
    
    Args:
        model: Model name (default: "gpt-4o-mini")
        temperature: Sampling temperature
    
    Returns:
        Chat model instance
    """
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        # Return a stub for testing without API key
        return StubLLM()
    
    return ChatOpenAI(model=model_name, temperature=temperature, api_key=api_key)


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
