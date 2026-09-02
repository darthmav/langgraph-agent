"""Custom exception classes for the LangGraph Agent package.

This module defines custom exception classes that provide more specific
error handling than generic built-in exceptions. All exceptions inherit
from a common base class for easy catching and identification.
"""


class LangGraphAgentError(Exception):
    """Base exception class for all LangGraph Agent errors.

    All custom exceptions in this package inherit from this class.
    This allows users to catch all package-specific errors with a
    single exception handler.

    Attributes:
        message: The error message describing what went wrong.
        details: Optional additional details about the error.
    """

    def __init__(self, message: str, details: str | None = None) -> None:
        """Initialize the LangGraphAgentError.

        Args:
            message: A descriptive error message.
            details: Optional additional context or debugging information.
        """
        self.message = message
        self.details = details
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the complete error message including details if present.

        Returns:
            The formatted error message string.
        """
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class ConfigurationError(LangGraphAgentError):
    """Exception raised when there is an issue with configuration.

    This exception is raised when configuration values are missing,
    invalid, or incompatible with each other.

    Example:
        >>> raise ConfigurationError("API key not found", "Check .env file")
    """

    pass


class StateError(LangGraphAgentError):
    """Exception raised when there is an issue with agent state.

    This exception is raised when the agent state is invalid,
    corrupted, or in an unexpected condition.

    Example:
        >>> raise StateError("Invalid state transition")
    """

    pass


class ToolError(LangGraphAgentError):
    """Exception raised when a tool execution fails.

    This exception is raised when a tool binding fails to execute,
    returns an unexpected result, or encounters an error during
    execution.

    Example:
        >>> raise ToolError("Tool execution failed", "Timeout after 30s")
    """

    pass


class GraphError(LangGraphAgentError):
    """Exception raised when there is an issue with the agent graph.

    This exception is raised when the graph structure is invalid,
    a node fails to execute, or there are issues with graph traversal.

    Example:
        >>> raise GraphError("Circular dependency detected")
    """

    pass


class InferenceError(LangGraphAgentError):
    """Exception raised when inference operations fail.

    This exception is raised when communication with inference
    providers (Ollama Cloud, Anthropic, OpenAI) fails or returns
    unexpected results.

    Example:
        >>> raise InferenceError("Model inference failed", "Rate limit exceeded")
    """

    pass
