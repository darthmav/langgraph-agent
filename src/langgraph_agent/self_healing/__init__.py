"""
Self-Healing Module for 4-Agent AI System

This module provides runtime resilience mechanisms including:
- Retry logic with exponential backoff (via tenacity)
- Circuit breaker patterns (via pybreaker)
- Structured logging with severity levels

All healing mechanisms are implemented as decorators or middleware
to avoid modifying core business logic directly.

Status: ACTIVE
"""

from .decorators import circuit_breaker, retry_with_backoff, self_healing_wrapper
from .logger import SelfHealingLogger, get_healing_logger

__all__ = [
    "SelfHealingLogger",
    "get_healing_logger",
    "retry_with_backoff",
    "circuit_breaker",
    "self_healing_wrapper",
]

__version__ = "1.0.0"
__status__ = "active"
