# agents/coding_agent/llm.py
"""
Coding Agent LLM - Deprecated

This file is kept for backward compatibility.
New code should use:
- CodingLLMHelpers mixin (this agent inherits from it)
- BaseAgent.llm_* methods for generic operations

See llm_helpers.py for the implementation.
"""
import warnings
from .llm_helpers import CodingLLMHelpers, CanvasDecision, CANVAS_DECISION_SYSTEM

# Deprecated - use CodingLLMHelpers mixin instead
class CodingAgentLLM(CodingLLMHelpers):
    """Deprecated: Use CodingLLMHelpers mixin with BaseAgent."""
    
    def __init__(self):
        warnings.warn(
            "CodingAgentLLM is deprecated. Use CodingLLMHelpers mixin.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()

# For backward compatibility
llm_client = CodingAgentLLM()
