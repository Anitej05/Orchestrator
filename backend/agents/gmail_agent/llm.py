# agents/gmail_agent/llm.py
"""
Gmail Agent LLM - Deprecated

This file is kept for backward compatibility.
New code should use:
- BaseAgent.llm_* methods (inherited from AgentLLMHelpers)
- GmailLLMHelpers mixin for Gmail-specific methods

See llm_helpers.py for the new implementation.
"""
import warnings
from .llm_helpers import GmailLLMHelpers, strip_think_tags

# Deprecated - use GmailLLMHelpers mixin instead
class LLMClient(GmailLLMHelpers):
    """Deprecated: Use GmailLLMHelpers mixin with BaseAgent instead."""
    
    def __init__(self):
        warnings.warn(
            "LLMClient is deprecated. Use GmailLLMHelpers mixin with BaseAgent.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()

# For backward compatibility
llm_client = LLMClient()
