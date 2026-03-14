# agents/mail_agent/llm.py
"""
Mail Agent LLM - Deprecated

This file is kept for backward compatibility.
New code should use:
- MailLLMHelpers mixin (this agent inherits from it)
- BaseAgent.llm_* methods for generic operations

See llm_helpers.py for the implementation.
"""
import warnings
from .llm_helpers import MailLLMHelpers

# Deprecated - use MailLLMHelpers mixin instead
class LLMClient(MailLLMHelpers):
    """Deprecated: Use MailLLMHelpers mixin with BaseAgent."""
    
    def __init__(self):
        warnings.warn(
            "LLMClient is deprecated. Use MailLLMHelpers mixin.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()

# For backward compatibility
llm_client = LLMClient()
