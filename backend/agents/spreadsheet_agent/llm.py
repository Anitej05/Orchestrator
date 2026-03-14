# agents/spreadsheet_agent/llm.py
"""
Spreadsheet Agent LLM - Deprecated

This file is kept for backward compatibility.
New code should use:
- SpreadsheetLLMHelpers mixin (this agent inherits from it)
- BaseAgent.llm_* methods for generic operations

See llm_helpers.py for the implementation.
"""
import warnings
from .llm_helpers import SpreadsheetLLMHelpers, strip_think_tags

# Deprecated - use SpreadsheetLLMHelpers mixin instead
class LLMClient(SpreadsheetLLMHelpers):
    """Deprecated: Use SpreadsheetLLMHelpers mixin with BaseAgent."""
    
    def __init__(self):
        warnings.warn(
            "LLMClient is deprecated. Use SpreadsheetLLMHelpers mixin with BaseAgent.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()

# For backward compatibility
llm_client = LLMClient()
