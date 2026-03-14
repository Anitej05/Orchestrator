# agents/document_agent_lib/llm.py
"""
Document Agent LLM - Deprecated

This file is kept for backward compatibility.
New code should use:
- DocumentLLMHelpers mixin (this agent inherits from it)
- BaseAgent.llm_* methods for generic operations

See llm_helpers.py for the implementation.
"""
import warnings
from .llm_helpers import DocumentLLMHelpers

# Deprecated - use DocumentLLMHelpers mixin instead
class DocumentLLMClient(DocumentLLMHelpers):
    """Deprecated: Use DocumentLLMHelpers mixin with BaseAgent."""
    
    def __init__(self):
        warnings.warn(
            "DocumentLLMClient is deprecated. Use DocumentLLMHelpers mixin.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__()

# For backward compatibility
llm_client = DocumentLLMClient()
