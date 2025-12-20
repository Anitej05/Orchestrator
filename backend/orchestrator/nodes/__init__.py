"""
Orchestrator Nodes Package

This package contains the individual node implementations for the LangGraph orchestrator.
Each module groups related nodes by purpose:
- parsing.py: Request analysis and task parsing
- searching.py: Agent discovery and ranking
- planning.py: Execution plan creation and validation
- execution.py: Agent execution logic
- evaluation.py: Response evaluation and user interaction
- response.py: Final response generation and history management
- routing.py: Conditional routing functions
- utils.py: Shared utilities

All nodes should be imported from this package or directly from graph.py.
The modules here provide organizational structure and documentation.
"""

# Utility functions (fully implemented here)
from .utils import (
    extract_json_from_response,
    serialize_complex_object,
    transform_payload_types,
    save_plan_to_file,
    get_hf_embeddings,
    invoke_llm_with_fallback,
    CustomJSONEncoder,
)

# Routing functions (fully implemented here)
from .routing import (
    route_after_search,
    route_after_approval,
    route_after_validation,
    route_after_load_history,
    route_after_analysis,
    route_after_parse,
    should_continue_or_finish,
    route_after_plan_creation,
    route_after_execute_batch,
    route_after_ask_user,
)

# Pydantic schemas from modules
from .planning import PlanValidation, PlanValidationResult
from .evaluation import AgentResponseEvaluation

# Searching utilities
from .searching import get_all_capabilities, cached_capabilities, CACHE_DURATION_SECONDS

# Execution cache
from .execution import get_request_cache, GET_CACHE_DURATION_SECONDS, clear_get_cache

# Response directory
from .response import CONVERSATION_HISTORY_DIR

__all__ = [
    # Utils
    'extract_json_from_response',
    'serialize_complex_object',
    'transform_payload_types',
    'save_plan_to_file',
    'get_hf_embeddings',
    'invoke_llm_with_fallback',
    'CustomJSONEncoder',
    
    # Routing
    'route_after_search',
    'route_after_approval',
    'route_after_validation',
    'route_after_load_history',
    'route_after_analysis',
    'route_after_parse',
    'should_continue_or_finish',
    'route_after_plan_creation',
    'route_after_execute_batch',
    'route_after_ask_user',
    
    # Schemas
    'PlanValidation',
    'PlanValidationResult',
    'AgentResponseEvaluation',
    
    # Searching
    'get_all_capabilities',
    'cached_capabilities',
    'CACHE_DURATION_SECONDS',
    
    # Execution
    'get_request_cache',
    'GET_CACHE_DURATION_SECONDS',
    'clear_get_cache',
    
    # Response
    'CONVERSATION_HISTORY_DIR',
]

