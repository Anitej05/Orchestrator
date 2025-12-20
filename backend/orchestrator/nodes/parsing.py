"""
Parsing Nodes - Request analysis and task parsing.

Contains nodes that process user input and break it down into tasks:
- analyze_request: Determines if request needs complex processing
- parse_prompt: Super-parser for task extraction and chitchat detection
- preprocess_files: Handles file preprocessing for uploaded attachments
"""

import os
import logging
from typing import Dict, Any, List
from orchestrator.state import State
from schemas import ParsedRequest, FileObject, AnalysisResult
from langchain_cerebras import ChatCerebras
from langchain_nvidia_ai_endpoints import ChatNVIDIA

logger = logging.getLogger("AgentOrchestrator")


def get_llm_clients():
    """Initialize LLM clients with fallback."""
    primary = ChatCerebras(model="gpt-oss-120b")
    fallback = ChatNVIDIA(model="openai/gpt-oss-120b") if os.getenv("NVIDIA_API_KEY") else None
    return primary, fallback


async def preprocess_files(state: State):
    """
    Processes uploaded files asynchronously with caching.
    - Images: Validates path only
    - Documents: Creates vector store + displays (async + cached)
    - Spreadsheets: Uploads to agent + displays (async)
    
    NOTE: Full implementation remains in graph.py due to complex async handling.
    This is a reference/documentation stub.
    """
    # Implementation in graph.py - this module serves as organizational structure
    raise NotImplementedError("Use graph.preprocess_files - async implementation in main module")


def analyze_request(state: State):
    """
    Sophisticated analysis of user request to determine processing approach.
    
    Determines:
    - needs_complex_processing: Whether agents are needed
    - analysis_reasoning: Why the decision was made
    - final_response: Direct response for simple requests
    
    NOTE: Full implementation remains in graph.py due to LLM client dependencies.
    This is a reference/documentation stub.
    """
    # Implementation in graph.py
    raise NotImplementedError("Use graph.analyze_request - LLM implementation in main module")


def parse_prompt(state: State):
    """
    SUPER-PARSER: 3-in-1 optimization
    1. Chitchat detection (direct response)
    2. Task parsing with parameter extraction  
    3. Title generation (first turn only)
    
    Returns:
    - parsed_tasks: List of extracted tasks
    - user_expectations: Preferences extracted from prompt
    - final_response: Direct response for chitchat
    - suggested_title: Conversation title (first turn)
    
    NOTE: Full implementation remains in graph.py due to LLM client dependencies.
    This is a reference/documentation stub.
    """
    # Implementation in graph.py
    raise NotImplementedError("Use graph.parse_prompt - LLM implementation in main module")


# Export node functions (will be aliased from graph.py)
__all__ = [
    'preprocess_files',
    'analyze_request', 
    'parse_prompt',
    'get_llm_clients',
]
