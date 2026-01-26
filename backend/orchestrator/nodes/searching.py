"""
Searching Nodes - Agent discovery and capability matching.

Contains nodes that find and rank agents for tasks:
- get_all_capabilities: Fetches all registered agent capabilities
- agent_directory_search: LLM-based semantic agent selection
- _fallback_text_search: Keyword-based fallback when LLM fails
- rank_agents: Enhanced LLM ranking with rich metadata
"""

import os
import time
import logging
from typing import Dict, Any, List, Tuple
from orchestrator.state import State
from sqlalchemy import select
from database import SessionLocal
from models import AgentCapability

logger = logging.getLogger("AgentOrchestrator")

# Cache for capabilities
cached_capabilities = {
    "texts": [],
    "embeddings": None,
    "timestamp": 0
}
CACHE_DURATION_SECONDS = 30


def get_all_capabilities() -> Tuple[List[str], Any]:
    """
    Fetches all capability texts and embeddings from the database.
    Uses caching to avoid repeated DB queries.
    
    Returns:
        Tuple of (capability_texts, embeddings_array)
    """
    global cached_capabilities
    
    current_time = time.time()
    if (current_time - cached_capabilities["timestamp"]) < CACHE_DURATION_SECONDS:
        return cached_capabilities["texts"], cached_capabilities["embeddings"]
    
    try:
        db = SessionLocal()
        capabilities = db.query(AgentCapability).all()
        texts = [cap.capability_text for cap in capabilities]
        embeddings = [cap.embedding for cap in capabilities if cap.embedding is not None]
        agent_ids = [cap.agent_id for cap in capabilities if cap.embedding is not None]
        db.close()
        
        cached_capabilities = {
            "texts": texts,
            "embeddings": embeddings,
            "agent_ids": agent_ids,
            "timestamp": current_time
        }
        
        return texts, embeddings, agent_ids
    except Exception as e:
        logger.error(f"Failed to fetch capabilities: {e}")
        return [], None


def agent_directory_search(state: State):
    """
    LLM-BASED SEMANTIC AGENT SELECTION (Primary) with vector similarity as fallback.
    
    This approach:
    1. Fetches ALL active agents with their names, descriptions, and capabilities
    2. Uses the LLM to semantically match tasks to the most appropriate agents
    3. Falls back to vector similarity if LLM selection fails
    
    Benefits:
    - Respects explicit user preferences (e.g., "use browser automation agent")
    - Better semantic understanding of task requirements
    - More accurate agent selection for complex/ambiguous tasks
    
    NOTE: Full implementation remains in graph.py due to LLM client dependencies.
    """
    raise NotImplementedError("Use graph.agent_directory_search - LLM implementation in main module")


def _fallback_text_search(parsed_tasks, all_agents, agent_lookup):
    """
    Fallback text-based search when LLM selection fails.
    Uses simple keyword matching on capabilities.
    
    NOTE: Full implementation remains in graph.py.
    """
    raise NotImplementedError("Use graph._fallback_text_search - implementation in main module")


def rank_agents(state: State):
    """
    ENHANCED LLM RANKING: Uses LLM for intelligent agent selection with rich metadata.
    Optimized with Groq for speed/cost while maintaining quality.
    
    For each task:
    - Considers agent descriptions, capabilities, ratings
    - Provides reasoning for the selection
    - Returns ordered list of agent IDs
    
    NOTE: Full implementation remains in graph.py due to LLM client dependencies.
    """
    raise NotImplementedError("Use graph.rank_agents - LLM implementation in main module")


__all__ = [
    'get_all_capabilities',
    'agent_directory_search',
    '_fallback_text_search',
    'rank_agents',
]
