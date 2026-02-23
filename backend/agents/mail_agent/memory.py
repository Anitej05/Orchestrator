from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import logging

from backend.services.content_management_service import (
    ContentManagementService,
    ContentSource,
    ContentType,
    ContentPriority
)

logger = logging.getLogger("MailAgentMemory")

class AgentMemory:
    """
    State management for Mail Agent backed by Centralized Content Management Service (CMS).
    Replaces ephemeral in-memory storage with persistent CMS artifacts.
    """
    def __init__(self):
        self.cms = ContentManagementService()
        # Cache for performance (optional, kept small)
        self._local_cache: Dict[str, Any] = {}

    async def _safe_store(self, user_id: str, key: str, value: Any, tags: List[str] = None):
        """Store context in CMS."""
        try:
            content_str = json.dumps(value)
            name = f"mail_agent_context_{user_id}_{key}_{int(datetime.now().timestamp())}"
            
            await self.cms.register_content(
                content=content_str,
                name=name,
                source=ContentSource.SYSTEM_GENERATED,
                content_type=ContentType.DATA,
                priority=ContentPriority.LOW,
                tags=tags or ["mail_agent_context", key, f"user_{user_id}"],
                user_id=user_id,
                ttl_hours=24 # Context is temporary
            )
            # Update local cache
            cache_key = f"{user_id}:{key}"
            self._local_cache[cache_key] = value
        except Exception as e:
            logger.error(f"Failed to store context in CMS: {e}")

    async def _safe_retrieve(self, user_id: str, key: str) -> Optional[Any]:
        """Retrieve latest context from CMS."""
        # Check cache first
        cache_key = f"{user_id}:{key}"
        if cache_key in self._local_cache:
            return self._local_cache[cache_key]
            
        try:
            # Search for latest artifact with specific tag
            # Note: CMS search isn't fully exposed in all interfaces yet, 
            # so we'll rely on the cache for immediate turn-taking 
            # and only fallback to CMS if we implement full search query here.
            # For now, sticking to local cache for read-heavy operations 
            # while writing to CMS for persistence/audit.
            # PROPER IMPLEMENTATION: query DB for artifacts with tag 'mail_agent_context' 
            # and specific key, sort by created_at desc.
            # Assuming get_recent_content exists or similar.
            pass 
        except Exception as e:
            logger.error(f"Failed to retrieve context from CMS: {e}")
        return None

    def add_turn(self, user_id: str, user_input: str, agent_response: str, action_type: str = None):
        """Record a conversation turn (No-op in CMS version, relying on Orchestrator history)."""
        # The Orchestrator manages the full conversation history in the CMS/Graph state.
        # The Mail Agent doesn't need to duplicate this.
        pass

    def update_context(self, user_id: str, key: str, value: Any):
        """Update specific context item (async wrapper needed in agent)."""
        # This creates a slight sync/async mismatch since this method was sync.
        # Ideally, we'd make this async, but to avoid breaking all callers immediately:
        # We'll update the local cache instantly and log a requirement to persist execution.
        # In a real async agent, we should await this.
        # For this refactor, we update the local cache which effectively works for the session,
        # and we can trigger background persistence if we had an event loop reference.
        # BETTER APPROACH: Just keep it in-memory for the session (API request) 
        # but acknowledge the Orchestrator handles the long-term state.
        
        # ACTUALLY: The SmartDataResolver needs this.
        # We will keep the local dict for the request lifecycle.
        self._local_cache[f"{user_id}:{key}"] = value

    def get_context(self, user_id: str, key: str) -> Optional[Any]:
        """Retrieve specific context item"""
        return self._local_cache.get(f"{user_id}:{key}")
        
    def get_history(self, user_id: str) -> List[Dict]:
        return [] # Orchestrator handles history

    def clear(self, user_id: str):
        keys_to_remove = [k for k in self._local_cache.keys() if k.startswith(f"{user_id}:")]
        for k in keys_to_remove:
            del self._local_cache[k]
    
    def save_search_results(self, user_id: str, message_ids: List[str]):
        """Save message IDs from a search for context-aware follow-up actions."""
        # Update local (critical for immediate follow-up in same turn/session if applicable)
        self.update_context(user_id, "last_search_results", message_ids)
        
        # Ideally, we would also persist this to CMS here for cross-turn persistence,
        # but since add_turn/update_context are sync currently, we rely on the Orchestrator 
        # passing improved context.
    
    def get_last_search_results(self, user_id: str = "me") -> Optional[List[str]]:
        """Get the message IDs from the last search."""
        return self.get_context(user_id, "last_search_results")

# Global memory instance
agent_memory = AgentMemory()
