from typing import List, Dict, Any, Optional
from datetime import datetime
import json

class AgentMemory:
    """
    Simple in-memory state management for the Mail Agent.
    Stores conversation history and context (like last search results) per user.
    """
    def __init__(self):
        # Structure: {user_id: {"history": [], "context": {}}}
        self.store: Dict[str, Dict[str, Any]] = {}

    def _get_user_store(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.store:
            self.store[user_id] = {
                "history": [],
                "context": {}
            }
        return self.store[user_id]

    def add_turn(self, user_id: str, user_input: str, agent_response: str, action_type: str = None):
        """Record a conversation turn"""
        store = self._get_user_store(user_id)
        store["history"].append({
            "timestamp": datetime.now().isoformat(),
            "role": "user",
            "content": user_input
        })
        store["history"].append({
            "timestamp": datetime.now().isoformat(),
            "role": "agent",
            "content": agent_response,
            "action_type": action_type
        })
        # Keep history limited (last 10 turns)
        if len(store["history"]) > 20:
            store["history"] = store["history"][-20:]

    def update_context(self, user_id: str, key: str, value: Any):
        """Update specific context item (e.g., last_search_results)"""
        store = self._get_user_store(user_id)
        store["context"][key] = value

    def get_context(self, user_id: str, key: str) -> Optional[Any]:
        """Retrieve specific context item"""
        store = self._get_user_store(user_id)
        return store["context"].get(key)
        
    def get_history(self, user_id: str) -> List[Dict]:
        return self._get_user_store(user_id)["history"]

    def clear(self, user_id: str):
        if user_id in self.store:
            del self.store[user_id]
    
    def save_search_results(self, user_id: str, message_ids: List[str]):
        """Save message IDs from a search for context-aware follow-up actions."""
        self.update_context(user_id, "last_search_results", message_ids)
    
    def get_last_search_results(self, user_id: str = "me") -> Optional[List[str]]:
        """Get the message IDs from the last search."""
        return self.get_context(user_id, "last_search_results")

# Global memory instance
agent_memory = AgentMemory()
