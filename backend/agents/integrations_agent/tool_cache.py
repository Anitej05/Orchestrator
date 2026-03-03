# agents/integrations_agent/tool_cache.py
"""
Tool Cache for General Agent

Caches discovered Composio tools to reduce API calls.
TTL-based expiration and user-specific invalidation.
"""

import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

logger = logging.getLogger("integrations_agent")

class ToolCache:
    """
    TTL-based cache for Composio tools.
    
    Structure:
    {
        "user_123": {
            "Slack": (tools_list, expiry_timestamp),
            "Notion": (tools_list, expiry_timestamp)
        }
    }
    """
    
    def __init__(self, ttl_seconds: int = 300):
        """
        Initialize tool cache.
        
        Args:
            ttl_seconds: Time-to-live for cached entries (default: 5 minutes)
        """
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Tuple[List[Any], float]]] = defaultdict(dict)
        self._hits = 0
        self._misses = 0
        
        logger.info(f"[ToolCache] Initialized with TTL={ttl_seconds}s")
    
    def get(self, user_id: str, app_name: str) -> Optional[List[Any]]:
        """
        Get cached tools for user and app.
        
        Args:
            user_id: User ID
            app_name: App name (e.g., "Slack")
        
        Returns:
            List of tools or None if cache miss/expired
        """
        if user_id not in self._cache:
            self._misses += 1
            return None
        
        user_cache = self._cache[user_id]
        
        if app_name not in user_cache:
            self._misses += 1
            return None
        
        tools, expiry = user_cache[app_name]
        
        # Check if expired
        if time.time() > expiry:
            logger.info(f"[ToolCache] Expired entry for {user_id}/{app_name}")
            del user_cache[app_name]
            self._misses += 1
            return None
        
        self._hits += 1
        logger.debug(f"[ToolCache] Hit for {user_id}/{app_name}")
        return tools
    
    def set(self, user_id: str, app_name: str, tools: List[Any]) -> None:
        """
        Cache tools for user and app.
        
        Args:
            user_id: User ID
            app_name: App name
            tools: List of tools to cache
        """
        expiry = time.time() + self.ttl_seconds
        self._cache[user_id][app_name] = (tools, expiry)
        logger.info(f"[ToolCache] Cached {len(tools)} tools for {user_id}/{app_name} (expires in {self.ttl_seconds}s)")
    
    def invalidate_user(self, user_id: str) -> None:
        """
        Invalidate all cached tools for a user.
        
        Call this when:
        - User disconnects an app
        - User connects a new app
        - Tool execution fails
        
        Args:
            user_id: User ID
        """
        if user_id in self._cache:
            del self._cache[user_id]
            logger.info(f"[ToolCache] Invalidated all entries for {user_id}")
        else:
            logger.debug(f"[ToolCache] No entries to invalidate for {user_id}")
    
    def invalidate_app(self, user_id: str, app_name: str) -> None:
        """
        Invalidate cached tools for specific app.
        
        Args:
            user_id: User ID
            app_name: App name
        """
        if user_id in self._cache and app_name in self._cache[user_id]:
            del self._cache[user_id][app_name]
            logger.info(f"[ToolCache] Invalidated {user_id}/{app_name}")
        else:
            logger.debug(f"[ToolCache] No entry to invalidate for {user_id}/{app_name}")
    
    def clear_all(self) -> None:
        """Clear entire cache"""
        self._cache.clear()
        logger.info("[ToolCache] Cleared all entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache metrics
        """
        total_users = len(self._cache)
        total_entries = sum(len(apps) for apps in self._cache.values())
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_users": total_users,
            "total_entries": total_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round(hit_rate, 2),
            "ttl_seconds": self.ttl_seconds
        }
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        removed_count = 0
        current_time = time.time()
        
        for user_id in list(self._cache.keys()):
            user_cache = self._cache[user_id]
            
            for app_name in list(user_cache.keys()):
                _, expiry = user_cache[app_name]
                
                if current_time > expiry:
                    del user_cache[app_name]
                    removed_count += 1
            
            # Remove user entry if no apps left
            if not user_cache:
                del self._cache[user_id]
        
        if removed_count > 0:
            logger.info(f"[ToolCache] Cleaned up {removed_count} expired entries")
        
        return removed_count
