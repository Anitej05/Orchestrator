"""
Memory and caching system for spreadsheet agent
Provides fast access to frequently used data and context management
"""
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, List
from collections import OrderedDict
from threading import Lock

from .config import (
    MEMORY_CACHE_DIR,
    MEMORY_CACHE_MAX_SIZE,
    MEMORY_CACHE_TTL_SECONDS,
    CONTEXT_MEMORY_MAX_TOKENS
)

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Simple LRU (Least Recently Used) cache with TTL
    """
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
        self.lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        with self.lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            if self._is_expired(key):
                del self.cache[key]
                del self.timestamps[key]
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key: str, value: Any):
        """Put item in cache"""
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.max_size:
                    # Remove least recently used
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
            
            self.cache[key] = value
            self.timestamps[key] = datetime.now()
    
    def invalidate(self, key: str):
        """Remove item from cache"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.timestamps[key]
    
    def clear(self):
        """Clear all cache"""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.timestamps:
            return True
        age = (datetime.now() - self.timestamps[key]).total_seconds()
        return age > self.ttl_seconds


class SpreadsheetMemory:
    """
    Memory system for spreadsheet agent
    - Caches dataframe metadata
    - Stores recent queries and results
    - Maintains context for conversations
    - Provides fast access to frequently used information
    """
    
    def __init__(self):
        self.metadata_cache = LRUCache(max_size=MEMORY_CACHE_MAX_SIZE, ttl_seconds=MEMORY_CACHE_TTL_SECONDS)
        self.query_cache = LRUCache(max_size=500, ttl_seconds=1800)  # 30 min for queries
        self.context_cache = LRUCache(max_size=200, ttl_seconds=3600)  # 1 hour for context
        self.memory_dir = Path(MEMORY_CACHE_DIR)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"SpreadsheetMemory initialized with cache dir: {self.memory_dir}")
    
    def _make_key(self, *args) -> str:
        """Create cache key from arguments"""
        combined = ":".join(str(arg) for arg in args)
        return hashlib.md5(combined.encode()).hexdigest()
    
    # ============== DATAFRAME METADATA CACHE ==============
    
    def cache_df_metadata(self, file_id: str, metadata: Dict[str, Any]):
        """
        Cache dataframe metadata for fast access
        
        Args:
            file_id: File identifier
            metadata: Metadata dict (shape, columns, dtypes, etc.)
        """
        key = f"df_meta:{file_id}"
        existing = self.metadata_cache.get(key) or {}
        merged = {**existing, **metadata}
        self.metadata_cache.put(key, merged)
        logger.debug(f"Cached metadata for {file_id}")
    
    def get_df_metadata(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get cached dataframe metadata"""
        key = f"df_meta:{file_id}"
        return self.metadata_cache.get(key)
    
    def invalidate_df_metadata(self, file_id: str):
        """Invalidate cached metadata (after operations)"""
        key = f"df_meta:{file_id}"
        self.metadata_cache.invalidate(key)
    
    # ============== QUERY RESULT CACHE ==============
    
    def cache_query_result(self, file_id: str, query: str, result: Any, thread_id: str = None):
        """
        Cache query result for reuse
        
        Args:
            file_id: File identifier
            query: Query string
            result: Query result
            thread_id: Optional thread ID
        """
        key = self._make_key("query", file_id, query, thread_id or "default")
        self.query_cache.put(key, {
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'file_id': file_id,
            'query': query
        })
        logger.debug(f"Cached query result for file {file_id}")
    
    def get_cached_query(self, file_id: str, query: str, thread_id: str = None) -> Optional[Any]:
        """
        Get cached query result
        
        Args:
            file_id: File identifier
            query: Query string
            thread_id: Optional thread ID
        
        Returns:
            Cached result or None
        """
        key = self._make_key("query", file_id, query, thread_id or "default")
        cached = self.query_cache.get(key)
        if cached:
            logger.debug(f"Query cache HIT for file {file_id}")
            return cached['result']
        logger.debug(f"Query cache MISS for file {file_id}")
        return None
    
    # ============== CONTEXT MEMORY ==============
    
    def store_context(self, thread_id: str, file_id: str, context: str):
        """
        Store conversation context
        
        Args:
            thread_id: Thread identifier
            file_id: File identifier
            context: Context string
        """
        key = self._make_key("context", thread_id, file_id)
        self.context_cache.put(key, context)
    
    def get_context(self, thread_id: str, file_id: str) -> Optional[str]:
        """Get stored context"""
        key = self._make_key("context", thread_id, file_id)
        return self.context_cache.get(key)
    
    # ============== FILE SUMMARIES ==============
    
    def cache_file_summary(self, file_id: str, summary: Dict[str, Any]):
        """Cache file summary (headers, sample, stats)"""
        key = f"summary:{file_id}"
        self.metadata_cache.put(key, summary)
    
    def get_file_summary(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get cached file summary"""
        key = f"summary:{file_id}"
        return self.metadata_cache.get(key)
    
    # ============== PERSISTENT MEMORY ==============
    
    def save_to_disk(self, key: str, data: Any):
        """Save data to persistent storage"""
        try:
            file_path = self.memory_dir / f"{key}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {key} to disk")
        except Exception as e:
            logger.error(f"Failed to save {key} to disk: {e}")
    
    def load_from_disk(self, key: str = "state") -> Optional[Any]:
        """Load data from persistent storage.

        The caller can optionally provide a key; defaults to "state" to keep
        backward compatibility with older call sites that didn't pass a key.
        """
        try:
            file_path = self.memory_dir / f"{key}.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {key} from disk: {e}")
        return None
    
    # ============== STATISTICS ==============
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'metadata_cache_size': len(self.metadata_cache.cache),
            'query_cache_size': len(self.query_cache.cache),
            'context_cache_size': len(self.context_cache.cache),
            'metadata_cache_max': self.metadata_cache.max_size,
            'query_cache_max': self.query_cache.max_size,
            'context_cache_max': self.context_cache.max_size
        }
    
    def clear_all(self):
        """Clear all caches"""
        self.metadata_cache.clear()
        self.query_cache.clear()
        self.context_cache.clear()
        logger.info("All caches cleared")


# Global memory instance
spreadsheet_memory = SpreadsheetMemory()
