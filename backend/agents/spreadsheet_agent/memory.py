"""
Memory and caching system for spreadsheet agent
Provides fast access to frequently used data and context management
Enhanced with advanced performance optimizations.
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

# Import advanced performance components
try:
    from .performance_optimizer import (
        AdvancedLRUCache,
        memory_optimizer,
        token_optimizer,
        performance_monitor
    )
    ADVANCED_PERFORMANCE_AVAILABLE = True
    logger.info("✅ Advanced performance optimizations enabled")
except ImportError as e:
    logger.warning(f"⚠️ Advanced performance optimizations not available: {e}")
    ADVANCED_PERFORMANCE_AVAILABLE = False


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
    - Enhanced with advanced performance optimizations
    """
    
    def __init__(self):
        if ADVANCED_PERFORMANCE_AVAILABLE:
            # Use advanced caching with memory monitoring
            self.metadata_cache = AdvancedLRUCache(
                max_size=MEMORY_CACHE_MAX_SIZE, 
                ttl_seconds=MEMORY_CACHE_TTL_SECONDS,
                max_memory_mb=200
            )
            self.query_cache = AdvancedLRUCache(
                max_size=500, 
                ttl_seconds=1800,  # 30 min for queries
                max_memory_mb=300
            )
            self.context_cache = AdvancedLRUCache(
                max_size=200, 
                ttl_seconds=3600,  # 1 hour for context
                max_memory_mb=100
            )
            logger.info("🚀 Using advanced LRU caches with memory optimization")
        else:
            # Fallback to basic LRU caches
            self.metadata_cache = LRUCache(max_size=MEMORY_CACHE_MAX_SIZE, ttl_seconds=MEMORY_CACHE_TTL_SECONDS)
            self.query_cache = LRUCache(max_size=500, ttl_seconds=1800)  # 30 min for queries
            self.context_cache = LRUCache(max_size=200, ttl_seconds=3600)  # 1 hour for context
            logger.info("📦 Using basic LRU caches (fallback mode)")
        
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
        Cache dataframe metadata for fast access with performance tracking
        
        Args:
            file_id: File identifier
            metadata: Metadata dict (shape, columns, dtypes, etc.)
        """
        key = f"df_meta:{file_id}"
        existing = self.metadata_cache.get(key) or {}
        merged = {**existing, **metadata}
        
        # Estimate memory usage for advanced cache
        if ADVANCED_PERFORMANCE_AVAILABLE:
            estimated_size = len(str(merged)) / (1024 * 1024)  # Rough MB estimate
            self.metadata_cache.put(key, merged, estimated_size)
        else:
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
        Cache query result for reuse with performance optimization
        
        Args:
            file_id: File identifier
            query: Query string
            result: Query result
            thread_id: Optional thread ID
        """
        key = self._make_key("query", file_id, query, thread_id or "default")
        cache_data = {
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'file_id': file_id,
            'query': query
        }
        
        # Estimate memory usage for advanced cache
        if ADVANCED_PERFORMANCE_AVAILABLE:
            estimated_size = len(str(cache_data)) / (1024 * 1024)  # Rough MB estimate
            self.query_cache.put(key, cache_data, estimated_size)
        else:
            self.query_cache.put(key, cache_data)
        
        logger.debug(f"Cached query result for file {file_id}")
    
    def get_cached_query(self, file_id: str, query: str, thread_id: str = None) -> Optional[Any]:
        """
        Get cached query result with performance tracking
        
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
        Store conversation context with token optimization
        
        Args:
            thread_id: Thread identifier
            file_id: File identifier
            context: Context string
        """
        key = self._make_key("context", thread_id, file_id)
        
        # Optimize context for token usage if advanced performance is available
        if ADVANCED_PERFORMANCE_AVAILABLE:
            # Truncate context if it's too long
            optimized_context = token_optimizer.truncate_to_token_limit(
                context, CONTEXT_MEMORY_MAX_TOKENS
            )
            estimated_size = len(optimized_context) / (1024 * 1024)
            self.context_cache.put(key, optimized_context, estimated_size)
        else:
            self.context_cache.put(key, context)
    
    def get_context(self, thread_id: str, file_id: str) -> Optional[str]:
        """Get stored context"""
        key = self._make_key("context", thread_id, file_id)
        return self.context_cache.get(key)
    
    # ============== FILE SUMMARIES ==============
    
    def cache_file_summary(self, file_id: str, summary: Dict[str, Any]):
        """Cache file summary (headers, sample, stats) with optimization"""
        key = f"summary:{file_id}"
        
        if ADVANCED_PERFORMANCE_AVAILABLE:
            estimated_size = len(str(summary)) / (1024 * 1024)
            self.metadata_cache.put(key, summary, estimated_size)
        else:
            self.metadata_cache.put(key, summary)
    
    def get_file_summary(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Get cached file summary"""
        key = f"summary:{file_id}"
        return self.metadata_cache.get(key)
    
    # ============== PERSISTENT MEMORY ==============
    
    def save_to_disk(self, key: str = "state", data: Any = None):
        """Save data to persistent storage with performance monitoring"""
        try:
            if ADVANCED_PERFORMANCE_AVAILABLE:
                with performance_monitor.time_operation(f"save_to_disk_{key}"):
                    file_path = self.memory_dir / f"{key}.json"
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data or self.get_cache_stats(), f, indent=2)
                    logger.debug(f"Saved {key} to disk with performance tracking")
            else:
                file_path = self.memory_dir / f"{key}.json"
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data or self.get_cache_stats(), f, indent=2)
                logger.debug(f"Saved {key} to disk")
        except Exception as e:
            logger.error(f"Failed to save {key} to disk: {e}")
    
    def load_from_disk(self, key: str = "state") -> Optional[Any]:
        """Load data from persistent storage with performance monitoring"""
        try:
            if ADVANCED_PERFORMANCE_AVAILABLE:
                with performance_monitor.time_operation(f"load_from_disk_{key}"):
                    file_path = self.memory_dir / f"{key}.json"
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            return json.load(f)
            else:
                file_path = self.memory_dir / f"{key}.json"
                if file_path.exists():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {key} from disk: {e}")
        return None
    
    # ============== STATISTICS ==============
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        if ADVANCED_PERFORMANCE_AVAILABLE:
            return {
                'metadata_cache': self.metadata_cache.get_stats(),
                'query_cache': self.query_cache.get_stats(),
                'context_cache': self.context_cache.get_stats(),
                'performance_enabled': True,
                'memory_optimizer_stats': {
                    'total_session_memory_mb': memory_optimizer.get_total_memory_usage(),
                    'system_memory_info': memory_optimizer.get_system_memory_info()
                }
            }
        else:
            return {
                'metadata_cache_size': len(self.metadata_cache.cache),
                'query_cache_size': len(self.query_cache.cache),
                'context_cache_size': len(self.context_cache.cache),
                'metadata_cache_max': self.metadata_cache.max_size,
                'query_cache_max': self.query_cache.max_size,
                'context_cache_max': self.context_cache.max_size,
                'performance_enabled': False
            }
    
    def clear_all(self):
        """Clear all caches with performance monitoring"""
        if ADVANCED_PERFORMANCE_AVAILABLE:
            with performance_monitor.time_operation("clear_all_caches"):
                self.metadata_cache.clear()
                self.query_cache.clear()
                self.context_cache.clear()
                logger.info("All advanced caches cleared")
        else:
            self.metadata_cache.clear()
            self.query_cache.clear()
            self.context_cache.clear()
            logger.info("All caches cleared")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if ADVANCED_PERFORMANCE_AVAILABLE:
            return {
                'cache_stats': self.get_cache_stats(),
                'performance_report': performance_monitor.get_performance_report(),
                'memory_should_cleanup': memory_optimizer.should_trigger_cleanup()
            }
        else:
            return {
                'cache_stats': self.get_cache_stats(),
                'performance_enabled': False
            }


# Global memory instance
spreadsheet_memory = SpreadsheetMemory()
