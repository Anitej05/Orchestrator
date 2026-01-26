"""
Spreadsheet Agent v3.0 - Session State Management

Unified session state that replaces multiple session managers and caches.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from threading import RLock
from collections import OrderedDict
import pandas as pd

logger = logging.getLogger("spreadsheet_agent.state")


# ============================================================================
# LRU CACHE (Unified)
# ============================================================================

class LRUCache:
    """
    Thread-safe LRU cache with TTL.
    Replaces the 4 separate caching systems.
    """
    
    def __init__(self, max_size: int = 100, ttl_hours: float = 2):
        self.max_size = max_size
        self.ttl = timedelta(hours=ttl_hours)
        self._cache: OrderedDict = OrderedDict()
        self._timestamps: Dict[str, datetime] = {}
        self._lock = RLock()
        
        # Stats
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            # Check expiration
            if self._is_expired(key):
                self._remove(key)
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self.hits += 1
            return self._cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache."""
        with self._lock:
            # Remove if exists
            if key in self._cache:
                self._remove(key)
            
            # Evict oldest if at capacity
            while len(self._cache) >= self.max_size:
                oldest = next(iter(self._cache))
                self._remove(oldest)
            
            # Add new item
            self._cache[key] = value
            self._timestamps[key] = datetime.now()
    
    def invalidate(self, key: str) -> None:
        """Remove item from cache."""
        with self._lock:
            self._remove(key)
    
    def clear(self) -> None:
        """Clear all cache."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def _is_expired(self, key: str) -> bool:
        """Check if entry is expired."""
        if key not in self._timestamps:
            return True
        return datetime.now() - self._timestamps[key] > self.ttl
    
    def _remove(self, key: str) -> None:
        """Remove key without lock (internal use)."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total > 0 else 0
            }


# ============================================================================
# SESSION STATE
# ============================================================================

@dataclass
class Session:
    """Single session state."""
    thread_id: str
    dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)  # file_id -> df
    file_paths: Dict[str, str] = field(default_factory=dict)           # file_id -> path
    file_metadata: Dict[str, Dict] = field(default_factory=dict)       # file_id -> metadata
    history: List[Dict[str, Any]] = field(default_factory=list)        # Operation history
    context: Dict[str, Any] = field(default_factory=dict)              # Conversation context
    pending_tasks: Dict[str, Dict] = field(default_factory=dict)       # task_id -> paused state
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    
    def touch(self):
        """Update last accessed time."""
        self.last_accessed = datetime.now()
    
    def add_operation(self, action: str, description: str, result_summary: Any = None):
        """Add operation to history."""
        self.history.append({
            "action": action,
            "description": description,
            "result_summary": result_summary,
            "timestamp": datetime.now().isoformat()
        })
        # Keep only last 20 operations
        if len(self.history) > 20:
            self.history = self.history[-20:]
    
    def get_recent_history(self, limit: int = 5) -> List[Dict]:
        """Get recent operation history."""
        return self.history[-limit:]
    
    def get_latest_file_id(self) -> Optional[str]:
        """Get the most recently accessed file ID."""
        if not self.dataframes:
            return None
        return list(self.dataframes.keys())[-1]


class SessionState:
    """
    Unified session state manager.
    Replaces session.py and spreadsheet_session_manager.py.
    """
    
    def __init__(self, session_timeout_hours: float = 24):
        self._sessions: Dict[str, Session] = {}
        self._lock = RLock()
        self.session_timeout = timedelta(hours=session_timeout_hours)
        
        # Shared cache for expensive computations
        self.cache = LRUCache(max_size=100, ttl_hours=2)
        
        logger.info("SessionState initialized")
    
    def get_or_create(self, thread_id: str) -> Session:
        """Get existing session or create new one."""
        with self._lock:
            if thread_id not in self._sessions:
                self._sessions[thread_id] = Session(thread_id=thread_id)
                logger.info(f"Created new session: {thread_id}")
            
            session = self._sessions[thread_id]
            session.touch()
            return session
    
    def get(self, thread_id: str) -> Optional[Session]:
        """Get session if exists."""
        with self._lock:
            session = self._sessions.get(thread_id)
            if session:
                session.touch()
            return session
    
    def store_dataframe(
        self, 
        thread_id: str, 
        file_id: str, 
        df: pd.DataFrame,
        file_path: str = "",
        metadata: Dict = None
    ) -> None:
        """Store DataFrame in session."""
        session = self.get_or_create(thread_id)
        
        with self._lock:
            session.dataframes[file_id] = df
            session.file_paths[file_id] = file_path
            session.file_metadata[file_id] = metadata or {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist(),
                "dtypes": {k: str(v) for k, v in df.dtypes.items()}
            }
            session.touch()
            
            # Also cache for quick access
            cache_key = f"{thread_id}:{file_id}"
            self.cache.put(cache_key, df)
            
            logger.info(f"Stored DataFrame {file_id} in session {thread_id}: {df.shape}")
    
    def get_dataframe(
        self, 
        thread_id: str, 
        file_id: str = None
    ) -> Optional[pd.DataFrame]:
        """Get DataFrame from session."""
        session = self.get(thread_id)
        if not session:
            return None
        
        with self._lock:
            # If no file_id, return latest
            if file_id is None:
                file_id = session.get_latest_file_id()
            
            if file_id is None:
                return None
            
            # Try cache first
            cache_key = f"{thread_id}:{file_id}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
            
            # Fallback to session
            return session.dataframes.get(file_id)
    
    def pause_task(
        self,
        thread_id: str,
        task_id: str,
        question: str,
        context: Dict[str, Any]
    ) -> None:
        """Pause a task waiting for user input."""
        session = self.get_or_create(thread_id)
        
        with self._lock:
            session.pending_tasks[task_id] = {
                "question": question,
                "context": context,
                "paused_at": datetime.now().isoformat()
            }
            logger.info(f"Task {task_id} paused in session {thread_id}")
    
    def resume_task(self, thread_id: str, task_id: str) -> Optional[Dict]:
        """Resume a paused task and return its context."""
        session = self.get(thread_id)
        if not session:
            return None
        
        with self._lock:
            return session.pending_tasks.pop(task_id, None)
    
    def cleanup_expired(self) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        now = datetime.now()
        expired = []
        
        with self._lock:
            for thread_id, session in self._sessions.items():
                if now - session.last_accessed > self.session_timeout:
                    expired.append(thread_id)
            
            for thread_id in expired:
                del self._sessions[thread_id]
                logger.info(f"Expired session removed: {thread_id}")
        
        return len(expired)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session state statistics."""
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "total_dataframes": sum(len(s.dataframes) for s in self._sessions.values()),
                "cache_stats": self.cache.get_stats()
            }


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

session_state = SessionState()
