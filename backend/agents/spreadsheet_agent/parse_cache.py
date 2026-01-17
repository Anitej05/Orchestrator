"""
Parse Result Cache

Caches parsed spreadsheet representations and generated contexts to avoid re-parsing.
Implements Requirements 8.4, 8.5 from the intelligent spreadsheet parsing spec.

This cache stores:
- ParsedSpreadsheet objects (complete parsing results)
- Generated contexts (structured and compact)
- Schema information
- Metadata

The cache is thread-isolated and integrates with the existing DataFrameCache.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from threading import RLock
from datetime import datetime
from collections import defaultdict
import hashlib
import json

from backend.agents.spreadsheet_agent.parsing_models import (
    ParsedSpreadsheet,
    StructuredContext
)

logger = logging.getLogger(__name__)


class ParseCache:
    """
    Thread-isolated cache for parsed spreadsheet results.
    
    Features:
    - Caches ParsedSpreadsheet objects to avoid re-parsing
    - Caches generated contexts (structured and compact)
    - Thread-safe operations with RLock
    - Strict thread isolation
    - Cache invalidation support
    - Context window optimization tracking
    
    Storage Structure:
    {
        "thread_123": {
            "file_456": {
                "parsed": ParsedSpreadsheet(...),
                "contexts": {
                    "structured_8000": StructuredContext(...),
                    "compact_8000": "...",
                    "full": "..."
                },
                "timestamp": "2024-01-15T10:30:00Z",
                "access_count": 5,
                "last_accessed": "2024-01-15T10:35:00Z"
            }
        }
    }
    """
    
    def __init__(self):
        """Initialize the parse cache with thread-isolated storage."""
        self._lock = RLock()
        # Thread-scoped storage: thread_id -> file_id -> cache_data
        self._storage: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        logger.info("ParseCache initialized with thread isolation")
    
    def store_parsed(
        self,
        thread_id: str,
        file_id: str,
        parsed: ParsedSpreadsheet
    ) -> None:
        """
        Store parsed spreadsheet result.
        
        Args:
            thread_id: Thread identifier for isolation
            file_id: File identifier
            parsed: ParsedSpreadsheet object to cache
        
        Validates: Requirement 8.5
        """
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        if not file_id:
            raise ValueError("file_id cannot be empty")
        if parsed is None:
            raise ValueError("parsed cannot be None")
        
        with self._lock:
            # Ensure thread storage exists
            if thread_id not in self._storage:
                self._storage[thread_id] = {}
            
            # Store or update parsed result
            if file_id not in self._storage[thread_id]:
                self._storage[thread_id][file_id] = {
                    'parsed': parsed,
                    'contexts': {},
                    'timestamp': datetime.now().isoformat(),
                    'access_count': 0,
                    'last_accessed': datetime.now().isoformat()
                }
            else:
                # Update existing entry
                self._storage[thread_id][file_id]['parsed'] = parsed
                self._storage[thread_id][file_id]['timestamp'] = datetime.now().isoformat()
            
            logger.info(
                f"Stored parsed result for thread={thread_id}, file={file_id}, "
                f"document_type={parsed.document_type.value}, "
                f"tables={parsed.table_count}"
            )
    
    def retrieve_parsed(
        self,
        thread_id: str,
        file_id: str,
        allow_cross_thread: bool = False
    ) -> Optional[ParsedSpreadsheet]:
        """
        Retrieve cached parsed spreadsheet.
        
        Args:
            thread_id: Thread identifier
            file_id: File identifier
            allow_cross_thread: If True, allows retrieving from other threads (Requirement 13.4)
        
        Returns:
            ParsedSpreadsheet if cached, None otherwise
        
        Validates: Requirements 8.5, 13.3, 13.4
        """
        if not thread_id or not file_id:
            return None
        
        with self._lock:
            if thread_id not in self._storage:
                # If cross-thread access allowed, search all threads
                if allow_cross_thread:
                    return self._retrieve_parsed_cross_thread(file_id, thread_id)
                return None
            
            if file_id not in self._storage[thread_id]:
                # If cross-thread access allowed, search other threads
                if allow_cross_thread:
                    return self._retrieve_parsed_cross_thread(file_id, thread_id)
                return None
            
            # Update access tracking
            cache_entry = self._storage[thread_id][file_id]
            cache_entry['access_count'] += 1
            cache_entry['last_accessed'] = datetime.now().isoformat()
            
            parsed = cache_entry['parsed']
            
            logger.info(
                f"Retrieved cached parsed result for thread={thread_id}, file={file_id}, "
                f"access_count={cache_entry['access_count']}"
            )
            
            return parsed
    
    def _retrieve_parsed_cross_thread(
        self,
        file_id: str,
        requesting_thread_id: str
    ) -> Optional[ParsedSpreadsheet]:
        """
        Retrieve parsed result from any thread (cross-thread access).
        
        Args:
            file_id: File identifier to search for
            requesting_thread_id: Thread making the request
        
        Returns:
            ParsedSpreadsheet if found, None otherwise
        
        Validates: Requirement 13.4
        """
        # Search all threads for the file
        for thread_id, thread_storage in self._storage.items():
            if file_id in thread_storage:
                cache_entry = thread_storage[file_id]
                cache_entry['access_count'] += 1
                cache_entry['last_accessed'] = datetime.now().isoformat()
                
                parsed = cache_entry['parsed']
                logger.info(
                    f"Cross-thread access: Retrieved parsed result file={file_id} "
                    f"from thread={thread_id} for requesting_thread={requesting_thread_id}"
                )
                return parsed
        
        logger.warning(
            f"Parsed result for file {file_id} not found in any thread "
            f"(requested by thread={requesting_thread_id})"
        )
        return None
    
    def store_context(
        self,
        thread_id: str,
        file_id: str,
        context_type: str,
        context: Any,
        max_tokens: Optional[int] = None
    ) -> None:
        """
        Store generated context.
        
        Args:
            thread_id: Thread identifier
            file_id: File identifier
            context_type: Type of context ("structured", "compact", "full")
            context: Context object or string
            max_tokens: Token limit used for generation (for cache key)
        
        Validates: Requirement 8.4
        """
        if not thread_id or not file_id:
            raise ValueError("thread_id and file_id cannot be empty")
        
        with self._lock:
            # Ensure entry exists
            if thread_id not in self._storage or file_id not in self._storage[thread_id]:
                logger.warning(
                    f"Cannot store context: no parsed result for thread={thread_id}, file={file_id}"
                )
                return
            
            # Create cache key with token limit
            cache_key = context_type
            if max_tokens:
                cache_key = f"{context_type}_{max_tokens}"
            
            # Store context
            self._storage[thread_id][file_id]['contexts'][cache_key] = context
            
            logger.info(
                f"Stored {context_type} context for thread={thread_id}, file={file_id}, "
                f"cache_key={cache_key}"
            )
    
    def retrieve_context(
        self,
        thread_id: str,
        file_id: str,
        context_type: str,
        max_tokens: Optional[int] = None,
        allow_cross_thread: bool = False
    ) -> Optional[Any]:
        """
        Retrieve cached context.
        
        Args:
            thread_id: Thread identifier
            file_id: File identifier
            context_type: Type of context ("structured", "compact", "full")
            max_tokens: Token limit to match (for cache key)
            allow_cross_thread: If True, allows retrieving from other threads (Requirement 13.4)
        
        Returns:
            Cached context if available, None otherwise
        
        Validates: Requirements 8.4, 13.3, 13.4
        """
        if not thread_id or not file_id:
            return None
        
        with self._lock:
            if thread_id not in self._storage or file_id not in self._storage[thread_id]:
                # If cross-thread access allowed, search other threads
                if allow_cross_thread:
                    return self._retrieve_context_cross_thread(
                        file_id, context_type, max_tokens, thread_id
                    )
                return None
            
            # Create cache key
            cache_key = context_type
            if max_tokens:
                cache_key = f"{context_type}_{max_tokens}"
            
            contexts = self._storage[thread_id][file_id].get('contexts', {})
            
            if cache_key in contexts:
                logger.info(
                    f"Cache HIT for {context_type} context: thread={thread_id}, file={file_id}, "
                    f"cache_key={cache_key}"
                )
                return contexts[cache_key]
            
            # If cross-thread access allowed and not found in current thread
            if allow_cross_thread:
                return self._retrieve_context_cross_thread(
                    file_id, context_type, max_tokens, thread_id
                )
            
            logger.debug(
                f"Cache MISS for {context_type} context: thread={thread_id}, file={file_id}, "
                f"cache_key={cache_key}"
            )
            return None
    
    def _retrieve_context_cross_thread(
        self,
        file_id: str,
        context_type: str,
        max_tokens: Optional[int],
        requesting_thread_id: str
    ) -> Optional[Any]:
        """
        Retrieve context from any thread (cross-thread access).
        
        Args:
            file_id: File identifier to search for
            context_type: Type of context
            max_tokens: Token limit to match
            requesting_thread_id: Thread making the request
        
        Returns:
            Context if found, None otherwise
        
        Validates: Requirement 13.4
        """
        # Create cache key
        cache_key = context_type
        if max_tokens:
            cache_key = f"{context_type}_{max_tokens}"
        
        # Search all threads for the file and context
        for thread_id, thread_storage in self._storage.items():
            if file_id in thread_storage:
                contexts = thread_storage[file_id].get('contexts', {})
                if cache_key in contexts:
                    logger.info(
                        f"Cross-thread cache HIT: Retrieved {context_type} context "
                        f"file={file_id} from thread={thread_id} "
                        f"for requesting_thread={requesting_thread_id}, cache_key={cache_key}"
                    )
                    return contexts[cache_key]
        
        logger.debug(
            f"Cross-thread cache MISS: Context not found for file={file_id}, "
            f"context_type={context_type}, requesting_thread={requesting_thread_id}"
        )
        return None
    
    def has_parsed(
        self,
        thread_id: str,
        file_id: str
    ) -> bool:
        """
        Check if parsed result is cached.
        
        Args:
            thread_id: Thread identifier
            file_id: File identifier
        
        Returns:
            True if cached, False otherwise
        """
        if not thread_id or not file_id:
            return False
        
        with self._lock:
            return (
                thread_id in self._storage and
                file_id in self._storage[thread_id] and
                'parsed' in self._storage[thread_id][file_id]
            )
    
    def has_context(
        self,
        thread_id: str,
        file_id: str,
        context_type: str,
        max_tokens: Optional[int] = None
    ) -> bool:
        """
        Check if context is cached.
        
        Args:
            thread_id: Thread identifier
            file_id: File identifier
            context_type: Type of context
            max_tokens: Token limit to match
        
        Returns:
            True if cached, False otherwise
        """
        if not thread_id or not file_id:
            return False
        
        with self._lock:
            if thread_id not in self._storage or file_id not in self._storage[thread_id]:
                return False
            
            cache_key = context_type
            if max_tokens:
                cache_key = f"{context_type}_{max_tokens}"
            
            contexts = self._storage[thread_id][file_id].get('contexts', {})
            return cache_key in contexts
    
    def invalidate_file(
        self,
        thread_id: str,
        file_id: str
    ) -> bool:
        """
        Invalidate cache for a specific file.
        
        Use this when file content changes or needs to be re-parsed.
        
        Args:
            thread_id: Thread identifier
            file_id: File identifier
        
        Returns:
            True if invalidated, False if not found
        """
        if not thread_id or not file_id:
            return False
        
        with self._lock:
            if thread_id in self._storage and file_id in self._storage[thread_id]:
                del self._storage[thread_id][file_id]
                logger.info(f"Invalidated cache for thread={thread_id}, file={file_id}")
                return True
            
            return False
    
    def clear_thread(
        self,
        thread_id: str
    ) -> None:
        """
        Clear all cached data for a thread.
        
        Args:
            thread_id: Thread identifier
        """
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        
        with self._lock:
            if thread_id in self._storage:
                file_count = len(self._storage[thread_id])
                del self._storage[thread_id]
                logger.info(f"Cleared parse cache for thread {thread_id} ({file_count} files)")
            else:
                logger.warning(f"Thread {thread_id} not found in parse cache")
    
    def list_files(
        self,
        thread_id: str
    ) -> list:
        """
        List all cached file IDs in a thread.
        
        Args:
            thread_id: Thread identifier
        
        Returns:
            List of file IDs
        """
        if not thread_id:
            return []
        
        with self._lock:
            if thread_id not in self._storage:
                return []
            
            return list(self._storage[thread_id].keys())
    
    def get_cache_stats(
        self,
        thread_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Args:
            thread_id: Optional thread to get stats for (None for all threads)
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            if thread_id:
                # Stats for specific thread
                if thread_id not in self._storage:
                    return {
                        'thread_id': thread_id,
                        'file_count': 0,
                        'files': []
                    }
                
                thread_storage = self._storage[thread_id]
                file_stats = []
                
                for file_id, cache_entry in thread_storage.items():
                    file_stats.append({
                        'file_id': file_id,
                        'access_count': cache_entry.get('access_count', 0),
                        'last_accessed': cache_entry.get('last_accessed'),
                        'cached_contexts': list(cache_entry.get('contexts', {}).keys()),
                        'context_count': len(cache_entry.get('contexts', {}))
                    })
                
                return {
                    'thread_id': thread_id,
                    'file_count': len(thread_storage),
                    'files': file_stats
                }
            
            else:
                # Stats for all threads
                total_files = sum(len(files) for files in self._storage.values())
                total_contexts = sum(
                    len(cache_entry.get('contexts', {}))
                    for thread_storage in self._storage.values()
                    for cache_entry in thread_storage.values()
                )
                
                thread_stats = {}
                for tid, thread_storage in self._storage.items():
                    thread_stats[tid] = {
                        'file_count': len(thread_storage),
                        'file_ids': list(thread_storage.keys())
                    }
                
                return {
                    'total_threads': len(self._storage),
                    'total_files': total_files,
                    'total_contexts': total_contexts,
                    'threads': thread_stats
                }
    
    def get_cache_efficiency(
        self,
        thread_id: str,
        file_id: str
    ) -> Dict[str, Any]:
        """
        Get cache efficiency metrics for a specific file.
        
        Args:
            thread_id: Thread identifier
            file_id: File identifier
        
        Returns:
            Dictionary with efficiency metrics
        """
        if not thread_id or not file_id:
            return {}
        
        with self._lock:
            if thread_id not in self._storage or file_id not in self._storage[thread_id]:
                return {
                    'cached': False,
                    'access_count': 0
                }
            
            cache_entry = self._storage[thread_id][file_id]
            
            return {
                'cached': True,
                'access_count': cache_entry.get('access_count', 0),
                'timestamp': cache_entry.get('timestamp'),
                'last_accessed': cache_entry.get('last_accessed'),
                'context_count': len(cache_entry.get('contexts', {})),
                'cached_contexts': list(cache_entry.get('contexts', {}).keys())
            }
    
    def clear_all(self) -> None:
        """
        Clear all cached data (use with caution).
        
        This removes all threads and files from the cache.
        """
        with self._lock:
            thread_count = len(self._storage)
            file_count = sum(len(files) for files in self._storage.values())
            self._storage.clear()
            logger.warning(
                f"Cleared ALL parse cache data: {thread_count} threads, {file_count} files"
            )
    
    def switch_thread_context(
        self,
        from_thread_id: str,
        to_thread_id: str,
        file_id: Optional[str] = None
    ) -> Optional[ParsedSpreadsheet]:
        """
        Switch from one thread context to another and load the correct parsed result.
        
        This method facilitates thread context switching by:
        1. Validating the target thread exists
        2. Loading the appropriate parsed result from the target thread
        3. Logging the context switch for debugging
        
        Args:
            from_thread_id: Current thread identifier
            to_thread_id: Target thread identifier to switch to
            file_id: Optional file identifier. If None, loads most recent file from target thread.
        
        Returns:
            ParsedSpreadsheet from target thread, or None if not found
        
        Validates: Requirement 13.3
        """
        if not from_thread_id or not to_thread_id:
            raise ValueError("from_thread_id and to_thread_id cannot be empty")
        
        logger.info(
            f"Switching parse cache thread context: {from_thread_id} -> {to_thread_id}, "
            f"file_id={file_id or 'most_recent'}"
        )
        
        with self._lock:
            # Check if target thread exists
            if to_thread_id not in self._storage:
                logger.warning(
                    f"Target thread {to_thread_id} not found in parse cache. "
                    f"Available threads: {list(self._storage.keys())}"
                )
                return None
            
            thread_storage = self._storage[to_thread_id]
            
            # If no file_id specified, get most recent
            if file_id is None:
                if not thread_storage:
                    logger.warning(f"No files in target thread {to_thread_id}")
                    return None
                
                # Get most recent file by timestamp
                most_recent = max(
                    thread_storage.items(),
                    key=lambda x: x[1]['timestamp']
                )
                file_id = most_recent[0]
                logger.info(f"No file_id specified, using most recent: {file_id}")
            
            # Retrieve from target thread
            if file_id not in thread_storage:
                logger.warning(
                    f"File {file_id} not found in target thread {to_thread_id}. "
                    f"Available files: {list(thread_storage.keys())}"
                )
                return None
            
            cache_entry = thread_storage[file_id]
            cache_entry['access_count'] += 1
            cache_entry['last_accessed'] = datetime.now().isoformat()
            
            parsed = cache_entry['parsed']
            
            logger.info(
                f"Successfully switched to thread {to_thread_id}, "
                f"loaded parsed result for file={file_id}, "
                f"document_type={parsed.document_type.value}"
            )
            
            return parsed
    
    def get_thread_context(
        self,
        thread_id: str
    ) -> Dict[str, Any]:
        """
        Get complete context information for a thread.
        
        This provides a snapshot of all files and cached contexts in a thread,
        useful for context switching and debugging.
        
        Args:
            thread_id: Thread identifier
        
        Returns:
            Dictionary with thread context information
        
        Validates: Requirement 13.3
        """
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        
        with self._lock:
            if thread_id not in self._storage:
                return {
                    'thread_id': thread_id,
                    'exists': False,
                    'file_count': 0,
                    'files': []
                }
            
            thread_storage = self._storage[thread_id]
            
            files_info = []
            for file_id, cache_entry in thread_storage.items():
                parsed = cache_entry['parsed']
                files_info.append({
                    'file_id': file_id,
                    'document_type': parsed.document_type.value,
                    'table_count': parsed.table_count,
                    'timestamp': cache_entry['timestamp'],
                    'access_count': cache_entry['access_count'],
                    'last_accessed': cache_entry['last_accessed'],
                    'cached_contexts': list(cache_entry.get('contexts', {}).keys())
                })
            
            return {
                'thread_id': thread_id,
                'exists': True,
                'file_count': len(thread_storage),
                'files': files_info
            }


# Global cache instance
parse_cache = ParseCache()
