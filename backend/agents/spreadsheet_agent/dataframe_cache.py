"""
DataFrame Cache with Thread Isolation

Provides thread-safe storage for parsed dataframes with strict thread isolation.
Implements Requirements 13.1, 13.2, 13.5 from the intelligent spreadsheet parsing spec.
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from threading import RLock
from datetime import datetime
from collections import defaultdict
import pandas as pd

logger = logging.getLogger(__name__)


class DataFrameCache:
    """
    Thread-isolated storage for parsed dataframes.
    
    Features:
    - Thread-safe operations with RLock
    - Strict thread isolation (no cross-thread access by default)
    - File-thread association tracking
    - Metadata storage alongside dataframes
    - Thread clearing for cleanup
    
    Storage Structure:
    {
        "thread_123": {
            "file_456": {
                "df": pd.DataFrame(...),
                "metadata": {...},
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }
    }
    """
    
    def __init__(self):
        """Initialize the DataFrame cache with thread-isolated storage."""
        self._lock = RLock()
        # Thread-scoped storage: thread_id -> file_id -> data
        self._storage: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        logger.info("DataFrameCache initialized with thread isolation")
    
    def store(
        self, 
        thread_id: str, 
        file_id: str, 
        df: pd.DataFrame, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store dataframe for a specific thread.
        
        Args:
            thread_id: Thread identifier for isolation
            file_id: File identifier
            df: Pandas DataFrame to store
            metadata: Optional metadata dict (shape, columns, dtypes, etc.)
        
        Validates: Requirements 13.1, 13.2
        """
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        if not file_id:
            raise ValueError("file_id cannot be empty")
        if df is None:
            raise ValueError("df cannot be None")
        
        with self._lock:
            # Ensure thread storage exists
            if thread_id not in self._storage:
                self._storage[thread_id] = {}
            
            # Store dataframe with metadata
            self._storage[thread_id][file_id] = {
                'df': df,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(
                f"Stored dataframe for thread={thread_id}, file={file_id}, "
                f"shape={df.shape}, columns={len(df.columns)}"
            )
    
    def retrieve(
        self, 
        thread_id: str, 
        file_id: Optional[str] = None,
        allow_cross_thread: bool = False
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Retrieve dataframe for a specific thread.
        
        Args:
            thread_id: Thread identifier
            file_id: Optional file identifier. If None, returns the most recently stored file.
            allow_cross_thread: If True, allows retrieving files from other threads (Requirement 13.4)
        
        Returns:
            Tuple of (DataFrame, metadata dict). Returns (None, {}) if not found.
        
        Validates: Requirements 13.1, 13.2, 13.3, 13.4
        """
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        
        with self._lock:
            # Check if thread exists
            if thread_id not in self._storage:
                logger.warning(f"Thread {thread_id} not found in cache")
                
                # If cross-thread access allowed, search all threads
                if allow_cross_thread and file_id:
                    return self._retrieve_cross_thread(file_id, thread_id)
                
                return None, {}
            
            thread_storage = self._storage[thread_id]
            
            # If no file_id specified, return most recent
            if file_id is None:
                if not thread_storage:
                    logger.warning(f"No files in thread {thread_id}")
                    return None, {}
                
                # Get most recent file by timestamp
                most_recent = max(
                    thread_storage.items(),
                    key=lambda x: x[1]['timestamp']
                )
                file_id = most_recent[0]
                logger.info(f"No file_id specified, using most recent: {file_id}")
            
            # Retrieve specific file
            if file_id not in thread_storage:
                logger.warning(
                    f"File {file_id} not found in thread {thread_id}. "
                    f"Available files: {list(thread_storage.keys())}"
                )
                
                # If cross-thread access allowed, search other threads
                if allow_cross_thread:
                    return self._retrieve_cross_thread(file_id, thread_id)
                
                return None, {}
            
            data = thread_storage[file_id]
            logger.info(
                f"Retrieved dataframe for thread={thread_id}, file={file_id}, "
                f"shape={data['df'].shape}"
            )
            
            return data['df'], data['metadata']
    
    def _retrieve_cross_thread(
        self,
        file_id: str,
        requesting_thread_id: str
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Retrieve file from any thread (cross-thread access).
        
        This is used when allow_cross_thread=True and the file is not
        found in the requesting thread.
        
        Args:
            file_id: File identifier to search for
            requesting_thread_id: Thread making the request
        
        Returns:
            Tuple of (DataFrame, metadata dict). Returns (None, {}) if not found.
        
        Validates: Requirement 13.4
        """
        # Search all threads for the file
        for thread_id, thread_storage in self._storage.items():
            if file_id in thread_storage:
                data = thread_storage[file_id]
                logger.info(
                    f"Cross-thread access: Retrieved file={file_id} from thread={thread_id} "
                    f"for requesting_thread={requesting_thread_id}, shape={data['df'].shape}"
                )
                return data['df'], data['metadata']
        
        logger.warning(
            f"File {file_id} not found in any thread (requested by thread={requesting_thread_id})"
        )
        return None, {}
    
    def clear_thread(self, thread_id: str) -> None:
        """
        Clear all data for a specific thread.
        
        Args:
            thread_id: Thread identifier
        
        Validates: Requirement 13.5
        """
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        
        with self._lock:
            if thread_id in self._storage:
                file_count = len(self._storage[thread_id])
                del self._storage[thread_id]
                logger.info(f"Cleared thread {thread_id} ({file_count} files removed)")
            else:
                logger.warning(f"Thread {thread_id} not found, nothing to clear")
    
    def list_files(self, thread_id: str) -> List[str]:
        """
        List all file IDs available in a thread.
        
        Args:
            thread_id: Thread identifier
        
        Returns:
            List of file IDs in the thread
        
        Validates: Requirements 13.1, 13.2
        """
        if not thread_id:
            raise ValueError("thread_id cannot be empty")
        
        with self._lock:
            if thread_id not in self._storage:
                return []
            
            files = list(self._storage[thread_id].keys())
            logger.debug(f"Thread {thread_id} has {len(files)} files: {files}")
            return files
    
    def has_file(self, thread_id: str, file_id: str) -> bool:
        """
        Check if a file exists in a thread.
        
        Args:
            thread_id: Thread identifier
            file_id: File identifier
        
        Returns:
            True if file exists in thread, False otherwise
        """
        if not thread_id or not file_id:
            return False
        
        with self._lock:
            return (
                thread_id in self._storage and 
                file_id in self._storage[thread_id]
            )
    
    def get_metadata(self, thread_id: str, file_id: str) -> Dict[str, Any]:
        """
        Get metadata for a specific file without retrieving the dataframe.
        
        Args:
            thread_id: Thread identifier
            file_id: File identifier
        
        Returns:
            Metadata dict, or empty dict if not found
        """
        if not thread_id or not file_id:
            return {}
        
        with self._lock:
            if thread_id in self._storage and file_id in self._storage[thread_id]:
                return self._storage[thread_id][file_id]['metadata'].copy()
            return {}
    
    def update_metadata(
        self, 
        thread_id: str, 
        file_id: str, 
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Update metadata for a specific file.
        
        Args:
            thread_id: Thread identifier
            file_id: File identifier
            metadata: New metadata dict (merged with existing)
        
        Returns:
            True if updated, False if file not found
        """
        if not thread_id or not file_id:
            return False
        
        with self._lock:
            if thread_id in self._storage and file_id in self._storage[thread_id]:
                existing = self._storage[thread_id][file_id]['metadata']
                existing.update(metadata)
                logger.debug(f"Updated metadata for thread={thread_id}, file={file_id}")
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache statistics
        """
        with self._lock:
            total_files = sum(len(files) for files in self._storage.values())
            thread_stats = {
                thread_id: {
                    'file_count': len(files),
                    'file_ids': list(files.keys())
                }
                for thread_id, files in self._storage.items()
            }
            
            return {
                'total_threads': len(self._storage),
                'total_files': total_files,
                'threads': thread_stats
            }
    
    def switch_thread_context(
        self,
        from_thread_id: str,
        to_thread_id: str,
        file_id: Optional[str] = None
    ) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
        """
        Switch from one thread context to another and load the correct dataframe.
        
        This method facilitates thread context switching by:
        1. Validating the target thread exists
        2. Loading the appropriate dataframe from the target thread
        3. Logging the context switch for debugging
        
        Args:
            from_thread_id: Current thread identifier
            to_thread_id: Target thread identifier to switch to
            file_id: Optional file identifier. If None, loads most recent file from target thread.
        
        Returns:
            Tuple of (DataFrame, metadata dict) from target thread. Returns (None, {}) if not found.
        
        Validates: Requirement 13.3
        """
        if not from_thread_id or not to_thread_id:
            raise ValueError("from_thread_id and to_thread_id cannot be empty")
        
        logger.info(
            f"Switching thread context: {from_thread_id} -> {to_thread_id}, "
            f"file_id={file_id or 'most_recent'}"
        )
        
        with self._lock:
            # Check if target thread exists
            if to_thread_id not in self._storage:
                logger.warning(
                    f"Target thread {to_thread_id} not found in cache. "
                    f"Available threads: {list(self._storage.keys())}"
                )
                return None, {}
            
            # Retrieve from target thread
            df, metadata = self.retrieve(to_thread_id, file_id)
            
            if df is not None:
                logger.info(
                    f"Successfully switched to thread {to_thread_id}, "
                    f"loaded file={file_id or 'most_recent'}, shape={df.shape}"
                )
            else:
                logger.warning(
                    f"Failed to load dataframe from thread {to_thread_id}, "
                    f"file_id={file_id}"
                )
            
            return df, metadata
    
    def get_thread_context(
        self,
        thread_id: str
    ) -> Dict[str, Any]:
        """
        Get complete context information for a thread.
        
        This provides a snapshot of all files and metadata in a thread,
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
            for file_id, data in thread_storage.items():
                files_info.append({
                    'file_id': file_id,
                    'shape': data['df'].shape,
                    'columns': list(data['df'].columns),
                    'timestamp': data['timestamp'],
                    'metadata': data['metadata']
                })
            
            return {
                'thread_id': thread_id,
                'exists': True,
                'file_count': len(thread_storage),
                'files': files_info
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
                f"Cleared ALL cache data: {thread_count} threads, {file_count} files"
            )


# Global cache instance
dataframe_cache = DataFrameCache()
