"""
Session management wrapper for spreadsheet operations.

Provides thread-scoped dataframe storage and session tracking.
"""

import logging
from typing import Dict, Any, Optional
from threading import local
from pathlib import Path

import pandas as pd

from .config import STORAGE_DIR
from .memory import spreadsheet_memory

logger = logging.getLogger(__name__)


# Thread-local storage for dataframes (prevents cross-conversation contamination)
_thread_local = local()


def get_conversation_dataframes(thread_id: str) -> Dict[str, pd.DataFrame]:
    """Get thread-scoped dataframe storage, keyed by thread_id for isolation."""
    if not hasattr(_thread_local, 'dataframes_by_thread'):
        _thread_local.dataframes_by_thread = {}
    if thread_id not in _thread_local.dataframes_by_thread:
        _thread_local.dataframes_by_thread[thread_id] = {}
    return _thread_local.dataframes_by_thread[thread_id]


def get_conversation_file_paths(thread_id: str) -> Dict[str, str]:
    """Get thread-scoped file paths, keyed by thread_id for isolation."""
    if not hasattr(_thread_local, 'file_paths_by_thread'):
        _thread_local.file_paths_by_thread = {}
    if thread_id not in _thread_local.file_paths_by_thread:
        _thread_local.file_paths_by_thread[thread_id] = {}
    return _thread_local.file_paths_by_thread[thread_id]


def ensure_file_loaded(file_id: str, thread_id: str = "default", file_manager=None) -> bool:
    """
    Ensure a file is loaded into the dataframes dict (thread-scoped).
    If not, attempt to reload from file_manager or memory cache.
    
    Args:
        file_id: The file identifier
        thread_id: Conversation thread ID for isolation
        file_manager: AgentFileManager instance
    
    Returns:
        True if file is available, False otherwise
    """
    # Validate file_id
    if not file_id or file_id is None:
        logger.error(f"[ENSURE_LOADED] file_id is None or empty - cannot load file")
        return False
    
    logger.info(f"[ENSURE_LOADED] thread_id={thread_id}, file_id={file_id}")
    
    # Get thread-scoped storage
    dfs = get_conversation_dataframes(thread_id)
    file_mapping = get_conversation_file_paths(thread_id)
    
    logger.info(f"[ENSURE_LOADED] Current dataframes keys in thread: {list(dfs.keys())}")
    logger.info(f"[ENSURE_LOADED] Current file_paths keys in thread: {list(file_mapping.keys())}")
    
    # Check if already loaded
    if file_id in dfs:
        df = dfs[file_id]
        logger.info(f"[ENSURE_LOADED] ✅ file_id {file_id} FOUND in thread {thread_id}")
        logger.info(f"[ENSURE_LOADED] DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
        return True
    
    logger.info(f"[ENSURE_LOADED] ❌ file_id {file_id} NOT in thread {thread_id}, attempting reload...")
    
    # Check memory cache
    cached_metadata = spreadsheet_memory.get_df_metadata(file_id)
    if cached_metadata and 'file_path' in cached_metadata:
        file_path = cached_metadata['file_path']
        logger.info(f"[ENSURE_LOADED] Found in memory cache: {file_path}")
        try:
            file_ext = Path(file_path).suffix.lower()
            if file_ext == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            dfs[file_id] = df
            file_mapping[file_id] = file_path
            logger.info(f"[ENSURE_LOADED] ✅ Reloaded from cache. Shape: {df.shape}")
            return True
        except Exception as e:
            logger.error(f"[ENSURE_LOADED] Failed to reload from cache: {e}")
    
    # Try file_paths mapping
    if file_id in file_mapping:
        file_path = file_mapping[file_id]
        logger.info(f"[ENSURE_LOADED] Found in file_paths: {file_path}")
        try:
            file_ext = Path(file_path).suffix.lower()
            if file_ext == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            dfs[file_id] = df
            logger.info(f"[ENSURE_LOADED] ✅ Reloaded from file_paths. Shape: {df.shape}")
            return True
        except Exception as e:
            logger.error(f"[ENSURE_LOADED] Failed to reload from file_paths: {e}")
    
    # Fallback to file_manager
    if file_manager:
        metadata = file_manager.get_file(file_id)
        if metadata and metadata.storage_path:
            logger.info(f"[ENSURE_LOADED] Found in file_manager: {metadata.storage_path}")
            try:
                file_ext = Path(metadata.original_name).suffix.lower()
                if file_ext == ".csv":
                    df = pd.read_csv(metadata.storage_path)
                else:
                    df = pd.read_excel(metadata.storage_path)
                dfs[file_id] = df
                file_mapping[file_id] = metadata.storage_path
                
                # Cache for future use
                spreadsheet_memory.cache_df_metadata(file_id, {
                    'file_path': metadata.storage_path,
                    'shape': df.shape,
                    'columns': df.columns.tolist()
                })
                
                logger.info(f"[ENSURE_LOADED] ✅ Reloaded from file_manager. Shape: {df.shape}")
                return True
            except Exception as e:
                logger.error(f"[ENSURE_LOADED] Failed to reload from file_manager: {e}")
    
    logger.error(f"[ENSURE_LOADED] ❌ Could not find file_id {file_id} anywhere")
    return False


def get_dataframe_state(df: pd.DataFrame, filename: str = "unknown") -> Dict[str, Any]:
    """Get current state of dataframe for session tracking."""
    return {
        'filename': filename,
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()},
        'memory_usage': int(df.memory_usage(deep=True).sum()),
        'has_nulls': bool(df.isnull().any().any()),
        'null_counts': {k: int(v) for k, v in df.isnull().sum().to_dict().items()}
    }


def store_dataframe(file_id: str, df: pd.DataFrame, file_path: str, thread_id: str = "default"):
    """Store dataframe in thread-scoped storage and cache."""
    dfs = get_conversation_dataframes(thread_id)
    paths = get_conversation_file_paths(thread_id)
    
    dfs[file_id] = df
    paths[file_id] = file_path
    
    # Cache metadata
    spreadsheet_memory.cache_df_metadata(file_id, {
        'file_path': file_path,
        'shape': df.shape,
        'columns': df.columns.tolist()
    })
    
    logger.info(f"Stored dataframe {file_id} in thread {thread_id}")


def get_dataframe(file_id: str, thread_id: str = "default") -> Optional[pd.DataFrame]:
    """Get dataframe from thread-scoped storage."""
    dfs = get_conversation_dataframes(thread_id)
    return dfs.get(file_id)


def clear_thread_data(thread_id: str):
    """Clear all data for a specific thread."""
    if hasattr(_thread_local, 'dataframes_by_thread'):
        _thread_local.dataframes_by_thread.pop(thread_id, None)
    if hasattr(_thread_local, 'file_paths_by_thread'):
        _thread_local.file_paths_by_thread.pop(thread_id, None)
    logger.info(f"Cleared thread data for {thread_id}")
