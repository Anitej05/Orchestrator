"""
Session management wrapper for spreadsheet operations.

Provides thread-scoped dataframe storage and session tracking.
"""

import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from threading import Lock

import pandas as pd

from .config import STORAGE_DIR
from .memory import spreadsheet_memory

logger = logging.getLogger(__name__)


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean up common spreadsheet issues so the LLM sees a sane schema.

    - Trim column headers
    - Drop empty/unnamed columns
    - If a single column header contains commas (e.g., "YearsExperience,Salary") and the
      remaining columns are empty, split that column into multiple columns.
    - Best-effort type coercion for newly split columns.
    """
    cleaned = df.copy()

    # Trim headers
    cleaned.columns = [str(c).strip() for c in cleaned.columns]

    # Drop completely empty columns (all NaN or all blank strings)
    def _is_empty_series(series: pd.Series) -> bool:
        if series.isna().all():
            return True
        try:
            return series.astype(str).str.strip().eq("").all()
        except Exception:
            return False

    empty_cols = [c for c in cleaned.columns if _is_empty_series(cleaned[c]) or str(c).lower().startswith("unnamed")]
    if empty_cols:
        cleaned = cleaned.drop(columns=empty_cols)

    if not cleaned.columns.any():
        return cleaned

    # Detect "CSV-in-a-cell" style: first column header has commas and others were empty/unnamed
    # BUT: only apply this if we truly have a single-column file with comma-separated values
    first_col = cleaned.columns[0]
    other_cols_empty = len(cleaned.columns) == 1
    if not other_cols_empty:
        other_cols_empty = all(_is_empty_series(cleaned[c]) for c in cleaned.columns[1:])

    # Check if first row values also contain commas (indicates CSV-in-cell pattern)
    first_row_has_commas = False
    if len(cleaned) > 0:
        first_val = str(cleaned[first_col].iloc[0])
        first_row_has_commas = "," in first_val and len(first_val.split(",")) >= 2

    # Only split if: header has commas, other columns empty, AND first row also has commas
    if "," in str(first_col) and other_cols_empty and first_row_has_commas:
        target_cols = [part.strip() for part in str(first_col).split(",") if part.strip()]
        if len(target_cols) >= 2:
            try:
                # Split the combined column into separate columns
                split_df = cleaned[first_col].astype(str).str.split(",", expand=True)
                split_df.columns = target_cols[: split_df.shape[1]]
                # Coerce numerics where possible (use convert_dtypes instead of deprecated errors='ignore')
                for col in split_df.columns:
                    try:
                        split_df[col] = pd.to_numeric(split_df[col].str.strip())
                    except (ValueError, TypeError):
                        pass  # Keep as string if not numeric

                cleaned = split_df
                logger.info(f"[NORMALIZE] Split CSV-in-cell format into {len(target_cols)} columns")
            except Exception as e:
                logger.warning(f"[NORMALIZE] Failed to split combined column '{first_col}': {e}")
    elif len(cleaned.columns) > 1:
        # Multi-column file detected - this is normal, don't attempt splitting
        logger.info(f"[NORMALIZE] Multi-column file detected ({len(cleaned.columns)} columns): {list(cleaned.columns)[:5]}")

    return cleaned


# Thread-scoped storage backed by process-level dicts to avoid cross-thread reuse.
_store_lock = Lock()
_dataframes_by_thread: Dict[str, Dict[str, pd.DataFrame]] = defaultdict(dict)
_file_paths_by_thread: Dict[str, Dict[str, str]] = defaultdict(dict)
_versions_by_thread: Dict[str, Dict[str, int]] = defaultdict(dict)

# Compatibility: keep an object with the same attributes some callers inspect directly.
class _ThreadView:
    pass


_thread_local = _ThreadView()
_thread_local.dataframes_by_thread = _dataframes_by_thread
_thread_local.file_paths_by_thread = _file_paths_by_thread


def get_conversation_dataframes(thread_id: str) -> Dict[str, pd.DataFrame]:
    """Get thread-scoped dataframe storage, keyed by thread_id for isolation."""
    with _store_lock:
        return _dataframes_by_thread[thread_id]


def get_conversation_file_paths(thread_id: str) -> Dict[str, str]:
    """Get thread-scoped file paths, keyed by thread_id for isolation."""
    with _store_lock:
        return _file_paths_by_thread[thread_id]


def _get_conversation_versions(thread_id: str) -> Dict[str, int]:
    """Track monotonically increasing versions per dataframe for cache safety."""
    with _store_lock:
        return _versions_by_thread[thread_id]


def _log_dtype_drift(file_id: str, previous: Optional[Dict[str, Any]], current_dtypes: Dict[str, str]):
    """Warn when dtypes drift between versions to surface silent coercions."""
    if not previous or not previous.get('dtypes'):
        return

    old_dtypes = previous.get('dtypes', {})
    changed = {
        col: (old_dtypes.get(col), current_dtypes.get(col))
        for col in set(old_dtypes) | set(current_dtypes)
        if old_dtypes.get(col) != current_dtypes.get(col)
    }

    if changed:
        logger.warning(f"[DTYPE-DRIFT] file_id={file_id}: {changed}")


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
    versions = _get_conversation_versions(thread_id)
    
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
            df = _normalize_dataframe(df)
            dfs[file_id] = df
            file_mapping[file_id] = file_path
            versions[file_id] = cached_metadata.get('version', versions.get(file_id, 0))
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
            df = _normalize_dataframe(df)
            dfs[file_id] = df
            versions[file_id] = versions.get(file_id, 0)
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
                df = _normalize_dataframe(df)
                dfs[file_id] = df
                file_mapping[file_id] = metadata.storage_path
                versions[file_id] = versions.get(file_id, 0) + 1
                
                # Cache for future use
                spreadsheet_memory.cache_df_metadata(file_id, {
                    'file_path': metadata.storage_path,
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'dtypes': {k: str(v) for k, v in df.dtypes.to_dict().items()},
                    'version': versions[file_id]
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
    versions = _get_conversation_versions(thread_id)

    prev_meta = spreadsheet_memory.get_df_metadata(file_id)

    normalized = _normalize_dataframe(df)
    current_dtypes = {k: str(v) for k, v in normalized.dtypes.to_dict().items()}

    dfs[file_id] = normalized
    paths[file_id] = file_path
    versions[file_id] = versions.get(file_id, 0) + 1

    _log_dtype_drift(file_id, prev_meta, current_dtypes)

    # Cache metadata with version and dtypes for downstream validation
    spreadsheet_memory.cache_df_metadata(file_id, {
        'file_path': file_path,
        'shape': normalized.shape,
        'columns': normalized.columns.tolist(),
        'dtypes': current_dtypes,
        'version': versions[file_id]
    })

    logger.info(f"Stored dataframe {file_id} in thread {thread_id} (v{versions[file_id]})")


def get_dataframe(file_id: str, thread_id: str = "default") -> Optional[pd.DataFrame]:
    """Get dataframe from thread-scoped storage."""
    dfs = get_conversation_dataframes(thread_id)
    return dfs.get(file_id)


def clear_thread_data(thread_id: str):
    """Clear all data for a specific thread."""
    _dataframes_by_thread.pop(thread_id, None)
    _file_paths_by_thread.pop(thread_id, None)
    _versions_by_thread.pop(thread_id, None)
    logger.info(f"Cleared thread data for {thread_id}")
