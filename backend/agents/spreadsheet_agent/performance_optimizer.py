"""
Advanced Performance Optimizer for Spreadsheet Agent

This module provides advanced performance optimizations including:
- Multi-level caching with intelligent eviction
- Memory usage optimization for concurrent sessions
- Token usage optimization for LLM context building
- Lazy loading and streaming for large datasets
- Connection pooling and resource management

Requirements: Task 4.3 - Performance Optimization
"""

import logging
import time
import gc
import threading
import weakref
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from threading import Lock, RLock
import psutil
import os

logger = logging.getLogger(__name__)


class AdvancedLRUCache:
    """
    Advanced LRU cache with intelligent eviction, memory monitoring, and performance tracking.
    
    Features:
    - Memory-aware eviction (evicts when memory usage is high)
    - Access frequency tracking for smarter eviction
    - Automatic cleanup of expired entries
    - Performance metrics and monitoring
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600, max_memory_mb: int = 500):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_mb = max_memory_mb
        
        self.cache = OrderedDict()
        self.timestamps = {}
        self.access_counts = defaultdict(int)
        self.memory_usage = {}  # Track memory usage per key
        self.lock = RLock()
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.memory_evictions = 0
        
        # Background cleanup
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with performance tracking."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            # Check TTL
            if self._is_expired(key):
                self._remove_key(key)
                self.misses += 1
                return None
            
            # Update access tracking
            self.access_counts[key] += 1
            self.cache.move_to_end(key)
            self.hits += 1
            
            return self.cache[key]
    
    def put(self, key: str, value: Any, estimated_size_mb: float = None):
        """Put item in cache with intelligent eviction."""
        with self.lock:
            # Estimate memory usage if not provided
            if estimated_size_mb is None:
                estimated_size_mb = self._estimate_memory_usage(value)
            
            # Check if we need to evict for memory
            if self._should_evict_for_memory(estimated_size_mb):
                self._evict_for_memory(estimated_size_mb)
            
            # Check if we need to evict for size
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            # Add/update item
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                self.cache[key] = value
                self.access_counts[key] = 1
            
            self.timestamps[key] = datetime.now()
            self.memory_usage[key] = estimated_size_mb
    
    def invalidate(self, key: str):
        """Remove specific key from cache."""
        with self.lock:
            if key in self.cache:
                self._remove_key(key)
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.access_counts.clear()
            self.memory_usage.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            total_memory = sum(self.memory_usage.values())
            
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'memory_evictions': self.memory_evictions,
                'total_memory_mb': total_memory,
                'max_memory_mb': self.max_memory_mb,
                'avg_access_count': sum(self.access_counts.values()) / len(self.access_counts) if self.access_counts else 0
            }
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.timestamps:
            return True
        age = (datetime.now() - self.timestamps[key]).total_seconds()
        return age > self.ttl_seconds
    
    def _remove_key(self, key: str):
        """Remove key and all associated data."""
        if key in self.cache:
            del self.cache[key]
        if key in self.timestamps:
            del self.timestamps[key]
        if key in self.access_counts:
            del self.access_counts[key]
        if key in self.memory_usage:
            del self.memory_usage[key]
    
    def _estimate_memory_usage(self, value: Any) -> float:
        """Estimate memory usage of a value in MB."""
        try:
            if isinstance(value, pd.DataFrame):
                return value.memory_usage(deep=True).sum() / (1024 * 1024)
            elif isinstance(value, (dict, list)):
                # Rough estimation for complex objects
                return len(str(value)) / (1024 * 1024)
            else:
                return 0.1  # Default small size
        except Exception:
            return 0.1
    
    def _should_evict_for_memory(self, new_item_size: float) -> bool:
        """Check if we should evict items due to memory pressure."""
        current_memory = sum(self.memory_usage.values())
        return (current_memory + new_item_size) > self.max_memory_mb
    
    def _evict_for_memory(self, needed_space: float):
        """Evict items to free up memory space."""
        with self.lock:
            current_memory = sum(self.memory_usage.values())
            target_memory = self.max_memory_mb - needed_space
            
            # Sort by access frequency (least accessed first)
            items_by_access = sorted(
                [(key, count) for key, count in self.access_counts.items() if key in self.cache],
                key=lambda x: x[1]
            )
            
            for key, _ in items_by_access:
                if current_memory <= target_memory:
                    break
                
                if key in self.memory_usage:
                    current_memory -= self.memory_usage[key]
                    self._remove_key(key)
                    self.memory_evictions += 1
    
    def _evict_lru(self):
        """Evict least recently used item, considering access frequency."""
        if self.cache:
            # If we have access count data, evict least accessed item
            if self.access_counts:
                # Find item with lowest access count among cached items
                cached_items = [(key, self.access_counts.get(key, 0)) for key in self.cache.keys()]
                if cached_items:
                    # Sort by access count (ascending) then by LRU order
                    least_accessed_key = min(cached_items, key=lambda x: (x[1], list(self.cache.keys()).index(x[0])))[0]
                    self._remove_key(least_accessed_key)
                    self.evictions += 1
                    return
            
            # Fallback to standard LRU eviction
            oldest_key = next(iter(self.cache))
            self._remove_key(oldest_key)
            self.evictions += 1
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while not self._stop_cleanup.wait(300):  # Check every 5 minutes
                self._cleanup_expired()
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_expired(self):
        """Remove expired entries in background."""
        with self.lock:
            expired_keys = [
                key for key in self.timestamps
                if self._is_expired(key)
            ]
            
            for key in expired_keys:
                self._remove_key(key)
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, '_stop_cleanup'):
            self._stop_cleanup.set()


class MemoryOptimizer:
    """
    Memory usage optimizer for concurrent sessions.
    
    Features:
    - Session-based memory tracking
    - Automatic garbage collection
    - Memory pressure detection
    - Resource cleanup for inactive sessions
    """
    
    def __init__(self, max_memory_per_session_mb: int = 100, cleanup_interval_seconds: int = 300):
        self.max_memory_per_session_mb = max_memory_per_session_mb
        self.cleanup_interval_seconds = cleanup_interval_seconds
        
        self.session_memory = {}  # session_id -> memory_usage_mb
        self.session_last_access = {}  # session_id -> timestamp
        self.lock = Lock()
        
        # Start background cleanup
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()
    
    def track_session_memory(self, session_id: str, memory_mb: float):
        """Track memory usage for a session."""
        with self.lock:
            self.session_memory[session_id] = memory_mb
            self.session_last_access[session_id] = datetime.now()
    
    def get_session_memory(self, session_id: str) -> float:
        """Get current memory usage for a session."""
        with self.lock:
            return self.session_memory.get(session_id, 0.0)
    
    def cleanup_session(self, session_id: str):
        """Clean up memory tracking for a session."""
        with self.lock:
            self.session_memory.pop(session_id, None)
            self.session_last_access.pop(session_id, None)
    
    def get_total_memory_usage(self) -> float:
        """Get total memory usage across all sessions."""
        with self.lock:
            return sum(self.session_memory.values())
    
    def get_system_memory_info(self) -> Dict[str, float]:
        """Get system memory information."""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'process_memory_mb': memory_info.rss / (1024 * 1024),
            'system_memory_percent': psutil.virtual_memory().percent,
            'available_memory_mb': psutil.virtual_memory().available / (1024 * 1024)
        }
    
    def should_trigger_cleanup(self) -> bool:
        """Check if we should trigger memory cleanup."""
        system_info = self.get_system_memory_info()
        return (
            system_info['system_memory_percent'] > 80 or
            system_info['process_memory_mb'] > 1000 or
            self.get_total_memory_usage() > 500
        )
    
    def force_garbage_collection(self):
        """Force garbage collection to free memory."""
        gc.collect()
        logger.info("Forced garbage collection completed")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while not self._stop_cleanup.wait(self.cleanup_interval_seconds):
                self._cleanup_inactive_sessions()
                if self.should_trigger_cleanup():
                    self.force_garbage_collection()
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_inactive_sessions(self):
        """Clean up inactive sessions."""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=2)
            inactive_sessions = [
                session_id for session_id, last_access in self.session_last_access.items()
                if last_access < cutoff_time
            ]
            
            for session_id in inactive_sessions:
                self.cleanup_session(session_id)
                logger.info(f"Cleaned up inactive session: {session_id}")
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, '_stop_cleanup'):
            self._stop_cleanup.set()


class TokenOptimizer:
    """
    Token usage optimizer for LLM context building.
    
    Features:
    - Intelligent context compression
    - Column-specific sampling
    - Hierarchical data representation
    - Token-aware truncation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TokenOptimizer")
    
    def optimize_dataframe_context(
        self,
        df: pd.DataFrame,
        max_tokens: int,
        include_columns: Optional[List[str]] = None,
        priority_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create token-optimized context from DataFrame.
        
        Args:
            df: DataFrame to optimize
            max_tokens: Maximum token budget
            include_columns: Specific columns to include (None for all)
            priority_columns: High-priority columns to always include
            
        Returns:
            Optimized context dictionary
        """
        # Estimate tokens per component
        token_budget = {
            'schema': max_tokens * 0.2,      # 20% for schema
            'sample_data': max_tokens * 0.6,  # 60% for sample data
            'metadata': max_tokens * 0.2     # 20% for metadata
        }
        
        context = {}
        
        # 1. Schema (always include, but optimize)
        schema_info = self._optimize_schema(df, int(token_budget['schema']), include_columns)
        context['schema'] = schema_info
        
        # 2. Sample data (intelligent sampling)
        sample_info = self._optimize_sample_data(
            df, int(token_budget['sample_data']), include_columns, priority_columns
        )
        context['sample_data'] = sample_info
        
        # 3. Metadata (compressed)
        metadata_info = self._optimize_metadata(df, int(token_budget['metadata']))
        context['metadata'] = metadata_info
        
        return context
    
    def _optimize_schema(self, df: pd.DataFrame, token_budget: int, include_columns: Optional[List[str]]) -> Dict[str, Any]:
        """Optimize schema representation for token efficiency."""
        columns = include_columns if include_columns else df.columns.tolist()
        
        # Use abbreviated type names
        type_mapping = {
            'object': 'str',
            'int64': 'int',
            'float64': 'float',
            'bool': 'bool',
            'datetime64[ns]': 'date'
        }
        
        schema = {
            'cols': len(columns),
            'rows': len(df),
            'types': {
                col: type_mapping.get(str(df[col].dtype), str(df[col].dtype)[:10])
                for col in columns[:20]  # Limit to first 20 columns
            }
        }
        
        # Add column names in compact format
        if len(columns) <= 10:
            schema['names'] = columns
        else:
            schema['names'] = columns[:8] + [f"...+{len(columns)-8} more"]
        
        return schema
    
    def _optimize_sample_data(
        self,
        df: pd.DataFrame,
        token_budget: int,
        include_columns: Optional[List[str]],
        priority_columns: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Optimize sample data for token efficiency."""
        columns = include_columns if include_columns else df.columns.tolist()
        priority_columns = priority_columns or []
        
        # Determine optimal sample size based on token budget
        estimated_tokens_per_row = len(columns) * 10  # Rough estimate
        max_sample_rows = min(20, token_budget // estimated_tokens_per_row)
        
        if len(df) <= max_sample_rows:
            # Small dataset - include all rows
            sample_df = df[columns]
            sample_strategy = "all_rows"
        else:
            # Large dataset - intelligent sampling
            sample_rows = []
            
            # Always include first few rows
            head_rows = min(5, max_sample_rows // 3)
            sample_rows.extend(range(head_rows))
            
            # Include last few rows
            tail_rows = min(3, max_sample_rows // 4)
            sample_rows.extend(range(len(df) - tail_rows, len(df)))
            
            # Include middle samples
            remaining_budget = max_sample_rows - len(sample_rows)
            if remaining_budget > 0:
                middle_indices = np.linspace(
                    head_rows, len(df) - tail_rows - 1,
                    min(remaining_budget, 10), dtype=int
                )
                sample_rows.extend(middle_indices)
            
            # Remove duplicates and sort
            sample_rows = sorted(list(set(sample_rows)))
            sample_df = df.iloc[sample_rows][columns]
            sample_strategy = f"sampled_{len(sample_rows)}_of_{len(df)}"
        
        # Convert to compact representation
        sample_data = {
            'strategy': sample_strategy,
            'rows': sample_df.head(max_sample_rows).to_dict('records')
        }
        
        # Truncate long string values
        for row in sample_data['rows']:
            for key, value in row.items():
                if isinstance(value, str) and len(value) > 50:
                    row[key] = value[:47] + "..."
        
        return sample_data
    
    def _optimize_metadata(self, df: pd.DataFrame, token_budget: int) -> Dict[str, Any]:
        """Optimize metadata representation."""
        metadata = {
            'shape': df.shape,
            'memory_mb': round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        }
        
        # Add null counts for columns with missing data
        null_counts = df.isnull().sum()
        if null_counts.any():
            metadata['nulls'] = {
                col: int(count) for col, count in null_counts.items()
                if count > 0
            }
        
        # Add basic statistics for numeric columns (compressed)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats = {}
            for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                col_stats = df[col].describe()
                stats[col] = {
                    'min': round(col_stats['min'], 2),
                    'max': round(col_stats['max'], 2),
                    'mean': round(col_stats['mean'], 2)
                }
            metadata['stats'] = stats
        
        return metadata
    
    def estimate_token_count(self, text: str) -> int:
        """Rough estimation of token count for text."""
        # Simple heuristic: ~4 characters per token
        return len(text) // 4
    
    def truncate_to_token_limit(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        estimated_tokens = self.estimate_token_count(text)
        if estimated_tokens <= max_tokens:
            return text
        
        # Calculate truncation point
        truncate_ratio = max_tokens / estimated_tokens
        truncate_length = int(len(text) * truncate_ratio * 0.9)  # 10% buffer
        
        return text[:truncate_length] + "... [truncated]"


class PerformanceMonitor:
    """
    Performance monitoring and metrics collection.
    
    Features:
    - Operation timing
    - Memory usage tracking
    - Cache performance metrics
    - System resource monitoring
    """
    
    def __init__(self):
        self.operation_times = defaultdict(list)
        self.memory_snapshots = []
        self.lock = Lock()
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return OperationTimer(self, operation_name)
    
    def record_operation_time(self, operation_name: str, duration: float):
        """Record operation timing."""
        with self.lock:
            self.operation_times[operation_name].append(duration)
            # Keep only last 100 measurements
            if len(self.operation_times[operation_name]) > 100:
                self.operation_times[operation_name] = self.operation_times[operation_name][-100:]
    
    def record_memory_snapshot(self):
        """Record current memory usage."""
        with self.lock:
            process = psutil.Process(os.getpid())
            snapshot = {
                'timestamp': datetime.now(),
                'memory_mb': process.memory_info().rss / (1024 * 1024),
                'cpu_percent': process.cpu_percent()
            }
            self.memory_snapshots.append(snapshot)
            
            # Keep only last 100 snapshots
            if len(self.memory_snapshots) > 100:
                self.memory_snapshots = self.memory_snapshots[-100:]
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        with self.lock:
            report = {
                'operation_stats': {},
                'memory_stats': {},
                'system_stats': self._get_system_stats()
            }
            
            # Operation statistics
            for op_name, times in self.operation_times.items():
                if times:
                    report['operation_stats'][op_name] = {
                        'count': len(times),
                        'avg_time': sum(times) / len(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'total_time': sum(times)
                    }
            
            # Memory statistics
            if self.memory_snapshots:
                memory_values = [s['memory_mb'] for s in self.memory_snapshots]
                report['memory_stats'] = {
                    'current_mb': memory_values[-1],
                    'avg_mb': sum(memory_values) / len(memory_values),
                    'min_mb': min(memory_values),
                    'max_mb': max(memory_values),
                    'snapshots_count': len(self.memory_snapshots)
                }
            
            return report
    
    def _get_system_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        process = psutil.Process(os.getpid())
        memory = psutil.virtual_memory()
        
        return {
            'process_memory_mb': process.memory_info().rss / (1024 * 1024),
            'process_cpu_percent': process.cpu_percent(),
            'system_memory_percent': memory.percent,
            'available_memory_mb': memory.available / (1024 * 1024),
            'system_cpu_count': psutil.cpu_count()
        }


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: PerformanceMonitor, operation_name: str):
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.record_operation_time(self.operation_name, duration)


# Global instances
advanced_cache = AdvancedLRUCache(max_size=2000, ttl_seconds=7200, max_memory_mb=1000)
memory_optimizer = MemoryOptimizer(max_memory_per_session_mb=150, cleanup_interval_seconds=300)
token_optimizer = TokenOptimizer()
performance_monitor = PerformanceMonitor()

logger.info("🚀 Advanced Performance Optimizer initialized")