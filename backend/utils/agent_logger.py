"""
Agent logging utilities for structured, debug-friendly logging across all agents.
Provides context-aware logging with consistent formatting for easier error diagnosis.
"""

import logging
import time
from typing import Any, Dict, Optional
from datetime import datetime


class AgentLogger:
    """
    Structured logger for agents with context tracking and emoji-based visual indicators.
    Helps create an audit trail for debugging when errors occur.
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logging.getLogger(agent_name)
        self.context_stack = []
        self.operation_start_times = {}
    
    def start_operation(self, operation_name: str, **params) -> str:
        """
        Log operation start. Returns operation_id for tracking completion.
        
        Args:
            operation_name: Name of the operation (e.g., "nl_query", "document_analysis")
            **params: Key parameters to log with the operation
        """
        op_id = f"{operation_name}_{int(time.time()*1000)}"
        self.operation_start_times[op_id] = time.time()
        
        params_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        self.logger.info(f"🚀 [{operation_name}] Starting with params: {params_str}")
        
        return op_id
    
    def end_operation(self, op_id: str, success: bool = True, **result_data) -> float:
        """
        Log operation completion. Returns elapsed time in milliseconds.
        
        Args:
            op_id: Operation ID from start_operation()
            success: Whether operation succeeded
            **result_data: Key result data to log (e.g., rows_processed=100)
        """
        if op_id not in self.operation_start_times:
            self.logger.warning(f"⚠️  Operation {op_id} not found in start times")
            return 0.0
        
        elapsed_ms = (time.time() - self.operation_start_times[op_id]) * 1000
        del self.operation_start_times[op_id]
        
        status_emoji = "✅" if success else "❌"
        operation_name = op_id.rsplit("_", 1)[0]
        
        result_str = ", ".join([f"{k}={v}" for k, v in result_data.items()])
        result_suffix = f" | {result_str}" if result_str else ""
        
        self.logger.info(f"{status_emoji} [{operation_name}] Completed in {elapsed_ms:.1f}ms{result_suffix}")
        
        return elapsed_ms
    
    def log_llm_call(self, provider: str, model: str, input_tokens: int, 
                     temperature: float = 0.7, max_tokens: Optional[int] = None):
        """Log before making LLM API call."""
        max_tokens_str = f", max_tokens={max_tokens}" if max_tokens else ""
        self.logger.info(
            f"🤖 [LLM_CALL] Provider={provider}, Model={model}, "
            f"Input_tokens={input_tokens}, Temp={temperature}{max_tokens_str}"
        )
    
    def log_llm_response(self, output_tokens: int, latency_ms: float, 
                        success: bool = True, error_msg: Optional[str] = None):
        """Log after receiving LLM API response."""
        status_emoji = "✅" if success else "❌"
        error_suffix = f" | Error: {error_msg}" if error_msg else ""
        self.logger.info(
            f"{status_emoji} [LLM_RESPONSE] Output_tokens={output_tokens}, "
            f"Latency={latency_ms:.1f}ms{error_suffix}"
        )
    
    def log_cache_operation(self, operation: str, key: str, hit: Optional[bool] = None, 
                           size_mb: Optional[float] = None):
        """
        Log cache hit/miss/clear operations.
        
        Args:
            operation: "hit", "miss", "clear", "update"
            key: Cache key being accessed
            hit: True for hit, False for miss (optional)
            size_mb: Current cache size in MB (optional)
        """
        if operation == "hit":
            self.logger.info(f"💾 [CACHE] HIT for key: {key[:50]}")
        elif operation == "miss":
            self.logger.info(f"💾 [CACHE] MISS for key: {key[:50]}")
        elif operation == "clear":
            self.logger.info(f"💾 [CACHE] CLEARED, freed ~{size_mb:.2f}MB")
        elif operation == "update":
            self.logger.info(f"💾 [CACHE] UPDATED key: {key[:50]}")
    
    def log_rag_operation(self, operation: str, chunks_retrieved: Optional[int] = None,
                         retrieval_time_ms: Optional[float] = None, 
                         top_k: Optional[int] = None, documents: Optional[int] = None):
        """
        Log RAG (Retrieval Augmented Generation) operations.
        
        Args:
            operation: "retrieve", "loaded_vectors", "embedding"
            chunks_retrieved: Number of chunks retrieved
            retrieval_time_ms: Time taken for retrieval
            top_k: Number of chunks requested
            documents: Number of documents loaded
        """
        if operation == "retrieve":
            self.logger.info(
                f"📚 [RAG] Retrieved {chunks_retrieved} chunks (top_k={top_k}) "
                f"in {retrieval_time_ms:.1f}ms"
            )
        elif operation == "loaded_vectors":
            self.logger.info(f"📚 [RAG] Loaded vector store with {documents} documents")
        elif operation == "embedding":
            self.logger.info(f"📚 [RAG] Generated query embedding in {retrieval_time_ms:.1f}ms")
    
    def log_decision(self, decision_type: str, choice: str, reason: Optional[str] = None, 
                    alternatives: Optional[list] = None):
        """
        Log important decision points for debugging flow.
        
        Args:
            decision_type: "provider_selected", "fallback_triggered", "cache_strategy", etc.
            choice: What was chosen
            reason: Why it was chosen
            alternatives: Other options that were considered
        """
        reason_suffix = f" (reason: {reason})" if reason else ""
        alt_suffix = f" | Alternatives: {alternatives}" if alternatives else ""
        self.logger.info(f"🔀 [DECISION] {decision_type}={choice}{reason_suffix}{alt_suffix}")
    
    def log_data_transformation(self, transformation_type: str, input_size: Any, 
                               output_size: Any, latency_ms: float):
        """Log data transformation operations."""
        self.logger.info(
            f"⚙️  [TRANSFORM] {transformation_type}: "
            f"{input_size} → {output_size} in {latency_ms:.1f}ms"
        )
    
    def log_retry(self, attempt: int, max_attempts: int, reason: str, 
                 backoff_ms: Optional[int] = None):
        """Log retry attempts."""
        backoff_suffix = f", backoff={backoff_ms}ms" if backoff_ms else ""
        self.logger.warning(
            f"🔄 [RETRY] Attempt {attempt}/{max_attempts}: {reason}{backoff_suffix}"
        )
    
    def log_error(self, operation: str, error_type: str, error_message: str, 
                 context: Optional[Dict[str, Any]] = None, recoverable: bool = False):
        """
        Log error with full context for debugging.
        
        Args:
            operation: What operation failed
            error_type: Type of error (e.g., "LLMError", "RAGError", "ValidationError")
            error_message: Error message
            context: Additional context data (inputs, state, etc.)
            recoverable: Whether the system can recover from this error
        """
        recovery_indicator = "⚠️ " if recoverable else "❌"
        context_str = ""
        if context:
            context_str = " | Context: " + ", ".join([f"{k}={v}" for k, v in context.items()])
        
        self.logger.error(
            f"{recovery_indicator} [{operation}] {error_type}: {error_message}{context_str}"
        )
    
    def log_metric(self, metric_name: str, value: float, unit: str = "", 
                  threshold: Optional[float] = None):
        """
        Log individual metrics with optional threshold comparison.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement (ms, MB, tokens, %)
            threshold: Alert if value exceeds this (for warnings)
        """
        warning_indicator = ""
        if threshold and value > threshold:
            warning_indicator = " ⚠️ EXCEEDS THRESHOLD"
        
        self.logger.info(f"📊 [{metric_name}] {value:.1f}{unit}{warning_indicator}")
    
    def log_debug(self, message: str, data: Optional[Dict[str, Any]] = None):
        """
        Log debug information (trace execution path).
        
        Args:
            message: Debug message
            data: Additional debug data
        """
        data_str = ""
        if data:
            data_str = " | Data: " + str(data)
        self.logger.debug(f"🔍 {message}{data_str}")
