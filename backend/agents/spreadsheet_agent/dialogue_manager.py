"""
Dialogue Manager for Spreadsheet Agent

Manages conversation state and orchestrator communication patterns.
Formats responses in orchestrator-compatible format with proper status codes,
execution metrics, and structured data.

Requirements: 9.1, 9.2, 9.3, 12.1, 12.2, 12.4, 12.5, 10.2
"""

import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# RESPONSE STATUS ENUM
# ============================================================================

class ResponseStatus(str, Enum):
    """Status codes for agent responses"""
    COMPLETE = "complete"
    ERROR = "error"
    NEEDS_INPUT = "needs_input"
    PARTIAL = "partial"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ExecutionMetrics:
    """Execution metrics for agent operations"""
    latency_ms: float = 0.0
    rows_processed: int = 0
    columns_affected: int = 0
    llm_calls: int = 0
    cache_hits: int = 0
    token_usage: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "latency_ms": round(self.latency_ms, 2),
            "rows_processed": self.rows_processed,
            "columns_affected": self.columns_affected,
            "llm_calls": self.llm_calls,
            "cache_hits": self.cache_hits
        }
        if self.token_usage:
            result["token_usage"] = self.token_usage
        return result


@dataclass
class AgentResponse:
    """
    Standardized response format for orchestrator communication.
    
    Supports:
    - Success/error status
    - Result data with explanations
    - NEEDS_INPUT for user clarification
    - Execution metrics
    - Partial results with progress
    """
    status: ResponseStatus
    result: Optional[Any] = None
    explanation: str = ""
    error: Optional[str] = None
    
    # For NEEDS_INPUT status
    question: Optional[str] = None
    question_type: Optional[str] = None  # 'choice', 'text', 'confirmation'
    choices: Optional[List[Dict[str, Any]]] = None
    context: Optional[Dict[str, Any]] = None
    
    # For PARTIAL status
    partial_result: Optional[Any] = None
    progress: Optional[float] = None  # 0.0 to 1.0
    
    # Execution metrics
    metrics: Optional[ExecutionMetrics] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        response = {
            "status": self.status.value,
            "result": self.result,
            "explanation": self.explanation
        }
        
        if self.error:
            response["error"] = self.error
        
        # NEEDS_INPUT fields
        if self.status == ResponseStatus.NEEDS_INPUT:
            response["question"] = self.question
            response["question_type"] = self.question_type
            if self.choices:
                response["choices"] = self.choices
            if self.context:
                response["context"] = self.context
        
        # PARTIAL fields
        if self.status == ResponseStatus.PARTIAL:
            response["partial_result"] = self.partial_result
            if self.progress is not None:
                response["progress"] = self.progress
        
        # Metrics
        if self.metrics:
            response["metrics"] = self.metrics.to_dict()
        
        # Metadata
        if self.metadata:
            response["metadata"] = self.metadata
        
        return response


@dataclass
class DialogueState:
    """State of an ongoing dialogue with the orchestrator"""
    thread_id: str
    file_id: Optional[str] = None
    operation: Optional[str] = None
    pending_question: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


# ============================================================================
# DIALOGUE MANAGER
# ============================================================================

class DialogueManager:
    """
    Manages conversation state and orchestrator communication.
    
    Responsibilities:
    - Format responses in orchestrator-compatible format
    - Track execution metrics
    - Manage dialogue state for multi-turn conversations
    - Create NEEDS_INPUT responses for anomalies
    """
    
    def __init__(self):
        """Initialize dialogue manager"""
        self.dialogue_states: Dict[str, DialogueState] = {}
        self.logger = logging.getLogger(f"{__name__}.DialogueManager")
    
    # ========================================================================
    # RESPONSE FORMATTING (Task 9.1)
    # ========================================================================
    
    def create_success_response(
        self,
        result: Any,
        explanation: str,
        metrics: Optional[ExecutionMetrics] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Create a successful completion response.
        
        Args:
            result: The result data (can be dict, list, DataFrame, etc.)
            explanation: Human-readable explanation of the result
            metrics: Optional execution metrics
            metadata: Optional additional metadata
            
        Returns:
            AgentResponse with COMPLETE status
            
        Requirements: 12.1, 12.2
        """
        return AgentResponse(
            status=ResponseStatus.COMPLETE,
            result=result,
            explanation=explanation,
            metrics=metrics,
            metadata=metadata or {}
        )
    
    def create_error_response(
        self,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        metrics: Optional[ExecutionMetrics] = None
    ) -> AgentResponse:
        """
        Create an error response.
        
        Args:
            error_message: Human-readable error message
            error_details: Optional detailed error information
            metrics: Optional execution metrics
            
        Returns:
            AgentResponse with ERROR status
            
        Requirements: 12.1
        """
        metadata = {}
        if error_details:
            metadata["error_details"] = error_details
        
        return AgentResponse(
            status=ResponseStatus.ERROR,
            error=error_message,
            explanation=f"Operation failed: {error_message}",
            metrics=metrics,
            metadata=metadata
        )
    
    def create_partial_response(
        self,
        partial_result: Any,
        explanation: str,
        progress: float,
        metrics: Optional[ExecutionMetrics] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Create a partial result response.
        
        Args:
            partial_result: The partial result data
            explanation: Human-readable explanation of progress
            progress: Progress indicator (0.0 to 1.0)
            metrics: Optional execution metrics
            metadata: Optional additional metadata
            
        Returns:
            AgentResponse with PARTIAL status
            
        Requirements: 12.5
        """
        return AgentResponse(
            status=ResponseStatus.PARTIAL,
            partial_result=partial_result,
            explanation=explanation,
            progress=progress,
            metrics=metrics,
            metadata=metadata or {}
        )
    
    # ========================================================================
    # NEEDS_INPUT RESPONSE FOR ANOMALIES (Task 9.2)
    # ========================================================================
    
    def create_needs_input_response(
        self,
        question: str,
        question_type: str = "choice",
        choices: Optional[List[Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
        metrics: Optional[ExecutionMetrics] = None
    ) -> AgentResponse:
        """
        Create a NEEDS_INPUT response for user clarification.
        
        Args:
            question: Clear question for the user
            question_type: Type of question ('choice', 'text', 'confirmation')
            choices: Available choices for 'choice' type questions
            context: Additional context about the situation
            metrics: Optional execution metrics
            
        Returns:
            AgentResponse with NEEDS_INPUT status
            
        Requirements: 9.1, 9.2
        """
        return AgentResponse(
            status=ResponseStatus.NEEDS_INPUT,
            question=question,
            question_type=question_type,
            choices=choices,
            context=context,
            explanation=f"User input required: {question}",
            metrics=metrics
        )
    
    def create_anomaly_response(
        self,
        anomaly: Any,  # Anomaly object from anomaly_detector
        metrics: Optional[ExecutionMetrics] = None
    ) -> AgentResponse:
        """
        Create a NEEDS_INPUT response for detected anomalies.
        
        This formats anomaly information into a user-friendly question
        with clear choices for resolution.
        
        Args:
            anomaly: Anomaly object with type, columns, sample_values, suggested_fixes
            metrics: Optional execution metrics
            
        Returns:
            AgentResponse with NEEDS_INPUT status and anomaly choices
            
        Requirements: 9.1, 9.2, 10.2
        """
        # Format the question based on anomaly type
        if anomaly.type == 'dtype_drift':
            question = (
                f"Column '{', '.join(anomaly.columns)}' contains mostly numeric values "
                f"but is stored as text. How would you like to proceed?"
            )
        elif anomaly.type == 'missing_values':
            question = (
                f"Column '{', '.join(anomaly.columns)}' has significant missing values. "
                f"How would you like to handle them?"
            )
        elif anomaly.type == 'outliers':
            question = (
                f"Column '{', '.join(anomaly.columns)}' contains outlier values. "
                f"How would you like to handle them?"
            )
        else:
            question = f"Data quality issue detected in column '{', '.join(anomaly.columns)}'. How would you like to proceed?"
        
        # Convert suggested fixes to choice format
        choices = []
        for fix in anomaly.suggested_fixes:
            choice = {
                "id": fix.action,
                "label": fix.action.replace('_', ' ').title(),
                "description": fix.description,
                "is_safe": fix.safe
            }
            choices.append(choice)
        
        # Build context with anomaly details
        context = {
            "anomaly_type": anomaly.type,
            "affected_columns": anomaly.columns,
            "severity": anomaly.severity,
            "message": anomaly.message,
            "sample_values": anomaly.sample_values,
            "metadata": anomaly.metadata
        }
        
        return self.create_needs_input_response(
            question=question,
            question_type="choice",
            choices=choices,
            context=context,
            metrics=metrics
        )
    
    # ========================================================================
    # DIALOGUE STATE MANAGEMENT
    # ========================================================================
    
    def save_state(self, thread_id: str, state: Dict[str, Any]) -> None:
        """
        Save conversation state for a thread.
        
        Args:
            thread_id: Thread identifier
            state: State dictionary to save
            
        Requirements: 9.3, 9.4
        """
        if thread_id not in self.dialogue_states:
            self.dialogue_states[thread_id] = DialogueState(thread_id=thread_id)
        
        dialogue_state = self.dialogue_states[thread_id]
        dialogue_state.context.update(state)
        dialogue_state.updated_at = time.time()
        
        self.logger.debug(f"Saved state for thread {thread_id}: {list(state.keys())}")
    
    def load_state(self, thread_id: str) -> Dict[str, Any]:
        """
        Load conversation state for a thread.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            State dictionary (empty if no state exists)
            
        Requirements: 9.3
        """
        if thread_id in self.dialogue_states:
            return self.dialogue_states[thread_id].context.copy()
        return {}
    
    def clear_state(self, thread_id: str) -> None:
        """
        Clear conversation state for a thread.
        
        Args:
            thread_id: Thread identifier
        """
        if thread_id in self.dialogue_states:
            del self.dialogue_states[thread_id]
            self.logger.debug(f"Cleared state for thread {thread_id}")
    
    def set_pending_question(self, thread_id: str, question: str, context: Dict[str, Any]) -> None:
        """
        Mark a question as pending for a thread.
        
        Args:
            thread_id: Thread identifier
            question: The pending question
            context: Context for resuming after user response
        """
        if thread_id not in self.dialogue_states:
            self.dialogue_states[thread_id] = DialogueState(thread_id=thread_id)
        
        dialogue_state = self.dialogue_states[thread_id]
        dialogue_state.pending_question = question
        dialogue_state.context.update(context)
        dialogue_state.updated_at = time.time()
        
        self.logger.info(f"Set pending question for thread {thread_id}: {question[:50]}...")
    
    def get_pending_question(self, thread_id: str) -> Optional[str]:
        """
        Get the pending question for a thread.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            Pending question or None
        """
        if thread_id in self.dialogue_states:
            return self.dialogue_states[thread_id].pending_question
        return None
    
    def clear_pending_question(self, thread_id: str) -> None:
        """
        Clear the pending question for a thread.
        
        Args:
            thread_id: Thread identifier
        """
        if thread_id in self.dialogue_states:
            self.dialogue_states[thread_id].pending_question = None
            self.logger.debug(f"Cleared pending question for thread {thread_id}")
    
    # ========================================================================
    # METRICS TRACKING
    # ========================================================================
    
    def create_metrics(
        self,
        start_time: float,
        rows_processed: int = 0,
        columns_affected: int = 0,
        llm_calls: int = 0,
        cache_hits: int = 0,
        token_usage: Optional[Dict[str, int]] = None
    ) -> ExecutionMetrics:
        """
        Create execution metrics from operation data.
        
        Args:
            start_time: Operation start time (from time.time())
            rows_processed: Number of rows processed
            columns_affected: Number of columns affected
            llm_calls: Number of LLM API calls made
            cache_hits: Number of cache hits
            token_usage: Optional token usage statistics
            
        Returns:
            ExecutionMetrics object
            
        Requirements: 12.4
        """
        latency_ms = (time.time() - start_time) * 1000
        
        return ExecutionMetrics(
            latency_ms=latency_ms,
            rows_processed=rows_processed,
            columns_affected=columns_affected,
            llm_calls=llm_calls,
            cache_hits=cache_hits,
            token_usage=token_usage
        )


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# Create a global dialogue manager instance for use across the agent
dialogue_manager = DialogueManager()
