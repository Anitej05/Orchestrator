"""
Base Agent Types and Schemas
Core type definitions for the Base Agent architecture.
"""

from typing import Dict, Any, List, Optional, Callable, Literal
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class AgentStatus(str, Enum):
    """Agent lifecycle statuses."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    TERMINATED = "terminated"


class CapabilityType(str, Enum):
    """Types of capabilities."""
    SIMPLE = "simple"      # Single operation
    COMPOUND = "compound"  # Multi-step workflow


@dataclass
class ParameterSchema:
    """Schema for capability parameters."""
    name: str
    type: str  # string, number, boolean, array, object
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None  # For enum types


@dataclass
class ExecutionContext:
    """Context passed during capability execution."""
    thread_id: str
    user_id: str
    task_id: Optional[str] = None
    conversation_history: List[Dict] = field(default_factory=list)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default=None):
        """Get value from intermediate results."""
        return self.intermediate_results.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set value in intermediate results."""
        self.intermediate_results[key] = value


@dataclass
class CapabilityResult:
    """Result from capability execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def ok(cls, data: Any = None, metadata: Dict = None):
        """Create successful result."""
        return cls(success=True, data=data, metadata=metadata or {})
    
    @classmethod
    def fail(cls, error: str, metadata: Dict = None):
        """Create failed result."""
        return cls(success=False, error=error, metadata=metadata or {})


@dataclass
class ExecutionPlan:
    """Plan for executing a task."""
    reasoning: str
    steps: List['ExecutionStep']
    estimated_complexity: Literal["simple", "medium", "complex"]
    fallback_strategy: Optional[str] = None


@dataclass
class ExecutionStep:
    """Single step in execution plan."""
    step_number: int
    capability_name: str
    description: str
    parameters: Dict[str, Any]
    expected_outcome: str
    fallback_capability: Optional[str] = None


@dataclass
class RecoveryPlan:
    """Plan for recovering from errors."""
    strategy: Literal["retry", "alternate", "escalate", "fail"]
    reasoning: str
    confidence: float  # 0.0 to 1.0
    
    # For retry strategy
    adjusted_parameters: Optional[Dict[str, Any]] = None
    
    # For alternate strategy
    alternative_capability: Optional[str] = None
    
    # For escalate strategy
    user_question: Optional[str] = None
    
    @property
    def should_retry(self) -> bool:
        return self.strategy == "retry"
    
    @property
    def should_alternate(self) -> bool:
        return self.strategy == "alternate"
    
    @property
    def should_escalate(self) -> bool:
        return self.strategy == "escalate"


@dataclass
class AgentRequest:
    """Request to execute a task."""
    prompt: str
    action: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    task_id: Optional[str] = None
    thread_id: Optional[str] = None
    user_id: Optional[str] = None
    attachments: List[Dict] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Response from agent execution."""
    status: Literal["success", "error", "partial", "needs_input"]
    result: Any = None
    error_message: Optional[str] = None
    
    # For v2 standard response
    summary: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    canvas_display: Optional[Dict[str, Any]] = None
    
    # For needs_input status
    question: Optional[str] = None
    question_type: Optional[Literal["choice", "text", "confirmation"]] = None
    options: Optional[List[str]] = None
    
    # Metadata
    execution_time_ms: Optional[float] = None
    capabilities_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def success(cls, result: Any, summary: str = None, **kwargs):
        """Create success response."""
        return cls(
            status="success",
            result=result,
            summary=summary,
            **kwargs
        )
    
    @classmethod
    def error(cls, message: str, **kwargs):
        """Create error response."""
        return cls(
            status="error",
            error_message=message,
            **kwargs
        )
    
    @classmethod
    def needs_input(cls, question: str, question_type: str = "text", 
                    options: List[str] = None, **kwargs):
        """Create needs_input response."""
        return cls(
            status="needs_input",
            question=question,
            question_type=question_type,
            options=options,
            **kwargs
        )


# Type aliases
CapabilityHandler = Callable[[Any, Dict[str, Any], ExecutionContext], 
                              Any]
