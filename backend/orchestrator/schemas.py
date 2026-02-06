from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Any, Dict, List
from datetime import datetime


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PhaseStatus(str, Enum):
    """Status of a phase in the execution plan."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"


class TaskItem(BaseModel):
    task_id: str = Field(..., description="Unique identifier for the task")
    description: str = Field(..., description="Human-readable description of the task")
    priority: int = Field(
        default=0, ge=0, description="Priority level (higher = more urgent)"
    )
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    payload: Dict[str, Any] = Field(
        default_factory=dict, description="Task-specific data"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class BrainAction(BaseModel):
    action_id: str = Field(..., description="Unique identifier for the action")
    action_type: str = Field(..., description="Type of action to perform")
    target_task: Optional[str] = Field(None, description="Associated task ID")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Action parameters"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ActionResult(BaseModel):
    action_id: str = Field(..., description="Associated action ID")
    success: bool = Field(..., description="Whether the action succeeded")
    output: Optional[Any] = Field(None, description="Action output data")
    error_message: Optional[str] = Field(None, description="Error details if failed")
    execution_time_ms: Optional[float] = Field(None, description="Execution duration")
    completed_at: datetime = Field(default_factory=datetime.utcnow)


# ============= ADAPTIVE PLANNING =============

class PlanPhase(BaseModel):
    """A phase in the execution plan for complex tasks."""
    phase_id: str = Field(..., description="Unique identifier for the phase")
    name: str = Field(..., description="Human-readable phase name, e.g. 'Data Collection'")
    goal: str = Field(..., description="What this phase should achieve")
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    depends_on: List[str] = Field(
        default_factory=list, 
        description="List of phase_ids this phase depends on"
    )
    result_summary: Optional[str] = Field(None, description="Summary of phase outcome")


# ============= PARALLEL EXECUTION =============

class ParallelAction(BaseModel):
    """An action to be executed in parallel with others."""
    action_type: str = Field(
        ..., 
        description="Type: 'agent' | 'tool' | 'python' | 'terminal'"
    )
    resource_id: str = Field(..., description="Agent ID or tool name")
    payload: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Parameters for the action"
    )
    description: Optional[str] = Field(None, description="What this action does")
