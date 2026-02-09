"""
Unified Agent Protocol (UAP) Schemas

Standard request/response models for all agent communication.
All agents MUST implement endpoints that accept these formats.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# UAP REQUEST SCHEMAS
# =============================================================================

class UAPExecuteRequest(BaseModel):
    """
    Standard request format for /execute endpoint.
    
    All agents receive tasks through this unified format.
    The agent's internal LLM interprets the prompt and determines actions.
    """
    prompt: str = Field(
        ..., 
        description="Natural language instruction for the agent to execute"
    )
    payload: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional structured data: file paths, configurations, etc."
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Task ID for stateful/multi-turn operations"
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Thread ID for session persistence across requests"
    )


class UAPContinueRequest(BaseModel):
    """
    Standard request format for /continue endpoint.
    
    Used to resume paused tasks when agent needs user input.
    """
    task_id: str = Field(
        ..., 
        description="Task ID returned from /execute when status was 'needs_input'"
    )
    answer: str = Field(
        ..., 
        description="User's response to the agent's question"
    )
    thread_id: Optional[str] = Field(
        default=None,
        description="Thread ID for session context"
    )


# =============================================================================
# UAP RESPONSE SCHEMAS
# =============================================================================

class UAPResponse(BaseModel):
    """
    Standard response format from all UAP endpoints.
    
    All agents MUST return responses in this format.
    """
    success: bool = Field(
        ..., 
        description="Whether the operation completed successfully"
    )
    result: Any = Field(
        default=None,
        description="The result data from the operation"
    )
    status: Literal["completed", "needs_input", "in_progress", "error"] = Field(
        ...,
        description="Current status of the task"
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Task ID for multi-turn operations (present when status='needs_input')"
    )
    question: Optional[str] = Field(
        default=None,
        description="Clarification question for user (present when status='needs_input')"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if success=False"
    )
    
    # Optional metadata
    execution_time_ms: Optional[float] = Field(
        default=None,
        description="Execution time in milliseconds"
    )


class UAPHealthResponse(BaseModel):
    """
    Standard response format for /health endpoint.
    """
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Current health status of the agent"
    )
    agent_id: str = Field(
        ...,
        description="Unique identifier for this agent"
    )
    agent_name: str = Field(
        ...,
        description="Human-readable name of the agent"
    )
    version: str = Field(
        ...,
        description="Agent version string"
    )
    capabilities: Optional[List[str]] = Field(
        default=None,
        description="List of capability keywords"
    )
    uptime_seconds: Optional[float] = Field(
        default=None,
        description="Time since agent started"
    )


class UAPMetricsResponse(BaseModel):
    """
    Standard response format for /metrics endpoint (optional).
    """
    uptime_seconds: float
    total_calls: int
    successful_calls: int
    failed_calls: int
    latency_p50_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    active_tasks: int = 0


# =============================================================================
# SKILL CONFIGURATION (parsed from SKILL.md frontmatter)
# =============================================================================

class AgentSkillConfig(BaseModel):
    """
    Minimal agent configuration parsed from SKILL.md frontmatter.
    
    Replaces the verbose JSON configuration files.
    """
    id: str = Field(..., description="Unique agent identifier")
    name: str = Field(..., description="Human-readable agent name")
    port: int = Field(..., description="Port the agent runs on")
    version: str = Field(default="1.0.0", description="Agent version")
    host: str = Field(default="localhost", description="Host address")
    
    # Computed property
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


# =============================================================================
# CONSTANTS
# =============================================================================

# Standard UAP endpoints - orchestrator ONLY calls these
UAP_ENDPOINTS = {
    "execute": "/execute",
    "continue": "/continue", 
    "health": "/health",
    "metrics": "/metrics",  # Optional
}

# Standard status values
class TaskStatus:
    COMPLETED = "completed"
    NEEDS_INPUT = "needs_input"
    IN_PROGRESS = "in_progress"
    ERROR = "error"
