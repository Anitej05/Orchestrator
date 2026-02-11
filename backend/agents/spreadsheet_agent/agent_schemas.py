"""
Spreadsheet Agent v3.0 - Schemas

Request/Response models for the redesigned spreadsheet agent.
Follows the same patterns as mail_agent and browser_agent.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator
from enum import Enum


# ============================================================================
# ENUMS
# ============================================================================

class TaskStatus(str, Enum):
    """Status of a task execution."""
    COMPLETE = "complete"
    NEEDS_INPUT = "needs_input"
    IN_PROGRESS = "in_progress"
    ERROR = "error"


class FileFormat(str, Enum):
    """Supported file formats."""
    CSV = "csv"
    XLSX = "xlsx"
    XLS = "xls"
    JSON = "json"


# ============================================================================
# REQUEST MODELS
# ============================================================================

class ExecuteRequest(BaseModel):
    """
    Unified request model for /execute endpoint.
    
    Supports two modes:
    1. Complex prompt mode: Provide 'prompt' for LLM decomposition
    2. Direct action mode: Provide 'action' with 'params'
    """
    prompt: Optional[str] = Field(None, description="Natural language prompt for LLM decomposition")
    action: Optional[str] = Field(None, description="Direct action to execute")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parameters for action")
    thread_id: Optional[str] = Field(None, description="Session thread ID")
    task_id: Optional[str] = Field(None, description="Task ID for tracking")
    
    # File upload fields
    file_content: Optional[bytes] = Field(None, description="File content for upload")
    filename: Optional[str] = Field(None, description="Filename for upload")
    
    # File reference fields
    file_path: Optional[str] = Field(None, description="Local file path to load")
    file_id: Optional[str] = Field(None, description="Existing file ID to use")

    class Config:
        extra = "allow"  # Allow extra fields like 'column_name', 'instruction' sent by orchestrator

    @model_validator(mode='before')
    @classmethod
    def validate_intent(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce strict intent: If operating on an existing file, prompt or action is REQUIRED.
        Prevents 'silent summary' bugs where instructions are lost.
        """
        prompt = values.get('prompt') or values.get('instruction')
        action = values.get('action')
        file_content = values.get('file_content')
        file_to_load = values.get('file_id') or values.get('file_path') or values.get('filename')
        
        # If we are strictly just uploading a NEW file (file_content present), prompt is optional (defaults to summary).
        if file_content:
            return values
            
        # If we are operating on an EXISTING file (ref by ID/Path), we MUST have an instruction.
        if file_to_load and not prompt and not action:
            raise ValueError(
                "Missing Intent: When operating on an existing file, you MUST provide a 'prompt' (instruction) or 'action'. "
                "Defaulting to summary is disabled for existing files to prevent ambiguity."
            )
            
        return values


class ContinueRequest(BaseModel):
    """Request model for /continue endpoint."""
    task_id: str = Field(..., description="ID of paused task to resume")
    user_response: str = Field(..., description="User's response to the question")
    thread_id: Optional[str] = Field(None, description="Session thread ID")


# ============================================================================
# RESPONSE MODELS
# ============================================================================

class ExecuteResponse(BaseModel):
    """
    Unified response model for /execute endpoint.
    Compatible with orchestrator's AgentResponse.
    """
    status: TaskStatus = Field(..., description="Task status")
    success: bool = Field(..., description="Success status (Orchestrator compatibility)")
    result: Optional[Any] = Field(None, description="Result data")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    # For NEEDS_INPUT status
    question: Optional[str] = Field(None, description="Question for user")
    question_type: Optional[str] = Field(None, description="Type: choice, text, confirmation")
    options: Optional[List[str]] = Field(None, description="Options if question_type is choice")
    context: Optional[Dict[str, Any]] = Field(None, description="Context for resuming")
    
    # Canvas display for UI
    canvas_display: Optional[Dict[str, Any]] = Field(None, description="Canvas data for frontend")
    
    # Standard V2 Output
    standard_response: Optional[Dict[str, Any]] = Field(None, description="Standardized V2 response structure")
    
    # Metrics
    metrics: Optional[Dict[str, Any]] = Field(None, description="Execution metrics")
    
    class Config:
        use_enum_values = True


class FileInfo(BaseModel):
    """File metadata model."""
    file_id: str
    filename: str
    file_path: str
    rows: int
    columns: int
    column_names: List[str]
    dtypes: Dict[str, str]
    size_bytes: int
    created_at: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    agent: str = "spreadsheet_agent"
    version: str = "3.0"
    cache_stats: Optional[Dict[str, Any]] = None


# ============================================================================
# INTERNAL MODELS
# ============================================================================

class StepPlan(BaseModel):
    """A single step in an execution plan."""
    action: str
    params: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None


class ExecutionPlan(BaseModel):
    """Plan generated by LLM for complex requests."""
    steps: List[StepPlan]
    reasoning: Optional[str] = None
    needs_clarification: bool = False
    question: Optional[str] = None
    options: Optional[List[str]] = None


class StepResult(BaseModel):
    """Result of executing a single step."""
    action: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    df_modified: bool = False
