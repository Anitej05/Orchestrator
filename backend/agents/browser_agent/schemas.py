"""
Browser Agent - Pydantic Schemas

Clean, simple schemas for LLM responses and action planning.
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal, Dict, Any, List


class ActionParams(BaseModel):
    """Parameters for actions"""
    url: Optional[str] = None
    selector: Optional[str] = None
    text: Optional[str] = None
    x: Optional[int] = None
    y: Optional[int] = None


class AtomicAction(BaseModel):
    """Single executable browser action"""
    name: Literal[
        "navigate", "click", "type", "scroll", "extract", "done",
        "hover", "press", "wait", "go_back", "go_forward", "screenshot", "save_info", "skip_subtask", "select"
    ]
    params: Dict[str, Any] = Field(default_factory=dict)

class ActionPlan(BaseModel):
    """LLM response containing a sequence of actions"""
    reasoning: str = Field(description="Why this sequence of actions")
    actions: List[AtomicAction] = Field(description="Sequence of actions to execute in order")
    confidence: float = Field(default=0.8, ge=0, le=1)
    next_mode: Literal["text", "vision"] = Field(
        default="text", 
        description="Which model to use for the NEXT step. Use 'vision' if visual analysis is needed."
    )
    completed_subtasks: List[Any] = Field(
        default_factory=list,
        description="List of subtask IDs that will be completed by these actions"
    )
    updated_plan: Optional[List[str]] = Field(
        default=None,
        description="A NEW list of subtasks to replace the current remaining plan (Dynamic Replanning)"
    )


class ActionResult(BaseModel):
    """Result of action execution"""
    success: bool
    action: str
    message: str = ""
    data: Optional[Dict[str, Any]] = None
    screenshot_id: Optional[str] = None
    # Timeout handling fields
    timeout_occurred: bool = False
    timeout_context: Optional[Dict[str, Any]] = None  # Contains: action, params, elapsed_ms, url


class PageState(BaseModel):
    """Current page state"""
    url: str
    title: str
    body_text: str = ""
    elements: List[Dict[str, Any]] = Field(default_factory=list)
    element_count: int = 0


class BrowserTask(BaseModel):
    """Task request"""
    task: str
    headless: bool = False
    thread_id: Optional[str] = None


class BrowserResult(BaseModel):
    """Final result"""
    success: bool
    task_summary: str
    actions_taken: List[Dict[str, Any]] = Field(default_factory=list)
    extracted_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
