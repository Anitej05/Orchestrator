"""
Browser Agent - Pydantic Schemas

Clean, simple schemas for LLM responses and action planning.
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal, Dict, Any, List


class AtomicAction(BaseModel):
    """Single executable browser action"""
    name: Literal[
        "navigate", "click", "type", "scroll", "extract", "done",
        "hover", "press", "wait", "go_back", "go_forward", "save_screenshot", 
        "save_info", "skip_subtask", "select", "upload_file", "download_file",
        "run_js", "press_keys",
        # Persistent memory actions
        "save_credential", "get_credential", "save_learning"
    ]
    params: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def restructure_params(cls, data: Any) -> Any:
        # DEBUG: Check if validator is running
        # print(f"DEBUG: restructure_params called with {data}")
        """Handle flat JSON where params are mixed with name"""
        if isinstance(data, dict) and 'params' not in data:
            # Move all non-name fields into params
            restructured = {
                "name": data.get("name"),
                "params": {k: v for k, v in data.items() if k != "name"}
            }
            # print(f"DEBUG: restructured to {restructured}")
            return restructured
        return data

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
    """
    Final result - Minimal & Flexible Schema
    
    REQUIRED (for orchestrator routing):
    - success: bool - determines success/failure routing
    - task_summary: str - human-readable result for LLM interpretation
    
    OPTIONAL (flexible content):
    - extracted_data: any dict - LLM interprets this, can have any structure
    - actions_taken: optional - for debugging/logging only
    - error: optional - only when errors occur
    - metrics: optional - performance data
    """
    # REQUIRED - Orchestrator needs these for routing decisions
    success: bool
    task_summary: str
    
    # FLEXIBLE - Content varies by task, LLM interprets
    extracted_data: Optional[Any] = None  # Can be any structure
    actions_taken: Optional[Any] = None   # Optional, any format
    error: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

