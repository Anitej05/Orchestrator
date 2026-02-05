# In Orbimesh Backend/orchestrator/state.py

from typing import Annotated, Any, List, Optional, Dict, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator

# Import the Pydantic models from the single source of truth: schemas.py
# This resolves the circular import error.
from schemas import (
    Task,
    AgentCard, # This is a Pydantic model
    TaskAgentPair, # This is a Pydantic model
    PlannedTask, # This is a Pydantic model
    FileObject,
    DialogueTask # For multi-turn agent conversations
)

def or_overwrite(a: bool | None, b: bool | None) -> bool:
    """Reducer that returns true if any value is true, handling Nones."""
    if a is True or b is True:
        return True
    return False

def concat_reducer(a: str | None, b: str | None) -> str | None:
    """Reducer that concatenates two strings with a newline, handling Nones."""
    if a and b:
        return f"{a}\n\n---\n\n{b}"
    return a or b

def overwrite_reducer(a, b):
    """Reducer that always takes the second value, effectively overwriting."""
    return b

class CompletedTask(TypedDict):
    """A dictionary to hold the result of a completed task."""
    task_name: str
    result: Any
    raw_response: Any  # Store full agent response including extracted_data

class State(TypedDict):
    """
    The central state of the orchestration graph.
    Fields with Pydantic models will store dictionaries at runtime to ensure serialization.
    """
    original_prompt: str
    parsed_tasks: Annotated[List[Task], overwrite_reducer]
    user_expectations: Annotated[Optional[Dict[str, float]], overwrite_reducer]
    messages: Annotated[List[BaseMessage], add_messages]
    
    # These fields will store lists of DICTIONARIES, not Pydantic objects
    candidate_agents: Annotated[Dict[str, List[Dict]], overwrite_reducer]
    task_agent_pairs: Annotated[List[Dict], overwrite_reducer]
    task_plan: Annotated[List[List[Dict]], overwrite_reducer]
    
    completed_tasks: Annotated[List[CompletedTask], operator.add]
    latest_completed_tasks: Annotated[List[CompletedTask], overwrite_reducer]
    final_response: Annotated[Optional[str], overwrite_reducer]
    pending_user_input: Annotated[bool, overwrite_reducer]  # Changed from or_overwrite to overwrite_reducer
    question_for_user: Annotated[Optional[str], overwrite_reducer]
    user_response: Optional[str]
    parsing_error_feedback: Annotated[Optional[str], overwrite_reducer]
    parse_retry_count: Annotated[int, overwrite_reducer]
    
    # This field will also store a list of DICTIONARIES
    uploaded_files: Annotated[List[Dict], overwrite_reducer]

    needs_complex_processing: Annotated[bool, overwrite_reducer]
    analysis_reasoning: Annotated[Optional[str], overwrite_reducer]
    
    # Canvas feature fields
    needs_canvas: Annotated[bool, overwrite_reducer]
    canvas_content: Annotated[Optional[str], overwrite_reducer]
    canvas_data: Annotated[Optional[Dict], overwrite_reducer]  # Structured canvas data
    canvas_type: Annotated[Optional[str], overwrite_reducer]  # Changed from Literal to str to support more types
    canvas_title: Annotated[Optional[str], overwrite_reducer]  # Canvas title
    has_canvas: Annotated[bool, overwrite_reducer]
    
    # Planning mode flag
    planning_mode: Annotated[bool, overwrite_reducer]
    
    # Plan approval fields
    approval_required: Annotated[bool, overwrite_reducer]
    estimated_cost: Annotated[Optional[float], overwrite_reducer]
    task_count: Annotated[Optional[int], overwrite_reducer]
    plan_approved: Annotated[bool, overwrite_reducer]
    needs_approval: Annotated[bool, overwrite_reducer]
    
    # Canvas confirmation fields
    pending_confirmation: Annotated[bool, overwrite_reducer]
    pending_confirmation_task: Annotated[Optional[Dict], overwrite_reducer]
    canvas_displays: Annotated[List[Dict], overwrite_reducer]
    canvas_confirmation_action: Annotated[Optional[str], overwrite_reducer]
    canvas_confirmation_message: Annotated[Optional[str], overwrite_reducer]
    canvas_requires_confirmation: Annotated[bool, overwrite_reducer]
    skip_preview_on_next_execution: Annotated[bool, overwrite_reducer]
    
    # Evaluation fields
    eval_status: Annotated[Optional[str], overwrite_reducer]
    replan_reason: Annotated[Optional[str], overwrite_reducer]
    replan_count: Annotated[int, overwrite_reducer]
    
    # Tool routing fields (Phase 1 implementation)
    tool_routed_count: Annotated[int, overwrite_reducer]  # Number of tasks handled by direct tools
    task_events: Annotated[List[Dict], overwrite_reducer]
    
    # Multi-Turn Agent Dialogue fields
    dialogue_task: Annotated[Optional[Dict], overwrite_reducer]  # Current DialogueTask (serialized)
    dialogue_result: Annotated[Optional[Dict], overwrite_reducer]  # Result from dialogue loop
    needs_dialogue_mode: Annotated[bool, overwrite_reducer]  # Flag to trigger dialogue loop
    
    # Bidirectional Dialogue State
    dialogue_contexts: Annotated[Dict[str, Dict], overwrite_reducer]
    pending_agent_questions: Annotated[List[Dict], overwrite_reducer]

    # --- NEW: Dynamic Orchestrator State ---
    todo_list: Annotated[List[Dict], overwrite_reducer] # List of TaskItem dicts
    memory: Annotated[Dict[str, Any], overwrite_reducer] # Shared long-term memory/context
    code_sandbox_state: Annotated[Dict[str, Any], overwrite_reducer] # Persisted sandbox session
    
    # Tracking
    current_task_id: Annotated[Optional[str], overwrite_reducer]
    iteration_count: Annotated[int, overwrite_reducer]
    max_iterations: Annotated[int, overwrite_reducer] # Safety limit
    
    # Error tracking
    error: Annotated[Optional[str], overwrite_reducer]
    failure_count: Annotated[int, overwrite_reducer] # Count of consecutive failed tasks
    last_failure_id: Annotated[Optional[str], overwrite_reducer] # ID of the last failed task
