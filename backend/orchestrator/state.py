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
    The central state of the modular OMNI-DISPATCHER system.
    """
    original_prompt: str
    messages: Annotated[List[BaseMessage], add_messages]
    
    # --- Dynamic Orchestrator State ---
    todo_list: Annotated[List[Dict], overwrite_reducer] # List of TaskItem dicts
    memory: Annotated[Dict[str, Any], overwrite_reducer] # Shared long-term memory/context
    
    # Execution Decision/Result
    decision: Annotated[Optional[Dict], overwrite_reducer] # BrainDecision dict
    execution_result: Annotated[Optional[Dict], overwrite_reducer] # ActionResult dict
    
    # Tracking
    current_task_id: Annotated[Optional[str], overwrite_reducer]
    iteration_count: Annotated[int, overwrite_reducer]
    max_iterations: Annotated[int, overwrite_reducer] # Safety limit
    
    # Final Response
    final_response: Annotated[Optional[str], overwrite_reducer]
    
    # Error tracking
    error: Annotated[Optional[str], overwrite_reducer]
    failure_count: Annotated[int, overwrite_reducer] # Count of consecutive failed tasks
    last_failure_id: Annotated[Optional[str], overwrite_reducer] # ID of the last failed task
    
    # Metadata
    thread_id: str
    user_id: str
    uploaded_files: Annotated[List[Dict], overwrite_reducer]

