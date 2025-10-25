# In Project_Agent_Directory/orchestrator/state.py

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
    FileObject
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
    final_response: Annotated[Optional[str], overwrite_reducer]
    pending_user_input: Annotated[bool, or_overwrite]
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
    canvas_type: Annotated[Optional[Literal["html", "markdown"]], overwrite_reducer]
    has_canvas: Annotated[bool, overwrite_reducer]
    
    # Orchestration pause/resume fields
    orchestration_paused: Annotated[bool, overwrite_reducer]
    waiting_for_continue: Annotated[bool, overwrite_reducer]
    pause_reason: Annotated[Optional[str], overwrite_reducer]
