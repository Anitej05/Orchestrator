# In Orbimesh Backend/orchestrator/state.py

from typing import Annotated, Any, List, Optional, Dict, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator

# Import the Pydantic models from the single source of truth: schemas.py
# This resolves the circular import error.
from backend.schemas import (
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

def append_reducer(a: List, b: List) -> List:
    """Reducer that appends new items to existing list."""
    if a is None:
        a = []
    if b is None:
        return a
    return a + b

class ActionHistoryEntry(TypedDict):
    """Records a single action execution for full context awareness."""
    iteration: int
    action_type: str  # agent, tool, python, terminal
    resource_id: Optional[str]
    instruction: str  # What was asked
    success: bool
    result_summary: str  # Compressed result for quick reference
    result_full: Any  # Full result (may be content_id reference)
    timestamp: float
    execution_time_ms: float

class CompletedTask(TypedDict):
    """A dictionary to hold the result of a completed task."""
    task_name: str
    result: Any
    raw_response: Any  # Store full agent response including extracted_data

class State(TypedDict):
    """
    The central state of the modular OMNI-DISPATCHER system.
    
    Full Context Awareness:
    - action_history: Complete log of ALL actions with full results
    - insights: Key learnings extracted from significant results (never compressed)
    - memory: Persistent facts and context
    """
    original_prompt: str
    messages: Annotated[List[BaseMessage], add_messages]
    
    # --- Dynamic Orchestrator State ---
    todo_list: Annotated[List[Dict], overwrite_reducer] # List of TaskItem dicts
    memory: Annotated[Dict[str, Any], overwrite_reducer] # Shared long-term memory/context
    
    # --- FULL CONTEXT AWARENESS ---
    # Complete action history - ALL actions with results (append-only)
    action_history: Annotated[List[Dict], append_reducer]
    
    # Key insights extracted from significant results (never compressed)
    # Format: {"step_1": "Q4 revenue: $2.3M", "phase_1_complete": "Analysis done"}
    insights: Annotated[Dict[str, str], overwrite_reducer]
    
    # --- ADAPTIVE PLANNING ---
    # Execution plan for complex tasks (list of PlanPhase dicts)
    execution_plan: Annotated[Optional[List[Dict]], overwrite_reducer]
    # Current phase being executed
    current_phase_id: Annotated[Optional[str], overwrite_reducer]
    
    # Execution Decision/Result
    decision: Annotated[Optional[Dict], overwrite_reducer] # BrainDecision dict
    execution_result: Annotated[Optional[Dict], overwrite_reducer] # ActionResult dict
    
    # Tracking
    current_task_id: Annotated[Optional[str], overwrite_reducer]
    iteration_count: Annotated[int, overwrite_reducer]
    max_iterations: Annotated[int, overwrite_reducer] # Safety limit
    
    # Final Response
    final_response: Annotated[Optional[str], overwrite_reducer]
    
    # --- HUMAN-IN-THE-LOOP ---
    # When Brain sets requires_approval=True, execution pauses here
    pending_approval: Annotated[bool, overwrite_reducer]  # True = waiting for user approval
    pending_decision: Annotated[Optional[Dict], overwrite_reducer]  # Snapshot of decision awaiting approval
    
    # Error tracking
    error: Annotated[Optional[str], overwrite_reducer]
    failure_count: Annotated[int, overwrite_reducer] # Count of consecutive failed tasks
    last_failure_id: Annotated[Optional[str], overwrite_reducer] # ID of the last failed task
    
    # Metadata
    thread_id: str
    user_id: str
    uploaded_files: Annotated[List[Dict], overwrite_reducer]
