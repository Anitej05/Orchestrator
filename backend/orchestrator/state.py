from typing import Annotated, Any, List, Optional, Dict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import operator

# Import the Pydantic models from the single source of truth: schemas.py
# This resolves the circular import error.
from schemas import (
    Task,
    AgentCard,
    TaskAgentPair,
    PlannedTask,
)

class CompletedTask(TypedDict):
    """A dictionary to hold the result of a completed task."""
    task_name: str
    result: Any

class State(TypedDict):
    """
    The central state of the orchestration graph. It's a dictionary that
    gets passed between nodes, each node updating its fields.
    
    Note: For memory persistence with LangGraph checkpointers, 
    complex Pydantic objects need to be serialized to avoid 
    "Type is not msgpack serializable" errors.
    """
    original_prompt: str
    parsed_tasks: List[Task]
    user_expectations: Optional[Dict[str, float]]
    messages: Annotated[List[BaseMessage], add_messages]
    candidate_agents: Dict[str, List[AgentCard]]
    task_agent_pairs: List[TaskAgentPair]
    task_plan: List[List[PlannedTask]]
    completed_tasks: Annotated[List[CompletedTask], operator.add]
    final_response: Optional[str]
    pending_user_input: bool
    question_for_user: Optional[str]
    user_response: Optional[str]
    parsing_error_feedback: Optional[str]
    parse_retry_count: int  # Track how many times we've tried to parse