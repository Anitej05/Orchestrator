
# In Orbimesh Backend/orchestrator/graph.py

import logging
import os
import json
import asyncio
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import messages_to_dict, messages_from_dict

from orchestrator.state import State
from orchestrator.nodes.brain_nodes import manage_todo_list, execute_next_action
from orchestrator.nodes.utils import serialize_complex_object, CustomJSONEncoder
from services.telemetry_service import telemetry_service

# Configure logger
logger = logging.getLogger("AgentOrchestrator")

class ForceJsonSerializer:
    """Helper for LangGraph checkpointers to ensure JSON compatibility."""
    def dumps(self, obj: Any) -> str:
        return json.dumps(obj, cls=CustomJSONEncoder)
    def loads(self, data: str) -> Any:
        return json.loads(data)

def get_orchestrator_metrics() -> Dict[str, Any]:
    """Get comprehensive orchestrator metrics."""
    return telemetry_service.get_metrics()

def log_orchestrator_metrics(operation: str, success: bool):
    """Log orchestrator metrics with clean formatting."""
    telemetry_service.print_metrics_report(operation, success)

def route_brain_output(state: State):
    """
    Decides where to go after the Brain node.
    - If finished (final_response is set) -> END
    - If task selected (current_task_id is set) -> execute_next_action
    - If error/wait (neither) -> manage_todo_list (loop)
    """
    if state.get("final_response"):
        return END
        
    if state.get("current_task_id"):
        return "execute_next_action"
        
    return "manage_todo_list" # Loop back if no task selected (e.g., waiting for input)

def create_graph_with_checkpointer(checkpointer):
    """Factory function used by main.py to create the orchestrator graph."""
    workflow = StateGraph(State)
    
    # Define Nodes
    workflow.add_node("manage_todo_list", manage_todo_list)
    workflow.add_node("execute_next_action", execute_next_action)
    
    # Define Edges
    workflow.add_edge(START, "manage_todo_list")
    
    workflow.add_conditional_edges(
        "manage_todo_list",
        route_brain_output,
        {
            "execute_next_action": "execute_next_action",
            END: END,
            "manage_todo_list": "manage_todo_list"
        }
    )
    
    # Loop back from execution to brain
    workflow.add_edge("execute_next_action", "manage_todo_list")
    
    return workflow.compile(checkpointer=checkpointer)

def create_execution_subgraph(checkpointer):
    """Legacy factory function for execution subgraph. 
    In the new architecture, the main graph handles this dynamically."""
    return create_graph_with_checkpointer(checkpointer)


# ==================================================================================
# CONVERSATION HISTORY AND STATE SERIALIZATION
# ==================================================================================

def save_conversation_history(state: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Save conversation history to a JSON file for persistence.
    Called after orchestration completes to preserve conversation state.
    """
    try:
        thread_id = config.get("configurable", {}).get("thread_id", "unknown")
        history_dir = os.path.join(os.path.dirname(__file__), "..", "conversation_history")
        os.makedirs(history_dir, exist_ok=True)
        
        # Extract messages and serialize
        messages = state.get("messages", [])
        serialized_messages = messages_to_dict(messages) if messages else []
        
        history_data = {
            "thread_id": thread_id,
            "messages": serialized_messages,
            "todo_list": state.get("todo_list", []),
            "memory": state.get("memory", {}),
            "final_response": state.get("final_response"),
        }
        
        filepath = os.path.join(history_dir, f"{thread_id}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history_data, f, cls=CustomJSONEncoder, indent=2)
        
        logger.debug(f"Saved conversation history for thread {thread_id}")
    except Exception as e:
        logger.warning(f"Failed to save conversation history: {e}")


def save_plan_to_file(state: Dict[str, Any]) -> None:
    """
    Save the task plan/todo list to a file for debugging/persistence.
    """
    try:
        thread_id = state.get("thread_id", "unknown")
        plans_dir = os.path.join(os.path.dirname(__file__), "..", "plans")
        os.makedirs(plans_dir, exist_ok=True)
        
        plan_data = {
            "thread_id": thread_id,
            "todo_list": state.get("todo_list", []),
            "completed_tasks": state.get("completed_tasks", []),
            "memory": state.get("memory", {}),
        }
        
        filepath = os.path.join(plans_dir, f"{thread_id}_plan.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(plan_data, f, cls=CustomJSONEncoder, indent=2)
        
        logger.debug(f"Saved plan for thread {thread_id}")
    except Exception as e:
        logger.warning(f"Failed to save plan: {e}")


def get_serializable_state(state: Dict[str, Any], thread_id: str) -> Dict[str, Any]:
    """
    Convert the orchestrator state to a JSON-serializable dictionary.
    Handles special types like messages, timestamps, etc.
    """
    try:
        messages = state.get("messages", [])
        serialized_messages = messages_to_dict(messages) if messages else []
        
        return {
            "thread_id": thread_id,
            "messages": serialized_messages,
            "todo_list": state.get("todo_list", []),
            "memory": state.get("memory", {}),
            "final_response": state.get("final_response"),
            "completed_tasks": state.get("completed_tasks", []),
            "canvas_data": state.get("canvas_data"),
            "canvas_content": state.get("canvas_content"),
            "canvas_type": state.get("canvas_type"),
            "has_canvas": state.get("has_canvas", False),
        }
    except Exception as e:
        logger.warning(f"Failed to serialize state: {e}")
        return {"thread_id": thread_id, "error": str(e)}

# ==================================================================================
# DEFAULT INSTANCE (Backward compatibility)
# ==================================================================================

from langgraph.checkpoint.memory import MemorySaver
default_checkpointer = MemorySaver()

# Export a default graph and related functions
graph = create_graph_with_checkpointer(default_checkpointer)
# Re-exports for convenience
__all__ = [
    'graph', 
    'create_graph_with_checkpointer', 
    'create_execution_subgraph',
    'ForceJsonSerializer',
    'messages_to_dict',
    'messages_from_dict',
    'serialize_complex_object',
    'save_conversation_history',
    'save_plan_to_file',
    'get_serializable_state'
]
