
# In Orbimesh Backend/orchestrator/graph.py

import logging
import os
import json
import asyncio
from typing import Dict, Any, List, Optional

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import messages_to_dict, messages_from_dict

from .state import State
from .omni_dispatcher import brain, hands, omni_route_condition
from .nodes.utils import (
    serialize_complex_object, 
    CustomJSONEncoder,
    save_plan_to_file,
    save_conversation_history,
    get_serializable_state
)

# Configure logger
logger = logging.getLogger("AgentOrchestrator")

class ForceJsonSerializer:
    """Helper for LangGraph checkpointers to ensure JSON compatibility."""
    def dumps(self, obj: Any) -> str:
        return json.dumps(obj, cls=CustomJSONEncoder)
    def loads(self, data: str) -> Any:
        return json.loads(data)


def create_graph_with_checkpointer(checkpointer):
    """Factory function used by main.py to create the modular orchestrator graph."""
    workflow = StateGraph(State)
    
    # Define Nodes using the new modular OMNI-DISPATCHER logic
    workflow.add_node("brain", brain.think)
    workflow.add_node("hands", hands.execute)
    
    # Define Core Logic Cycle
    workflow.add_edge(START, "brain")
    
    workflow.add_conditional_edges(
        "brain",
        omni_route_condition,
        {
            "hands": "hands",
            "finish": END
        }
    )
    
    # Hands always loop back to Brain for the next decision
    workflow.add_edge("hands", "brain")
    
    return workflow.compile(checkpointer=checkpointer)


def create_execution_subgraph(checkpointer):
    """Execution subgraph - same as main graph in modular design."""
    return create_graph_with_checkpointer(checkpointer)

# Default Instance for backward compatibility
from langgraph.checkpoint.memory import MemorySaver
default_checkpointer = MemorySaver()
graph = create_graph_with_checkpointer(default_checkpointer)

# Re-exports for main.py
__all__ = [
    'graph', 
    'create_graph_with_checkpointer', 
    'create_execution_subgraph',
    'ForceJsonSerializer',
    'messages_to_dict',
    'messages_from_dict',
    'serialize_complex_object',
    'save_plan_to_file',
    'save_conversation_history',
    'get_serializable_state'
]
