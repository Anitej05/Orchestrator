import asyncio
import logging
import sys
import os

# Add backend root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- MOCKING ---
from unittest.mock import MagicMock
import orchestrator.nodes.brain_nodes as brain_nodes

# Mock LLM response
mock_llm = MagicMock()
# Mock httpx for Agent calls
import httpx
from unittest.mock import patch, Mock

# Mock LLM response side effect
def mock_invoke(*args, **kwargs):
    # First call: Add task to use Tool
    if mock_llm.call_count == 1:
        return MagicMock(content="""
        {
            "thought": "I will search for news.",
            "new_tasks": [
                {
                    "description": "Search for AI news",
                    "status": "pending",
                    "priority": "high",
                    "id": "task_tool"
                }
            ],
            "next_task_id": "task_tool",
            "suggested_command": "TOOL: news_search {\\"query\\": \\"AI funding\\"}" 
        }
        """)
    # Second call (after execution): Mark finished
    else:
        return MagicMock(content="""
        {
            "thought": "Tool finished. Workflow done.",
            "completed_task_id": "task_tool",
            "is_finished": true
        }
        """)

mock_llm.invoke.side_effect = mock_invoke

# Patch agent registry
brain_nodes.agent_registry.list_active_agents = MagicMock(return_value=[])

# Patch tool registry to intercept execution
async def mock_exec_tool(name, args):
    print(f"MOCK TOOL EXEC: {name} with {args}")
    return {"success": True, "result": "Found 5 news articles about AI."}

brain_nodes.tool_registry.execute_tool = mock_exec_tool

# ----------------

from orchestrator.graph import graph
from orchestrator.state import State


logging.basicConfig(level=logging.INFO)
logging.getLogger("TerminalService").setLevel(logging.INFO)

async def test_orchestrator():

    print("🚀 Starting Orchestrator Verification...")
    
    # Mock Initial State
    initial_state = {
        "original_prompt": "Calculate 25 * 4 using python code and tell me the result.",
        "todo_list": [],
        "memory": {"user_context": "testing"},
        "iteration_count": 0,
        "max_iterations": 10
    }
    
    config = {"configurable": {"thread_id": "test_thread_1"}}
    
    # Run the graph
    async for event in graph.astream(initial_state, config):
        print(f"DEBUG EVENT: {event}")
        for node_name, values in event.items():
            print(f"\n📍 Node: {node_name}")
            if values is None:
                print("❌ Values is None!")
                continue
                
            if "todo_list" in values:
                print("📋 Task List:")
                for task in values['todo_list']:
                    print(f"  - [{task['status']}] {task['description']} (ID: {task.get('id')})")
                    if task.get('result'):
                        print(f"    Result: {task['result']}")
            
            if "final_response" in values and values["final_response"]:
                print(f"\n✅ Final Response: {values['final_response']}")

if __name__ == "__main__":
    asyncio.run(test_orchestrator())
