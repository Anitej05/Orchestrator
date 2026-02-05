
import asyncio
import sys
import os
# Add backend root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orchestrator.nodes.brain_nodes import execute_next_action
from orchestrator.state import State
from unittest.mock import MagicMock
import orchestrator.nodes.brain_nodes as brain_nodes

# Patch Tool Registry
async def mock_exec(name, args):
    return {"success": True, "result": f"MOCKED EXEC: {name} {args}"}

brain_nodes.tool_registry.execute_tool = mock_exec

async def test():
    state = {
        "current_task_id": "abc",
        "todo_list": [{
            "id": "abc",
            "description": "Test Tool",
            "status": "in_progress",
            "code_snippet": 'TOOL: news_search {"query": "foobar"}'
        }]
    }
    
    print("Running execute_next_action...")
    res = await execute_next_action(state)
    print("Result:", res)
    
    task = res['todo_list'][0]
    print(f"Task Result: {task.get('result')}")

if __name__ == "__main__":
    asyncio.run(test())
