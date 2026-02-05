
import asyncio
import sys
import io
import os
import json
import logging
from typing import Dict, Any

# Force UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging to show Orchestrator logs clearly
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
# Mute overly verbose libs
logging.getLogger("httpx").setLevel(logging.WARNING)

from backend.orchestrator.graph import create_graph_with_checkpointer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async def run_scenario(graph, name: str, prompt: str):
    with open("orchestrator_trace.log", "w", encoding="utf-8") as log_file:
        def log(msg):
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()

        log(f"\n{'='*60}")
        log(f"🚀 SCENARIO: {name}")
        log(f"📄 Prompt: {prompt}")
        log(f"{'='*60}\n")
        
        initial_state = {
            "todo_list": [],
            "memory": {},
            "original_prompt": prompt,
            "iteration_count": 0,
            "final_response": None,
            "current_task_id": None
        }
        
        iteration = 0
        max_steps = 50
        config = {"configurable": {"thread_id": "test_1"}, "recursion_limit": max_steps}
        
        try:
            # Using astream to watch transitions
            async for output in graph.astream(initial_state, config):
                iteration += 1
                for node, state in output.items():
                    log(f"DEBUG: Node={node}, Type={type(state)}")
                    log(f"\n--- Step {iteration}: Node [{node}] ---")
                    
                    if state.get('error'):
                        log(f"❌ STATE ERROR: {state.get('error')}")

                    # Print Task Updates
                    if node == "manage_todo_list":
                        tasks = state.get("todo_list", [])
                        pending = [t for t in tasks if t['status'] == "pending"]
                        in_progress = [t for t in tasks if t['status'] == "in_progress"]
                        completed = [t for t in tasks if t['status'] == "completed"]
                        
                        log(f"📊 Tasks: {len(completed)} Done, {len(in_progress)} Active, {len(pending)} Pending")
                        for t in in_progress:
                            log(f"👉 Current Task: {t['description']}")
                            if t.get('code_snippet'):
                                log(f"   💻 Snippet: {t['code_snippet']}")
                        for t in completed:
                             log(f"✅ Completed: {t['description']} (Result: {t.get('result')})")
                                
                    elif node == "execute_next_action":
                        mem = state.get("memory", {})
                        last_res = mem.get("last_result", "")
                        log(f"✅ Execution Output (Head): {str(last_res)[:500]}...")

                    if state.get("final_response"):
                        log(f"\n🏁 FINAL RESPONSE: {state.get('final_response')}")
                        return

        except Exception as e:
            log(f"\n❌ ERROR in Scenario: {e}")
            import traceback
            traceback.print_exc(file=log_file)

async def main():
    db_path = os.path.join("backend", "storage", "system", "orchestrator.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        graph = create_graph_with_checkpointer(checkpointer)
        
        if len(sys.argv) > 1:
            # Run specific prompt from args
            await run_scenario(graph, "Custom", sys.argv[1])
        else:
            # Default Level 1 Tests
            await run_scenario(graph, "Basic Chat", "Hello, are you online?")
            await run_scenario(graph, "File Ops", "Create a file named 'orchestrator_test.txt' with content 'Test Successful' in the storage folder.")

if __name__ == "__main__":
    asyncio.run(main())
