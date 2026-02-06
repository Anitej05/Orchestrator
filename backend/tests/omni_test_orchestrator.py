import asyncio
import sys
import os
import json
import logging
from typing import Dict, Any

# Add backend to path - but we need to be careful with the directory
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, backend_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

from orchestrator.graph import graph


# Cleanup function
def cleanup_test_files():
    """Remove any test files created during testing."""
    test_files = ["omni_test.txt", "test_output.txt"]
    for f in test_files:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"🧹 Cleaned up: {f}")
            except Exception as e:
                print(f"⚠️ Could not clean up {f}: {e}")


async def run_scenario(name: str, prompt: str):
    print(f"\n{'=' * 60}")
    print(f"🚀 OMNI SCENARIO: {name}")
    print(f"📄 Prompt: {prompt}")
    print(f"{'=' * 60}\n")

    initial_state = {
        "original_prompt": prompt,
        "todo_list": [],
        "memory": {},
        "iteration_count": 0,
        "failure_count": 0,
        "max_iterations": 20,
        "final_response": None,
        "current_task_id": None,
        "thread_id": f"test_{name.replace(' ', '_')}",
        "user_id": "test_user",
        "messages": [],
        "uploaded_files": [],
    }

    config = {
        "configurable": {"thread_id": initial_state["thread_id"]},
        "recursion_limit": 50,
    }

    try:
        async for output in graph.astream(initial_state, config):
            for node, node_state in output.items():
                if node_state is None:
                    print(f"\n--- Node: [{node}] returned None ---")
                    continue
                
                print(f"\n--- Node: [{node}] ---")

                if node == "brain":
                    decision = node_state.get("decision") or {}
                    if not decision:
                        print(
                            f"🧠 Brain No Decision - checking state: {list(node_state.keys())}"
                        )
                        continue
                    action_type = decision.get("action_type", "unknown")
                    resource_id = decision.get("resource_id", "N/A")
                    print(f"🧠 Brain Decision: {action_type} -> {resource_id}")
                    print(f"🤔 Reasoning: {decision.get('reasoning', 'N/A')}")
                    if decision.get("memory_updates"):
                        print(f"💾 Memory Updates: {decision.get('memory_updates')}")
                    todo_list = node_state.get("todo_list", [])
                    if todo_list:
                        print(f"📋 To-Do Items: {len(todo_list)}")
                        for t in todo_list[:3]:
                            status = t.get("status", "unknown").upper()
                            print(f"   [{status}] {t.get('description', 'N/A')[:50]}")

                elif node == "hands":
                    res = node_state.get("execution_result") or {}
                    if not res:
                        print(f"⚡ Hands No Result - possibly skipped")
                        continue
                    print(
                        f"⚡ Execution {'✅ SUCCESS' if res.get('success') else '❌ FAILED'}"
                    )
                    if not res.get("success"):
                        print(f"❗ Error: {res.get('error_message', 'Unknown')}")
                    else:
                        out = str(res.get("output", ""))
                        print(
                            f"📦 Output: {out[:200]}..."
                            if len(out) > 200
                            else f"📦 Output: {out}"
                        )

                if node_state.get("final_response"):
                    print(f"\n🏁 FINAL RESPONSE: {node_state.get('final_response')}")
                    return

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()


async def main():
    # 1. Simple Task: General Greeting / Information
    await run_scenario("Basic Chat", "Hi there! Who are you and what can you do?")

    # 2. Terminal Task: File Operations
    await run_scenario(
        "Terminal Ops",
        "Create a file named 'omni_test.txt' containing 'Manus logic test' in the root directory, then read it back.",
    )


if __name__ == "__main__":
    asyncio.run(main())
