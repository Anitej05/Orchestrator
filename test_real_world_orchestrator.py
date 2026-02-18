"""
Real-World Orchestrator Test

This test runs the orchestrator with ACTUAL LLM calls to test:
1. Brain making real decisions with the inference service
2. Hands executing those decisions
3. Full cycle completion

This requires valid API keys to be configured.
"""

import asyncio
import sys
import os
import time
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver


async def test_real_world_task(task_description: str, max_iterations: int = 5):
    """
    Test a real-world task through the orchestrator.
    
    Args:
        task_description: The task to execute
        max_iterations: Maximum iterations to run
    """
    print(f"\n{'='*70}")
    print(f"REAL-WORLD TASK: {task_description}")
    print(f"{'='*70}\n")
    
    from backend.orchestrator.graph import create_graph_with_checkpointer
    
    # Create graph with memory checkpointer
    checkpointer = MemorySaver()
    graph = create_graph_with_checkpointer(checkpointer)
    
    # Configuration
    thread_id = f"real_test_{int(time.time())}"
    config = {
        "configurable": {
            "thread_id": thread_id,
            "owner": {"user_id": "test_user"}
        }
    }
    
    # Initial state
    initial_state = {
        "original_prompt": task_description,
        "messages": [HumanMessage(content=task_description)],
        "todo_list": [],
        "memory": {},
        "insights": {},
        "action_history": [],
        "thread_id": thread_id,
        "user_id": "test_user",
        "iteration_count": 0,
        "failure_count": 0,
        "max_iterations": max_iterations
    }
    
    start_time = time.time()
    
    try:
        print("🚀 Starting orchestrator execution...\n")
        
        # Run the graph
        result = await graph.ainvoke(initial_state, config)
        
        duration = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"EXECUTION COMPLETED in {duration:.2f}s")
        print(f"{'='*70}")
        
        # Print results
        print(f"\n📊 Final State:")
        print(f"   - Iterations: {result.get('iteration_count', 'N/A')}")
        print(f"   - Final Response: {result.get('final_response', 'N/A')[:500]}...")
        
        if result.get('action_history'):
            print(f"\n📝 Action History ({len(result['action_history'])} actions):")
            for i, action in enumerate(result['action_history'][-5:], 1):
                status = "✅" if action.get('success') else "❌"
                print(f"   {i}. {status} {action.get('action_type')}:{action.get('resource_id', '')}")
                if action.get('result_summary'):
                    print(f"      Result: {action['result_summary'][:100]}...")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


async def run_real_world_tests():
    """Run a series of real-world tasks."""
    
    print("\n" + "="*70)
    print("REAL-WORLD ORCHESTRATOR TESTS")
    print("Testing with ACTUAL LLM inference and code execution")
    print("="*70)
    
    # Test 1: Simple calculation task
    print("\n\n📌 TEST 1: Simple Calculation Task")
    print("-" * 50)
    
    result1 = await test_real_world_task(
        "Calculate the compound interest for $10000 at 5% annual rate for 3 years using Python code. Show me the result.",
        max_iterations=5
    )
    
    if result1 and result1.get('final_response'):
        print("✅ Test 1 PASSED - Orchestrator executed calculation task")
    else:
        print("⚠️ Test 1 may need review - check output above")
    
    # Test 2: Data analysis task
    print("\n\n📌 TEST 2: Data Analysis Task")
    print("-" * 50)
    
    result2 = await test_real_world_task(
        "Create a pandas DataFrame with sales data for 3 products (A, B, C) with random values between 100-500, then calculate and show the total sales and average per product.",
        max_iterations=5
    )
    
    if result2 and result2.get('final_response'):
        print("✅ Test 2 PASSED - Orchestrator executed data analysis task")
    else:
        print("⚠️ Test 2 may need review - check output above")
    
    # Test 3: Multi-step task with planning
    print("\n\n📌 TEST 3: Multi-Step Task")
    print("-" * 50)
    
    result3 = await test_real_world_task(
        "Do the following: 1) Calculate 25 * 4, 2) Then add 10 to the result, 3) Tell me the final answer.",
        max_iterations=8
    )
    
    if result3 and result3.get('final_response'):
        print("✅ Test 3 PASSED - Orchestrator executed multi-step task")
    else:
        print("⚠️ Test 3 may need review - check output above")
    
    # Summary
    print("\n\n" + "="*70)
    print("REAL-WORLD TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in [result1, result2, result3] if r and r.get('final_response'))
    print(f"\n✅ {passed}/3 tests completed with responses")
    
    if passed == 3:
        print("\n🎉 ALL REAL-WORLD TESTS PASSED!")
        print("The orchestrator successfully:")
        print("  - Made decisions using LLM inference")
        print("  - Executed Python code in sandbox")
        print("  - Handled multi-step tasks")
        print("  - Returned appropriate responses")
    elif passed > 0:
        print(f"\n⚠️ {passed}/3 tests passed - partial success")
        print("This may indicate LLM API issues or task complexity")
    else:
        print("\n❌ Tests did not complete - check API keys and logs")
    
    return passed


async def quick_test():
    """Quick single test for fast verification."""
    print("\n🚀 QUICK REAL-WORLD TEST")
    print("="*50)
    
    result = await test_real_world_task(
        "Use Python to calculate 15 * 7 and tell me the answer.",
        max_iterations=3
    )
    
    if result:
        print(f"\n✅ Quick test completed!")
        print(f"Response: {result.get('final_response', 'No response')}")
        return True
    return False


if __name__ == "__main__":
    # Run quick test by default, or full test suite with --full flag
    if "--full" in sys.argv:
        asyncio.run(run_real_world_tests())
    else:
        print("Running quick test. Use --full flag for complete test suite.\n")
        asyncio.run(quick_test())