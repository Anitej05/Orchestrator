"""
Quick test to verify the brain properly terminates tasks
"""
import asyncio
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Fix unicode printing on Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

async def test_simple_task():
    """Test a simple task that should complete quickly."""
    print("Testing simple calculation task...")
    print("="*70)
    
    from backend.orchestrator.graph import create_graph_with_checkpointer
    
    checkpointer = MemorySaver()
    graph = create_graph_with_checkpointer(checkpointer)
    
    thread_id = f'simple_test_{int(time.time())}'
    config = {
        'configurable': {
            'thread_id': thread_id,
            'owner': {'user_id': 'test_user'}
        }
    }
    
    # Simple task that requires calculation
    task = 'Calculate 15 * 23 using Python and tell me the answer.'
    
    initial_state = {
        'original_prompt': task,
        'messages': [HumanMessage(content=task)],
        'todo_list': [],
        'memory': {},
        'insights': {},
        'action_history': [],
        'thread_id': thread_id,
        'user_id': 'test_user',
        'iteration_count': 0,
        'failure_count': 0,
        'max_iterations': 5
    }
    
    start_time = time.time()
    
    try:
        print('Starting execution...')
        result = await graph.ainvoke(initial_state, config)
        duration = time.time() - start_time
        
        print(f'\nExecution completed in {duration:.2f}s!')
        print(f'Iterations: {result.get("iteration_count", "N/A")}')
        
        final_response = result.get("final_response", "No response")
        # Remove emojis for safe printing
        final_response = final_response.encode('ascii', 'ignore').decode('ascii')
        print(f'Final response: {final_response[:200]}...')
        
        if result.get('action_history'):
            print(f'\nAction History ({len(result["action_history"])} actions):')
            for i, action in enumerate(result['action_history'], 1):
                status = 'OK' if action.get('success') else 'FAIL'
                print(f'  {i}. {status} {action.get("action_type")}:{action.get("resource_id", "")}')
        
        return result
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return None

async def test_multi_step_task():
    """Test a multi-step task similar to Tesla analysis."""
    print("\nTesting multi-step task...")
    print("="*70)
    
    from backend.orchestrator.graph import create_graph_with_checkpointer
    
    checkpointer = MemorySaver()
    graph = create_graph_with_checkpointer(checkpointer)
    
    thread_id = f'multi_test_{int(time.time())}'
    config = {
        'configurable': {
            'thread_id': thread_id,
            'owner': {'user_id': 'test_user'}
        }
    }
    
    # Multi-step task with web search and analysis
    task = 'Search for current Bitcoin price, then search for Ethereum price, and tell me which is higher.'
    
    initial_state = {
        'original_prompt': task,
        'messages': [HumanMessage(content=task)],
        'todo_list': [],
        'memory': {},
        'insights': {},
        'action_history': [],
        'thread_id': thread_id,
        'user_id': 'test_user',
        'iteration_count': 0,
        'failure_count': 0,
        'max_iterations': 8
    }
    
    start_time = time.time()
    
    try:
        print('Starting execution...')
        result = await graph.ainvoke(initial_state, config)
        duration = time.time() - start_time
        
        print(f'\nExecution completed in {duration:.2f}s!')
        print(f'Iterations: {result.get("iteration_count", "N/A")}')
        
        final_response = result.get("final_response", "No response")
        # Remove emojis for safe printing
        final_response = final_response.encode('ascii', 'ignore').decode('ascii')
        print(f'Final response length: {len(final_response)} chars')
        print(f'Final response: {final_response[:300]}...')
        
        if result.get('action_history'):
            print(f'\nAction History ({len(result["action_history"])} actions):')
            for i, action in enumerate(result['action_history'], 1):
                status = 'OK' if action.get('success') else 'FAIL'
                atype = action.get("action_type")
                resid = action.get("resource_id", "")
                result_summary = action.get('result_summary', '')[:60]
                print(f'  {i}. {status} {atype}:{resid}')
                if result_summary:
                    # Remove emojis
                    safe_summary = result_summary.encode('ascii', 'ignore').decode('ascii')
                    print(f'      -> {safe_summary}...')
        
        return result
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Run both tests."""
    print("BRAIN TERMINATION TEST")
    print("Testing if the updated brain properly terminates tasks")
    print("="*70)
    
    # Test 1: Simple task
    result1 = await test_simple_task()
    
    if result1 and result1.get('final_response'):
        print("\n[SUCCESS] Simple task completed!")
        iterations1 = result1.get('iteration_count', 0)
        if iterations1 <= 3:
            print(f"[GOOD] Completed in {iterations1} iterations (efficient)")
        else:
            print(f"[WARNING] Took {iterations1} iterations for simple task")
    else:
        print("\n[FAIL] Simple task did not complete properly")
    
    # Test 2: Multi-step task
    result2 = await test_multi_step_task()
    
    if result2 and result2.get('final_response'):
        print("\n[SUCCESS] Multi-step task completed!")
        iterations2 = result2.get('iteration_count', 0)
        if iterations2 <= 6:
            print(f"[GOOD] Completed in {iterations2} iterations (efficient)")
        elif iterations2 <= 10:
            print(f"[OK] Completed in {iterations2} iterations (acceptable)")
        else:
            print(f"[WARNING] Took {iterations2} iterations (may indicate looping)")
    else:
        print("\n[FAIL] Multi-step task did not complete properly")
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    passed = sum(1 for r in [result1, result2] if r and r.get('final_response'))
    print(f"Passed: {passed}/2 tests")
    
    if passed == 2:
        print("\n*** Both tests passed - brain properly terminates tasks! ***")

if __name__ == "__main__":
    asyncio.run(main())
