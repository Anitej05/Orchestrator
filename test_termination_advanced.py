"""
Test 2: Multi-step task with web search and code sandbox
This tests if the brain properly terminates after completing a multi-step workflow
"""
import asyncio
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

async def test_weather_comparison_task():
    """Test a task that requires web search + code execution + visualization."""
    print("\n" + "="*70)
    print("TEST: Weather Comparison with Visualization")
    print("="*70)
    print("Task: Search for current weather in NY and SF, compare them,")
    print("      then create a Python chart showing the comparison.")
    print("="*70)
    
    from backend.orchestrator.graph import create_graph_with_checkpointer
    
    checkpointer = MemorySaver()
    graph = create_graph_with_checkpointer(checkpointer)
    
    thread_id = f'weather_test_{int(time.time())}'
    config = {
        'configurable': {
            'thread_id': thread_id,
            'owner': {'user_id': 'test_user'}
        }
    }
    
    task = 'Search for current weather in New York City, then search for current weather in San Francisco, compare the temperatures, and create a simple Python bar chart showing both temperatures side by side.'
    
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
        'max_iterations': 10
    }
    
    start_time = time.time()
    
    try:
        print('\nStarting execution...')
        result = await graph.ainvoke(initial_state, config)
        duration = time.time() - start_time
        
        iterations = result.get('iteration_count', 0)
        
        print(f'\n[COMPLETE] Execution finished in {duration:.2f}s')
        print(f'[INFO] Total iterations: {iterations}')
        
        final_response = result.get('final_response', 'No response')
        print(f'\nFinal response preview:')
        print('-' * 50)
        print(final_response[:400] if len(final_response) > 400 else final_response)
        print('-' * 50)
        
        if result.get('action_history'):
            print(f'\nAction History ({len(result["action_history"])} actions):')
            for i, action in enumerate(result['action_history'], 1):
                status = 'OK' if action.get('success') else 'FAIL'
                atype = action.get('action_type', 'unknown')
                resource = action.get('resource_id', 'N/A')
                print(f'  {i}. [{status}] {atype}:{resource}')
                
                # Show result summary if available
                result_summary = action.get('result_summary', '')
                if result_summary:
                    summary = result_summary[:80] + '...' if len(result_summary) > 80 else result_summary
                    print(f'      -> {summary}')
        
        # Analysis
        print('\n' + '='*70)
        print('ANALYSIS')
        print('='*70)
        
        if iterations <= 5:
            print(f'[PASS] Task completed efficiently in {iterations} iterations')
            print('       Brain properly recognized completion and terminated.')
            return True
        elif iterations <= 8:
            print(f'[WARNING] Task took {iterations} iterations (acceptable but watch for patterns)')
            return True
        else:
            print(f'[FAIL] Task took {iterations} iterations (may indicate looping behavior)')
            print('       Brain should have terminated earlier.')
            return False
            
    except Exception as e:
        print(f'\n[ERROR] {e}')
        import traceback
        traceback.print_exc()
        return False

async def test_data_analysis_task():
    """Test a data analysis task that requires multiple searches and Python processing."""
    print("\n" + "="*70)
    print("TEST: Market Data Analysis")
    print("="*70)
    print("Task: Get stock prices for Apple and Microsoft, calculate which")
    print("      performed better, and show the percentage difference.")
    print("="*70)
    
    from backend.orchestrator.graph import create_graph_with_checkpointer
    
    checkpointer = MemorySaver()
    graph = create_graph_with_checkpointer(checkpointer)
    
    thread_id = f'stock_test_{int(time.time())}'
    config = {
        'configurable': {
            'thread_id': thread_id,
            'owner': {'user_id': 'test_user'}
        }
    }
    
    task = 'Search for current Apple stock price (AAPL), then search for Microsoft stock price (MSFT). Use Python to calculate which stock has a higher price and by what percentage difference. Show the calculation and result.'
    
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
        'max_iterations': 10
    }
    
    start_time = time.time()
    
    try:
        print('\nStarting execution...')
        result = await graph.ainvoke(initial_state, config)
        duration = time.time() - start_time
        
        iterations = result.get('iteration_count', 0)
        
        print(f'\n[COMPLETE] Execution finished in {duration:.2f}s')
        print(f'[INFO] Total iterations: {iterations}')
        
        final_response = result.get('final_response', 'No response')
        print(f'\nFinal response preview:')
        print('-' * 50)
        print(final_response[:400] if len(final_response) > 400 else final_response)
        print('-' * 50)
        
        if result.get('action_history'):
            print(f'\nAction History ({len(result["action_history"])} actions):')
            for i, action in enumerate(result['action_history'], 1):
                status = 'OK' if action.get('success') else 'FAIL'
                atype = action.get('action_type', 'unknown')
                resource = action.get('resource_id', 'N/A')
                print(f'  {i}. [{status}] {atype}:{resource}')
                
                result_summary = action.get('result_summary', '')
                if result_summary:
                    summary = result_summary[:80] + '...' if len(result_summary) > 80 else result_summary
                    print(f'      -> {summary}')
        
        print('\n' + '='*70)
        print('ANALYSIS')
        print('='*70)
        
        if iterations <= 5:
            print(f'[PASS] Task completed efficiently in {iterations} iterations')
            return True
        elif iterations <= 8:
            print(f'[WARNING] Task took {iterations} iterations')
            return True
        else:
            print(f'[FAIL] Task took {iterations} iterations (possible looping)')
            return False
            
    except Exception as e:
        print(f'\n[ERROR] {e}')
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("ADDITIONAL TERMINATION TESTS")
    print("Testing complex multi-tool tasks with code sandbox")
    print("="*70)
    
    # Test 1: Weather comparison
    result1 = await test_weather_comparison_task()
    
    # Small delay between tests
    await asyncio.sleep(2)
    
    # Test 2: Stock analysis
    result2 = await test_data_analysis_task()
    
    # Summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    passed = sum([result1, result2])
    total = 2
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n[EXCELLENT] All tests passed!")
        print("The brain correctly terminates after completing multi-step tasks.")
        print("No infinite loops detected.")
    elif passed > 0:
        print(f"\n[PARTIAL] {passed}/{total} tests passed")
        print("Some tasks may need optimization.")
    else:
        print("\n[FAIL] No tests passed")
        print("The termination logic may need review.")

if __name__ == "__main__":
    asyncio.run(main())
