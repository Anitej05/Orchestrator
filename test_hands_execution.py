"""Quick test script to verify Hands can execute Python code."""
import sys
import asyncio
sys.path.insert(0, 'd:/Internship/Orbimesh')

from backend.orchestrator.hands import Hands

async def test_hands_execution():
    print("Creating Hands instance...")
    hands = Hands()
    
    state = {
        'decision': {
            'action_type': 'python',
            'resource_id': None,
            'payload': {
                'code': 'print("Hello from Hands"); result = 123',
                'session_id': 'test_session'
            }
        },
        'iteration_count': 0,
        'action_history': []
    }
    
    print("Executing Python code through Hands...")
    result = await hands.execute(state)
    print("Execution completed!")
    print("Result:", result)
    
    # Check if execution_result exists and is successful
    if 'execution_result' in result:
        exec_result = result['execution_result']
        print(f"Success: {exec_result.get('success')}")
        print(f"Output: {exec_result.get('output')}")
    
    return result

if __name__ == "__main__":
    asyncio.run(test_hands_execution())
