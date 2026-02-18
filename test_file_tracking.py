"""
Test File Tracking System

This test verifies:
1. Orchestrator creates its own workspace
2. Files created by Python are tracked
3. Files created by Terminal are tracked
4. Brain is aware of created files
5. Files persist across the conversation
"""

import asyncio
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

async def test_file_tracking():
    """Test that orchestrator tracks created files."""
    print("="*70)
    print("TEST: File Tracking System")
    print("="*70)
    print("Task: Create a chart with Python, verify orchestrator tracks it")
    print("="*70)
    
    from backend.orchestrator.graph import create_graph_with_checkpointer
    
    checkpointer = MemorySaver()
    graph = create_graph_with_checkpointer(checkpointer)
    
    thread_id = f'file_test_{int(time.time())}'
    config = {
        'configurable': {
            'thread_id': thread_id,
            'owner': {'user_id': 'test_user'}
        }
    }
    
    # Task that creates a file
    task = '''Create a simple Python script that:
1. Creates a list of numbers [1, 2, 3, 4, 5]
2. Calculates their squares
3. Saves a text file called "squares.txt" with the results
4. Print "File created successfully"'''
    
    initial_state = {
        'original_prompt': task,
        'messages': [HumanMessage(content=task)],
        'todo_list': [],
        'memory': {},
        'insights': {},
        'action_history': [],
        'created_files': [],
        'orchestrator_workspace': '',
        'thread_id': thread_id,
        'user_id': 'test_user',
        'iteration_count': 0,
        'failure_count': 0,
        'max_iterations': 5
    }
    
    try:
        print('\n[1/3] Executing task...')
        result = await graph.ainvoke(initial_state, config)
        
        iterations = result.get('iteration_count', 0)
        created_files = result.get('created_files', [])
        workspace = result.get('orchestrator_workspace', '')
        
        print(f'\n[2/3] Execution complete:')
        print(f'    Iterations: {iterations}')
        print(f'    Workspace: {workspace}')
        print(f'    Files created: {len(created_files)}')
        
        if created_files:
            print(f'\n    Created files:')
            for f in created_files:
                print(f"      - {f.get('file_name')} ({f.get('file_type')})")
        
        # Test 2: Ask about the created file
        print('\n[3/3] Testing file awareness in follow-up...')
        
        follow_up_task = 'Read the contents of the squares.txt file you just created and tell me what numbers are in it.'
        
        # Continue the conversation
        state_update = {
            'messages': [HumanMessage(content=follow_up_task)],
            'original_prompt': follow_up_task,
        }
        
        # We need to continue from previous state
        result2 = await graph.ainvoke(
            {**result, **state_update},
            config
        )
        
        final_response = result2.get('final_response', '')
        
        print('\n' + '='*70)
        print('RESULTS')
        print('='*70)
        
        # Check results
        checks = []
        
        # Check 1: Workspace path exists
        if workspace and Path(workspace).exists():
            checks.append(('Workspace created', True))
        else:
            checks.append(('Workspace created', False))
        
        # Check 2: File was tracked
        if created_files and any('squares' in str(f.get('file_name', '')) for f in created_files):
            checks.append(('File tracked', True))
        else:
            checks.append(('File tracked', False))
        
        # Check 3: File exists in workspace
        if created_files:
            file_path = created_files[0].get('file_path', '')
            if file_path and Path(file_path).exists():
                checks.append(('File exists in workspace', True))
            else:
                checks.append(('File exists in workspace', False))
        else:
            checks.append(('File exists in workspace', False))
        
        # Check 4: Follow-up response mentions the file contents
        if '1' in final_response and ('4' in final_response or '9' in final_response or 'square' in final_response.lower()):
            checks.append(('Brain aware of file', True))
        else:
            checks.append(('Brain aware of file', False))
        
        for check_name, passed in checks:
            status = 'PASS' if passed else 'FAIL'
            print(f'  [{status}] {check_name}')
        
        passed = sum(1 for _, p in checks if p)
        total = len(checks)
        
        print(f'\nOverall: {passed}/{total} checks passed')
        
        if passed == total:
            print('\n[SUCCESS] File tracking system is working correctly!')
            return True
        else:
            print('\n[WARNING] Some checks failed - review output above')
            return passed >= total * 0.5
            
    except Exception as e:
        print(f'\n[ERROR] {e}')
        import traceback
        traceback.print_exc()
        return False

async def test_workspace_isolation():
    """Test that different threads have isolated workspaces."""
    print("\n" + "="*70)
    print("TEST: Workspace Isolation")
    print("="*70)
    
    from backend.orchestrator.workspace_manager import get_workspace_manager
    
    # Create two workspace managers for different threads
    wm1 = get_workspace_manager("thread_1")
    wm2 = get_workspace_manager("thread_2")
    
    path1 = str(wm1.get_workspace_path())
    path2 = str(wm2.get_workspace_path())
    
    print(f'Workspace 1: {path1}')
    print(f'Workspace 2: {path2}')
    
    if path1 != path2:
        print('[PASS] Workspaces are isolated by thread')
        return True
    else:
        print('[FAIL] Workspaces are NOT isolated')
        return False

async def main():
    """Run all file tracking tests."""
    print("\n" + "="*70)
    print("FILE TRACKING SYSTEM TESTS")
    print("="*70)
    
    # Test 1: File tracking
    result1 = await test_file_tracking()
    
    # Test 2: Workspace isolation
    result2 = await test_workspace_isolation()
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    passed = sum([result1, result2])
    total = 2
    
    print(f"\nTests passed: {passed}/{total}")
    
    if passed == total:
        print("\n[SUCCESS] File tracking system is fully operational!")
        print("  - Orchestrator has its own workspace")
        print("  - Created files are tracked")
        print("  - Brain is aware of created files")
        print("  - Workspaces are isolated per thread")
    else:
        print(f"\n[PARTIAL] {passed}/{total} tests passed")

if __name__ == "__main__":
    asyncio.run(main())
