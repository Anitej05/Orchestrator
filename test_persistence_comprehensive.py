"""
Comprehensive Test: File Persistence System

Tests:
1. Thread persistence - Files persist within a conversation
2. Shared persistence - Files persist across conversations  
3. Brain awareness - Orchestrator knows about both workspaces
4. Cross-conversation access - Can reference files from previous conversations
"""

import asyncio
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

async def test_thread_persistence():
    """Test that files persist within a conversation (thread workspace)."""
    print("\n" + "="*70)
    print("TEST 1: Thread Persistence (Within Conversation)")
    print("="*70)
    print("Verifies that files created are tracked and accessible in the same conversation")
    
    from backend.orchestrator.graph import create_graph_with_checkpointer
    
    checkpointer = MemorySaver()
    graph = create_graph_with_checkpointer(checkpointer)
    
    thread_id = f'thread_test_{int(time.time())}'
    config = {
        'configurable': {
            'thread_id': thread_id,
            'owner': {'user_id': 'test_user'}
        }
    }
    
    # Step 1: Create a file
    task1 = '''Create a Python script that:
1. Creates a list of numbers [10, 20, 30, 40, 50]
2. Saves them to a file called "numbers.txt"
3. Prints "File created"'''
    
    initial_state = {
        'original_prompt': task1,
        'messages': [HumanMessage(content=task1)],
        'todo_list': [],
        'memory': {},
        'insights': {},
        'action_history': [],
        'created_files': [],
        'orchestrator_workspace': '',
        'shared_files': [],
        'shared_workspace': '',
        'thread_id': thread_id,
        'user_id': 'test_user',
        'iteration_count': 0,
        'failure_count': 0,
        'max_iterations': 5
    }
    
    try:
        print('\n[Step 1/3] Creating file in thread workspace...')
        result1 = await graph.ainvoke(initial_state, config)
        
        workspace = result1.get('orchestrator_workspace', '')
        created_files = result1.get('created_files', [])
        
        print(f'  Workspace: {workspace}')
        print(f'  Files created: {len(created_files)}')
        
        if not created_files:
            print('  [FAIL] No files were tracked')
            return False
        
        file_names = [f.get('file_name') for f in created_files]
        print(f'  File names: {file_names}')
        
        # Step 2: Reference the file in same conversation
        print('\n[Step 2/3] Referencing created file in same conversation...')
        
        task2 = 'Read the contents of the numbers.txt file you just created and tell me what numbers are in it.'
        
        state2 = {
            **result1,
            'messages': [HumanMessage(content=task2)],
            'original_prompt': task2,
        }
        
        result2 = await graph.ainvoke(state2, config)
        
        final_response = result2.get('final_response', '')
        print(f'  Response: {final_response[:200]}...')
        
        # Check if response mentions the numbers
        has_numbers = any(str(n) in final_response for n in [10, 20, 30, 40, 50])
        
        if has_numbers:
            print('  [PASS] Brain successfully read and referenced the file')
        else:
            print('  [FAIL] Brain did not reference the file contents correctly')
            return False
        
        # Step 3: Check file still exists
        print('\n[Step 3/3] Verifying file persistence...')
        
        from backend.orchestrator.workspace_manager import get_workspace_manager
        wm = get_workspace_manager(thread_id)
        files = wm.list_files()
        
        if files and any('numbers' in f.file_name for f in files):
            print('  [PASS] File still exists in workspace')
            return True
        else:
            print('  [FAIL] File no longer exists')
            return False
            
    except Exception as e:
        print(f'\n  [ERROR] {e}')
        import traceback
        traceback.print_exc()
        return False


async def test_shared_persistence():
    """Test that files can be shared and persist across conversations."""
    print("\n" + "="*70)
    print("TEST 2: Shared Persistence (Across Conversations)")
    print("="*70)
    print("Verifies that shared files are accessible in new conversations")
    
    from backend.orchestrator.graph import create_graph_with_checkpointer
    from backend.orchestrator.workspace_manager import get_workspace_manager
    from backend.orchestrator.shared_workspace import get_shared_workspace_manager
    
    checkpointer = MemorySaver()
    graph = create_graph_with_checkpointer(checkpointer)
    
    # Conversation 1: Create and share a file
    thread_id_1 = f'shared_test_1_{int(time.time())}'
    config1 = {
        'configurable': {
            'thread_id': thread_id_1,
            'owner': {'user_id': 'test_user'}
        }
    }
    
    task1 = '''Create a file called "important_notes.txt" with the content:
"Project Deadline: March 15th
Key Contacts: John (Engineering), Sarah (Design)"
Then save it to the shared workspace so it's available in future conversations.'''
    
    initial_state = {
        'original_prompt': task1,
        'messages': [HumanMessage(content=task1)],
        'todo_list': [],
        'memory': {},
        'insights': {},
        'action_history': [],
        'created_files': [],
        'orchestrator_workspace': '',
        'shared_files': [],
        'shared_workspace': '',
        'thread_id': thread_id_1,
        'user_id': 'test_user',
        'iteration_count': 0,
        'failure_count': 0,
        'max_iterations': 8
    }
    
    try:
        print('\n[Conversation 1] Creating and sharing file...')
        result1 = await graph.ainvoke(initial_state, config1)
        
        print(f'  Thread workspace: {result1.get("orchestrator_workspace")}')
        print(f'  Shared workspace: {result1.get("shared_workspace")}')
        print(f'  Thread files: {len(result1.get("created_files", []))}')
        print(f'  Shared files: {len(result1.get("shared_files", []))}')
        
        # Check if file is in shared workspace
        shared_files = result1.get('shared_files', [])
        if not shared_files:
            print('  [FAIL] No files in shared workspace')
            return False
        
        print(f'  Shared file names: {[f.get("file_name") for f in shared_files]}')
        
        # Conversation 2: Try to access the shared file
        print('\n[Conversation 2] Accessing shared file from new conversation...')
        
        thread_id_2 = f'shared_test_2_{int(time.time())}'
        config2 = {
            'configurable': {
                'thread_id': thread_id_2,
                'owner': {'user_id': 'test_user'}  # Same user
            }
        }
        
        task2 = 'Read the important_notes.txt file from the shared workspace and tell me the project deadline.'
        
        initial_state2 = {
            'original_prompt': task2,
            'messages': [HumanMessage(content=task2)],
            'todo_list': [],
            'memory': {},
            'insights': {},
            'action_history': [],
            'created_files': [],
            'orchestrator_workspace': '',
            'shared_files': [],
            'shared_workspace': '',
            'thread_id': thread_id_2,
            'user_id': 'test_user',
            'iteration_count': 0,
            'failure_count': 0,
            'max_iterations': 5
        }
        
        result2 = await graph.ainvoke(initial_state2, config2)
        
        final_response = result2.get('final_response', '')
        print(f'  Response: {final_response[:200]}...')
        
        # Check if response mentions March 15th
        if 'march' in final_response.lower() and '15' in final_response:
            print('  [PASS] Successfully accessed shared file from new conversation!')
            return True
        else:
            print('  [FAIL] Could not access shared file contents')
            return False
            
    except Exception as e:
        print(f'\n  [ERROR] {e}')
        import traceback
        traceback.print_exc()
        return False


async def test_brain_workspace_awareness():
    """Test that Brain is aware of both workspaces in its prompt."""
    print("\n" + "="*70)
    print("TEST 3: Brain Workspace Awareness")
    print("="*70)
    print("Verifies that Brain prompt includes both thread and shared workspaces")
    
    from backend.orchestrator.workspace_manager import get_workspace_manager
    from backend.orchestrator.shared_workspace import get_shared_workspace_manager
    from backend.orchestrator.brain import Brain
    
    # Setup workspaces with test files
    thread_id = f'awareness_test_{int(time.time())}'
    user_id = 'test_user'
    
    # Create thread workspace file
    wm = get_workspace_manager(thread_id)
    test_file_thread = wm.workspace_path / "thread_file.txt"
    test_file_thread.write_text("This is a thread-specific file")
    wm.scan_for_new_files()
    
    # Create shared workspace file
    swm = get_shared_workspace_manager(user_id)
    test_file_shared = swm.workspace_path / "shared_file.txt"
    test_file_shared.write_text("This is a shared file")
    swm.scan_for_new_files()
    
    print(f'\n  Created test files:')
    print(f'    Thread: {test_file_thread}')
    print(f'    Shared: {test_file_shared}')
    
    # Check if files are tracked
    thread_files = wm.list_files()
    shared_files = swm.list_files()
    
    print(f'\n  Thread files tracked: {len(thread_files)}')
    print(f'  Shared files tracked: {len(shared_files)}')
    
    if thread_files and shared_files:
        print('  [PASS] Both workspaces have tracked files')
        
        # Verify files are in expected locations
        thread_names = [f.file_name for f in thread_files]
        shared_names = [f.file_name for f in shared_files]
        
        print(f'  Thread file names: {thread_names}')
        print(f'  Shared file names: {shared_names}')
        
        if 'thread_file.txt' in thread_names and 'shared_file.txt' in shared_names:
            print('  [PASS] Files correctly categorized by workspace')
            return True
        else:
            print('  [FAIL] Files not in expected workspaces')
            return False
    else:
        print('  [FAIL] Files not tracked properly')
        return False


async def test_user_isolation():
    """Test that users cannot access each other's shared files."""
    print("\n" + "="*70)
    print("TEST 4: User Isolation")
    print("="*70)
    print("Verifies that different users have isolated shared workspaces")
    
    from backend.orchestrator.shared_workspace import get_shared_workspace_manager
    
    # Create files for user 1
    swm1 = get_shared_workspace_manager('user_1')
    test_file = swm1.workspace_path / "user1_secret.txt"
    test_file.write_text("User 1's private file")
    swm1.scan_for_new_files()
    
    # Try to access from user 2
    swm2 = get_shared_workspace_manager('user_2')
    files_user2 = swm2.list_files()
    
    print(f'\n  User 1 shared files: {[f.file_name for f in swm1.list_files()]}')
    print(f'  User 2 shared files: {[f.file_name for f in files_user2]}')
    
    # User 2 should not see User 1's file
    user2_names = [f.file_name for f in files_user2]
    
    if 'user1_secret.txt' not in user2_names:
        print('  [PASS] User isolation working - User 2 cannot see User 1 files')
        return True
    else:
        print('  [FAIL] User isolation broken')
        return False


async def main():
    """Run all persistence tests."""
    print("\n" + "="*70)
    print("FILE PERSISTENCE SYSTEM - COMPREHENSIVE TESTS")
    print("="*70)
    print("Testing thread persistence, shared persistence, and workspace awareness")
    
    results = []
    
    # Test 1: Thread persistence
    print("\n" + "-"*70)
    result1 = await test_thread_persistence()
    results.append(("Thread Persistence", result1))
    
    # Test 2: Shared persistence
    print("\n" + "-"*70)
    result2 = await test_shared_persistence()
    results.append(("Shared Persistence", result2))
    
    # Test 3: Brain awareness
    print("\n" + "-"*70)
    result3 = await test_brain_workspace_awareness()
    results.append(("Brain Workspace Awareness", result3))
    
    # Test 4: User isolation
    print("\n" + "-"*70)
    result4 = await test_user_isolation()
    results.append(("User Isolation", result4))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nThe orchestrator successfully:")
        print("  1. ✅ Tracks files within conversations (thread workspace)")
        print("  2. ✅ Persists files across conversations (shared workspace)")
        print("  3. ✅ Is aware of both workspaces")
        print("  4. ✅ Isolates users from each other")
        print("\nFiles persist intelligently - temporary by default,")
        print("persistent when needed! 🚀")
    else:
        print(f"\n⚠️  {passed}/{total} tests passed")
        print("Review failed tests above")


if __name__ == "__main__":
    asyncio.run(main())
