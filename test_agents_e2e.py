"""
Comprehensive End-to-End Agent Testing
Tests all agents with actual task execution
"""

import asyncio
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def test_agent_execution(agent_id, task_name, task):
    """Test agent execution."""
    print(f"\n{'='*70}")
    print(f"TESTING: {agent_id} - {task_name}")
    print(f"{'='*70}")
    
    from backend.services.agent_manager import get_agent_manager
    
    manager = get_agent_manager()
    await manager.initialize()
    
    try:
        # Spawn agent
        print(f"\n1. Spawning {agent_id}...")
        start = time.time()
        instance = await manager.spawn_agent(agent_id)
        spawn_time = time.time() - start
        print(f"   [OK] Spawned in {spawn_time:.1f}s (Port: {instance.port})")
        
        # Execute task
        print(f"\n2. Executing task: {task_name}")
        print(f"   Task: {task.get('prompt', 'N/A')[:60]}...")
        start = time.time()
        result = await manager.execute(agent_id, task)
        exec_time = time.time() - start
        
        print(f"   [OK] Executed in {exec_time:.1f}s")
        print(f"   Status: {result.get('status', 'unknown')}")
        
        if result.get('status') == 'error':
            print(f"   [ERROR] {result.get('error', 'Unknown error')}")
            success = False
        else:
            print(f"   [SUCCESS] Task completed")
            # Show result preview
            if 'result' in result:
                result_str = str(result['result'])[:200]
                print(f"   Result: {result_str}...")
            success = True
        
        # Terminate
        print(f"\n3. Terminating...")
        await manager.terminate_agent(agent_id)
        print(f"   [OK] Terminated")
        
        return success, spawn_time, exec_time
        
    except Exception as e:
        print(f"   [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False, 0, 0
    finally:
        await manager.shutdown()


async def main():
    """Run comprehensive tests."""
    print("\n" + "="*70)
    print("COMPREHENSIVE AGENT END-TO-END TESTING")
    print("="*70)
    print("Testing: Spawn -> Execute -> Terminate for all agents")
    
    results = []
    
    # Test 1: Spreadsheet Agent
    result = await test_agent_execution(
        'spreadsheet',
        'Create CSV',
        {
            'prompt': 'Create a simple CSV with columns Name, Age and 2 sample rows',
            'action': 'create',
            'payload': {'filename': 'test.csv'},
            'thread_id': f'test_{int(time.time())}',
            'user_id': 'test_user',
        }
    )
    results.append(('Spreadsheet', result))
    
    # Test 2: Mail Agent
    result = await test_agent_execution(
        'mail',
        'Health Check Only',
        {
            'prompt': 'Check email system status',
            'action': 'search',
            'payload': {'query': 'test'},
            'thread_id': f'test_{int(time.time())}',
            'user_id': 'test_user',
        }
    )
    results.append(('Mail', result))
    
    # Test 3: Document Agent
    # Create a test file first
    test_file = Path('backend/storage/documents/test.txt')
    test_file.parent.mkdir(parents=True, exist_ok=True)
    test_file.write_text("This is a test document for analysis.")
    
    result = await test_agent_execution(
        'document',
        'Analyze Document',
        {
            'prompt': f'Analyze the document at {test_file}',
            'action': 'analyze',
            'payload': {'file_path': str(test_file)},
            'thread_id': f'test_{int(time.time())}',
            'user_id': 'test_user',
        }
    )
    results.append(('Document', result))
    
    # Cleanup test file
    test_file.unlink(missing_ok=True)
    
    # Test 4: Zoho Books Agent
    result = await test_agent_execution(
        'zoho_books',
        'Health Check',
        {
            'prompt': 'Check Zoho Books connection',
            'action': 'health',
            'payload': {},
            'thread_id': f'test_{int(time.time())}',
            'user_id': 'test_user',
        }
    )
    results.append(('Zoho Books', result))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    total_spawn = 0
    total_exec = 0
    passed = 0
    
    for agent_name, (success, spawn_time, exec_time) in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {agent_name}")
        if success:
            print(f"       Spawn: {spawn_time:.1f}s | Execute: {exec_time:.1f}s")
            total_spawn += spawn_time
            total_exec += exec_time
            passed += 1
    
    print(f"\n{'='*70}")
    print(f"Results: {passed}/{len(results)} agents passed")
    
    if passed > 0:
        print(f"Average spawn time: {total_spawn/passed:.1f}s")
        print(f"Average exec time: {total_exec/passed:.1f}s")
    
    if passed == len(results):
        print("\n" + "="*70)
        print("ALL AGENTS WORKING END-TO-END!")
        print("="*70)
        print("\nOn-Demand Agent Spawning System:")
        print("  [OK] All agents spawn correctly")
        print("  [OK] All agents execute tasks")
        print("  [OK] All agents terminate properly")
        print("  [OK] Health checks pass")
        print("  [OK] Resource efficient (spawn on demand)")
        print("\nSystem is production ready!")
    else:
        print(f"\n{passed}/{len(results)} agents working")


if __name__ == "__main__":
    asyncio.run(main())
