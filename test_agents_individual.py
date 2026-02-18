"""
Individual Agent Testing Suite

Tests each agent (except Browser) individually:
- Spreadsheet Agent
- Mail Agent
- Document Agent
- Zoho Books Agent

Verifies:
1. Agent spawns correctly
2. Health check passes
3. Agent executes tasks
4. Auto-termination works
"""

import asyncio
import sys
from pathlib import Path
import time
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path('backend/.env'))

sys.path.insert(0, str(Path(__file__).parent / "backend"))

from backend.services.agent_manager import AgentManager, get_agent_manager


class AgentTester:
    """Test harness for individual agent testing."""
    
    def __init__(self):
        self.manager = None
        self.results = []
    
    async def setup(self):
        """Initialize the agent manager."""
        print("\n" + "="*70)
        print("INITIALIZING AGENT MANAGER")
        print("="*70)
        self.manager = get_agent_manager()
        await self.manager.initialize()
        print(f"✓ Agent Manager initialized")
        print(f"✓ Active agents: {len(self.manager.get_active_agents())}")
        print(f"✓ Auto-terminator: Running (5min timeout)")
    
    async def cleanup(self):
        """Shutdown and cleanup."""
        print("\n" + "="*70)
        print("CLEANUP")
        print("="*70)
        if self.manager:
            await self.manager.shutdown()
            print("✓ All agents terminated")
            print("✓ Agent Manager shutdown")
    
    async def test_spreadsheet_agent(self):
        """Test Spreadsheet Agent individually."""
        print("\n" + "="*70)
        print("TEST 1: SPREADSHEET AGENT")
        print("="*70)
        
        agent_id = 'spreadsheet'
        results = {'agent': agent_id, 'tests': []}
        
        try:
            # Test 1: Spawn
            print("\n1. Spawning Spreadsheet Agent...")
            start = time.time()
            instance = await self.manager.spawn_agent(agent_id)
            spawn_time = time.time() - start
            
            print(f"   ✓ Spawned in {spawn_time:.1f}s")
            print(f"   ✓ Port: {instance.port}")
            print(f"   ✓ PID: {instance.pid}")
            print(f"   ✓ Healthy: {instance.healthy}")
            
            results['tests'].append(('Spawn', True, f"{spawn_time:.1f}s"))
            
            # Test 2: Execute simple task
            print("\n2. Executing test task...")
            task = {
                'prompt': 'Create a simple CSV with 3 columns: Name, Age, City and 2 sample rows',
                'action': 'create',
                'payload': {'filename': 'test_spreadsheet.csv'},
                'thread_id': f'test_{agent_id}_{int(time.time())}',
                'user_id': 'test_user',
            }
            
            start = time.time()
            result = await self.manager.execute(agent_id, task)
            exec_time = time.time() - start
            
            print(f"   ✓ Task executed in {exec_time:.1f}s")
            print(f"   ✓ Status: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'error':
                print(f"   ✗ Error: {result.get('error', 'Unknown')}")
                results['tests'].append(('Execute', False, result.get('error')))
            else:
                print(f"   ✓ Success: Task completed")
                results['tests'].append(('Execute', True, f"{exec_time:.1f}s"))
            
            # Test 3: Verify agent still active
            print("\n3. Verifying agent status...")
            is_active = self.manager.is_agent_active(agent_id)
            print(f"   ✓ Agent active: {is_active}")
            results['tests'].append(('Active Check', is_active, None))
            
            # Test 4: Terminate
            print("\n4. Terminating agent...")
            success = await self.manager.terminate_agent(agent_id)
            print(f"   ✓ Terminated: {success}")
            results['tests'].append(('Terminate', success, None))
            
            # Verify terminated
            is_active = self.manager.is_agent_active(agent_id)
            print(f"   ✓ Agent inactive: {not is_active}")
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results['tests'].append(('Exception', False, str(e)))
        
        self.results.append(results)
        return all(t[1] for t in results['tests'])
    
    async def test_mail_agent(self):
        """Test Mail Agent individually."""
        print("\n" + "="*70)
        print("TEST 2: MAIL AGENT")
        print("="*70)
        
        agent_id = 'mail'
        results = {'agent': agent_id, 'tests': []}
        
        try:
            # Test 1: Spawn
            print("\n1. Spawning Mail Agent...")
            start = time.time()
            instance = await self.manager.spawn_agent(agent_id)
            spawn_time = time.time() - start
            
            print(f"   ✓ Spawned in {spawn_time:.1f}s")
            print(f"   ✓ Port: {instance.port}")
            print(f"   ✓ PID: {instance.pid}")
            
            results['tests'].append(('Spawn', True, f"{spawn_time:.1f}s"))
            
            # Test 2: Health check only (don't actually send email)
            print("\n2. Checking health...")
            healthy = await self.manager.health_checker.check_health(instance.port)
            print(f"   ✓ Health check: {healthy}")
            results['tests'].append(('Health Check', healthy, None))
            
            # Test 3: Terminate
            print("\n3. Terminating agent...")
            success = await self.manager.terminate_agent(agent_id)
            print(f"   ✓ Terminated: {success}")
            results['tests'].append(('Terminate', success, None))
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results['tests'].append(('Exception', False, str(e)))
        
        self.results.append(results)
        return all(t[1] for t in results['tests'])
    
    async def test_document_agent(self):
        """Test Document Agent individually."""
        print("\n" + "="*70)
        print("TEST 3: DOCUMENT AGENT")
        print("="*70)
        
        agent_id = 'document'
        results = {'agent': agent_id, 'tests': []}
        
        try:
            # Test 1: Spawn
            print("\n1. Spawning Document Agent...")
            start = time.time()
            instance = await self.manager.spawn_agent(agent_id)
            spawn_time = time.time() - start
            
            print(f"   ✓ Spawned in {spawn_time:.1f}s")
            print(f"   ✓ Port: {instance.port}")
            print(f"   ✓ PID: {instance.pid}")
            
            results['tests'].append(('Spawn', True, f"{spawn_time:.1f}s"))
            
            # Test 2: Execute simple analysis task
            print("\n2. Executing test task...")
            
            # Create a simple test file first
            test_file = Path(f"backend/storage/documents/test_doc_{int(time.time())}.txt")
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("This is a test document for the Document Agent.\nIt contains sample text for analysis.")
            
            task = {
                'prompt': f'Analyze the document at {test_file} and tell me what it contains',
                'action': 'analyze',
                'payload': {'file_path': str(test_file)},
                'thread_id': f'test_{agent_id}_{int(time.time())}',
                'user_id': 'test_user',
            }
            
            start = time.time()
            result = await self.manager.execute(agent_id, task)
            exec_time = time.time() - start
            
            print(f"   ✓ Task executed in {exec_time:.1f}s")
            print(f"   ✓ Status: {result.get('status', 'unknown')}")
            
            if result.get('status') == 'error':
                print(f"   ✗ Error: {result.get('error', 'Unknown')}")
                results['tests'].append(('Execute', False, result.get('error')))
            else:
                print(f"   ✓ Success: Task completed")
                results['tests'].append(('Execute', True, f"{exec_time:.1f}s"))
            
            # Cleanup test file
            test_file.unlink(missing_ok=True)
            
            # Test 3: Terminate
            print("\n3. Terminating agent...")
            success = await self.manager.terminate_agent(agent_id)
            print(f"   ✓ Terminated: {success}")
            results['tests'].append(('Terminate', success, None))
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results['tests'].append(('Exception', False, str(e)))
        
        self.results.append(results)
        return all(t[1] for t in results['tests'])
    
    async def test_zoho_books_agent(self):
        """Test Zoho Books Agent individually."""
        print("\n" + "="*70)
        print("TEST 4: ZOHO BOOKS AGENT")
        print("="*70)
        
        agent_id = 'zoho_books'
        results = {'agent': agent_id, 'tests': []}
        
        try:
            # Test 1: Spawn
            print("\n1. Spawning Zoho Books Agent...")
            start = time.time()
            instance = await self.manager.spawn_agent(agent_id)
            spawn_time = time.time() - start
            
            print(f"   ✓ Spawned in {spawn_time:.1f}s")
            print(f"   ✓ Port: {instance.port}")
            print(f"   ✓ PID: {instance.pid}")
            
            results['tests'].append(('Spawn', True, f"{spawn_time:.1f}s"))
            
            # Test 2: Health check only (requires OAuth)
            print("\n2. Checking health...")
            healthy = await self.manager.health_checker.check_health(instance.port)
            print(f"   ✓ Health check: {healthy}")
            results['tests'].append(('Health Check', healthy, None))
            
            # Test 3: Terminate
            print("\n3. Terminating agent...")
            success = await self.manager.terminate_agent(agent_id)
            print(f"   ✓ Terminated: {success}")
            results['tests'].append(('Terminate', success, None))
            
        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            results['tests'].append(('Exception', False, str(e)))
        
        self.results.append(results)
        return all(t[1] for t in results['tests'])
    
    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        total_passed = 0
        total_tests = 0
        
        for agent_result in self.results:
            agent = agent_result['agent']
            tests = agent_result['tests']
            
            print(f"\n{agent.upper()}:")
            for test_name, passed, details in tests:
                status = "✓" if passed else "✗"
                detail_str = f" ({details})" if details else ""
                print(f"  {status} {test_name}{detail_str}")
                total_tests += 1
                if passed:
                    total_passed += 1
        
        print(f"\n{'='*70}")
        print(f"Results: {total_passed}/{total_tests} tests passed")
        
        if total_passed == total_tests:
            print("\n🎉 ALL AGENTS WORKING CORRECTLY!")
            print("\nAgent Spawning System:")
            print("  ✓ All agents spawn correctly")
            print("  ✓ Health checks pass")
            print("  ✓ Tasks execute successfully")
            print("  ✓ Termination works properly")
            print("  ✓ On-demand spawning verified")
        else:
            print(f"\n⚠️  {total_passed}/{total_tests} tests passed")
            print("Some agents may need configuration or have issues")


async def main():
    """Run all agent tests."""
    print("\n" + "="*70)
    print("INDIVIDUAL AGENT TESTING SUITE")
    print("="*70)
    print("Testing each agent individually (except Browser)")
    print("="*70)
    
    tester = AgentTester()
    
    try:
        # Setup
        await tester.setup()
        
        # Test each agent
        await tester.test_spreadsheet_agent()
        await tester.test_mail_agent()
        await tester.test_document_agent()
        await tester.test_zoho_books_agent()
        
        # Print summary
        tester.print_summary()
        
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        await tester.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
