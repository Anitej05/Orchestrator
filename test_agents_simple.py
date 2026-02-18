"""
Simple Agent Spawn Test - No Unicode
Tests if agents can be spawned
"""

import asyncio
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def test_agent(agent_id):
    """Test spawning a single agent."""
    print(f"\nTesting {agent_id} agent...")
    print("-" * 50)
    
    from backend.services.agent_manager import AgentManager
    
    manager = AgentManager()
    await manager.initialize()
    
    try:
        print(f"1. Spawning {agent_id}...")
        start = time.time()
        instance = await manager.spawn_agent(agent_id)
        spawn_time = time.time() - start
        
        print(f"   [OK] Spawned in {spawn_time:.1f}s")
        print(f"   Port: {instance.port}")
        print(f"   PID: {instance.pid}")
        print(f"   Healthy: {instance.healthy}")
        
        print(f"2. Checking health...")
        healthy = await manager.health_checker.check_health(instance.port)
        print(f"   [OK] Health: {healthy}")
        
        print(f"3. Terminating...")
        success = await manager.terminate_agent(agent_id)
        print(f"   [OK] Terminated: {success}")
        
        print(f"   [PASS] {agent_id} agent working!")
        return True
        
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        await manager.shutdown()


async def main():
    """Test all agents."""
    print("="*70)
    print("AGENT SPAWN TESTING (No Browser)")
    print("="*70)
    
    agents = ['spreadsheet', 'mail', 'document', 'zoho_books']
    results = {}
    
    for agent in agents:
        try:
            success = await test_agent(agent)
            results[agent] = success
        except Exception as e:
            print(f"   [FAIL] Fatal error: {e}")
            results[agent] = False
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for agent, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {agent}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nResults: {passed}/{total} agents working")
    
    if passed == total:
        print("\n[OK] All agents spawn correctly!")


if __name__ == "__main__":
    asyncio.run(main())
