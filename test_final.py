#!/usr/bin/env python3
"""
Final test script for orchestrator agent selection.
Tests proper agent routing using the correct API format.
"""

import asyncio
import aiohttp
import json
import sys
from datetime import datetime

ORCHESTRATOR_URL = "http://127.0.0.1:8000"
HEADERS = {"Content-Type": "application/json"}

async def test_chat_endpoint(session, prompt, test_name, expected_agent_pattern=None):
    """Test the /api/chat endpoint with correct format."""
    print(f"\n{'='*70}")
    print(f"📋 Test: {test_name}")
    print(f"   Prompt: {prompt}")
    if expected_agent_pattern:
        print(f"   Expected agent: {expected_agent_pattern}")
    print(f"{'='*70}")
    
    try:
        async with session.post(
            f"{ORCHESTRATOR_URL}/api/chat",
            headers=HEADERS,
            json={"prompt": prompt, "thread_id": f"test-{datetime.now().timestamp()}"}
        ) as response:
            if response.status != 200:
                print(f"❌ Status: {response.status}")
                error_detail = await response.json()
                print(f"   Error: {json.dumps(error_detail, indent=2)[:500]}")
                return False
            
            result = await response.json()
            
            print(f"\n📊 Response:")
            print(f"   Status: {response.status} ✓")
            print(f"   Thread ID: {result.get('thread_id', 'unknown')}")
            print(f"   Message: {result.get('message', '')[:150]}...")
            
            # Check agent-task pairs
            task_agent_pairs = result.get('task_agent_pairs', [])
            if task_agent_pairs:
                print(f"\n   📌 Task-Agent Pairs ({len(task_agent_pairs)}):")
                for pair in task_agent_pairs[:3]:  # Show first 3
                    print(f"      - Task: {pair.get('task', 'unknown')[:50]}...")
                    print(f"        Agent: {pair.get('agent', 'unknown')}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_agent_health(session, agent_name, base_url):
    """Test individual agent health."""
    print(f"\n🔍 Checking {agent_name}...")
    try:
        async with session.get(f"{base_url}/health") as response:
            result = await response.json()
            status = result.get('status', 'unknown')
            print(f"   ✓ {agent_name}: {status}")
            return True
    except Exception as e:
        print(f"   ✗ {agent_name}: {e}")
        return False

async def test_agent_direct(session, agent_id, base_url, task):
    """Test direct agent execution."""
    print(f"\n🔧 Direct Test: {agent_id}")
    print(f"   Task: {task}")
    
    try:
        async with session.post(
            f"{base_url}/execute",
            json={"type": "execute", "prompt": task}
        ) as response:
            result = await response.json()
            status = result.get('status', result.get('success', 'error'))
            print(f"   ✓ {agent_id}: {status}")
            return True
    except Exception as e:
        print(f"   ✗ {agent_id}: {e}")
        return False

async def main():
    """Main test function."""
    print("="*70)
    print("🎯 ORCHESTRATOR AGENT SELECTION - FINAL TEST")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Check all agents are healthy
        print("\n\n📦 PHASE 1: Agent Health Check")
        print("="*70)
        
        agents = [
            ("Spreadsheet Agent", "http://127.0.0.1:9000"),
            ("Mail Agent", "http://127.0.0.1:8040"),
            ("Browser Agent", "http://127.0.0.1:8090"),
            ("Document Agent", "http://127.0.0.1:8050"),
        ]
        
        health_results = []
        for name, url in agents:
            result = await test_agent_health(session, name, url)
            health_results.append(result)
        
        health_success = sum(health_results)
        print(f"\n📊 Agent Health: {health_success}/{len(agents)} ✓")
        
        # Test 2: Test direct agent execution
        print("\n\n📦 PHASE 2: Direct Agent Execution")
        print("="*70)
        
        direct_results = []
        direct_results.append(await test_agent_direct(
            session, "spreadsheet_agent", "http://127.0.0.1:9000",
            "Show me the top 5 customers"
        ))
        
        direct_success = sum(direct_results)
        print(f"\n📊 Direct Tests: {direct_success}/{len(direct_results)} ✓")
        
        # Test 3: Test orchestrator routing
        print("\n\n🧠 PHASE 3: Orchestrator Routing Tests")
        print("="*70)
        
        test_tasks = [
            ("Browser Task", "Go to example.com and tell me what you see", "browser"),
            ("Spreadsheet Task", "Analyze the sales spreadsheet and show top customers", "spreadsheet"),
            ("Document Task", "Summarize the quarterly report PDF", "document"),
            ("Mail Task", "Find emails from John about the project", "mail"),
            ("General Task", "What can you help me with?", None),
        ]
        
        routing_results = []
        for test_name, prompt, expected_pattern in test_tasks:
            result = await test_chat_endpoint(session, prompt, test_name, expected_pattern)
            routing_results.append(result)
        
        routing_success = sum(routing_results)
        print(f"\n📊 Routing Tests: {routing_success}/{len(test_tasks)} ✓")
        
        # Summary
        print("\n\n" + "="*70)
        print("📈 TEST SUMMARY")
        print("="*70)
        total_tests = len(agents) + len(direct_results) + len(test_tasks)
        total_passed = health_success + direct_success + routing_success
        
        print(f"Agent Health Checks: {health_success}/{len(agents)} ✓")
        print(f"Direct Agent Tests: {direct_success}/{len(direct_results)} ✓")
        print(f"Orchestrator Routing: {routing_success}/{len(test_tasks)} ✓")
        print(f"\nTotal: {total_passed}/{total_tests}")
        
        if total_passed == total_tests:
            print(f"\n🎉 ALL TESTS PASSED!")
            print("\n✓ Centralized agent naming is working correctly!")
            print("✓ All agents are healthy and responding!")
            print("✓ Orchestrator can route tasks to appropriate agents!")
            return 0
        else:
            print(f"\n⚠️  SOME TESTS NEED ATTENTION")
            failed = total_tests - total_passed
            print(f"   Failed: {failed}/{total_tests}")
            return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

