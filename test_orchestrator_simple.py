#!/usr/bin/env python3
"""
Simple test script for orchestrator agent selection.
"""

import asyncio
import aiohttp
import json
import sys
from datetime import datetime

ORCHESTRATOR_URL = "http://127.0.0.1:8000"
HEADERS = {"Content-Type": "application/json"}

async def test_chat_endpoint(session, task, test_name):
    """Test the /api/chat endpoint."""
    print(f"\n{'='*70}")
    print(f"📋 Test: {test_name}")
    print(f"   Task: {task}")
    print(f"{'='*70}")
    
    try:
        async with session.post(
            f"{ORCHESTRATOR_URL}/api/chat",
            headers=HEADERS,
            json={"input": task}
        ) as response:
            result = await response.json()
            
            print(f"\n📊 Response:")
            print(f"   Status: {response.status}")
            print(f"   Agent Used: {result.get('agent_used', 'unknown')}")
            
            # Check if it's a final response or needs continuation
            status = result.get('status', '')
            if 'input' in result:
                print(f"   Result: {str(result.get('input', ''))[:200]}...")
            elif 'output' in result:
                print(f"   Result: {str(result.get('output', ''))[:200]}...")
            
            print(f"   Full response keys: {list(result.keys())}")
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
            print(f"   ✓ {agent_name}: {result.get('status', 'unknown')}")
            return True
    except Exception as e:
        print(f"   ✗ {agent_name}: {e}")
        return False

async def main():
    """Main test function."""
    print("="*70)
    print("🎯 ORCHESTRATOR AGENT SELECTION TEST")
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
        
        # Test 2: Test orchestrator routing
        print("\n\n🧠 PHASE 2: Orchestrator Routing Tests")
        print("="*70)
        
        test_tasks = [
            ("Browser Task", "Go to example.com and tell me what you see"),
            ("Spreadsheet Task", "Analyze the sales spreadsheet and show top customers"),
            ("Document Task", "Summarize the quarterly report PDF"),
            ("General Task", "What can you help me with?"),
        ]
        
        routing_results = []
        for test_name, task in test_tasks:
            result = await test_chat_endpoint(session, task, test_name)
            routing_results.append(result)
        
        routing_success = sum(routing_results)
        print(f"\n📊 Routing Tests: {routing_success}/{len(test_tasks)} ✓")
        
        # Summary
        print("\n\n" + "="*70)
        print("📈 TEST SUMMARY")
        print("="*70)
        total = len(agents) + len(test_tasks)
        passed = health_success + routing_success
        
        print(f"Agent Health Checks: {health_success}/{len(agents)} ✓")
        print(f"Routing Tests: {routing_success}/{len(test_tasks)} ✓")
        print(f"Total: {passed}/{total}")
        
        if passed == total:
            print(f"\n🎉 ALL TESTS PASSED!")
            return 0
        else:
            print(f"\n⚠️  SOME TESTS NEED ATTENTION")
            return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

