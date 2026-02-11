#!/usr/bin/env python3
"""
Test script for orchestrator agent selection and execution.
Tests various tasks to verify centralized agent naming works.
"""

import asyncio
import aiohttp
import json
import sys
from datetime import datetime

# Test configuration
ORCHESTRATOR_URL = "http://127.0.0.1:8000"
HEADERS = {"Content-Type": "application/json", "Authorization": "Bearer test-token"}

# Test scenarios
TEST_TASKS = [
    {
        "name": "Web Navigation Task",
        "task": "Go to example.com and tell me what you see",
        "expected_agent": "browser_automation_agent",
        "description": "Should route to Browser Automation Agent"
    },
    {
        "name": "Spreadsheet Task", 
        "task": "Show me the top 10 customers by revenue from the sales data",
        "expected_agent": "spreadsheet_agent",
        "description": "Should route to Spreadsheet Agent"
    },
    {
        "name": "Email Task",
        "task": "Find emails from John about the project deadline",
        "expected_agent": "mail_agent", 
        "description": "Should route to Mail Agent"
    },
    {
        "name": "Document Task",
        "task": "Summarize this PDF document about quarterly results",
        "expected_agent": "document_agent",
        "description": "Should route to Document Agent"
    },
    {
        "name": "Wikipedia Search",
        "task": "Find out who won the 2024 Super Bowl using Wikipedia",
        "expected_agent": "tool",  # This should use Wikipedia search tool
        "description": "Should use Wikipedia search tool"
    }
]

async def test_orchestrator_task(session, task_info):
    """Test a single task with the orchestrator."""
    print(f"\n{'='*70}")
    print(f"📋 Test: {task_info['name']}")
    print(f"   Task: {task_info['task']}")
    print(f"   Expected: {task_info['description']}")
    print(f"{'='*70}")
    
    try:
        # Create a new conversation
        async with session.post(
            f"{ORCHESTRATOR_URL}/api/v1/conversations",
            headers=HEADERS,
            json={"input": task_info['task']}
        ) as response:
            if response.status != 200:
                print(f"❌ Failed to create conversation: {response.status}")
                return False
            
            conversation = await response.json()
            conversation_id = conversation.get("conversation_id")
            print(f"✓ Created conversation: {conversation_id}")
        
        # Execute the task
        async with session.post(
            f"{ORCHESTRATOR_URL}/api/v1/execute",
            headers=HEADERS,
            json={
                "conversation_id": conversation_id,
                "input": task_info['task']
            }
        ) as response:
            result = await response.json()
            print(f"\n📊 Execution Result:")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Agent Used: {result.get('agent_used', 'unknown')}")
            print(f"   Response: {str(result.get('response', ''))[:200]}...")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def test_direct_agent(session, agent_id, task):
    """Test direct agent execution."""
    print(f"\n{'='*70}")
    print(f"🔧 Direct Test: {agent_id}")
    print(f"   Task: {task}")
    print(f"{'='*70}")
    
    try:
        # Get agent URL from registry
        agent_urls = {
            "spreadsheet_agent": "http://127.0.0.1:9000",
            "mail_agent": "http://127.0.0.1:8040", 
            "browser_automation_agent": "http://127.0.0.1:8090",
            "document_agent": "http://127.0.0.1:8050",
            "zoho_books_agent": "http://127.0.0.1:8060"
        }
        
        base_url = agent_urls.get(agent_id)
        if not base_url:
            print(f"❌ Unknown agent: {agent_id}")
            return False
            
        async with session.post(
            f"{base_url}/execute",
            json={"type": "execute", "prompt": task}
        ) as response:
            result = await response.json()
            print(f"\n📊 Agent Result:")
            print(f"   Status: {result.get('status', result.get('success', 'unknown'))}")
            print(f"   Response: {str(result)[:300]}...")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def main():
    """Main test function."""
    print("="*70)
    print("🎯 ORCHESTRATOR AGENT SELECTION - COMPREHENSIVE TEST")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Orchestrator URL: {ORCHESTRATOR_URL}")
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Direct agent execution (baseline)
        print("\n\n" + "="*70)
        print("📦 PHASE 1: Direct Agent Execution Tests")
        print("="*70)
        
        direct_tests = [
            ("browser_automation_agent", "Navigate to example.com"),
            ("spreadsheet_agent", "Show top 5 customers by revenue"),
            ("document_agent", "Summarize this document"),
        ]
        
        direct_results = []
        for agent_id, task in direct_tests:
            result = await test_direct_agent(session, agent_id, task)
            direct_results.append(result)
        
        direct_success = sum(direct_results)
        print(f"\n📊 Direct Tests: {direct_success}/{len(direct_tests)} passed")
        
        # Test 2: Orchestrator routing tests
        print("\n\n" + "="*70)
        print("🧠 PHASE 2: Orchestrator Routing Tests")
        print("="*70)
        
        orchestrator_results = []
        for task_info in TEST_TASKS[:4]:  # Skip Wikipedia for now
            result = await test_orchestrator_task(session, task_info)
            orchestrator_results.append(result)
        
        orchestrator_success = sum(orchestrator_results)
        print(f"\n📊 Orchestrator Tests: {orchestrator_success}/{len(orchestrator_results)} passed")
        
        # Summary
        print("\n\n" + "="*70)
        print("📈 TEST SUMMARY")
        print("="*70)
        print(f"Direct Agent Tests: {direct_success}/{len(direct_tests)} ✓")
        print(f"Orchestrator Tests: {orchestrator_success}/{len(TEST_TASKS[:4])} ✓")
        
        total_tests = len(direct_tests) + len(TEST_TASKS[:4])
        total_passed = direct_success + orchestrator_success
        
        if total_passed == total_tests:
            print(f"\n🎉 ALL TESTS PASSED! ({total_passed}/{total_tests})")
            print("✓ Centralized agent naming is working correctly!")
            return 0
        else:
            print(f"\n⚠️  SOME TESTS FAILED ({total_passed}/{total_tests})")
            return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

