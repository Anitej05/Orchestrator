#!/usr/bin/env python3
"""
Quick test: Gmail Agent with BaseAgent architecture fixes.

Tests that the agent can:
1. Fetch unread emails without getting stuck in loops
2. Properly finish after single-step operations
3. Handle multi-step workflows correctly

Run:
    cd d:\Internship\Orbimesh
    python backend\tests\test_gmail_quick_live.py
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from backend.agents.gmail_agent.base_agent_impl import GmailAgent
from backend.agents.base.services import AgentServices
from backend.agents.base.types import AgentRequest

USER_ID = "user_374hMFRAc0nkaGdH8XtXNRIdfrk"


async def test_fetch_unread_emails():
    """Test 1: Fetch unread emails - should complete in one step."""
    print("\n" + "="*60)
    print("TEST 1: Fetch unread emails")
    print("="*60)
    
    agent = GmailAgent()
    services = AgentServices.create_default()
    agent = GmailAgent(services=services)
    
    try:
        await agent.initialize()
        
        result = await agent.execute(AgentRequest(
            prompt="Fetch my unread emails from the inbox",
            user_id=USER_ID,
            thread_id="test_fetch_unread"
        ))
        
        print(f"\n✅ Status: {result.status}")
        print(f"📝 Summary: {result.summary}")
        if result.result:
            print(f"📦 Result preview: {str(result.result)[:200]}...")
        
        await agent.terminate()
        return result.status == "success"
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await agent.terminate()
        return False


async def test_search_and_summarize():
    """Test 2: Search emails - multi-step workflow should work."""
    print("\n" + "="*60)
    print("TEST 2: Search and summarize emails")
    print("="*60)
    
    agent = GmailAgent()
    services = AgentServices.create_default()
    agent = GmailAgent(services=services)
    
    try:
        await agent.initialize()
        
        result = await agent.execute(AgentRequest(
            prompt="Search for emails about 'invoice' and tell me what you found",
            user_id=USER_ID,
            thread_id="test_search_summarize"
        ))
        
        print(f"\n✅ Status: {result.status}")
        print(f"📝 Summary: {result.summary}")
        if result.result:
            print(f"📦 Result preview: {str(result.result)[:200]}...")
        
        await agent.terminate()
        return result.status == "success"
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await agent.terminate()
        return False


async def test_send_email_needs_input():
    """Test 3: Send email should ask for missing info (needs_input)."""
    print("\n" + "="*60)
    print("TEST 3: Send email (should ask for recipient)")
    print("="*60)
    
    agent = GmailAgent()
    services = AgentServices.create_default()
    agent = GmailAgent(services=services)
    
    try:
        await agent.initialize()
        
        result = await agent.execute(AgentRequest(
            prompt="Send an email saying 'Hello, this is a test'",
            user_id=USER_ID,
            thread_id="test_send_needs_input"
        ))
        
        print(f"\n✅ Status: {result.status}")
        if result.status == "needs_input":
            print(f"❓ Question: {result.question}")
            print("✅ Correctly asked for missing information!")
        else:
            print(f"📝 Summary: {result.summary}")
        
        await agent.terminate()
        return result.status in ("success", "needs_input")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await agent.terminate()
        return False


async def main():
    print("\n" + "="*60)
    print("GMAIL AGENT - LIVE TEST WITH BASEAGENT FIXES")
    print("="*60)
    print("\nTesting that:")
    print("  1. Single-step operations finish correctly")
    print("  2. Multi-step workflows don't break prematurely")
    print("  3. needs_input bubbles up properly")
    
    results = []
    
    # Test 1: Simple fetch
    results.append(("Fetch unread", await test_fetch_unread_emails()))
    
    # Test 2: Search (may have 0 results, that's ok)
    results.append(("Search emails", await test_search_and_summarize()))
    
    # Test 3: Send with missing info
    results.append(("Send (needs input)", await test_send_email_needs_input()))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + ("="*60))
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  SOME TESTS FAILED")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
