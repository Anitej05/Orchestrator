#!/usr/bin/env python3
"""
Quick test: Document Agent with BaseAgent architecture fixes.

Tests that the agent can:
1. Create a document without getting stuck in loops
2. Properly finish after completing the task
3. Handle multi-step workflows (create + edit + save)
4. Not break prematurely on "terminal" operations

Run:
    cd d:\Internship\Orbimesh
    python backend\tests\test_document_agent_quick.py
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from backend.agents.document_agent_lib.base_agent_impl import DocumentAgent
from backend.agents.base.services import AgentServices
from backend.agents.base.types import AgentRequest


async def test_create_document():
    """Test 1: Create a simple document - should complete successfully."""
    print("\n" + "="*60)
    print("TEST 1: Create a new document")
    print("="*60)
    
    agent = DocumentAgent()
    
    try:
        await agent.initialize()
        
        result = await agent.execute(AgentRequest(
            prompt="Create a new document titled 'Test Report' with the content 'Hello, this is a test document created by the agent.'",
            thread_id="test_create_doc"
        ))
        
        print(f"\n✅ Status: {result.status}")
        print(f"📝 Summary: {result.summary}")
        if result.result:
            print(f"📦 Result preview: {str(result.result)[:300]}...")
        if result.canvas_display:
            print(f"🎨 Canvas: {result.canvas_display.get('type', 'unknown')}")
        
        await agent.terminate()
        return result.status == "success"
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await agent.terminate()
        return False


async def test_create_and_edit():
    """Test 2: Create document then add content - multi-step workflow."""
    print("\n" + "="*60)
    print("TEST 2: Create document and add section")
    print("="*60)
    
    agent = DocumentAgent()
    
    try:
        await agent.initialize()
        
        result = await agent.execute(AgentRequest(
            prompt="Create a document titled 'Meeting Notes' with a title, then add a section called 'Action Items' with 3 bullet points",
            thread_id="test_create_edit"
        ))
        
        print(f"\n✅ Status: {result.status}")
        print(f"📝 Summary: {result.summary}")
        if result.result:
            print(f"📦 Result preview: {str(result.result)[:300]}...")
        if result.canvas_display:
            print(f"🎨 Canvas: {result.canvas_display.get('type', 'unknown')}")
        
        await agent.terminate()
        return result.status == "success"
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await agent.terminate()
        return False


async def test_summarize_document():
    """Test 3: Create then summarize - verifies multi-step doesn't break."""
    print("\n" + "="*60)
    print("TEST 3: Create document and summarize it")
    print("="*60)
    
    agent = DocumentAgent()
    
    try:
        await agent.initialize()
        
        result = await agent.execute(AgentRequest(
            prompt="Create a short document about 'Benefits of Exercise' with 2 paragraphs, then provide a summary of what you created",
            thread_id="test_summarize"
        ))
        
        print(f"\n✅ Status: {result.status}")
        print(f"📝 Summary: {result.summary}")
        if result.result:
            print(f"📦 Result preview: {str(result.result)[:300]}...")
        
        await agent.terminate()
        return result.status == "success"
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await agent.terminate()
        return False


async def test_simple_query():
    """Test 4: Simple informational query - should finish in one step."""
    print("\n" + "="*60)
    print("TEST 4: Simple query (what can you do?)")
    print("="*60)
    
    agent = DocumentAgent()
    
    try:
        await agent.initialize()
        
        result = await agent.execute(AgentRequest(
            prompt="What types of documents can you create?",
            thread_id="test_simple"
        ))
        
        print(f"\n✅ Status: {result.status}")
        print(f"📝 Summary: {result.summary}")
        if result.result:
            print(f"📦 Result: {str(result.result)[:300]}...")
        
        await agent.terminate()
        return result.status == "success"
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        await agent.terminate()
        return False


async def main():
    print("\n" + "="*60)
    print("DOCUMENT AGENT - LIVE TEST WITH BASEAGENT FIXES")
    print("="*60)
    print("\nTesting that:")
    print("  1. Single-step operations finish correctly")
    print("  2. Multi-step workflows don't break prematurely")
    print("  3. Agent doesn't loop infinitely")
    print("  4. LLM controls finish decision (not hardcoded rules)")
    
    results = []
    
    # Test 1: Simple create
    results.append(("Create document", await test_create_document()))
    
    # Test 2: Multi-step create + edit
    results.append(("Create + edit", await test_create_and_edit()))
    
    # Test 3: Create + summarize (multi-step)
    results.append(("Create + summarize", await test_summarize_document()))
    
    # Test 4: Simple query
    results.append(("Simple query", await test_simple_query()))
    
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
