#!/usr/bin/env python3
"""
Final Gmail MCP Agent Test
Verifies everything is set up correctly
"""

import os
import sys
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def test_setup():
    """Test that everything is configured correctly"""
    
    print("=" * 70)
    print("Gmail MCP Agent - Final Setup Verification")
    print("=" * 70)
    print()
    
    all_good = True
    
    # Test 1: Environment variables
    print("✓ Test 1: Environment Variables")
    print("-" * 70)
    
    api_key = os.getenv("COMPOSIO_API_KEY")
    mcp_url = os.getenv("GMAIL_MCP_URL")
    connection_id = os.getenv("GMAIL_CONNECTION_ID")
    
    if api_key:
        print(f"  ✅ COMPOSIO_API_KEY: {api_key[:15]}...")
    else:
        print(f"  ❌ COMPOSIO_API_KEY: Not set")
        all_good = False
    
    if mcp_url:
        print(f"  ✅ GMAIL_MCP_URL: {mcp_url[:50]}...")
    else:
        print(f"  ❌ GMAIL_MCP_URL: Not set")
        all_good = False
    
    if connection_id:
        print(f"  ✅ GMAIL_CONNECTION_ID: {connection_id}")
    else:
        print(f"  ⚠️  GMAIL_CONNECTION_ID: Not set (optional)")
    
    print()
    
    # Test 2: Agent definition file
    print("✓ Test 2: Agent Definition")
    print("-" * 70)
    
    agent_json = Path("Agent_entries/gmail_mcp_agent.json")
    if agent_json.exists():
        print(f"  ✅ Agent JSON exists: {agent_json}")
        
        import json
        with open(agent_json) as f:
            data = json.load(f)
            print(f"  ✅ Agent ID: {data['id']}")
            print(f"  ✅ Agent Name: {data['name']}")
            print(f"  ✅ Agent Type: {data['agent_type']}")
            print(f"  ✅ Endpoints: {len(data['endpoints'])}")
            print(f"  ✅ Capabilities: {len(data['capabilities'])}")
    else:
        print(f"  ❌ Agent JSON not found: {agent_json}")
        all_good = False
    
    print()
    
    # Test 3: Database registration
    print("✓ Test 3: Database Registration")
    print("-" * 70)
    
    try:
        from database import SessionLocal
        from models import Agent
        
        db = SessionLocal()
        try:
            agent = db.query(Agent).filter_by(id='gmail_mcp_agent').first()
            
            if agent:
                print(f"  ✅ Agent registered in database")
                print(f"  ✅ Name: {agent.name}")
                print(f"  ✅ Type: {agent.agent_type}")
                print(f"  ✅ Endpoints: {len(agent.endpoints)}")
                print(f"  ✅ Capabilities: {len(agent.capabilities)}")
                
                # Show sample capabilities
                print(f"\n  Sample capabilities:")
                for cap in agent.capabilities[:5]:
                    print(f"    - {cap}")
                if len(agent.capabilities) > 5:
                    print(f"    ... and {len(agent.capabilities) - 5} more")
                
                # Show sample endpoints
                print(f"\n  Sample endpoints:")
                for ep in agent.endpoints[:3]:
                    print(f"    - {ep.endpoint} ({ep.http_method})")
                if len(agent.endpoints) > 3:
                    print(f"    ... and {len(agent.endpoints) - 3} more")
            else:
                print(f"  ❌ Agent not found in database")
                print(f"  Run: python manage.py sync")
                all_good = False
        finally:
            db.close()
            
    except Exception as e:
        print(f"  ❌ Database error: {e}")
        all_good = False
    
    print()
    
    # Test 4: Dependencies
    print("✓ Test 4: Dependencies")
    print("-" * 70)
    
    try:
        import composio
        print(f"  ✅ composio: {composio.__version__}")
    except ImportError:
        print(f"  ❌ composio: Not installed")
        all_good = False
    
    try:
        import mcp
        print(f"  ✅ mcp: Installed")
    except ImportError:
        print(f"  ❌ mcp: Not installed")
        all_good = False
    
    try:
        import httpx
        print(f"  ✅ httpx: Installed")
    except ImportError:
        print(f"  ❌ httpx: Not installed")
        all_good = False
    
    print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    if all_good:
        print("✅ All tests passed!")
        print()
        print("The Gmail MCP agent is fully configured and ready to use.")
        print()
        print("Next steps:")
        print("1. Start the backend server:")
        print("   python main.py")
        print()
        print("2. Open your frontend and try these queries:")
        print("   - 'Check my unread emails'")
        print("   - 'Show me my latest 5 emails'")
        print("   - 'Search for emails from john@example.com'")
        print("   - 'Do I have any new messages?'")
        print()
        print("The orchestrator will automatically:")
        print("  • Parse your query")
        print("  • Match it to the Gmail agent via semantic search")
        print("  • Call the appropriate Gmail operation")
        print("  • Return the results")
        print()
    else:
        print("❌ Some tests failed")
        print()
        print("Please fix the issues above and run this test again.")
        print()
        print("Common fixes:")
        print("1. Missing env vars: Add to backend/.env")
        print("2. Agent not in DB: Run 'python manage.py sync'")
        print("3. Missing deps: Run 'pip install composio mcp httpx-sse'")
    
    print("=" * 70)
    
    return all_good

if __name__ == "__main__":
    result = test_setup()
    sys.exit(0 if result else 1)
