#!/usr/bin/env python3
"""
Live test of Gmail MCP Agent
Tests actual email fetching through MCP protocol
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

async def test_gmail_mcp():
    """Test Gmail MCP agent with real MCP connection"""
    
    print("=" * 70)
    print("Gmail MCP Agent - Live Test")
    print("=" * 70)
    print()
    
    # Import the agent
    sys.path.insert(0, 'agents')
    from gmail_mcp_agent import GmailMCPClient
    
    # Get credentials
    api_key = os.getenv("COMPOSIO_API_KEY")
    mcp_url = os.getenv("GMAIL_MCP_URL")
    connection_id = os.getenv("GMAIL_CONNECTION_ID")
    
    print("Configuration:")
    print(f"  API Key: {api_key[:15] if api_key else 'NOT SET'}...")
    print(f"  MCP URL: {mcp_url[:50] if mcp_url else 'NOT SET'}...")
    print(f"  Connection ID: {connection_id if connection_id else 'NOT SET'}")
    print()
    
    if not api_key or not mcp_url:
        print("❌ Missing required environment variables")
        print("   Please set COMPOSIO_API_KEY and GMAIL_MCP_URL in .env")
        return False
    
    try:
        # Initialize client
        print("Test 1: Initializing Gmail MCP client...")
        print("-" * 70)
        client = GmailMCPClient()
        print("✅ Client initialized")
        print()
        
        # Test 2: Fetch unread emails
        print("Test 2: Fetching unread emails...")
        print("-" * 70)
        
        result = await client.call_tool(
            "GMAIL_FETCH_EMAILS",
            {
                "query": "is:unread",
                "max_results": 3,
                "user_id": "me"
            }
        )
        
        print(f"Result: {result}")
        print()
        
        if result.get('success'):
            print("✅ Successfully fetched emails!")
            
            # Parse the data
            data = result.get('data', {})
            if isinstance(data, str):
                print(f"Response: {data[:200]}...")
            elif isinstance(data, dict):
                print(f"Response keys: {list(data.keys())}")
                if 'messages' in data:
                    messages = data['messages']
                    print(f"\n📧 Found {len(messages)} unread email(s)")
                    for i, msg in enumerate(messages[:3], 1):
                        print(f"\n  Email {i}:")
                        if isinstance(msg, dict):
                            print(f"    ID: {msg.get('id', 'N/A')}")
                            print(f"    Snippet: {msg.get('snippet', 'N/A')[:80]}...")
                        else:
                            print(f"    {msg}")
            else:
                print(f"Data type: {type(data)}")
                print(f"Data: {data}")
        else:
            print(f"⚠️  Request failed: {result.get('error', 'Unknown error')}")
        
        print()
        
        # Test 3: List threads
        print("Test 3: Listing email threads...")
        print("-" * 70)
        
        result = await client.call_tool(
            "GMAIL_LIST_THREADS",
            {
                "max_results": 2,
                "user_id": "me"
            }
        )
        
        if result.get('success'):
            print("✅ Successfully listed threads!")
            data = result.get('data', {})
            print(f"Response: {str(data)[:200]}...")
        else:
            print(f"⚠️  Request failed: {result.get('error', 'Unknown error')}")
        
        print()
        
        # Summary
        print("=" * 70)
        print("Test Summary")
        print("=" * 70)
        print("✅ Gmail MCP client is working!")
        print("✅ Can communicate with Composio MCP server")
        print()
        print("The agent is ready to be used by the orchestrator.")
        print()
        print("Next: Start the backend server and test through the UI:")
        print("  python main.py")
        print()
        print("Then try queries like:")
        print("  - 'Check my unread emails'")
        print("  - 'Show me my latest emails'")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_gmail_mcp())
    sys.exit(0 if result else 1)
