#!/usr/bin/env python3
"""Simple Gmail MCP test without unicode"""

import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

async def test():
    sys.path.insert(0, 'agents')
    from gmail_mcp_agent import GmailMCPClient
    
    print("Testing Gmail MCP Agent...")
    print()
    
    client = GmailMCPClient()
    print(f"API Key: {client.api_key[:15] if client.api_key else 'NONE'}...")
    print(f"MCP URL: {client.mcp_url[:50] if client.mcp_url else 'NONE'}...")
    print()
    
    print("Fetching unread emails...")
    result = await client.call_tool(
        "GMAIL_FETCH_EMAILS",
        {"query": "is:unread", "max_results": 2, "user_id": "me"}
    )
    
    print(f"Success: {result.get('success')}")
    if result.get('success'):
        print(f"Data: {str(result.get('data'))[:200]}...")
    else:
        print(f"Error: {result.get('error')[:200]}...")
    
    return result.get('success', False)

if __name__ == "__main__":
    result = asyncio.run(test())
    sys.exit(0 if result else 1)
