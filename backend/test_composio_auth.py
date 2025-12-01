#!/usr/bin/env python3
"""Test Composio API authentication"""

import os
import httpx
from dotenv import load_dotenv

load_dotenv()

def test_auth():
    api_key = os.getenv("COMPOSIO_API_KEY")
    mcp_url = os.getenv("GMAIL_MCP_URL")
    
    print(f"API Key: {api_key[:15]}...")
    print(f"MCP URL: {mcp_url[:60]}...")
    print()
    
    # Test 1: Try the SSE endpoint with x-api-key header
    print("Test 1: SSE endpoint with x-api-key header")
    sse_url = mcp_url.replace("/mcp?", "/sse?")
    print(f"URL: {sse_url}")
    
    headers = {"x-api-key": api_key}
    print(f"Headers: {headers}")
    
    try:
        response = httpx.get(sse_url, headers=headers, timeout=10.0)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    
    # Test 2: Try with Authorization header
    print("Test 2: SSE endpoint with Authorization header")
    headers = {"Authorization": f"Bearer {api_key}"}
    print(f"Headers: {headers}")
    
    try:
        response = httpx.get(sse_url, headers=headers, timeout=10.0)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")
    
    print()
    
    # Test 3: Try the MCP endpoint
    print("Test 3: MCP endpoint with x-api-key header")
    headers = {"x-api-key": api_key}
    print(f"URL: {mcp_url}")
    print(f"Headers: {headers}")
    
    try:
        response = httpx.get(mcp_url, headers=headers, timeout=10.0)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_auth()
